#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import traceback
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.io import loadmat

EPS = 1e-8


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        a = np.asarray(x).squeeze()
        if a.size == 0:
            return default
        return float(a.flat[0])
    except Exception:
        return default


def field(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, np.void) and obj.dtype.names and name in obj.dtype.names:
        return obj[name]
    if isinstance(obj, np.ndarray) and obj.dtype.names and name in obj.dtype.names:
        return obj[name]
    return default


def as_items(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.ndarray):
        return list(value.ravel())
    return [value]


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    a = np.asarray(value).squeeze()
    if a.size == 0:
        return ""
    return str(a.flat[0])


def as_float_array(value: Any) -> np.ndarray:
    if value is None:
        return np.asarray([], dtype=float)
    try:
        return np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        out = []
        for item in np.asarray(value, dtype=object).reshape(-1):
            try:
                out.append(float(np.asarray(item).squeeze()))
            except Exception:
                pass
        return np.asarray(out, dtype=float)


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def logit(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=float), 1e-5, 1 - 1e-5)
    return np.log(x / (1 - x))


def geometric_mean(values: Iterable[float]) -> float:
    a = np.asarray(list(values), dtype=float)
    a = a[np.isfinite(a) & (a > 0)]
    if len(a) == 0:
        return float("inf")
    return float(np.exp(np.mean(np.log(a))))


def unique_sorted_xy(x: np.ndarray, *ys: np.ndarray) -> tuple[np.ndarray, ...]:
    order = np.argsort(x, kind="mergesort")
    x = np.asarray(x, float)[order]
    ysort = [np.asarray(y, float)[order] for y in ys]
    ux, inverse = np.unique(x, return_inverse=True)
    if len(ux) == len(x):
        return (x, *ysort)
    agg = []
    for y in ysort:
        sums = np.zeros(len(ux), float)
        counts = np.zeros(len(ux), float)
        np.add.at(sums, inverse, y)
        np.add.at(counts, inverse, 1.0)
        agg.append(sums / np.maximum(counts, 1.0))
    return (ux, *agg)


@dataclass
class Cycle:
    cell: str
    discharge_order: int
    source_cycle_index: int
    ambient_temperature: float
    mean_abs_current: float
    q: np.ndarray
    voltage: np.ndarray
    temperature: np.ndarray

    @property
    def q_end(self) -> float:
        return float(self.q[-1])


def extract_member(archive: Path, cell: str, destination: Path) -> dict[str, Any]:
    target_name = f"{cell}.mat".lower()
    with zipfile.ZipFile(archive) as z:
        matches = [n for n in z.namelist() if Path(n).name.lower() == target_name]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one {cell}.mat member, found {matches}")
        payload = z.read(matches[0])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    return {
        "cell": cell,
        "archive_member": matches[0],
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def load_cell_cycles(mat_path: Path, cell: str, protocol: dict[str, Any]) -> list[Cycle]:
    loaded = loadmat(mat_path, simplify_cells=True)
    root = loaded.get(cell)
    if root is None:
        candidates = [k for k in loaded if not k.startswith("__")]
        if len(candidates) != 1:
            raise KeyError(f"Cannot locate {cell} in {mat_path}; keys={candidates}")
        root = loaded[candidates[0]]
    raw_cycles = as_items(field(root, "cycle"))
    cfg = protocol["cycle_selection"]
    valid: list[Cycle] = []
    discharge_order = 0
    for source_index, cyc in enumerate(raw_cycles):
        if as_text(field(cyc, "type")).strip().lower() != "discharge":
            continue
        data = field(cyc, "data")
        time = as_float_array(field(data, "Time"))
        current = as_float_array(field(data, "Current_measured"))
        voltage = as_float_array(field(data, "Voltage_measured"))
        temperature = as_float_array(field(data, "Temperature_measured"))
        n = min(len(time), len(current), len(voltage), len(temperature))
        if n < int(cfg["minimum_raw_points"]):
            discharge_order += 1
            continue
        time, current, voltage, temperature = time[:n], current[:n], voltage[:n], temperature[:n]
        finite = np.isfinite(time) & np.isfinite(current) & np.isfinite(voltage) & np.isfinite(temperature)
        time, current, voltage, temperature = time[finite], current[finite], voltage[finite], temperature[finite]
        if len(time) < int(cfg["minimum_raw_points"]):
            discharge_order += 1
            continue
        order = np.argsort(time, kind="mergesort")
        time, current, voltage, temperature = time[order], current[order], voltage[order], temperature[order]
        dt = np.maximum(np.diff(time), 0.0)
        dq = 0.5 * (np.abs(current[1:]) + np.abs(current[:-1])) * dt / 3600.0
        q = np.r_[0.0, np.cumsum(dq)]
        q, voltage, temperature, current = unique_sorted_xy(q, voltage, temperature, current)
        if len(q) < int(cfg["minimum_raw_points"]):
            discharge_order += 1
            continue
        if q[-1] - q[0] < float(cfg["minimum_charge_span_Ah"]):
            discharge_order += 1
            continue
        if np.ptp(voltage) < float(cfg["minimum_voltage_range_V"]):
            discharge_order += 1
            continue
        # Bound computation without changing physical coordinates.
        if len(q) > 900:
            keep = np.unique(np.round(np.linspace(0, len(q) - 1, 900)).astype(int))
            q, voltage, temperature, current = q[keep], voltage[keep], temperature[keep], current[keep]
        valid.append(Cycle(
            cell=cell,
            discharge_order=discharge_order,
            source_cycle_index=source_index,
            ambient_temperature=safe_float(field(cyc, "ambient_temperature")),
            mean_abs_current=float(np.mean(np.abs(current))),
            q=q.astype(float),
            voltage=voltage.astype(float),
            temperature=temperature.astype(float),
        ))
        discharge_order += 1
    limit = int(cfg["maximum_cycles_per_cell"])
    if len(valid) > limit:
        idx = np.unique(np.round(np.linspace(0, len(valid) - 1, limit)).astype(int))
        valid = [valid[i] for i in idx]
    return valid


def mechanism_from_curves(curves: list[np.ndarray], grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    residuals = []
    template = None
    for y in curves:
        y = np.asarray(y, float)
        chord = y[0] + (y[-1] - y[0]) * grid
        r = y - chord
        r -= np.mean(r)
        norm = float(np.sqrt(np.mean(r * r)))
        if not np.isfinite(norm) or norm < 1e-8:
            continue
        r = r / norm
        if template is not None and float(np.dot(r, template)) < 0:
            r = -r
        residuals.append(r)
        template = r if template is None else np.median(np.vstack(residuals), axis=0)
    if len(residuals) < 3:
        raise RuntimeError("Too few nondegenerate projective residual curves")
    h0 = np.median(np.vstack(residuals), axis=0)
    h0 -= np.mean(h0)
    h0 /= max(float(np.sqrt(np.mean(h0 * h0))), 1e-8)
    x = logit(np.clip(grid, 1e-4, 1 - 1e-4))
    h1 = np.gradient(h0, x)
    h1 -= np.mean(h1)
    h1 -= float(np.dot(h1, h0) / max(np.dot(h0, h0), 1e-12)) * h0
    h1 /= max(float(np.sqrt(np.mean(h1 * h1))), 1e-8)
    correlations = [abs(float(np.corrcoef(r, h0)[0, 1])) for r in residuals]
    return h0, h1, {
        "curve_count": len(residuals),
        "median_projective_correlation": float(np.median(correlations)),
        "minimum_projective_correlation": float(np.min(correlations)),
    }


def build_library(cycles: list[Cycle], protocol: dict[str, Any]) -> dict[str, Any]:
    mcfg = protocol["mechanism_library"]
    grid = np.linspace(float(mcfg["grid_min"]), float(mcfg["grid_max"]), int(mcfg["grid_points"]))
    voltage_curves, temperature_curves = [], []
    for cyc in cycles:
        s = np.clip(cyc.q / max(cyc.q_end, EPS), 0.0, 1.0)
        voltage_curves.append(np.interp(grid, s, cyc.voltage))
        temperature_curves.append(np.interp(grid, s, cyc.temperature))
    hv0, hv1, vstats = mechanism_from_curves(voltage_curves, grid)
    ht0, ht1, tstats = mechanism_from_curves(temperature_curves, grid)
    return {
        "grid": grid,
        "voltage_h0": hv0,
        "voltage_h1": hv1,
        "temperature_h0": ht0,
        "temperature_h1": ht1,
        "voltage_stats": vstats,
        "temperature_stats": tstats,
        "training_cycle_count": len(cycles),
        "training_cells": sorted({c.cell for c in cycles}),
    }


def mechanism_eval(grid: np.ndarray, h0: np.ndarray, h1: np.ndarray, phase: np.ndarray, tau: float) -> np.ndarray:
    return np.interp(np.clip(phase, 0.0, 1.0), grid, h0) + float(tau) * np.interp(np.clip(phase, 0.0, 1.0), grid, h1)


def ridge_fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    penalty = np.diag([1e-10, 1e-7, 1e-5])
    try:
        return np.linalg.solve(X.T @ X + penalty, X.T @ y)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(X, y, rcond=None)[0]


def probe_indices(n: int, fractions: list[float]) -> np.ndarray:
    raw = np.round(np.asarray(fractions, float) * (n - 1)).astype(int)
    raw = np.clip(raw, 0, n - 1)
    chosen: list[int] = []
    for idx in raw:
        candidate = int(idx)
        if candidate in chosen:
            for radius in range(1, n):
                options = [candidate - radius, candidate + radius]
                found = next((x for x in options if 0 <= x < n and x not in chosen), None)
                if found is not None:
                    candidate = found
                    break
        chosen.append(candidate)
    return np.asarray(sorted(chosen), int)


def capacity_from_probes(q_probe: np.ndarray, fractions: np.ndarray) -> float:
    usable = fractions > 0.02
    estimates = q_probe[usable] / fractions[usable]
    estimates = estimates[np.isfinite(estimates) & (estimates > 0)]
    if len(estimates) == 0:
        return max(float(np.max(q_probe)), EPS)
    return max(float(np.median(estimates)), float(np.max(q_probe)) * 1.001, EPS)


def fit_local_kind(qp: np.ndarray, yp: np.ndarray, q_all: np.ndarray, kind: str) -> np.ndarray:
    qp, yp = unique_sorted_xy(np.asarray(qp, float), np.asarray(yp, float))
    if len(qp) < 2:
        return np.full_like(q_all, float(yp[0]) if len(yp) else float("nan"), dtype=float)
    if kind == "pchip":
        try:
            return np.asarray(PchipInterpolator(qp, yp, extrapolate=True)(q_all), float)
        except Exception:
            kind = "poly3"
    if kind == "poly3":
        center = float(np.mean(qp))
        scale = max(float(np.ptp(qp)), EPS)
        x = (qp - center) / scale
        xa = (q_all - center) / scale
        degree = min(3, len(qp) - 1)
        X = np.vander(x, degree + 1, increasing=True)
        ridge = np.eye(degree + 1) * 1e-8
        ridge[0, 0] = 1e-12
        coef = np.linalg.solve(X.T @ X + ridge, X.T @ yp)
        return np.vander(xa, degree + 1, increasing=True) @ coef
    raise ValueError(kind)


def choose_local_kind(qp: np.ndarray, yp: np.ndarray, scale: float) -> str:
    scores = {}
    for kind in ("pchip", "poly3"):
        errors = []
        for j in range(len(qp)):
            keep = np.arange(len(qp)) != j
            pred = fit_local_kind(qp[keep], yp[keep], np.asarray([qp[j]]), kind)[0]
            errors.append(((pred - yp[j]) / scale) ** 2)
        scores[kind] = float(np.mean(errors))
    return min(scores, key=scores.get)


def fit_transport(
    q_all: np.ndarray,
    voltage_all: np.ndarray,
    temperature_all: np.ndarray,
    pidx: np.ndarray,
    pfrac: np.ndarray,
    library: dict[str, Any],
    protocol: dict[str, Any],
    clock_regularization: float,
) -> dict[str, Any]:
    qp = q_all[pidx]
    vp = voltage_all[pidx]
    tp = temperature_all[pidx]
    C = capacity_from_probes(qp, pfrac)
    u_all = np.clip(q_all / C, 1e-5, 1 - 1e-5)
    u_probe = u_all[pidx]
    search = protocol["candidate_search"]
    grid = np.asarray(library["grid"], float)
    best = None
    for alpha in search["clock_alpha_offsets"]:
        for beta_clock in search["clock_beta_values"]:
            phase_all = sigmoid(float(alpha) + float(beta_clock) * logit(u_all))
            phase_probe = phase_all[pidx]
            for tau in search["shared_tangent_values"]:
                mv_all = mechanism_eval(grid, np.asarray(library["voltage_h0"]), np.asarray(library["voltage_h1"]), phase_all, float(tau))
                mv_probe = mv_all[pidx]
                Xv = np.column_stack([np.ones(len(pidx)), u_probe, mv_probe])
                bv = ridge_fit(Xv, vp)
                pv = Xv @ bv
                for lag in search["temperature_lag_values"]:
                    phase_t_all = np.clip(phase_all - float(lag), 0.0, 1.0)
                    mt_all = mechanism_eval(grid, np.asarray(library["temperature_h0"]), np.asarray(library["temperature_h1"]), phase_t_all, float(tau))
                    mt_probe = mt_all[pidx]
                    Xt = np.column_stack([np.ones(len(pidx)), u_probe, mt_probe])
                    bt = ridge_fit(Xt, tp)
                    pt = Xt @ bt
                    data_loss = float(np.mean(((pv - vp) / 1.0) ** 2 + ((pt - tp) / 20.0) ** 2) / 2.0)
                    reg = float(clock_regularization) * (
                        float(alpha) ** 2 + (float(beta_clock) - 1.0) ** 2 + 0.35 * float(tau) ** 2 + 4.0 * float(lag) ** 2
                    )
                    objective = data_loss + reg
                    if best is None or objective < best[0]:
                        Xv_all = np.column_stack([np.ones(len(q_all)), u_all, mv_all])
                        Xt_all = np.column_stack([np.ones(len(q_all)), u_all, mt_all])
                        best = (
                            objective,
                            np.asarray(Xv_all @ bv, float),
                            np.asarray(Xt_all @ bt, float),
                            {
                                "alpha": float(alpha),
                                "beta": float(beta_clock),
                                "tau": float(tau),
                                "temperature_lag": float(lag),
                                "C_probe_Ah": float(C),
                                "probe_data_loss": data_loss,
                                "regularized_objective": objective,
                            },
                        )
    assert best is not None
    return {"voltage": best[1], "temperature": best[2], "parameters": best[3]}


def clip_outputs(voltage: np.ndarray, temperature: np.ndarray, protocol: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    vc = protocol["outputs"]["voltage"]["prediction_clip"]
    tc = protocol["outputs"]["temperature"]["prediction_clip"]
    return np.clip(voltage, vc[0], vc[1]), np.clip(temperature, tc[0], tc[1])


def joint_error(vhat: np.ndarray, that: np.ndarray, v: np.ndarray, t: np.ndarray, mask: np.ndarray) -> float:
    if int(np.sum(mask)) == 0:
        return float("nan")
    z = ((vhat[mask] - v[mask]) / 1.0) ** 2 + ((that[mask] - t[mask]) / 20.0) ** 2
    return float(np.sqrt(np.mean(z) / 2.0))


def evaluate_cycle(
    cyc: Cycle,
    library: dict[str, Any],
    protocol: dict[str, Any],
    clock_regularization: float,
    blend_penalty: float,
) -> dict[str, Any]:
    q, v, t = cyc.q, cyc.voltage, cyc.temperature
    fractions = np.asarray(protocol["probes"]["rank_fractions"], float)
    pidx = probe_indices(len(q), list(fractions))
    pfrac = fractions.copy()
    if len(pidx) != len(pfrac):
        raise RuntimeError("probe count mismatch")
    qp, vp, tp = q[pidx], v[pidx], t[pidx]
    vkind = choose_local_kind(qp, vp, 1.0)
    tkind = choose_local_kind(qp, tp, 20.0)
    local_v = fit_local_kind(qp, vp, q, vkind)
    local_t = fit_local_kind(qp, tp, q, tkind)
    transport = fit_transport(q, v, t, pidx, pfrac, library, protocol, clock_regularization)
    tv, tt = transport["voltage"], transport["temperature"]

    # Probe-only certification of the convex fallback.
    held_v_local, held_t_local, held_v_transport, held_t_transport = [], [], [], []
    for j in range(len(pidx)):
        keep = np.arange(len(pidx)) != j
        sub_idx = pidx[keep]
        sub_frac = pfrac[keep]
        target_q = np.asarray([q[pidx[j]]])
        held_v_local.append(fit_local_kind(q[sub_idx], v[sub_idx], target_q, vkind)[0])
        held_t_local.append(fit_local_kind(q[sub_idx], t[sub_idx], target_q, tkind)[0])
        sub_transport = fit_transport(q, v, t, sub_idx, sub_frac, library, protocol, clock_regularization)
        held_v_transport.append(sub_transport["voltage"][pidx[j]])
        held_t_transport.append(sub_transport["temperature"][pidx[j]])
    held_v_local = np.asarray(held_v_local); held_t_local = np.asarray(held_t_local)
    held_v_transport = np.asarray(held_v_transport); held_t_transport = np.asarray(held_t_transport)
    lambda_scores = []
    for lam in (0.0, 0.25, 0.5, 0.75, 1.0):
        bv = (1 - lam) * held_v_local + lam * held_v_transport
        bt = (1 - lam) * held_t_local + lam * held_t_transport
        err = float(np.mean(((bv - vp) / 1.0) ** 2 + ((bt - tp) / 20.0) ** 2) / 2.0)
        lambda_scores.append((err + float(blend_penalty) * lam * lam, lam, err))
    _, selected_lambda, certification_error = min(lambda_scores, key=lambda x: (x[0], x[1]))
    safe_v = (1 - selected_lambda) * local_v + selected_lambda * tv
    safe_t = (1 - selected_lambda) * local_t + selected_lambda * tt
    safe_v, safe_t = clip_outputs(safe_v, safe_t, protocol)
    local_v, local_t = clip_outputs(local_v, local_t, protocol)
    tv, tt = clip_outputs(tv, tt, protocol)

    score_mask = np.ones(len(q), dtype=bool)
    score_mask[pidx] = False
    tail_mask = score_mask & (np.arange(len(q)) > int(pidx[-1]))
    safe_error = joint_error(safe_v, safe_t, v, t, score_mask)
    local_error = joint_error(local_v, local_t, v, t, score_mask)
    flow_error = joint_error(tv, tt, v, t, score_mask)
    safe_tail = joint_error(safe_v, safe_t, v, t, tail_mask)
    local_tail = joint_error(local_v, local_t, v, t, tail_mask)
    catastrophic = (not np.isfinite(safe_error)) or safe_error > max(0.50, 4.0 * max(local_error, 1e-8))
    return {
        "cell": cyc.cell,
        "discharge_order": cyc.discharge_order,
        "source_cycle_index": cyc.source_cycle_index,
        "raw_points": len(q),
        "scored_points": int(np.sum(score_mask)),
        "tail_points": int(np.sum(tail_mask)),
        "local_voltage_kind": vkind,
        "local_temperature_kind": tkind,
        "selected_lambda": float(selected_lambda),
        "probe_certification_error": float(certification_error),
        "safe_joint_error": safe_error,
        "local_joint_error": local_error,
        "flow_joint_error": flow_error,
        "safe_vs_local_ratio": safe_error / max(local_error, 1e-10),
        "safe_tail_error": safe_tail,
        "local_tail_error": local_tail,
        "tail_safe_vs_local_ratio": safe_tail / max(local_tail, 1e-10) if np.isfinite(safe_tail) and np.isfinite(local_tail) else float("inf"),
        "catastrophic": bool(catastrophic),
        "transport_parameters": transport["parameters"],
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [r for r in rows if np.isfinite(r["safe_joint_error"]) and np.isfinite(r["local_joint_error"])]
    ratios = np.asarray([r["safe_vs_local_ratio"] for r in valid], float)
    tail = np.asarray([r["tail_safe_vs_local_ratio"] for r in valid if np.isfinite(r["tail_safe_vs_local_ratio"])], float)
    errors = np.asarray([r["safe_joint_error"] for r in valid], float)
    return {
        "valid_target_cycles": len(valid),
        "catastrophic_failures": int(sum(bool(r["catastrophic"]) for r in valid)),
        "median_absolute_joint_error": float(np.median(errors)) if len(errors) else float("inf"),
        "safe_vs_local": {
            "median_ratio": float(np.median(ratios)) if len(ratios) else float("inf"),
            "geometric_mean_ratio": geometric_mean(ratios),
            "win_fraction": float(np.mean(ratios < 1.0)) if len(ratios) else 0.0,
            "p90_ratio": float(np.quantile(ratios, 0.9)) if len(ratios) else float("inf"),
            "n": len(ratios),
        },
        "tail_safe_vs_local": {
            "median_ratio": float(np.median(tail)) if len(tail) else float("inf"),
            "geometric_mean_ratio": geometric_mean(tail),
            "win_fraction": float(np.mean(tail < 1.0)) if len(tail) else 0.0,
            "n": len(tail),
        },
        "selected_lambda": {
            "median": float(np.median([r["selected_lambda"] for r in valid])) if valid else 0.0,
            "positive_fraction": float(np.mean([r["selected_lambda"] > 0 for r in valid])) if valid else 0.0,
            "full_transport_fraction": float(np.mean([r["selected_lambda"] == 1 for r in valid])) if valid else 0.0,
        },
    }


def gate_development(summary: dict[str, Any], protocol: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    g = protocol["development_gates"]
    values = {
        "minimum_valid_target_cycles": summary["valid_target_cycles"],
        "zero_catastrophic_failures": summary["catastrophic_failures"],
        "safe_vs_local_median_ratio_lte": summary["safe_vs_local"]["median_ratio"],
        "safe_vs_local_geometric_mean_ratio_lte": summary["safe_vs_local"]["geometric_mean_ratio"],
        "safe_vs_local_win_fraction_gte": summary["safe_vs_local"]["win_fraction"],
        "safe_vs_local_p90_ratio_lte": summary["safe_vs_local"]["p90_ratio"],
        "tail_safe_vs_local_geometric_mean_ratio_lte": summary["tail_safe_vs_local"]["geometric_mean_ratio"],
        "tail_safe_vs_local_win_fraction_gte": summary["tail_safe_vs_local"]["win_fraction"],
    }
    passes = {
        "minimum_valid_target_cycles": values["minimum_valid_target_cycles"] >= g["minimum_valid_target_cycles"],
        "zero_catastrophic_failures": values["zero_catastrophic_failures"] == 0,
        "safe_vs_local_median_ratio_lte": values["safe_vs_local_median_ratio_lte"] <= g["safe_vs_local_median_ratio_lte"],
        "safe_vs_local_geometric_mean_ratio_lte": values["safe_vs_local_geometric_mean_ratio_lte"] <= g["safe_vs_local_geometric_mean_ratio_lte"],
        "safe_vs_local_win_fraction_gte": values["safe_vs_local_win_fraction_gte"] >= g["safe_vs_local_win_fraction_gte"],
        "safe_vs_local_p90_ratio_lte": values["safe_vs_local_p90_ratio_lte"] <= g["safe_vs_local_p90_ratio_lte"],
        "tail_safe_vs_local_geometric_mean_ratio_lte": values["tail_safe_vs_local_geometric_mean_ratio_lte"] <= g["tail_safe_vs_local_geometric_mean_ratio_lte"],
        "tail_safe_vs_local_win_fraction_gte": values["tail_safe_vs_local_win_fraction_gte"] >= g["tail_safe_vs_local_win_fraction_gte"],
    }
    return all(passes.values()), {k: {"value": values[k], "threshold": g[k] if k in g else 0, "pass": passes[k]} for k in passes}


def gate_holdout(summary: dict[str, Any], protocol: dict[str, Any], strong: bool = False) -> tuple[bool, dict[str, Any]]:
    g = protocol["holdout_strong_claim_gates" if strong else "holdout_primary_gates"]
    if strong:
        values = {
            "safe_vs_local_geometric_mean_ratio_lte": summary["safe_vs_local"]["geometric_mean_ratio"],
            "safe_vs_local_win_fraction_gte": summary["safe_vs_local"]["win_fraction"],
            "tail_safe_vs_local_geometric_mean_ratio_lte": summary["tail_safe_vs_local"]["geometric_mean_ratio"],
            "tail_safe_vs_local_win_fraction_gte": summary["tail_safe_vs_local"]["win_fraction"],
        }
    else:
        values = {
            "minimum_valid_target_cycles": summary["valid_target_cycles"],
            "zero_catastrophic_failures": summary["catastrophic_failures"],
            "median_absolute_joint_error_lte": summary["median_absolute_joint_error"],
            "safe_vs_local_median_ratio_lte": summary["safe_vs_local"]["median_ratio"],
            "safe_vs_local_geometric_mean_ratio_lte": summary["safe_vs_local"]["geometric_mean_ratio"],
            "safe_vs_local_win_fraction_gte": summary["safe_vs_local"]["win_fraction"],
            "tail_safe_vs_local_geometric_mean_ratio_lte": summary["tail_safe_vs_local"]["geometric_mean_ratio"],
            "tail_safe_vs_local_win_fraction_gte": summary["tail_safe_vs_local"]["win_fraction"],
        }
    passes = {}
    for key, value in values.items():
        if key == "zero_catastrophic_failures":
            passes[key] = value == 0
        elif key.startswith("minimum_"):
            passes[key] = value >= g[key]
        elif key.endswith("_gte"):
            passes[key] = value >= g[key]
        else:
            passes[key] = value <= g[key]
    return all(passes.values()), {k: {"value": values[k], "threshold": g[k], "pass": passes[k]} for k in passes}


def rows_to_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    flat = []
    for r in rows:
        p = r["transport_parameters"]
        flat.append({
            **{k: v for k, v in r.items() if k != "transport_parameters"},
            **{f"transport_{k}": v for k, v in p.items()},
        })
    if not flat:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0].keys()))
        w.writeheader(); w.writerows(flat)


def library_to_json(library: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for k, v in library.items():
        out[k] = v.tolist() if isinstance(v, np.ndarray) else v
    return out


def library_from_json(value: dict[str, Any]) -> dict[str, Any]:
    out = dict(value)
    for k in ("grid", "voltage_h0", "voltage_h1", "temperature_h0", "temperature_h1"):
        out[k] = np.asarray(out[k], float)
    return out


def report_markdown(title: str, result: dict[str, Any]) -> str:
    s = result["summary"]
    lines = [f"# {title}", "", f"Primary pass: **{result.get('primary_pass', result.get('development_pass'))}**"]
    if "strong_claim_pass" in result:
        lines.append(f"Strong-claim pass: **{result['strong_claim_pass']}**")
    lines += [
        "",
        f"- Valid cycles: **{s['valid_target_cycles']}**",
        f"- Catastrophic failures: **{s['catastrophic_failures']}**",
        f"- Median absolute joint error: **{s['median_absolute_joint_error']:.6f}**",
        f"- Safe/local median ratio: **{s['safe_vs_local']['median_ratio']:.6f}**",
        f"- Safe/local geometric-mean ratio: **{s['safe_vs_local']['geometric_mean_ratio']:.6f}**",
        f"- Safe/local win fraction: **{s['safe_vs_local']['win_fraction']:.3%}**",
        f"- Safe/local p90 ratio: **{s['safe_vs_local']['p90_ratio']:.6f}**",
        f"- Tail safe/local geometric-mean ratio: **{s['tail_safe_vs_local']['geometric_mean_ratio']:.6f}**",
        f"- Tail safe/local win fraction: **{s['tail_safe_vs_local']['win_fraction']:.3%}**",
        f"- Positive atlas blend fraction: **{s['selected_lambda']['positive_fraction']:.3%}**",
        "",
        "The errors use fixed physical scales of 1.0 V and 20 °C. No trajectory-specific output normalization was used.",
    ]
    return "\n".join(lines) + "\n"


def run_development(archive: Path, protocol_path: Path, out: Path) -> int:
    protocol = json.loads(protocol_path.read_text())
    out.mkdir(parents=True, exist_ok=True)
    archive_hash = sha256_file(archive)
    extracted = []
    all_cycles = {}
    for cell in protocol["dataset"]["development_cells"]:
        mat = out / "_development_payloads" / f"{cell}.mat"
        extracted.append(extract_member(archive, cell, mat))
        all_cycles[cell] = load_cell_cycles(mat, cell, protocol)
    manifest = {
        "archive_sha256": archive_hash,
        "opened_cells": sorted(all_cycles),
        "explicitly_unopened_cells": protocol["dataset"]["untouched_holdout_cells"],
        "member_evidence": extracted,
        "cycles_per_cell": {k: len(v) for k, v in all_cycles.items()},
    }
    write_json(out / "DEVELOPMENT_ACCESS_MANIFEST.json", manifest)

    configs = [(float(r), float(b)) for r in protocol["candidate_search"]["clock_regularization_values"] for b in protocol["candidate_search"]["safe_blend_penalty_values"]]
    config_rows: dict[tuple[float, float], list[dict[str, Any]]] = {c: [] for c in configs}
    cells = protocol["dataset"]["development_cells"]
    fold_library_stats = []
    for target in cells:
        training = [cyc for cell in cells if cell != target for cyc in all_cycles[cell]]
        library = build_library(training, protocol)
        fold_library_stats.append({"target_cell": target, "library": {k: v for k, v in library_to_json(library).items() if k.endswith("_stats") or k.startswith("training_")}})
        for cyc in all_cycles[target]:
            cache_by_reg = {}
            for reg in protocol["candidate_search"]["clock_regularization_values"]:
                # blend selection is the only part that changes with blend penalty, so calculate once per pair below.
                cache_by_reg[float(reg)] = cyc
            for reg, blend in configs:
                row = evaluate_cycle(cyc, library, protocol, reg, blend)
                row["development_target_cell"] = target
                row["clock_regularization"] = reg
                row["blend_penalty"] = blend
                config_rows[(reg, blend)].append(row)

    candidates = []
    for (reg, blend), rows in config_rows.items():
        summary = summarize(rows)
        passed, gates = gate_development(summary, protocol)
        candidates.append({
            "clock_regularization": reg,
            "blend_penalty": blend,
            "summary": summary,
            "development_pass": passed,
            "gates": gates,
        })
    candidates.sort(key=lambda x: (
        not x["development_pass"],
        x["summary"]["safe_vs_local"]["geometric_mean_ratio"],
        x["summary"]["tail_safe_vs_local"]["geometric_mean_ratio"],
        x["clock_regularization"],
        x["blend_penalty"],
    ))
    selected = candidates[0]
    development_pass = bool(selected["development_pass"])
    selected_rows = config_rows[(selected["clock_regularization"], selected["blend_penalty"])]
    result = {
        "protocol_id": protocol["protocol_id"],
        "stage": "development-lock",
        "development_pass": development_pass,
        "archive_sha256": archive_hash,
        "selected_configuration": {
            "clock_regularization": selected["clock_regularization"],
            "blend_penalty": selected["blend_penalty"],
        },
        "summary": selected["summary"],
        "gates": selected["gates"],
        "all_configuration_summaries": candidates,
        "fold_library_stats": fold_library_stats,
    }
    write_json(out / "development_results.json", result)
    rows_to_csv(out / "development_per_cycle.csv", selected_rows)
    (out / "DEVELOPMENT_REPORT.md").write_text(report_markdown("Causal Loom v0.24 development-lock result", result))
    if development_pass:
        final_library = build_library([c for cell in cells for c in all_cycles[cell]], protocol)
        lock = {
            "protocol_id": protocol["protocol_id"],
            "archive_sha256": archive_hash,
            "development_cells": cells,
            "untouched_holdout_cells": protocol["dataset"]["untouched_holdout_cells"],
            "selected_configuration": result["selected_configuration"],
            "development_summary": result["summary"],
            "mechanism_library": library_to_json(final_library),
        }
        write_json(out / "CANDIDATE_LOCK.json", lock)
        (out / "CANDIDATE_LOCK_SHA256.txt").write_text(sha256_file(out / "CANDIDATE_LOCK.json") + "  CANDIDATE_LOCK.json\n")
    # Remove extracted payloads from evidence before upload.
    import shutil
    shutil.rmtree(out / "_development_payloads", ignore_errors=True)
    return 0 if development_pass else 2


def run_holdout(archive: Path, protocol_path: Path, lock_path: Path, out: Path) -> int:
    protocol = json.loads(protocol_path.read_text())
    lock = json.loads(lock_path.read_text())
    out.mkdir(parents=True, exist_ok=True)
    archive_hash = sha256_file(archive)
    if lock["protocol_id"] != protocol["protocol_id"]:
        raise RuntimeError("Candidate lock protocol mismatch")
    if archive_hash != lock["archive_sha256"]:
        raise RuntimeError("Holdout archive bytes differ from development archive")
    library = library_from_json(lock["mechanism_library"])
    config = lock["selected_configuration"]
    extracted = []
    all_cycles = []
    for cell in protocol["dataset"]["untouched_holdout_cells"]:
        mat = out / "_holdout_payloads" / f"{cell}.mat"
        extracted.append(extract_member(archive, cell, mat))
        all_cycles.extend(load_cell_cycles(mat, cell, protocol))
    manifest = {
        "archive_sha256": archive_hash,
        "candidate_lock_sha256": sha256_file(lock_path),
        "opened_cells": protocol["dataset"]["untouched_holdout_cells"],
        "member_evidence": extracted,
        "cycles_per_cell": {cell: sum(c.cell == cell for c in all_cycles) for cell in protocol["dataset"]["untouched_holdout_cells"]},
    }
    write_json(out / "HOLDOUT_ACCESS_MANIFEST.json", manifest)
    rows = [evaluate_cycle(c, library, protocol, float(config["clock_regularization"]), float(config["blend_penalty"])) for c in all_cycles]
    summary = summarize(rows)
    primary_pass, gates = gate_holdout(summary, protocol, strong=False)
    strong_pass, strong_gates = gate_holdout(summary, protocol, strong=True)
    result = {
        "protocol_id": protocol["protocol_id"],
        "stage": "first-and-permanent-holdout",
        "primary_pass": primary_pass,
        "strong_claim_pass": strong_pass,
        "archive_sha256": archive_hash,
        "candidate_lock_sha256": sha256_file(lock_path),
        "selected_configuration": config,
        "summary": summary,
        "gates": gates,
        "strong_claim_gates": strong_gates,
    }
    write_json(out / "holdout_results.json", result)
    rows_to_csv(out / "holdout_per_cycle.csv", rows)
    (out / "HOLDOUT_REPORT.md").write_text(report_markdown("Causal Loom v0.24 frozen NASA battery result", result))
    import shutil
    shutil.rmtree(out / "_holdout_payloads", ignore_errors=True)
    return 0 if primary_pass else 2


def self_test() -> int:
    grid = np.linspace(0, 1, 241)
    base = 4.15 - 0.65 * grid - 0.75 / (1 + np.exp(-(grid - 0.91) * 45))
    curves = []
    for alpha in (-0.12, 0.0, 0.10, 0.18):
        phase = sigmoid(alpha + logit(np.clip(grid, 1e-5, 1 - 1e-5)))
        curves.append(np.interp(phase, grid, base))
    h0, h1, stats = mechanism_from_curves(curves, grid)
    assert np.all(np.isfinite(h0)) and np.all(np.isfinite(h1))
    assert abs(float(np.mean(h0))) < 1e-8
    assert stats["curve_count"] == 4
    idx = probe_indices(101, [0.06, 0.27, 0.5, 0.73, 0.92])
    assert len(np.unique(idx)) == 5
    print("SELF_TEST_PASS", stats["median_projective_correlation"])
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["development-lock", "holdout"])
    ap.add_argument("--archive")
    ap.add_argument("--protocol")
    ap.add_argument("--candidate-lock")
    ap.add_argument("--output-dir")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if not all([args.stage, args.archive, args.protocol, args.output_dir]):
        ap.error("stage, archive, protocol, and output-dir are required")
    try:
        if args.stage == "development-lock":
            return run_development(Path(args.archive), Path(args.protocol), Path(args.output_dir))
        if not args.candidate_lock:
            ap.error("candidate-lock is required for holdout")
        return run_holdout(Path(args.archive), Path(args.protocol), Path(args.candidate_lock), Path(args.output_dir))
    except Exception:
        out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
        (out / "ERROR.txt").write_text(traceback.format_exc())
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
