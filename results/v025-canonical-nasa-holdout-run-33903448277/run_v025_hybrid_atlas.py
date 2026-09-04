#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.io import loadmat

EPS = 1e-10


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        result = float(np.asarray(value).squeeze())
        return result if math.isfinite(result) else default
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
    return default


def as_items(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.ndarray):
        return list(value.reshape(-1))
    return [value]


def as_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, np.ndarray) and value.size == 1:
        return as_text(value.reshape(-1)[0])
    return str(value)


def as_float_array(value: Any) -> np.ndarray:
    try:
        return np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        if isinstance(value, np.ndarray) and value.dtype == object:
            return np.asarray([safe_float(x) for x in value.reshape(-1)], dtype=float)
        raise


def geometric_mean(values: Iterable[float]) -> float:
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    return float(np.exp(np.mean(np.log(x)))) if len(x) else float("inf")


def pearson(x: Iterable[float], y: Iterable[float]) -> float:
    a = np.asarray(list(x), float)
    b = np.asarray(list(y), float)
    m = np.isfinite(a) & np.isfinite(b)
    if int(m.sum()) < 3 or np.std(a[m]) < EPS or np.std(b[m]) < EPS:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


@dataclass
class Cycle:
    cell: str
    discharge_order: int
    source_cycle_index: int
    ambient_temperature: float
    time: np.ndarray
    current: np.ndarray
    q: np.ndarray
    voltage: np.ndarray
    temperature: np.ndarray


def extract_member(archive: Path, cell: str, destination: Path) -> dict[str, Any]:
    target = f"{cell}.mat".lower()
    with zipfile.ZipFile(archive) as z:
        matches = [n for n in z.namelist() if Path(n).name.lower() == target]
        if len(matches) != 1:
            raise RuntimeError(f"Expected exactly one {cell}.mat, found {matches}")
        payload = z.read(matches[0])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    return {"cell": cell, "archive_member": matches[0], "bytes": len(payload), "sha256": sha256_bytes(payload)}


def load_cell_cycles(path: Path, cell: str, protocol: dict[str, Any]) -> list[Cycle]:
    loaded = loadmat(path, simplify_cells=True)
    root = loaded.get(cell)
    if root is None:
        keys = [k for k in loaded if not k.startswith("__")]
        if len(keys) != 1:
            raise KeyError(f"Cannot locate {cell}; keys={keys}")
        root = loaded[keys[0]]
    raw = as_items(field(root, "cycle"))
    cfg = protocol["cycle_selection"]
    valid: list[Cycle] = []
    discharge_order = 0
    for source_index, cyc in enumerate(raw):
        if as_text(field(cyc, "type")).strip().lower() != "discharge":
            continue
        data = field(cyc, "data")
        time = as_float_array(field(data, "Time"))
        current = as_float_array(field(data, "Current_measured"))
        voltage = as_float_array(field(data, "Voltage_measured"))
        temperature = as_float_array(field(data, "Temperature_measured"))
        n = min(len(time), len(current), len(voltage), len(temperature))
        time, current, voltage, temperature = time[:n], current[:n], voltage[:n], temperature[:n]
        finite = np.isfinite(time) & np.isfinite(current) & np.isfinite(voltage) & np.isfinite(temperature)
        time, current, voltage, temperature = time[finite], current[finite], voltage[finite], temperature[finite]
        order = np.argsort(time, kind="mergesort")
        time, current, voltage, temperature = time[order], current[order], voltage[order], temperature[order]
        if len(time) < int(cfg["minimum_raw_points"]):
            discharge_order += 1
            continue
        dt = np.maximum(np.diff(time), 0.0)
        dq = 0.5 * (np.abs(current[1:]) + np.abs(current[:-1])) * dt / 3600.0
        q = np.r_[0.0, np.cumsum(dq)]
        if float(q[-1] - q[0]) < float(cfg["minimum_charge_span_Ah"]):
            discharge_order += 1
            continue
        if float(np.ptp(voltage)) < float(cfg["minimum_voltage_range_V"]):
            discharge_order += 1
            continue
        valid.append(Cycle(
            cell=cell,
            discharge_order=discharge_order,
            source_cycle_index=source_index,
            ambient_temperature=safe_float(field(cyc, "ambient_temperature")),
            time=time.astype(float),
            current=current.astype(float),
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


def probe_indices(n: int, fractions: list[float]) -> np.ndarray:
    raw = np.round(np.asarray(fractions, float) * (n - 1)).astype(int)
    raw = np.clip(raw, 0, n - 1)
    chosen: list[int] = []
    for idx in raw:
        candidate = int(idx)
        if candidate in chosen:
            for radius in range(1, n):
                found = next((x for x in (candidate - radius, candidate + radius) if 0 <= x < n and x not in chosen), None)
                if found is not None:
                    candidate = found
                    break
        chosen.append(candidate)
    return np.asarray(sorted(chosen), int)


def hybrid_coordinate(cyc: Cycle, protocol: dict[str, Any]) -> tuple[np.ndarray, int, float, float]:
    threshold = float(protocol["inputs"]["current_off_threshold_A"])
    active = np.abs(cyc.current) > threshold
    off = int(np.where(active)[0][-1]) if np.any(active) else len(cyc.current) - 1
    qoff = max(float(cyc.q[off]), EPS)
    rest_duration = max(float(cyc.time[-1] - cyc.time[off]), EPS)
    z = np.empty(len(cyc.time), float)
    z[:off + 1] = np.clip(cyc.q[:off + 1] / qoff, 0.0, 1.0)
    if off + 1 < len(z):
        z[off + 1:] = 1.0 + np.clip((cyc.time[off + 1:] - cyc.time[off]) / rest_duration, 0.0, 1.0)
    return z, off, qoff, rest_duration


def projective_modes(curves: np.ndarray, orders: np.ndarray, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    g = (grid - grid[0]) / max(float(grid[-1] - grid[0]), EPS)
    chord = curves[:, [0]] + (curves[:, [-1]] - curves[:, [0]]) * g[None, :]
    residuals = curves - chord
    residuals -= np.mean(residuals, axis=1, keepdims=True)
    residuals /= np.maximum(np.sqrt(np.mean(residuals ** 2, axis=1, keepdims=True)), 1e-8)
    template = residuals[0].copy()
    for i in range(len(residuals)):
        if float(np.dot(residuals[i], template)) < 0:
            residuals[i] *= -1.0
        template = np.median(residuals[:i + 1], axis=0)
    h0 = np.median(residuals, axis=0)
    h0 -= np.mean(h0)
    h0 /= max(float(np.sqrt(np.mean(h0 ** 2))), 1e-8)
    centered = residuals - h0
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    h1 = vt[0].copy()
    h1 -= np.mean(h1)
    h1 -= float(np.dot(h1, h0) / max(np.dot(h0, h0), EPS)) * h0
    h1 /= max(float(np.sqrt(np.mean(h1 ** 2))), 1e-8)
    scores = np.mean(centered * h1[None, :], axis=1)
    score_order_corr = pearson(scores, orders)
    if math.isfinite(score_order_corr) and score_order_corr < 0:
        h1 *= -1.0
        scores *= -1.0
        score_order_corr *= -1.0
    correlations = [abs(float(np.corrcoef(r, h0)[0, 1])) for r in residuals]
    variance = singular ** 2
    return h0, h1, {
        "curve_count": int(len(residuals)),
        "median_projective_correlation": float(np.median(correlations)),
        "minimum_projective_correlation": float(np.min(correlations)),
        "pc1_residual_variance_fraction": float(variance[0] / max(np.sum(variance), EPS)),
        "training_tangent_score_vs_discharge_order_pearson": score_order_corr,
    }


def build_atlas(cycles: list[Cycle], protocol: dict[str, Any]) -> dict[str, Any]:
    cfg = protocol["atlas"]
    grid = np.linspace(float(cfg["grid_min"]), float(cfg["grid_max"]), int(cfg["grid_points"]))
    orders = np.asarray([c.discharge_order for c in cycles], float)
    atlas: dict[str, Any] = {
        "grid": grid,
        "training_cells": sorted({c.cell for c in cycles}),
        "training_cycle_count": len(cycles),
    }
    for key, attr in (("voltage", "voltage"), ("temperature", "temperature")):
        curves = []
        for cyc in cycles:
            z, _, _, _ = hybrid_coordinate(cyc, protocol)
            curves.append(np.interp(grid, z, getattr(cyc, attr)))
        h0, h1, stats = projective_modes(np.vstack(curves), orders, grid)
        atlas[key] = {"h0": h0, "h1": h1, "stats": stats}
    return atlas


def atlas_json(atlas: dict[str, Any]) -> dict[str, Any]:
    return {
        "grid": np.asarray(atlas["grid"]).tolist(),
        "training_cells": atlas["training_cells"],
        "training_cycle_count": atlas["training_cycle_count"],
        "voltage": {"h0": np.asarray(atlas["voltage"]["h0"]).tolist(), "h1": np.asarray(atlas["voltage"]["h1"]).tolist(), "stats": atlas["voltage"]["stats"]},
        "temperature": {"h0": np.asarray(atlas["temperature"]["h0"]).tolist(), "h1": np.asarray(atlas["temperature"]["h1"]).tolist(), "stats": atlas["temperature"]["stats"]},
    }


def atlas_from_json(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "grid": np.asarray(value["grid"], float),
        "training_cells": list(value["training_cells"]),
        "training_cycle_count": int(value["training_cycle_count"]),
        "voltage": {"h0": np.asarray(value["voltage"]["h0"], float), "h1": np.asarray(value["voltage"]["h1"], float), "stats": value["voltage"]["stats"]},
        "temperature": {"h0": np.asarray(value["temperature"]["h0"], float), "h1": np.asarray(value["temperature"]["h1"], float), "stats": value["temperature"]["stats"]},
    }


def eta_values(protocol: dict[str, Any]) -> np.ndarray:
    lo, hi, step = [float(x) for x in protocol["atlas"]["eta_grid"]]
    return np.arange(lo, hi + step * 0.5, step)


def fit_atlas(cyc: Cycle, atlas: dict[str, Any], fit_idx: np.ndarray, protocol: dict[str, Any]) -> dict[str, Any]:
    z, _, _, _ = hybrid_coordinate(cyc, protocol)
    zn = z / 2.0
    grid = np.asarray(atlas["grid"], float)
    ridge = float(protocol["atlas"]["coefficient_ridge"])
    eta_reg = float(protocol["atlas"]["eta_regularization"])
    interpolated: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for key in ("voltage", "temperature"):
        interpolated[key] = (
            np.interp(z, grid, np.asarray(atlas[key]["h0"], float)),
            np.interp(z, grid, np.asarray(atlas[key]["h1"], float)),
        )
    best = None
    for eta in eta_values(protocol):
        total = 0.0
        outputs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for key, y, scale in (
            ("voltage", cyc.voltage, float(protocol["outputs"]["voltage"]["scale"])),
            ("temperature", cyc.temperature, float(protocol["outputs"]["temperature"]["scale"])),
        ):
            h = interpolated[key][0] + float(eta) * interpolated[key][1]
            X = np.column_stack([np.ones(len(z)), zn, h])
            Xp = X[fit_idx]
            penalty = np.diag([1e-10, 1e-8, ridge])
            try:
                coef = np.linalg.solve(Xp.T @ Xp + penalty, Xp.T @ y[fit_idx])
            except np.linalg.LinAlgError:
                coef = np.linalg.lstsq(Xp.T @ Xp + penalty, Xp.T @ y[fit_idx], rcond=None)[0]
            pred = X @ coef
            total += float(np.mean(((pred[fit_idx] - y[fit_idx]) / scale) ** 2) / 2.0)
            outputs[key] = (pred, coef)
        objective = total + eta_reg * float(eta) ** 2
        tie = (objective, abs(float(eta)), float(eta))
        if best is None or tie < best[0]:
            best = (tie, float(eta), total, outputs)
    assert best is not None
    return {
        "eta": best[1],
        "probe_loss": best[2],
        "voltage": np.asarray(best[3]["voltage"][0], float),
        "temperature": np.asarray(best[3]["temperature"][0], float),
        "voltage_coefficients": np.asarray(best[3]["voltage"][1], float),
        "temperature_coefficients": np.asarray(best[3]["temperature"][1], float),
    }


def normalized_coordinate(cyc: Cycle, kind: str, protocol: dict[str, Any]) -> np.ndarray:
    if kind == "time":
        return (cyc.time - cyc.time[0]) / max(float(cyc.time[-1] - cyc.time[0]), EPS)
    if kind == "charge":
        return cyc.q / max(float(cyc.q[-1]), EPS)
    if kind == "rank":
        return np.linspace(0.0, 1.0, len(cyc.time))
    if kind == "hybrid":
        return hybrid_coordinate(cyc, protocol)[0] / 2.0
    raise ValueError(kind)


def fit_local(x: np.ndarray, y: np.ndarray, fit_idx: np.ndarray, kind: str) -> np.ndarray:
    xp, yp = np.asarray(x[fit_idx], float), np.asarray(y[fit_idx], float)
    order = np.argsort(xp, kind="mergesort")
    xp, yp = xp[order], yp[order]
    ux, inv = np.unique(xp, return_inverse=True)
    if len(ux) != len(xp):
        sums = np.bincount(inv, weights=yp)
        counts = np.bincount(inv)
        xp, yp = ux, sums / np.maximum(counts, 1)
    if len(xp) < 2:
        return np.full_like(x, float(yp[0]) if len(yp) else float("nan"))
    if kind == "pchip":
        try:
            return np.asarray(PchipInterpolator(xp, yp, extrapolate=True)(x), float)
        except Exception:
            kind = "poly3"
    if kind == "poly3":
        center = float(np.mean(xp))
        scale = max(float(np.ptp(xp)), EPS)
        xx = (xp - center) / scale
        xa = (x - center) / scale
        degree = min(3, len(xp) - 1)
        X = np.vander(xx, degree + 1, increasing=True)
        penalty = np.eye(degree + 1) * 1e-8
        penalty[0, 0] = 1e-12
        coef = np.linalg.solve(X.T @ X + penalty, X.T @ yp)
        return np.vander(xa, degree + 1, increasing=True) @ coef
    raise ValueError(kind)


def choose_local(cyc: Cycle, y: np.ndarray, scale: float, pidx: np.ndarray, protocol: dict[str, Any]) -> tuple[str, str]:
    best = None
    for coordinate in ("time", "charge", "rank", "hybrid"):
        x = normalized_coordinate(cyc, coordinate, protocol)
        for kind in ("pchip", "poly3"):
            errors = []
            for j in range(len(pidx)):
                sub = np.delete(pidx, j)
                pred = fit_local(x, y, sub, kind)[pidx[j]]
                errors.append(((pred - y[pidx[j]]) / scale) ** 2)
            item = (float(np.mean(errors)), coordinate, kind)
            if best is None or item < best:
                best = item
    assert best is not None
    return best[1], best[2]


def fit_generic_hybrid_cubic(cyc: Cycle, y: np.ndarray, pidx: np.ndarray, protocol: dict[str, Any]) -> np.ndarray:
    x = normalized_coordinate(cyc, "hybrid", protocol)
    X = np.polynomial.chebyshev.chebvander(2.0 * x - 1.0, 3)
    Xp = X[pidx]
    penalty = np.eye(4) * 1e-6
    penalty[0, 0] = 1e-10
    coef = np.linalg.solve(Xp.T @ Xp + penalty, Xp.T @ y[pidx])
    return X @ coef


def clip_pair(voltage: np.ndarray, temperature: np.ndarray, protocol: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    vr = protocol["outputs"]["voltage"]["clip"]
    tr = protocol["outputs"]["temperature"]["clip"]
    return np.clip(voltage, vr[0], vr[1]), np.clip(temperature, tr[0], tr[1])


def joint_error(vhat: np.ndarray, that: np.ndarray, cyc: Cycle, mask: np.ndarray, protocol: dict[str, Any]) -> float:
    if int(np.sum(mask)) == 0:
        return float("nan")
    vs = float(protocol["outputs"]["voltage"]["scale"])
    ts = float(protocol["outputs"]["temperature"]["scale"])
    z = ((vhat[mask] - cyc.voltage[mask]) / vs) ** 2 + ((that[mask] - cyc.temperature[mask]) / ts) ** 2
    return float(np.sqrt(np.mean(z) / 2.0))


def evaluate_cycle(cyc: Cycle, atlas: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    pidx = probe_indices(len(cyc.time), list(protocol["probes"]["rank_fractions"]))
    vcoord, vkind = choose_local(cyc, cyc.voltage, float(protocol["outputs"]["voltage"]["scale"]), pidx, protocol)
    tcoord, tkind = choose_local(cyc, cyc.temperature, float(protocol["outputs"]["temperature"]["scale"]), pidx, protocol)
    local_v = fit_local(normalized_coordinate(cyc, vcoord, protocol), cyc.voltage, pidx, vkind)
    local_t = fit_local(normalized_coordinate(cyc, tcoord, protocol), cyc.temperature, pidx, tkind)
    candidate = fit_atlas(cyc, atlas, pidx, protocol)
    generic_v = fit_generic_hybrid_cubic(cyc, cyc.voltage, pidx, protocol)
    generic_t = fit_generic_hybrid_cubic(cyc, cyc.temperature, pidx, protocol)

    held = []
    for j in range(len(pidx)):
        sub = np.delete(pidx, j)
        fit = fit_atlas(cyc, atlas, sub, protocol)
        held.append((
            fit_local(normalized_coordinate(cyc, vcoord, protocol), cyc.voltage, sub, vkind)[pidx[j]],
            fit_local(normalized_coordinate(cyc, tcoord, protocol), cyc.temperature, sub, tkind)[pidx[j]],
            fit["voltage"][pidx[j]],
            fit["temperature"][pidx[j]],
        ))
    held = np.asarray(held, float)
    lambda_scores = []
    vs = float(protocol["outputs"]["voltage"]["scale"])
    ts = float(protocol["outputs"]["temperature"]["scale"])
    for lam in protocol["safe_fallback"]["lambda_grid"]:
        lam = float(lam)
        pv = (1.0 - lam) * held[:, 0] + lam * held[:, 2]
        pt = (1.0 - lam) * held[:, 1] + lam * held[:, 3]
        loss = float(np.mean(((pv - cyc.voltage[pidx]) / vs) ** 2 + ((pt - cyc.temperature[pidx]) / ts) ** 2) / 2.0)
        score = loss + float(protocol["safe_fallback"]["blend_penalty"]) * lam ** 2
        lambda_scores.append((score, lam, loss))
    _, lam, certification_loss = min(lambda_scores, key=lambda x: (x[0], x[1]))
    safe_v = (1.0 - lam) * local_v + lam * candidate["voltage"]
    safe_t = (1.0 - lam) * local_t + lam * candidate["temperature"]
    safe_v, safe_t = clip_pair(safe_v, safe_t, protocol)
    local_v, local_t = clip_pair(local_v, local_t, protocol)
    cand_v, cand_t = clip_pair(candidate["voltage"], candidate["temperature"], protocol)
    generic_v, generic_t = clip_pair(generic_v, generic_t, protocol)

    score_mask = np.ones(len(cyc.time), bool)
    score_mask[pidx] = False
    z, off, qoff, rest_duration = hybrid_coordinate(cyc, protocol)
    tail_mask = score_mask & (np.arange(len(cyc.time)) > int(pidx[-1]))
    rest_mask = score_mask & (np.arange(len(cyc.time)) > off)
    safe_error = joint_error(safe_v, safe_t, cyc, score_mask, protocol)
    local_error = joint_error(local_v, local_t, cyc, score_mask, protocol)
    candidate_error = joint_error(cand_v, cand_t, cyc, score_mask, protocol)
    generic_error = joint_error(generic_v, generic_t, cyc, score_mask, protocol)
    safe_tail = joint_error(safe_v, safe_t, cyc, tail_mask, protocol)
    local_tail = joint_error(local_v, local_t, cyc, tail_mask, protocol)
    safe_rest = joint_error(safe_v, safe_t, cyc, rest_mask, protocol)
    local_rest = joint_error(local_v, local_t, cyc, rest_mask, protocol)
    catastrophic = (not math.isfinite(safe_error)) or safe_error > max(0.50, 4.0 * max(local_error, EPS))
    return {
        "cell": cyc.cell,
        "discharge_order": cyc.discharge_order,
        "source_cycle_index": cyc.source_cycle_index,
        "raw_points": len(cyc.time),
        "scored_points": int(np.sum(score_mask)),
        "tail_points": int(np.sum(tail_mask)),
        "relaxation_points": int(np.sum(rest_mask)),
        "event_index": off,
        "event_rank_fraction": float(off / max(len(cyc.time) - 1, 1)),
        "q_off_Ah": qoff,
        "rest_duration_s": rest_duration,
        "last_probe_after_event": bool(int(pidx[-1]) > off),
        "eta": float(candidate["eta"]),
        "atlas_probe_loss": float(candidate["probe_loss"]),
        "selected_lambda": float(lam),
        "fallback_certification_loss": certification_loss,
        "local_voltage_coordinate": vcoord,
        "local_voltage_kind": vkind,
        "local_temperature_coordinate": tcoord,
        "local_temperature_kind": tkind,
        "safe_joint_error": safe_error,
        "local_joint_error": local_error,
        "atlas_joint_error": candidate_error,
        "generic_hybrid_cubic_joint_error": generic_error,
        "safe_vs_local_ratio": safe_error / max(local_error, EPS),
        "atlas_vs_generic_hybrid_cubic_ratio": candidate_error / max(generic_error, EPS),
        "safe_tail_error": safe_tail,
        "local_tail_error": local_tail,
        "tail_safe_vs_local_ratio": safe_tail / max(local_tail, EPS) if math.isfinite(safe_tail) and math.isfinite(local_tail) else float("nan"),
        "safe_relaxation_error": safe_rest,
        "local_relaxation_error": local_rest,
        "relaxation_safe_vs_local_ratio": safe_rest / max(local_rest, EPS) if math.isfinite(safe_rest) and math.isfinite(local_rest) else float("nan"),
        "catastrophic": bool(catastrophic),
    }


def ratio_summary(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    x = np.asarray([safe_float(r.get(key)) for r in rows], float)
    x = x[np.isfinite(x) & (x > 0)]
    return {
        "n": int(len(x)),
        "median_ratio": float(np.median(x)) if len(x) else float("inf"),
        "geometric_mean_ratio": geometric_mean(x),
        "win_fraction": float(np.mean(x < 1.0)) if len(x) else 0.0,
        "p90_ratio": float(np.quantile(x, 0.9)) if len(x) else float("inf"),
        "maximum_ratio": float(np.max(x)) if len(x) else float("inf"),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [r for r in rows if math.isfinite(safe_float(r.get("safe_joint_error"))) and math.isfinite(safe_float(r.get("local_joint_error")))]
    errors = np.asarray([r["safe_joint_error"] for r in valid], float)
    return {
        "valid_cycles": len(valid),
        "catastrophic_failures": int(sum(bool(r["catastrophic"]) for r in valid)),
        "median_safe_joint_error": float(np.median(errors)) if len(errors) else float("inf"),
        "safe_vs_local": ratio_summary(valid, "safe_vs_local_ratio"),
        "tail_safe_vs_local": ratio_summary(valid, "tail_safe_vs_local_ratio"),
        "relaxation_safe_vs_local": ratio_summary(valid, "relaxation_safe_vs_local_ratio"),
        "atlas_vs_generic_hybrid_cubic": ratio_summary(valid, "atlas_vs_generic_hybrid_cubic_ratio"),
        "selected_lambda": {
            "median": float(np.median([r["selected_lambda"] for r in valid])) if valid else float("nan"),
            "positive_fraction": float(np.mean([r["selected_lambda"] > 0 for r in valid])) if valid else 0.0,
            "full_atlas_fraction": float(np.mean([r["selected_lambda"] == 1 for r in valid])) if valid else 0.0,
        },
        "eta_vs_qoff_pearson": pearson([r["eta"] for r in valid], [r["q_off_Ah"] for r in valid]),
        "eta_vs_discharge_order_pearson": pearson([r["eta"] for r in valid], [r["discharge_order"] for r in valid]),
        "eta_vs_qoff_pearson_by_cell": {
            cell: pearson([r["eta"] for r in valid if r["cell"] == cell], [r["q_off_Ah"] for r in valid if r["cell"] == cell])
            for cell in sorted({r["cell"] for r in valid})
        },
    }


def compare_atlases(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for key in ("voltage", "temperature"):
        result[key] = {
            "h0_abs_correlation": abs(float(np.corrcoef(a[key]["h0"], b[key]["h0"])[0, 1])),
            "h1_abs_correlation": abs(float(np.corrcoef(a[key]["h1"], b[key]["h1"])[0, 1])),
        }
    return result


def bool_gate(value: Any, threshold: Any, mode: str) -> dict[str, Any]:
    if mode == "gte": passed = value >= threshold
    elif mode == "lte": passed = value <= threshold
    elif mode == "eq": passed = value == threshold
    else: raise ValueError(mode)
    return {"value": value, "threshold": threshold, "pass": bool(passed)}


def development_gates(summary: dict[str, Any], correlations: dict[str, Any], protocol: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    g = protocol["development_gates"]
    sl = summary["safe_vs_local"]
    tl = summary["tail_safe_vs_local"]
    rl = summary["relaxation_safe_vs_local"]
    gc = summary["atlas_vs_generic_hybrid_cubic"]
    min_h0 = min(correlations[k]["h0_abs_correlation"] for k in correlations)
    min_h1 = min(correlations[k]["h1_abs_correlation"] for k in correlations)
    gates = {
        "minimum_valid_cycles": bool_gate(summary["valid_cycles"], g["minimum_valid_cycles"], "gte"),
        "zero_catastrophic_failures": bool_gate(summary["catastrophic_failures"], 0, "eq"),
        "safe_vs_local_geometric_mean_ratio": bool_gate(sl["geometric_mean_ratio"], g["safe_vs_local_geometric_mean_ratio_lte"], "lte"),
        "safe_vs_local_median_ratio": bool_gate(sl["median_ratio"], g["safe_vs_local_median_ratio_lte"], "lte"),
        "safe_vs_local_win_fraction": bool_gate(sl["win_fraction"], g["safe_vs_local_win_fraction_gte"], "gte"),
        "safe_vs_local_p90_ratio": bool_gate(sl["p90_ratio"], g["safe_vs_local_p90_ratio_lte"], "lte"),
        "tail_geometric_mean_ratio": bool_gate(tl["geometric_mean_ratio"], g["tail_safe_vs_local_geometric_mean_ratio_lte"], "lte"),
        "tail_win_fraction": bool_gate(tl["win_fraction"], g["tail_safe_vs_local_win_fraction_gte"], "gte"),
        "relaxation_geometric_mean_ratio": bool_gate(rl["geometric_mean_ratio"], g["relaxation_safe_vs_local_geometric_mean_ratio_lte"], "lte"),
        "relaxation_win_fraction": bool_gate(rl["win_fraction"], g["relaxation_safe_vs_local_win_fraction_gte"], "gte"),
        "atlas_vs_generic_cubic_geometric_mean_ratio": bool_gate(gc["geometric_mean_ratio"], g["atlas_vs_generic_hybrid_cubic_geometric_mean_ratio_lte"], "lte"),
        "atlas_vs_generic_cubic_win_fraction": bool_gate(gc["win_fraction"], g["atlas_vs_generic_hybrid_cubic_win_fraction_gte"], "gte"),
        "minimum_cross_cell_h0_abs_correlation": bool_gate(min_h0, g["minimum_cross_cell_h0_abs_correlation"], "gte"),
        "minimum_cross_cell_h1_abs_correlation": bool_gate(min_h1, g["minimum_cross_cell_h1_abs_correlation"], "gte"),
        "eta_vs_qoff_pearson": bool_gate(summary["eta_vs_qoff_pearson"], g["eta_vs_qoff_pearson_lte"], "lte"),
    }
    return all(x["pass"] for x in gates.values()), gates


def holdout_gates(summary: dict[str, Any], protocol: dict[str, Any]) -> tuple[bool, dict[str, Any], bool, dict[str, Any]]:
    g = protocol["holdout_gates"]
    sl, tl, rl, gc = summary["safe_vs_local"], summary["tail_safe_vs_local"], summary["relaxation_safe_vs_local"], summary["atlas_vs_generic_hybrid_cubic"]
    gates = {
        "minimum_valid_cycles": bool_gate(summary["valid_cycles"], g["minimum_valid_cycles"], "gte"),
        "zero_catastrophic_failures": bool_gate(summary["catastrophic_failures"], 0, "eq"),
        "median_safe_joint_error": bool_gate(summary["median_safe_joint_error"], g["median_safe_joint_error_lte"], "lte"),
        "safe_vs_local_geometric_mean_ratio": bool_gate(sl["geometric_mean_ratio"], g["safe_vs_local_geometric_mean_ratio_lte"], "lte"),
        "safe_vs_local_median_ratio": bool_gate(sl["median_ratio"], g["safe_vs_local_median_ratio_lte"], "lte"),
        "safe_vs_local_win_fraction": bool_gate(sl["win_fraction"], g["safe_vs_local_win_fraction_gte"], "gte"),
        "safe_vs_local_p90_ratio": bool_gate(sl["p90_ratio"], g["safe_vs_local_p90_ratio_lte"], "lte"),
        "tail_geometric_mean_ratio": bool_gate(tl["geometric_mean_ratio"], g["tail_safe_vs_local_geometric_mean_ratio_lte"], "lte"),
        "tail_win_fraction": bool_gate(tl["win_fraction"], g["tail_safe_vs_local_win_fraction_gte"], "gte"),
        "relaxation_geometric_mean_ratio": bool_gate(rl["geometric_mean_ratio"], g["relaxation_safe_vs_local_geometric_mean_ratio_lte"], "lte"),
        "relaxation_win_fraction": bool_gate(rl["win_fraction"], g["relaxation_safe_vs_local_win_fraction_gte"], "gte"),
        "atlas_vs_generic_cubic_geometric_mean_ratio": bool_gate(gc["geometric_mean_ratio"], g["atlas_vs_generic_hybrid_cubic_geometric_mean_ratio_lte"], "lte"),
        "atlas_vs_generic_cubic_win_fraction": bool_gate(gc["win_fraction"], g["atlas_vs_generic_hybrid_cubic_win_fraction_gte"], "gte"),
    }
    strong_cfg = protocol["strong_claim_gates"]
    strong = {
        "eta_vs_qoff_pearson": bool_gate(summary["eta_vs_qoff_pearson"], strong_cfg["eta_vs_qoff_pearson_lte"], "lte"),
        "eta_vs_discharge_order_pearson": bool_gate(summary["eta_vs_discharge_order_pearson"], strong_cfg["eta_vs_discharge_order_pearson_gte"], "gte"),
    }
    return all(x["pass"] for x in gates.values()), gates, all(x["pass"] for x in strong.values()), strong


def rows_to_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_development(archive: Path, protocol_path: Path, out: Path) -> int:
    protocol = json.loads(protocol_path.read_text())
    expected = protocol["source"]["nested_archive_sha256"]
    actual = sha256_file(archive)
    if actual != expected:
        raise RuntimeError(f"Nested archive mismatch: {actual} != {expected}")
    out.mkdir(parents=True, exist_ok=True)
    payload_dir = out / "_development_payloads"
    member_evidence = []
    by_cell: dict[str, list[Cycle]] = {}
    for cell in protocol["access_policy"]["development_cells"]:
        path = payload_dir / f"{cell}.mat"
        member_evidence.append(extract_member(archive, cell, path))
        by_cell[cell] = load_cell_cycles(path, cell, protocol)
    unopened = list(protocol["access_policy"]["holdout_cells"])
    access = {
        "archive_sha256": actual,
        "opened_cells": sorted(by_cell),
        "explicitly_unopened_cells": unopened,
        "member_evidence": member_evidence,
        "cycles_per_cell": {k: len(v) for k, v in by_cell.items()},
    }
    write_json(out / "DEVELOPMENT_ACCESS_MANIFEST.json", access)

    atlases = {cell: build_atlas(cycles, protocol) for cell, cycles in by_cell.items()}
    cells = list(protocol["access_policy"]["development_cells"])
    rows = []
    for target in cells:
        train = [c for cell in cells if cell != target for c in by_cell[cell]]
        atlas = build_atlas(train, protocol)
        for cyc in by_cell[target]:
            row = evaluate_cycle(cyc, atlas, protocol)
            row["development_target_cell"] = target
            rows.append(row)
    summary = summarize(rows)
    correlations = compare_atlases(atlases[cells[0]], atlases[cells[1]])
    passed, gates = development_gates(summary, correlations, protocol)
    combined_cycles = [c for cell in cells for c in by_cell[cell]]
    combined_atlas = build_atlas(combined_cycles, protocol)
    lock = {
        "protocol_id": protocol["protocol_id"],
        "archive_sha256": actual,
        "development_cells": cells,
        "holdout_cells": protocol["access_policy"]["holdout_cells"],
        "development_pass": passed,
        "atlas": atlas_json(combined_atlas),
        "fixed_hyperparameters": {
            "current_off_threshold_A": protocol["inputs"]["current_off_threshold_A"],
            "eta_grid": protocol["atlas"]["eta_grid"],
            "coefficient_ridge": protocol["atlas"]["coefficient_ridge"],
            "eta_regularization": protocol["atlas"]["eta_regularization"],
            "blend_penalty": protocol["safe_fallback"]["blend_penalty"],
            "probe_rank_fractions": protocol["probes"]["rank_fractions"],
        },
        "development_summary": summary,
        "development_gates": gates,
        "cross_cell_atlas_correlations": correlations,
    }
    result = {
        "protocol_id": protocol["protocol_id"],
        "stage": "development-lock",
        "development_pass": passed,
        "summary": summary,
        "cross_cell_atlas_correlations": correlations,
        "gates": gates,
        "combined_atlas_stats": {
            "voltage": combined_atlas["voltage"]["stats"],
            "temperature": combined_atlas["temperature"]["stats"],
        },
    }
    write_json(out / "development_results.json", result)
    rows_to_csv(out / "development_per_cycle.csv", rows)
    write_json(out / "CANDIDATE_LOCK.json", lock)
    lock_hash = sha256_file(out / "CANDIDATE_LOCK.json")
    (out / "CANDIDATE_LOCK_SHA256.txt").write_text(f"{lock_hash}  CANDIDATE_LOCK.json\n")
    report = [
        "# Causal Loom v0.25 development-lock result", "",
        f"Development pass: **{passed}**", "",
        f"- Valid cycles: **{summary['valid_cycles']}**",
        f"- Catastrophic failures: **{summary['catastrophic_failures']}**",
        f"- Median safe joint error: **{summary['median_safe_joint_error']:.6f}**",
        f"- Safe/local geometric-mean ratio: **{summary['safe_vs_local']['geometric_mean_ratio']:.6f}**",
        f"- Safe/local wins: **{summary['safe_vs_local']['win_fraction']:.3%}**",
        f"- Tail geometric-mean ratio: **{summary['tail_safe_vs_local']['geometric_mean_ratio']:.6f}**",
        f"- Relaxation geometric-mean ratio: **{summary['relaxation_safe_vs_local']['geometric_mean_ratio']:.6f}**",
        f"- Atlas/generic-hybrid-cubic geometric-mean ratio: **{summary['atlas_vs_generic_hybrid_cubic']['geometric_mean_ratio']:.6f}**",
        f"- eta vs q_off Pearson correlation: **{summary['eta_vs_qoff_pearson']:.6f}**",
        "", "B0007 and B0018 were not opened during this stage.", ""
    ]
    (out / "DEVELOPMENT_REPORT.md").write_text("\n".join(report))
    return 0 if passed else 2


def run_holdout(archive: Path, protocol_path: Path, lock_path: Path, out: Path) -> int:
    protocol = json.loads(protocol_path.read_text())
    lock_bytes = lock_path.read_bytes()
    lock = json.loads(lock_bytes)
    if lock.get("development_pass") is not True:
        raise RuntimeError("Candidate lock does not authorize holdout")
    expected_archive = protocol["source"]["nested_archive_sha256"]
    actual_archive = sha256_file(archive)
    if actual_archive != expected_archive or lock.get("archive_sha256") != actual_archive:
        raise RuntimeError("Archive does not match frozen development lock")
    if lock.get("protocol_id") != protocol["protocol_id"]:
        raise RuntimeError("Protocol mismatch")
    out.mkdir(parents=True, exist_ok=True)
    payload_dir = out / "_holdout_payloads"
    member_evidence = []
    cycles: list[Cycle] = []
    for cell in protocol["access_policy"]["holdout_cells"]:
        path = payload_dir / f"{cell}.mat"
        member_evidence.append(extract_member(archive, cell, path))
        cycles.extend(load_cell_cycles(path, cell, protocol))
    access = {
        "archive_sha256": actual_archive,
        "candidate_lock_sha256": sha256_bytes(lock_bytes),
        "opened_cells": list(protocol["access_policy"]["holdout_cells"]),
        "member_evidence": member_evidence,
        "cycle_count": len(cycles),
    }
    write_json(out / "HOLDOUT_ACCESS_MANIFEST.json", access)
    atlas = atlas_from_json(lock["atlas"])
    rows = [evaluate_cycle(cyc, atlas, protocol) for cyc in cycles]
    summary = summarize(rows)
    primary, gates, strong_pass, strong_gates = holdout_gates(summary, protocol)
    result = {
        "protocol_id": protocol["protocol_id"],
        "stage": "frozen-holdout",
        "primary_pass": primary,
        "strong_claim_pass": strong_pass,
        "summary": summary,
        "gates": gates,
        "strong_claim_gates": strong_gates,
    }
    write_json(out / "holdout_results.json", result)
    rows_to_csv(out / "holdout_per_cycle.csv", rows)
    report = [
        "# Causal Loom v0.25 frozen NASA battery result", "",
        f"Primary pass: **{primary}**",
        f"Strong claim pass: **{strong_pass}**", "",
        f"- Valid cycles: **{summary['valid_cycles']}**",
        f"- Catastrophic failures: **{summary['catastrophic_failures']}**",
        f"- Median safe joint error: **{summary['median_safe_joint_error']:.6f}**",
        f"- Safe/local geometric-mean ratio: **{summary['safe_vs_local']['geometric_mean_ratio']:.6f}**",
        f"- Safe/local wins: **{summary['safe_vs_local']['win_fraction']:.3%}**",
        f"- Tail geometric-mean ratio: **{summary['tail_safe_vs_local']['geometric_mean_ratio']:.6f}**",
        f"- Relaxation geometric-mean ratio: **{summary['relaxation_safe_vs_local']['geometric_mean_ratio']:.6f}**",
        f"- Atlas/generic-hybrid-cubic geometric-mean ratio: **{summary['atlas_vs_generic_hybrid_cubic']['geometric_mean_ratio']:.6f}**",
        f"- eta vs q_off Pearson correlation: **{summary['eta_vs_qoff_pearson']:.6f}",
        ""
    ]
    (out / "HOLDOUT_REPORT.md").write_text("\n".join(report))
    return 0 if primary else 2


def self_test() -> int:
    n = 201
    t = np.linspace(0.0, 1000.0, n)
    current = np.r_[np.full(160, -2.0), np.zeros(n - 160)]
    dt = np.diff(t)
    q = np.r_[0.0, np.cumsum(0.5 * (np.abs(current[1:]) + np.abs(current[:-1])) * dt / 3600.0)]
    z_true = np.r_[np.linspace(0.0, 1.0, 160), 1.0 + np.linspace(1.0 / (n - 160), 1.0, n - 160)]
    voltage = 4.2 - 0.8 * z_true + 0.18 * np.sin(np.pi * z_true) + 0.15 * np.maximum(z_true - 1.0, 0.0)
    temperature = 24.0 + 15.0 * np.sin(np.pi * np.minimum(z_true, 1.0)) * np.exp(-0.8 * np.maximum(z_true - 1.0, 0.0))
    cyc = Cycle("SYN", 0, 0, 24.0, t, current, q, voltage, temperature)
    p = json.loads(Path(__file__).with_name("V025_FROZEN_PROTOCOL.json").read_text()) if Path(__file__).with_name("V025_FROZEN_PROTOCOL.json").exists() else {
        "inputs": {"current_off_threshold_A": 0.2},
        "atlas": {"grid_min": 0, "grid_max": 2, "grid_points": 401, "eta_grid": [-1.5, 1.5, 0.05], "coefficient_ridge": 0.01, "eta_regularization": 0.0001},
        "outputs": {"voltage": {"scale": 1.0, "clip": [2, 5]}, "temperature": {"scale": 20.0, "clip": [0, 80]}},
        "probes": {"rank_fractions": [0.06, 0.27, 0.50, 0.73, 0.92]},
        "safe_fallback": {"lambda_grid": [0, .25, .5, .75, 1], "blend_penalty": .001}
    }
    z, off, _, _ = hybrid_coordinate(cyc, p)
    if off != 159 or np.max(np.abs(z - z_true)) > 1e-8:
        raise AssertionError((off, np.max(np.abs(z - z_true))))
    train = []
    for j in range(8):
        eta = -0.4 + 0.1 * j
        train.append(Cycle("SYN", j, j, 24, t, current, q, voltage + eta * np.sin(2 * np.pi * z), temperature + 2 * eta * np.sin(2 * np.pi * z)))
    atlas = build_atlas(train, p)
    fit = fit_atlas(train[3], atlas, probe_indices(n, p["probes"]["rank_fractions"]), p)
    corr = abs(float(np.corrcoef(atlas["voltage"]["h0"], atlas["temperature"]["h0"])[0, 1]))
    if not math.isfinite(fit["eta"]) or corr < 0.4:
        raise AssertionError((fit["eta"], corr))
    print("SELF_TEST_PASS", off, fit["eta"], corr)
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
    required = [args.stage, args.archive, args.protocol, args.output_dir]
    if any(x is None for x in required):
        ap.error("--stage, --archive, --protocol, and --output-dir are required")
    if args.stage == "development-lock":
        return run_development(Path(args.archive), Path(args.protocol), Path(args.output_dir))
    if args.candidate_lock is None:
        ap.error("--candidate-lock is required for holdout")
    return run_holdout(Path(args.archive), Path(args.protocol), Path(args.candidate_lock), Path(args.output_dir))


if __name__ == "__main__":
    raise SystemExit(main())
