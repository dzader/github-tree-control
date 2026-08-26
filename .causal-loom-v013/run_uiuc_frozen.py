#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def stable_json(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n"


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def parse_static_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = np.genfromtxt(path, skip_header=1, dtype=float)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    if raw.ndim != 2 or raw.shape[1] < 2:
        raise ValueError(f"expected at least two numeric columns, got shape {raw.shape}")
    rpm = raw[:, 0]
    ct = raw[:, 1]
    mask = np.isfinite(rpm) & np.isfinite(ct) & (rpm > 0.0) & (ct > 0.0)
    rpm = rpm[mask]
    ct = ct[mask]
    if rpm.size == 0:
        raise ValueError("no positive finite RPM/C_T rows")
    order = np.argsort(rpm)
    rpm, ct = rpm[order], ct[order]
    # Collapse exact duplicate RPM rows without using outcome-dependent selection.
    unique = np.unique(rpm)
    ct_mean = np.array([np.mean(ct[rpm == value]) for value in unique], dtype=float)
    return unique.astype(float), ct_mean


def probe_indices(n: int) -> np.ndarray:
    idx = np.rint(np.array([0.2, 0.5, 0.8]) * (n - 1)).astype(int)
    if len(np.unique(idx)) != 3:
        raise ValueError(f"probe quantiles did not produce three distinct indices for n={n}: {idx.tolist()}")
    return idx


def mechanism(z: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    return coefficients[0] * z**2 + coefficients[1] * z**3


def fit_predict(name: str, z: np.ndarray, h: np.ndarray, y: np.ndarray, probes: np.ndarray) -> tuple[np.ndarray, int, float]:
    if name == "pmcd":
        x = np.column_stack([np.ones_like(z), z, h])
    elif name == "linear":
        x = np.column_stack([np.ones_like(z), z])
    elif name == "quadratic":
        x = np.column_stack([np.ones_like(z), z, z**2])
    elif name == "cubic_primitive":
        x = np.column_stack([np.ones_like(z), z, z**3])
    else:
        raise ValueError(name)
    xp = x[probes]
    beta, _, rank, singular = np.linalg.lstsq(xp, y[probes], rcond=None)
    cond = float(np.inf if singular.size == 0 or singular[-1] == 0 else singular[0] / singular[-1])
    return x @ beta, int(rank), cond


def nrmse(y_true: np.ndarray, y_pred: np.ndarray, evaluation_mask: np.ndarray, full_range: float) -> float:
    if full_range <= 0.0 or not np.isfinite(full_range):
        return math.inf
    err = y_pred[evaluation_mask] - y_true[evaluation_mask]
    return float(np.sqrt(np.mean(err**2)) / full_range)


def affine_residual_correlation(z: np.ndarray, y: np.ndarray, h: np.ndarray) -> float:
    a = np.column_stack([np.ones_like(z), z])
    y_res = y - a @ np.linalg.lstsq(a, y, rcond=None)[0]
    h_res = h - a @ np.linalg.lstsq(a, h, rcond=None)[0]
    ny = float(np.linalg.norm(y_res))
    nh = float(np.linalg.norm(h_res))
    if ny <= 1e-14 or nh <= 1e-14:
        return math.nan
    return float(abs(np.dot(y_res, h_res) / (ny * nh)))


def exact_two_sided_sign_p(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return float(min(1.0, 2.0 * tail))


def bootstrap_median_improvement(pmcd: np.ndarray, control: np.ndarray, seed: int = 13013, draws: int = 20000) -> list[float]:
    rng = np.random.default_rng(seed)
    n = len(pmcd)
    values = np.empty(draws, dtype=float)
    for i in range(draws):
        idx = rng.integers(0, n, size=n)
        values[i] = 1.0 - np.median(pmcd[idx]) / np.median(control[idx])
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def run(repo: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text())
    scale = float(protocol["frozen_mechanism"]["scale_rpm"])
    coefficients = np.asarray(protocol["frozen_mechanism"]["coefficients"], dtype=float)
    minimum_rows = int(protocol["measurement_model"]["minimum_numeric_rows"])
    catastrophe_threshold = float(protocol["frozen_gates"]["catastrophe_threshold_nrmse"])

    records: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    for spec in protocol["holdout"]["selected_files"]:
        path = repo / spec["path"]
        entry: dict[str, Any] = {"brand": spec["brand"], "family": spec["family"], "path": spec["path"], "expected_blob_sha": spec["blob_sha"]}
        try:
            data = path.read_bytes()
            actual_blob = git_blob_sha1(data)
            entry["actual_blob_sha"] = actual_blob
            if actual_blob != spec["blob_sha"]:
                raise ValueError(f"git blob mismatch: expected {spec['blob_sha']}, got {actual_blob}")
            rpm, ct = parse_static_file(path)
            if len(rpm) < minimum_rows:
                raise ValueError(f"only {len(rpm)} numeric rows; minimum is {minimum_rows}")
            z = rpm / scale
            y = ct * z**2
            h = mechanism(z, coefficients)
            probes = probe_indices(len(z))
            evaluation = np.ones(len(z), dtype=bool)
            evaluation[probes] = False
            if int(evaluation.sum()) < 4:
                raise ValueError("fewer than four held-out rows")
            full_range = float(np.ptp(y))
            models: dict[str, Any] = {}
            predictions: dict[str, np.ndarray] = {}
            for name in ["pmcd", "linear", "quadratic", "cubic_primitive"]:
                pred, rank, condition = fit_predict(name, z, h, y, probes)
                error = nrmse(y, pred, evaluation, full_range)
                predictions[name] = pred
                models[name] = {"nrmse": error, "rank": rank, "probe_condition_number": condition, "finite": bool(np.all(np.isfinite(pred)) and np.isfinite(error))}
            structural_corr = affine_residual_correlation(z, y, h)
            entry.update({
                "row_count": int(len(rpm)),
                "rpm_min": float(rpm.min()),
                "rpm_max": float(rpm.max()),
                "probe_indices": probes.tolist(),
                "probe_rpm": rpm[probes].tolist(),
                "models": models,
                "projective_residual_correlation": structural_corr,
                "pmcd_catastrophe": bool((not models["pmcd"]["finite"]) or models["pmcd"]["nrmse"] > catastrophe_threshold),
            })
            records.append(entry)
        except Exception as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
            invalid.append(entry)

    model_names = ["pmcd", "linear", "quadratic", "cubic_primitive"]
    valid_count = len(records)
    arrays = {name: np.array([r["models"][name]["nrmse"] for r in records], dtype=float) for name in model_names}
    correlations = np.array([r["projective_residual_correlation"] for r in records], dtype=float)
    finite_corr = correlations[np.isfinite(correlations)]
    medians = {name: (float(np.median(values)) if len(values) else math.inf) for name, values in arrays.items()}
    wins_linear = int(np.sum(arrays["pmcd"] < arrays["linear"])) if valid_count else 0
    losses_linear = int(np.sum(arrays["pmcd"] > arrays["linear"])) if valid_count else 0
    ties_linear = valid_count - wins_linear - losses_linear
    catastrophes = int(sum(r["pmcd_catastrophe"] for r in records)) + len(invalid)
    improvement_linear = (1.0 - medians["pmcd"] / medians["linear"]) if np.isfinite(medians["linear"]) and medians["linear"] > 0 else -math.inf
    aggregate_ratio_quadratic = medians["pmcd"] / medians["quadratic"] if np.isfinite(medians["quadratic"]) and medians["quadratic"] > 0 else math.inf
    win_fraction_linear = wins_linear / valid_count if valid_count else 0.0
    median_corr = float(np.median(finite_corr)) if len(finite_corr) else math.nan

    gates = {
        "all_selected_files_valid": len(invalid) == 0 and valid_count == int(protocol["holdout"]["selected_count"]),
        "zero_pmcd_catastrophes": catastrophes == 0,
        "median_pmcd_nrmse": medians["pmcd"] <= float(protocol["frozen_gates"]["median_pmcd_nrmse_max"]),
        "aggregate_pmcd_to_quadratic": aggregate_ratio_quadratic <= float(protocol["frozen_gates"]["aggregate_pmcd_to_quadratic_max_ratio"]),
        "median_improvement_over_linear": improvement_linear >= float(protocol["frozen_gates"]["median_improvement_over_linear_min"]),
        "win_fraction_over_linear": win_fraction_linear >= float(protocol["frozen_gates"]["win_fraction_over_linear_min"]),
        "median_projective_residual_correlation": np.isfinite(median_corr) and median_corr >= float(protocol["frozen_gates"]["median_projective_residual_correlation_min"]),
    }
    verdict = "PASS" if all(gates.values()) else "FAIL"
    bootstrap_ci = bootstrap_median_improvement(arrays["pmcd"], arrays["linear"]) if valid_count else [math.nan, math.nan]
    result = {
        "protocol_id": protocol["protocol_id"],
        "verdict": verdict,
        "first_selected_content_execution": True,
        "selected_count": int(protocol["holdout"]["selected_count"]),
        "valid_count": valid_count,
        "invalid_count": len(invalid),
        "catastrophe_count": catastrophes,
        "aggregate": {
            "median_nrmse": medians,
            "pmcd_to_quadratic_median_ratio": float(aggregate_ratio_quadratic),
            "pmcd_median_improvement_over_linear": float(improvement_linear),
            "pmcd_median_improvement_over_linear_bootstrap_95_ci": bootstrap_ci,
            "pmcd_wins_vs_linear": wins_linear,
            "pmcd_losses_vs_linear": losses_linear,
            "pmcd_ties_vs_linear": ties_linear,
            "pmcd_win_fraction_vs_linear": float(win_fraction_linear),
            "paired_sign_test_two_sided_p_vs_linear": exact_two_sided_sign_p(wins_linear, losses_linear),
            "median_projective_residual_correlation": median_corr,
        },
        "gates": gates,
        "invalid_files": invalid,
        "files": records,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(stable_json(result))
    lines = [
        "# Causal Loom v0.13 — frozen unseen-propeller replication",
        "",
        f"**Verdict: {verdict}**",
        "",
        f"- Frozen UIUC files: {result['selected_count']}",
        f"- Valid files: {valid_count}",
        f"- PMCD catastrophes: {catastrophes}",
        f"- Median PMCD NRMSE: {medians['pmcd']:.6f}",
        f"- Median supplied-quadratic NRMSE: {medians['quadratic']:.6f}",
        f"- PMCD / quadratic median-error ratio: {aggregate_ratio_quadratic:.6f}",
        f"- Median improvement over linear: {improvement_linear:.2%}",
        f"- Paired wins over linear: {wins_linear}/{valid_count} ({win_fraction_linear:.2%})",
        f"- Median projective residual correlation: {median_corr:.6f}",
        "",
        "## Frozen gates",
        "",
    ]
    for name, passed in gates.items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} — `{name}`")
    lines.extend(["", "The first run that read the selected UIUC files is permanent. No post-result retuning is permitted.", ""])
    (output / "verdict.md").write_text("\n".join(lines))
    return result


def self_test() -> None:
    z = np.linspace(0.1, 1.1, 13)
    h = mechanism(z, np.array([0.9996822704947699, -0.025206309892999604]))
    y = 0.2 + 0.4*z + 1.7*h
    p = probe_indices(len(z))
    pred, rank, _ = fit_predict("pmcd", z, h, y, p)
    if rank != 3 or float(np.max(np.abs(pred-y))) > 1e-10:
        raise RuntimeError("PMCD binding self-test failed")
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "synthetic_static.txt"
        path.write_text("RPM CT CP\n" + "\n".join(f"{1000+500*i} {0.1+0.0001*i} 0.05" for i in range(10)) + "\n")
        rpm, ct = parse_static_file(path)
        if len(rpm) != 10 or len(ct) != 10:
            raise RuntimeError("UIUC parser self-test failed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("self-test passed")
        return 0
    if args.repo is None or args.protocol is None or args.output is None:
        parser.error("--repo, --protocol, and --output are required unless --self-test is used")
    result = run(args.repo, args.protocol, args.output)
    print(stable_json({"verdict": result["verdict"], "aggregate": result["aggregate"], "gates": result["gates"]}))
    return 0 if result["verdict"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
