#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

EPS = 1e-12
OUTPUTS = ("ct", "cp")


def stable_fold(name: str, folds: int = 5) -> int:
    return int(hashlib.sha256(name.encode()).hexdigest()[:8], 16) % folds


def parse_uiuc_file(path: Path) -> np.ndarray:
    rows: List[List[float]] = []
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.replace(",", " ").strip()
        if not line:
            continue
        parts = line.split()
        vals: List[float] = []
        for token in parts:
            try:
                vals.append(float(token))
            except ValueError:
                break
        if len(vals) < 4:
            continue
        j, ct, cp, eta = vals[:4]
        if not all(math.isfinite(v) for v in (j, ct, cp, eta)):
            continue
        if not (-0.25 <= j <= 3.0 and abs(ct) <= 2.0 and abs(cp) <= 2.0):
            continue
        rows.append([j, ct, cp, eta])
    if len(rows) < 5:
        raise ValueError(f"insufficient numeric rows in {path}: {len(rows)}")
    a = np.asarray(rows, dtype=float)
    # Exact duplicate rows are metadata noise, not independent observations.
    a = np.unique(np.round(a, 12), axis=0)
    return a[np.argsort(a[:, 0], kind="mergesort")]


@dataclass
class Family:
    name: str
    diameter_in: float
    pitch_in: float
    j: np.ndarray
    rpm: np.ndarray
    ct: np.ndarray
    cp: np.ndarray

    @property
    def pitch_ratio(self) -> float:
        return self.pitch_in / self.diameter_in

    @property
    def log_d(self) -> float:
        return math.log(self.diameter_in / 10.0)

    @property
    def r(self) -> np.ndarray:
        R = (self.rpm / 5000.0) * (self.diameter_in / 10.0) ** 2
        return np.log(np.maximum(R, 1e-9))

    @property
    def z_fixed(self) -> np.ndarray:
        R = (self.rpm / 5000.0) * (self.diameter_in / 10.0) ** 2
        return self.j * np.power(np.maximum(R, 1e-9), -0.1)

    def y(self, output: str) -> np.ndarray:
        return self.ct if output == "ct" else self.cp


def load_families(inventory_path: Path, data_dir: Path) -> List[Family]:
    inv = json.loads(inventory_path.read_text())
    families: List[Family] = []
    for f in inv["families"]:
        js: List[np.ndarray] = []
        rpms: List[np.ndarray] = []
        cts: List[np.ndarray] = []
        cps: List[np.ndarray] = []
        for run in f["runs"]:
            arr = parse_uiuc_file(data_dir / run["filename"])
            n = len(arr)
            js.append(arr[:, 0])
            cts.append(arr[:, 1])
            cps.append(arr[:, 2])
            rpms.append(np.full(n, float(run["rpm"])))
        fam = Family(
            name=f["family"],
            diameter_in=float(f["diameter_in"]),
            pitch_in=float(f["pitch_in"]),
            j=np.concatenate(js),
            rpm=np.concatenate(rpms),
            ct=np.concatenate(cts),
            cp=np.concatenate(cps),
        )
        if len(fam.j) >= 20 and np.ptp(fam.ct) > 1e-5 and np.ptp(fam.cp) > 1e-5:
            families.append(fam)
    return families


def normalize_axis(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    if not hi > lo:
        return np.zeros_like(x)
    return 2.0 * (x - lo) / (hi - lo) - 1.0


def tensor_cheb(z: np.ndarray, r: np.ndarray, bounds: Dict[str, float], dz: int = 6, dr: int = 2) -> np.ndarray:
    uz = normalize_axis(z, bounds["z_lo"], bounds["z_hi"])
    ur = normalize_axis(r, bounds["r_lo"], bounds["r_hi"])
    bz = np.polynomial.chebyshev.chebvander(uz, dz)
    br = np.polynomial.chebyshev.chebvander(ur, dr)
    return np.einsum("ni,nj->nij", bz, br).reshape(len(z), -1)


def ridge_solve(X: np.ndarray, y: np.ndarray, lam: float, weights: np.ndarray | None = None) -> np.ndarray:
    if weights is not None:
        sw = np.sqrt(np.asarray(weights, float))
        X = X * sw[:, None]
        y = y * sw
    reg = np.eye(X.shape[1]) * lam
    reg[0, 0] = 0.0
    return np.linalg.solve(X.T @ X + reg, X.T @ y)


def fit_affine(h: np.ndarray, y: np.ndarray, lam: float = 1e-6) -> Tuple[float, float]:
    X = np.column_stack([np.ones(len(h)), h])
    beta = ridge_solve(X, y, lam)
    return float(beta[0]), float(beta[1])


def family_joint_error(f: Family, predictions: Dict[str, np.ndarray], idx: np.ndarray, floors: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
    errs: Dict[str, float] = {}
    for output in OUTPUTS:
        y = f.y(output)[idx]
        p = predictions[output][idx]
        den = max(float(np.ptp(y)), floors[output])
        errs[output] = float(np.sqrt(np.mean((p - y) ** 2)) / den)
    return float(math.sqrt(max(errs["ct"], EPS) * max(errs["cp"], EPS))), errs


def select_probes(f: Family, count: int = 5) -> np.ndarray:
    # Output-blind maximin design in the two physical coordinates available at test time.
    z = f.z_fixed
    r = f.r
    X = np.column_stack([
        normalize_axis(z, float(np.min(z)), float(np.max(z))),
        normalize_axis(r, float(np.min(r)), float(np.max(r))),
    ])
    # Collapse exact/near-exact coordinate duplicates, retaining a deterministic representative.
    keys = np.round(X, 8)
    _, unique_idx = np.unique(keys, axis=0, return_index=True)
    candidates = np.sort(unique_idx)
    if len(candidates) < count:
        candidates = np.arange(len(z))
    center = np.mean(X[candidates], axis=0)
    selected = [int(candidates[np.argmin(np.sum((X[candidates] - center) ** 2, axis=1))])]
    while len(selected) < count:
        remaining = np.asarray([i for i in candidates if int(i) not in selected], dtype=int)
        if len(remaining) == 0:
            remaining = np.asarray([i for i in range(len(z)) if i not in selected], dtype=int)
        d = np.min(
            np.sum((X[remaining, None, :] - X[np.asarray(selected)][None, :, :]) ** 2, axis=2),
            axis=1,
        )
        selected.append(int(remaining[np.argmax(d)]))
    return np.asarray(sorted(selected), dtype=int)


def fixed_cubic_predict(f: Family, probes: np.ndarray, coordinate: str = "fixed", ridge: float = 2e-3) -> Dict[str, np.ndarray]:
    x = f.z_fixed if coordinate == "fixed" else f.j
    lo, hi = float(np.min(x)), float(np.max(x))
    u = normalize_axis(x, lo, hi)
    X = np.column_stack([np.ones(len(u)), u, u ** 2, u ** 3])
    out: Dict[str, np.ndarray] = {}
    for output in OUTPUTS:
        beta = ridge_solve(X[probes], f.y(output)[probes], ridge)
        out[output] = X @ beta
    return out


def geometry_features(families: Sequence[Family]) -> np.ndarray:
    rows = []
    for f in families:
        ld = f.log_d
        pr = f.pitch_ratio
        rows.append([1.0, ld, pr, ld * pr, ld * ld, pr * pr])
    return np.asarray(rows, float)


@dataclass
class Atlas:
    bounds: Dict[str, float]
    coeffs: Dict[str, np.ndarray]
    tau_prior_beta: np.ndarray
    tau_lo: float
    tau_hi: float
    output_floors: Dict[str, float]
    prior_lambda: float

    def h(self, f: Family, tau: float, output: str) -> np.ndarray:
        return tensor_cheb(f.z_fixed + tau, f.r, self.bounds) @ self.coeffs[output]

    def prior_tau(self, f: Family) -> float:
        x = geometry_features([f])[0]
        return float(np.clip(x @ self.tau_prior_beta, self.tau_lo, self.tau_hi))


def normalized_family_targets(f: Family) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for output in OUTPUTS:
        y = f.y(output)
        center = float(np.median(y))
        scale = max(float(np.std(y)), float(np.percentile(y, 90) - np.percentile(y, 10)) / 2.563, 1e-4)
        out[output] = (y - center) / scale
    return out


def estimate_tau_full(f: Family, coeffs: Dict[str, np.ndarray], bounds: Dict[str, float], grid: np.ndarray, floors: Dict[str, float]) -> float:
    best = (float("inf"), 0.0)
    for tau in grid:
        score_parts = []
        for output in OUTPUTS:
            h = tensor_cheb(f.z_fixed + tau, f.r, bounds) @ coeffs[output]
            a, b = fit_affine(h, f.y(output))
            den = max(float(np.ptp(f.y(output))), floors[output])
            score_parts.append(float(np.sqrt(np.mean((a + b * h - f.y(output)) ** 2)) / den))
        score = math.sqrt(max(score_parts[0], EPS) * max(score_parts[1], EPS))
        if score < best[0]:
            best = (score, float(tau))
    return best[1]


def fit_atlas(train: Sequence[Family]) -> Atlas:
    all_z = np.concatenate([f.z_fixed for f in train])
    all_r = np.concatenate([f.r for f in train])
    z_pad = 0.35
    bounds = {
        "z_lo": float(np.min(all_z) - z_pad),
        "z_hi": float(np.max(all_z) + z_pad),
        "r_lo": float(np.min(all_r)),
        "r_hi": float(np.max(all_r)),
    }
    floors = {
        output: max(float(np.median([np.ptp(f.y(output)) for f in train])) * 0.15, 1e-4)
        for output in OUTPUTS
    }
    targets = {f.name: normalized_family_targets(f) for f in train}
    tau = {f.name: 0.0 for f in train}
    grid = np.linspace(-0.30, 0.30, 61)
    coeffs: Dict[str, np.ndarray] = {}
    for _ in range(4):
        for output in OUTPUTS:
            blocks = []
            ys = []
            ws = []
            for f in train:
                blocks.append(tensor_cheb(f.z_fixed + tau[f.name], f.r, bounds))
                ys.append(targets[f.name][output])
                ws.append(np.full(len(f.j), 1.0 / len(f.j)))
            coeffs[output] = ridge_solve(np.vstack(blocks), np.concatenate(ys), 4e-3, np.concatenate(ws))
        tau = {f.name: estimate_tau_full(f, coeffs, bounds, grid, floors) for f in train}
    G = geometry_features(train)
    t = np.asarray([tau[f.name] for f in train])
    prior_beta = ridge_solve(G, t, 0.35)
    return Atlas(
        bounds=bounds,
        coeffs=coeffs,
        tau_prior_beta=prior_beta,
        tau_lo=-0.30,
        tau_hi=0.30,
        output_floors=floors,
        prior_lambda=0.035,
    )


def atlas_predict(atlas: Atlas, f: Family, probes: np.ndarray) -> Tuple[Dict[str, np.ndarray], float]:
    prior = atlas.prior_tau(f)
    grid = np.unique(np.clip(np.r_[np.linspace(prior - 0.22, prior + 0.22, 57), prior, 0.0], atlas.tau_lo, atlas.tau_hi))
    best_score = float("inf")
    best_tau = prior
    best_params: Dict[str, Tuple[float, float]] = {}
    for tau in grid:
        params: Dict[str, Tuple[float, float]] = {}
        parts = []
        for output in OUTPUTS:
            h = atlas.h(f, float(tau), output)
            a, b = fit_affine(h[probes], f.y(output)[probes], 2e-4)
            params[output] = (a, b)
            parts.append(float(np.sqrt(np.mean((a + b * h[probes] - f.y(output)[probes]) ** 2)) / atlas.output_floors[output]))
        score = math.sqrt(max(parts[0], EPS) * max(parts[1], EPS)) + atlas.prior_lambda * ((tau - prior) / 0.18) ** 2
        if score < best_score:
            best_score = score
            best_tau = float(tau)
            best_params = params
    pred = {}
    for output in OUTPUTS:
        h = atlas.h(f, best_tau, output)
        a, b = best_params[output]
        pred[output] = a + b * h
    return pred, best_tau


def choose_safe_blend(atlas: Atlas, f: Family, probes: np.ndarray) -> float:
    loo_rows = []
    for held in probes:
        sub = probes[probes != held]
        ap, _ = atlas_predict(atlas, f, sub)
        fp = fixed_cubic_predict(f, sub, "fixed")
        loo_rows.append((int(held), ap, fp))
    best = (float("inf"), 0.0)
    for alpha in np.linspace(0.0, 1.0, 11):
        parts = {output: [] for output in OUTPUTS}
        for held, ap, fp in loo_rows:
            for output in OUTPUTS:
                p = alpha * ap[output][held] + (1.0 - alpha) * fp[output][held]
                parts[output].append((p - f.y(output)[held]) / atlas.output_floors[output])
        score = math.sqrt(
            max(float(np.sqrt(np.mean(np.square(parts["ct"])))), EPS)
            * max(float(np.sqrt(np.mean(np.square(parts["cp"])))), EPS)
        ) + 0.008 * alpha
        if score < best[0]:
            best = (score, float(alpha))
    return best[1]


def safe_atlas_predict(atlas: Atlas, f: Family, probes: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, float]:
    ap, tau = atlas_predict(atlas, f, probes)
    fp = fixed_cubic_predict(f, probes, "fixed")
    alpha = choose_safe_blend(atlas, f, probes)
    return {output: alpha * ap[output] + (1.0 - alpha) * fp[output] for output in OUTPUTS}, tau, alpha


def global_base_features(f: Family) -> np.ndarray:
    return np.column_stack([
        f.j,
        f.z_fixed,
        f.r,
        np.full(len(f.j), f.log_d),
        np.full(len(f.j), f.pitch_ratio),
    ])


@dataclass
class GlobalControls:
    models: Dict[str, Dict[str, object]]
    scales: Dict[str, Tuple[np.ndarray, np.ndarray]]


def fit_global_controls(train: Sequence[Family]) -> GlobalControls:
    X = np.vstack([global_base_features(f) for f in train])
    models: Dict[str, Dict[str, object]] = {"direct_poly": {}, "extra_trees": {}, "hist_gb": {}, "mlp": {}}
    scales: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for output in OUTPUTS:
        y = np.concatenate([f.y(output) for f in train])
        poly = make_pipeline(PolynomialFeatures(degree=3, include_bias=True), Ridge(alpha=0.02))
        poly.fit(X, y)
        models["direct_poly"][output] = poly
        et = ExtraTreesRegressor(n_estimators=180, min_samples_leaf=4, max_features=0.9, random_state=1701, n_jobs=-1)
        et.fit(X, y)
        models["extra_trees"][output] = et
        hg = HistGradientBoostingRegressor(max_iter=220, learning_rate=0.055, max_leaf_nodes=18, l2_regularization=0.02, random_state=1701)
        hg.fit(X, y)
        models["hist_gb"][output] = hg
        mlp = make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(40, 24), activation="tanh", alpha=0.01, max_iter=450, early_stopping=True, random_state=1701),
        )
        mlp.fit(X, y)
        models["mlp"][output] = mlp
    return GlobalControls(models=models, scales=scales)


def calibrated_global_predict(controls: GlobalControls, method: str, f: Family, probes: np.ndarray) -> Dict[str, np.ndarray]:
    X = global_base_features(f)
    out: Dict[str, np.ndarray] = {}
    for output in OUTPUTS:
        raw = np.asarray(controls.models[method][output].predict(X), float)
        a, b = fit_affine(raw[probes], f.y(output)[probes], 4e-4)
        out[output] = a + b * raw
    return out


def aggregate(records: Sequence[Dict[str, object]], method: str) -> Dict[str, float]:
    vals = np.asarray([float(r[f"{method}_joint"]) for r in records])
    fixed = np.asarray([float(r["fixed_joint"]) for r in records])
    ratio = vals / np.maximum(fixed, EPS)
    return {
        "median_joint": float(np.median(vals)),
        "geometric_mean_joint": float(np.exp(np.mean(np.log(np.maximum(vals, EPS))))),
        "median_ratio_vs_fixed": float(np.median(ratio)),
        "geometric_mean_ratio_vs_fixed": float(np.exp(np.mean(np.log(np.maximum(ratio, EPS))))),
        "win_fraction_vs_fixed": float(np.mean(ratio < 1.0)),
        "p90_ratio_vs_fixed": float(np.quantile(ratio, 0.9)),
        "worst_ratio_vs_fixed": float(np.max(ratio)),
    }


def run_cv(families: Sequence[Family]) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for fold in range(5):
        train = [f for f in families if stable_fold(f.name) != fold]
        test = [f for f in families if stable_fold(f.name) == fold]
        if not test:
            continue
        atlas = fit_atlas(train)
        controls = fit_global_controls(train)
        for f in test:
            probes = select_probes(f, 5)
            test_idx = np.asarray([i for i in range(len(f.j)) if i not in set(probes.tolist())], dtype=int)
            methods: Dict[str, Dict[str, np.ndarray]] = {
                "fixed": fixed_cubic_predict(f, probes, "fixed"),
                "plain_j": fixed_cubic_predict(f, probes, "j"),
            }
            safe, tau, alpha = safe_atlas_predict(atlas, f, probes)
            methods["safe_atlas"] = safe
            raw_atlas, _ = atlas_predict(atlas, f, probes)
            methods["raw_atlas"] = raw_atlas
            for name in controls.models:
                methods[name] = calibrated_global_predict(controls, name, f, probes)
            row: Dict[str, object] = {
                "family": f.name,
                "fold": fold,
                "diameter_in": f.diameter_in,
                "pitch_in": f.pitch_in,
                "points": len(f.j),
                "probe_indices": ";".join(str(x) for x in probes.tolist()),
                "atlas_tau": tau,
                "atlas_weight": alpha,
            }
            for name, pred in methods.items():
                joint, errs = family_joint_error(f, pred, test_idx, atlas.output_floors)
                row[f"{name}_joint"] = joint
                row[f"{name}_ct"] = errs["ct"]
                row[f"{name}_cp"] = errs["cp"]
            records.append(row)
    method_names = ["fixed", "plain_j", "safe_atlas", "raw_atlas", "direct_poly", "extra_trees", "hist_gb", "mlp"]
    summary: Dict[str, object] = {
        "protocol": "five-family-disjoint-folds; exactly five output-blind maximin probes; cubic fixed-coordinate control",
        "family_count": len(records),
        "methods": {name: aggregate(records, name) for name in method_names},
        "atlas_diagnostics": {
            "median_tau": float(np.median([float(r["atlas_tau"]) for r in records])),
            "median_atlas_weight": float(np.median([float(r["atlas_weight"]) for r in records])),
            "nonzero_atlas_weight_fraction": float(np.mean([float(r["atlas_weight"]) > 0 for r in records])),
            "full_atlas_weight_fraction": float(np.mean([float(r["atlas_weight"]) >= 0.999 for r in records])),
        },
    }
    a = summary["methods"]["safe_atlas"]
    control_gm = min(summary["methods"][x]["geometric_mean_joint"] for x in ("direct_poly", "extra_trees", "hist_gb", "mlp"))
    summary["development_gate"] = {
        "pass": bool(
            a["geometric_mean_ratio_vs_fixed"] <= 0.92
            and a["win_fraction_vs_fixed"] >= 0.65
            and a["worst_ratio_vs_fixed"] <= 1.80
            and a["geometric_mean_joint"] <= 0.90 * control_gm
        ),
        "requirements": {
            "geometric_mean_ratio_vs_fixed_lte": 0.92,
            "win_fraction_vs_fixed_gte": 0.65,
            "worst_ratio_vs_fixed_lte": 1.80,
            "geometric_mean_vs_best_learned_control_lte": 0.90,
        },
        "best_learned_control_geometric_mean_joint": float(control_gm),
        "atlas_ratio_vs_best_learned_control": float(a["geometric_mean_joint"] / control_gm),
    }
    return records, summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    families = load_families(Path(args.inventory), Path(args.data_dir))
    records, summary = run_cv(families)
    (out / "development_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if records:
        with (out / "per_family.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(records[0].keys()))
            w.writeheader()
            w.writerows(records)
    compact = {
        "family_count": summary["family_count"],
        "development_gate": summary["development_gate"],
        "safe_atlas": summary["methods"]["safe_atlas"],
        "fixed": summary["methods"]["fixed"],
        "direct_poly": summary["methods"]["direct_poly"],
        "extra_trees": summary["methods"]["extra_trees"],
        "hist_gb": summary["methods"]["hist_gb"],
        "mlp": summary["methods"]["mlp"],
        "atlas_diagnostics": summary["atlas_diagnostics"],
    }
    print("V019_FIVE_PROBE_DEVELOPMENT=" + json.dumps(compact, sort_keys=True))
    return 0 if summary["development_gate"]["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
