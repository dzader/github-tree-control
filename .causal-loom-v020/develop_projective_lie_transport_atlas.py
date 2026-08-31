#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence

import numpy as np

EPS = 1e-12
OUTPUTS = ("ct", "cp")


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def geometry_features(family):
    ld = family.log_d
    pr = family.pitch_ratio
    return np.asarray([1.0, ld, pr, ld * pr, ld * ld, pr * pr], float)


def generator_basis(shared, z, r, bounds):
    return shared.tensor_cheb(z, r, bounds, dz=2, dr=2)


def projective_corr(a, b):
    a = np.asarray(a, float) - float(np.mean(a))
    b = np.asarray(b, float) - float(np.mean(b))
    if float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
        return 0.0
    return float(abs(np.corrcoef(a, b)[0, 1]))


@dataclass
class LieAtlas:
    bounds: Dict[str, float]
    base_coeffs: Dict[str, np.ndarray]
    generator_coeffs: np.ndarray
    c_prior_beta: np.ndarray
    output_floors: Dict[str, float]
    c_lo: float = -1.7
    c_hi: float = 1.7
    prior_lambda: float = 0.025

    def vector_field(self, shared, family):
        return generator_basis(shared, family.z_fixed, family.r, self.bounds) @ self.generator_coeffs

    def h(self, shared, family, c_value, output):
        v = self.vector_field(shared, family)
        z = family.z_fixed + c_value * v
        margin = 1e-6
        z = np.clip(z, self.bounds["z_lo"] + margin, self.bounds["z_hi"] - margin)
        return shared.tensor_cheb(z, family.r, self.bounds) @ self.base_coeffs[output]

    def prior(self, family):
        return float(np.clip(geometry_features(family) @ self.c_prior_beta, self.c_lo, self.c_hi))


def fit_generator(shared, lowrank, lowrank_atlas, train):
    designs = []
    targets = []
    weights = []
    epsilon = 1e-4
    for family in train:
        Bv = generator_basis(shared, family.z_fixed, family.r, lowrank_atlas.bounds)
        family_weight = np.full(len(family.j), 1.0 / len(family.j))
        for output in OUTPUTS:
            C0 = lowrank_atlas.coeffs[output][:, 0]
            C1 = lowrank_atlas.coeffs[output][:, 1]
            hp = shared.tensor_cheb(family.z_fixed + epsilon, family.r, lowrank_atlas.bounds) @ C0
            hm = shared.tensor_cheb(family.z_fixed - epsilon, family.r, lowrank_atlas.bounds) @ C0
            derivative = (hp - hm) / (2.0 * epsilon)
            tangent = shared.tensor_cheb(family.z_fixed, family.r, lowrank_atlas.bounds) @ C1
            informative = np.abs(derivative) > np.percentile(np.abs(derivative), 15)
            if int(np.sum(informative)) < 5:
                informative = np.ones(len(derivative), dtype=bool)
            designs.append(Bv[informative] * derivative[informative, None])
            targets.append(tangent[informative])
            weights.append(family_weight[informative])
    coeffs = shared.ridge_solve(np.vstack(designs), np.concatenate(targets), 0.015, np.concatenate(weights))
    # Normalize the physical flow field; the family chart coordinate absorbs global scale.
    all_v = np.concatenate([
        generator_basis(shared, f.z_fixed, f.r, lowrank_atlas.bounds) @ coeffs
        for f in train
    ])
    scale = max(float(np.sqrt(np.mean(all_v * all_v))), 1e-8)
    return coeffs / scale


def normalized_target(family, output):
    y = family.y(output)
    center = float(np.median(y))
    scale = max(float(np.std(y)), float(np.percentile(y, 90) - np.percentile(y, 10)) / 2.563, 1e-4)
    return (y - center) / scale


def full_score(shared, atlas, family, c_value, indices=None):
    if indices is None:
        indices = np.arange(len(family.j))
    parts = []
    for output in OUTPUTS:
        h = atlas.h(shared, family, c_value, output)
        a, b = shared.fit_affine(h[indices], family.y(output)[indices], 2e-5)
        den = max(float(np.ptp(family.y(output)[indices])), atlas.output_floors[output])
        parts.append(float(np.sqrt(np.mean((a + b * h[indices] - family.y(output)[indices]) ** 2)) / den))
    return math.sqrt(max(parts[0], EPS) * max(parts[1], EPS))


def estimate_full_c(shared, atlas, family):
    grid = np.linspace(atlas.c_lo, atlas.c_hi, 101)
    scores = np.asarray([full_score(shared, atlas, family, float(c)) for c in grid])
    return float(grid[int(np.argmin(scores))])


def refit_base(shared, atlas, train, c_values):
    updated = {}
    for output in OUTPUTS:
        blocks = []
        targets = []
        weights = []
        for family in train:
            v = atlas.vector_field(shared, family)
            z = np.clip(
                family.z_fixed + c_values[family.name] * v,
                atlas.bounds["z_lo"] + 1e-6,
                atlas.bounds["z_hi"] - 1e-6,
            )
            blocks.append(shared.tensor_cheb(z, family.r, atlas.bounds))
            targets.append(normalized_target(family, output))
            weights.append(np.full(len(family.j), 1.0 / len(family.j)))
        updated[output] = shared.ridge_solve(np.vstack(blocks), np.concatenate(targets), 0.006, np.concatenate(weights))
    return updated


def fit_atlas(shared, lowrank, train):
    lowrank_atlas, _ = lowrank.fit_atlas(shared, train)
    floors = {
        output: max(float(np.median([np.ptp(f.y(output)) for f in train])) * 0.15, 1e-4)
        for output in OUTPUTS
    }
    atlas = LieAtlas(
        bounds=lowrank_atlas.bounds,
        base_coeffs={output: lowrank_atlas.coeffs[output][:, 0].copy() for output in OUTPUTS},
        generator_coeffs=fit_generator(shared, lowrank, lowrank_atlas, train),
        c_prior_beta=np.zeros(6),
        output_floors=floors,
    )
    c_values = {family.name: 0.0 for family in train}
    for _ in range(3):
        c_values = {family.name: estimate_full_c(shared, atlas, family) for family in train}
        atlas.base_coeffs = refit_base(shared, atlas, train, c_values)
    G = np.vstack([geometry_features(family) for family in train])
    c = np.asarray([c_values[family.name] for family in train])
    atlas.c_prior_beta = shared.ridge_solve(G, c, 0.35)
    return atlas, c_values, lowrank_atlas


def atlas_predict(shared, atlas, family, probes):
    prior = atlas.prior(family)
    grid = np.unique(np.clip(np.r_[np.linspace(prior - 1.05, prior + 1.05, 85), prior, 0.0], atlas.c_lo, atlas.c_hi))
    best = (float("inf"), prior, {})
    for c_value in grid:
        params = {}
        parts = []
        for output in OUTPUTS:
            h = atlas.h(shared, family, float(c_value), output)
            a, b = shared.fit_affine(h[probes], family.y(output)[probes], 2e-4)
            params[output] = (a, b)
            parts.append(float(np.sqrt(np.mean((a + b * h[probes] - family.y(output)[probes]) ** 2)) / atlas.output_floors[output]))
        score = math.sqrt(max(parts[0], EPS) * max(parts[1], EPS)) + atlas.prior_lambda * ((c_value - prior) / 0.85) ** 2
        if score < best[0]:
            best = (score, float(c_value), params)
    _, c_value, params = best
    prediction = {}
    for output in OUTPUTS:
        h = atlas.h(shared, family, c_value, output)
        a, b = params[output]
        prediction[output] = a + b * h
    return prediction, c_value


def choose_blend(shared, atlas, family, probes):
    rows = []
    for held in probes:
        subset = probes[probes != held]
        atlas_prediction, _ = atlas_predict(shared, atlas, family, subset)
        fixed_prediction = shared.fixed_cubic_predict(family, subset, "fixed")
        rows.append((int(held), atlas_prediction, fixed_prediction))
    best = (float("inf"), 0.0)
    for alpha in np.linspace(0, 1, 11):
        errors = {output: [] for output in OUTPUTS}
        for held, atlas_prediction, fixed_prediction in rows:
            for output in OUTPUTS:
                value = alpha * atlas_prediction[output][held] + (1.0 - alpha) * fixed_prediction[output][held]
                errors[output].append((value - family.y(output)[held]) / atlas.output_floors[output])
        score = math.sqrt(
            max(float(np.sqrt(np.mean(np.square(errors["ct"])))), EPS)
            * max(float(np.sqrt(np.mean(np.square(errors["cp"])))), EPS)
        ) + 0.008 * alpha
        if score < best[0]:
            best = (score, float(alpha))
    return best[1]


def safe_predict(shared, atlas, family, probes):
    atlas_prediction, c_value = atlas_predict(shared, atlas, family, probes)
    fixed_prediction = shared.fixed_cubic_predict(family, probes, "fixed")
    alpha = choose_blend(shared, atlas, family, probes)
    return {
        output: alpha * atlas_prediction[output] + (1.0 - alpha) * fixed_prediction[output]
        for output in OUTPUTS
    }, c_value, alpha


def generator_alignment(shared, atlas, lowrank_atlas, train):
    values = {output: [] for output in OUTPUTS}
    epsilon = 1e-4
    for family in train:
        v = atlas.vector_field(shared, family)
        for output in OUTPUTS:
            C0 = lowrank_atlas.coeffs[output][:, 0]
            C1 = lowrank_atlas.coeffs[output][:, 1]
            hp = shared.tensor_cheb(family.z_fixed + epsilon, family.r, atlas.bounds) @ C0
            hm = shared.tensor_cheb(family.z_fixed - epsilon, family.r, atlas.bounds) @ C0
            generated = v * (hp - hm) / (2 * epsilon)
            tangent = shared.tensor_cheb(family.z_fixed, family.r, atlas.bounds) @ C1
            values[output].append(projective_corr(generated, tangent))
    return {output: float(np.median(values[output])) for output in OUTPUTS}


def aggregate(records, name):
    values = np.asarray([row[f"{name}_joint"] for row in records], float)
    fixed = np.asarray([row["fixed_joint"] for row in records], float)
    ratio = values / np.maximum(fixed, EPS)
    return {
        "median_joint": float(np.median(values)),
        "geometric_mean_joint": float(np.exp(np.mean(np.log(np.maximum(values, EPS))))),
        "median_ratio_vs_fixed": float(np.median(ratio)),
        "geometric_mean_ratio_vs_fixed": float(np.exp(np.mean(np.log(np.maximum(ratio, EPS))))),
        "win_fraction_vs_fixed": float(np.mean(ratio < 1.0)),
        "p90_ratio_vs_fixed": float(np.quantile(ratio, 0.9)),
        "worst_ratio_vs_fixed": float(np.max(ratio)),
    }


def run(shared, lowrank, families):
    records = []
    alignments = []
    for fold in range(5):
        train = [f for f in families if shared.stable_fold(f.name) != fold]
        test = [f for f in families if shared.stable_fold(f.name) == fold]
        if not test:
            continue
        atlas, _, lowrank_atlas = fit_atlas(shared, lowrank, train)
        controls = shared.fit_global_controls(train)
        alignments.append(generator_alignment(shared, atlas, lowrank_atlas, train))
        for family in test:
            probes = shared.select_probes(family, 5)
            scored = np.asarray([i for i in range(len(family.j)) if i not in set(probes.tolist())], int)
            methods = {
                "fixed": shared.fixed_cubic_predict(family, probes, "fixed"),
                "plain_j": shared.fixed_cubic_predict(family, probes, "j"),
            }
            safe, c_value, alpha = safe_predict(shared, atlas, family, probes)
            raw, _ = atlas_predict(shared, atlas, family, probes)
            methods["safe_atlas"] = safe
            methods["raw_atlas"] = raw
            for name in controls.models:
                methods[name] = shared.calibrated_global_predict(controls, name, family, probes)
            row = {
                "family": family.name,
                "fold": fold,
                "diameter_in": family.diameter_in,
                "pitch_in": family.pitch_in,
                "points": len(family.j),
                "atlas_c": c_value,
                "atlas_weight": alpha,
            }
            for name, prediction in methods.items():
                joint, errors = shared.family_joint_error(family, prediction, scored, atlas.output_floors)
                row[f"{name}_joint"] = joint
                row[f"{name}_ct"] = errors["ct"]
                row[f"{name}_cp"] = errors["cp"]
            records.append(row)
    names = ["fixed", "plain_j", "safe_atlas", "raw_atlas", "direct_poly", "extra_trees", "hist_gb", "mlp"]
    methods = {name: aggregate(records, name) for name in names}
    candidate = methods["safe_atlas"]
    best_learned = min(methods[name]["geometric_mean_joint"] for name in ("direct_poly", "extra_trees", "hist_gb", "mlp"))
    alignment = {
        output: float(np.median([fold[output] for fold in alignments]))
        for output in OUTPUTS
    }
    summary = {
        "protocol": "five family-disjoint folds; exactly five output-blind probes; matched cubic and learned controls",
        "family_count": len(records),
        "methods": methods,
        "generator_alignment": {**alignment, "per_fold": alignments},
        "atlas_diagnostics": {
            "median_c": float(np.median([row["atlas_c"] for row in records])),
            "median_weight": float(np.median([row["atlas_weight"] for row in records])),
            "nonzero_weight_fraction": float(np.mean([row["atlas_weight"] > 0 for row in records])),
        },
    }
    summary["development_gate"] = {
        "pass": bool(
            candidate["geometric_mean_ratio_vs_fixed"] <= 0.90
            and candidate["win_fraction_vs_fixed"] >= 0.65
            and candidate["worst_ratio_vs_fixed"] <= 1.75
            and candidate["geometric_mean_joint"] <= 0.90 * best_learned
            and alignment["ct"] >= 0.80
            and alignment["cp"] >= 0.80
        ),
        "requirements": {
            "geometric_mean_ratio_vs_fixed_lte": 0.90,
            "win_fraction_vs_fixed_gte": 0.65,
            "worst_ratio_vs_fixed_lte": 1.75,
            "ratio_vs_best_learned_control_lte": 0.90,
            "generator_alignment_each_gte": 0.80,
        },
        "best_learned_control_geometric_mean_joint": best_learned,
        "atlas_ratio_vs_best_learned_control": candidate["geometric_mean_joint"] / best_learned,
    }
    return records, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared-module", required=True)
    parser.add_argument("--lowrank-module", required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    shared = load_module(Path(args.shared_module), "v020_lie_shared")
    lowrank = load_module(Path(args.lowrank_module), "v020_lie_lowrank")
    families = shared.load_families(Path(args.inventory), Path(args.data_dir))
    records, summary = run(shared, lowrank, families)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "development_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    with (output / "per_family.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print("V020_PROJECTIVE_LIE_ATLAS=" + json.dumps({
        "family_count": summary["family_count"],
        "development_gate": summary["development_gate"],
        "safe_atlas": summary["methods"]["safe_atlas"],
        "fixed": summary["methods"]["fixed"],
        "generator_alignment": summary["generator_alignment"],
        "atlas_diagnostics": summary["atlas_diagnostics"],
    }, sort_keys=True))
    return 0 if summary["development_gate"]["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
