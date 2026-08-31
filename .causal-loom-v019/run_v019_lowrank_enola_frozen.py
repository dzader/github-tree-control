#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def lowrank_atlas_from_json(lowrank, model):
    atlas = model["atlas"]
    return lowrank.LowRankAtlas(
        bounds={key: float(value) for key, value in atlas["bounds"].items()},
        coeffs={key: np.asarray(value, float) for key, value in atlas["coefficients"].items()},
        c_prior_beta=np.asarray(atlas["c_prior_beta"], float),
        output_floors={key: float(value) for key, value in atlas["output_floors"].items()},
        c_lo=float(atlas["c_lo"]),
        c_hi=float(atlas["c_hi"]),
        prior_lambda=float(atlas["prior_lambda"]),
    )


def aggregate(rows, method):
    values = np.asarray([float(row[f"{method}_joint"]) for row in rows])
    fixed = np.asarray([float(row["fixed_joint"]) for row in rows])
    ratio = values / np.maximum(fixed, 1e-12)
    return {
        "median_joint_nrmse": float(np.median(values)),
        "geometric_mean_joint_nrmse": float(np.exp(np.mean(np.log(np.maximum(values, 1e-12))))),
        "median_ratio_vs_fixed": float(np.median(ratio)),
        "geometric_mean_ratio_vs_fixed": float(np.exp(np.mean(np.log(np.maximum(ratio, 1e-12))))),
        "win_fraction_vs_fixed": float(np.mean(ratio < 1.0)),
        "worst_ratio_vs_fixed": float(np.max(ratio)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol")
    parser.add_argument("--model")
    parser.add_argument("--shared-module")
    parser.add_argument("--lowrank-module")
    parser.add_argument("--parser-module")
    parser.add_argument("--archive")
    parser.add_argument("--output-dir")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        shared = load_module(Path(args.shared_module), "v019_lowrank_selftest_shared") if args.shared_module else None
        lowrank = load_module(Path(args.lowrank_module), "v019_lowrank_selftest_impl") if args.lowrank_module else None
        if shared is None or lowrank is None:
            print("SELF_TEST_PASS_IMPORT_SKIPPED")
            return 0
        z = np.linspace(0.1, 0.9, 30)
        r = np.zeros_like(z)
        bounds = {"z_lo": 0.0, "z_hi": 1.0, "r_lo": -1.0, "r_hi": 1.0}
        B = shared.tensor_cheb(z, r, bounds)
        assert B.shape[0] == len(z)
        print("SELF_TEST_PASS", B.shape)
        return 0

    protocol_path = Path(args.protocol)
    model_path = Path(args.model)
    shared_path = Path(args.shared_module)
    lowrank_path = Path(args.lowrank_module)
    parser_path = Path(args.parser_module)
    archive_path = Path(args.archive)
    protocol = json.loads(protocol_path.read_text())
    model = json.loads(model_path.read_text())

    integrity = protocol["integrity"]
    checks = {
        model_path: integrity["model_sha256"],
        shared_path: integrity["shared_module_sha256"],
        lowrank_path: integrity["lowrank_module_sha256"],
        Path(__file__): integrity["frozen_runner_sha256"],
    }
    for path, expected in checks.items():
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"integrity mismatch {path}: {actual} != {expected}")
    if sha256(archive_path) != protocol["holdout"]["archive_sha256"]:
        raise RuntimeError("ENOLA archive SHA-256 mismatch")

    shared = load_module(shared_path, "v019_lowrank_frozen_shared")
    lowrank = load_module(lowrank_path, "v019_lowrank_frozen_impl")
    parser_module = load_module(parser_path, "v019_lowrank_enola_parser")
    atlas = lowrank_atlas_from_json(lowrank, model)
    families, invalid_members, invalid_families = parser_module.build_families(shared, archive_path, protocol)

    records = []
    for family in families:
        probes = shared.select_probes(family, int(protocol["binding"]["probe_count"]))
        probe_set = set(probes.tolist())
        scored = np.asarray([index for index in range(len(family.j)) if index not in probe_set], int)
        if len(scored) < int(protocol["validity"]["minimum_scored_rows"]):
            invalid_families.append({"family": family.name, "reason": "too_few_scored_rows", "rows": len(family.j)})
            continue
        safe, c_value, alpha = lowrank.safe_predict(shared, atlas, family, probes)
        raw, _ = lowrank.atlas_predict(shared, atlas, family, probes)
        fixed = shared.fixed_cubic_predict(family, probes, "fixed")
        plain = shared.fixed_cubic_predict(family, probes, "j")
        row = {
            "family": family.name,
            "diameter_in": family.diameter_in,
            "pitch_in": family.pitch_in,
            "rows": len(family.j),
            "probe_indices": ";".join(str(index) for index in probes.tolist()),
            "atlas_c": c_value,
            "atlas_weight": alpha,
        }
        for name, prediction in (("safe_atlas", safe), ("raw_atlas", raw), ("fixed", fixed), ("plain_j", plain)):
            joint, errors = shared.family_joint_error(family, prediction, scored, atlas.output_floors)
            row[f"{name}_joint"] = joint
            row[f"{name}_ct"] = errors["ct"]
            row[f"{name}_cp"] = errors["cp"]
        records.append(row)

    methods = {name: aggregate(records, name) for name in ("safe_atlas", "raw_atlas", "fixed", "plain_j")} if records else {}
    candidate = np.asarray([float(row["safe_atlas_joint"]) for row in records]) if records else np.asarray([])
    plain_values = np.asarray([float(row["plain_j_joint"]) for row in records]) if records else np.asarray([])
    plain_ratio = float(np.exp(np.mean(np.log(np.maximum(candidate / np.maximum(plain_values, 1e-12), 1e-12))))) if records else float("inf")
    plain_win = float(np.mean(candidate < plain_values)) if records else 0.0
    safe = methods.get("safe_atlas", {})
    gates = {
        "minimum_valid_families": {
            "value": len(records),
            "threshold": protocol["gates"]["minimum_valid_families"],
            "pass": len(records) >= protocol["gates"]["minimum_valid_families"],
        },
        "zero_catastrophic": {
            "value": sum(not math.isfinite(float(row["safe_atlas_joint"])) or float(row["safe_atlas_joint"]) >= protocol["gates"]["catastrophic_joint_nrmse"] for row in records),
            "threshold": 0,
            "pass": all(math.isfinite(float(row["safe_atlas_joint"])) and float(row["safe_atlas_joint"]) < protocol["gates"]["catastrophic_joint_nrmse"] for row in records),
        },
        "median_joint_nrmse": {
            "value": safe.get("median_joint_nrmse", float("inf")),
            "threshold": protocol["gates"]["median_joint_nrmse_lte"],
            "pass": safe.get("median_joint_nrmse", float("inf")) <= protocol["gates"]["median_joint_nrmse_lte"],
        },
        "gm_ratio_vs_fixed": {
            "value": safe.get("geometric_mean_ratio_vs_fixed", float("inf")),
            "threshold": protocol["gates"]["geometric_mean_ratio_vs_fixed_lte"],
            "pass": safe.get("geometric_mean_ratio_vs_fixed", float("inf")) <= protocol["gates"]["geometric_mean_ratio_vs_fixed_lte"],
        },
        "win_fraction_vs_fixed": {
            "value": safe.get("win_fraction_vs_fixed", 0.0),
            "threshold": protocol["gates"]["win_fraction_vs_fixed_gte"],
            "pass": safe.get("win_fraction_vs_fixed", 0.0) >= protocol["gates"]["win_fraction_vs_fixed_gte"],
        },
        "gm_ratio_vs_plain_j": {
            "value": plain_ratio,
            "threshold": protocol["gates"]["geometric_mean_ratio_vs_plain_j_lte"],
            "pass": plain_ratio <= protocol["gates"]["geometric_mean_ratio_vs_plain_j_lte"],
        },
        "win_fraction_vs_plain_j": {
            "value": plain_win,
            "threshold": protocol["gates"]["win_fraction_vs_plain_j_gte"],
            "pass": plain_win >= protocol["gates"]["win_fraction_vs_plain_j_gte"],
        },
    }
    result = {
        "protocol_id": protocol["protocol_id"],
        "primary_pass": all(gate["pass"] for gate in gates.values()),
        "summary": {
            "valid_family_count": len(records),
            "invalid_family_count": len(invalid_families),
            "invalid_member_count": len(invalid_members),
            "methods": methods,
            "safe_vs_plain_j": {"geometric_mean_ratio": plain_ratio, "win_fraction": plain_win},
            "median_c": float(np.median([row["atlas_c"] for row in records])) if records else None,
            "median_atlas_weight": float(np.median([row["atlas_weight"] for row in records])) if records else None,
        },
        "gates": gates,
        "invalid_members": invalid_members,
        "invalid_families": invalid_families,
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if records:
        with (output / "per_family.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)
    verdict = [
        "# Causal Loom v0.19 low-rank frozen ENOLA result",
        "",
        f"Primary pass: **{result['primary_pass']}**",
        f"Valid configurations: **{len(records)}**",
        f"Invalid configurations: **{len(invalid_families)}**",
        "",
    ]
    for key, gate in gates.items():
        verdict.append(f"- {key}: {gate['value']} (gate {gate['threshold']}) — {'PASS' if gate['pass'] else 'FAIL'}")
    (output / "verdict.md").write_text("\n".join(verdict) + "\n")
    print("V019_LOWRANK_FROZEN_RESULT=" + json.dumps({"primary_pass": result["primary_pass"], "summary": result["summary"], "gates": gates}, sort_keys=True))
    return 0 if result["primary_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
