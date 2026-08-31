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
from typing import Dict, Mapping, Sequence

import numpy as np

EPS = 1e-12


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("v021_frozen_shared", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen shared development module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def atlas_from_model(module, model: Mapping[str, object]):
    return module.Atlas(
        x_lower=float(model["x_lower"]),
        x_upper=float(model["x_upper"]),
        output_coefficients={name: np.asarray(values, float) for name, values in model["output_coefficients"].items()},
        output_floors={name: float(value) for name, value in model["output_floors"].items()},
        context_center=np.asarray(model["context_center"], float),
        context_scale=np.asarray(model["context_scale"], float),
        c_prior=np.asarray(model["c_prior"], float),
        c_lower=float(model["c_lower"]),
        c_upper=float(model["c_upper"]),
        c_penalty=float(model["c_penalty"]),
    )


def base_only_predict(module, atlas, curve, probes: np.ndarray) -> Dict[str, np.ndarray]:
    matrix = module.basis(curve.x, atlas.x_lower, atlas.x_upper)
    predictions = {}
    for output, coefficients in atlas.output_coefficients.items():
        if output not in curve.outputs:
            continue
        shape = matrix @ coefficients[:, 0]
        a, b = module.fit_affine(shape[probes], curve.outputs[output][probes], 2e-4)
        predictions[output] = a + b * shape
    return predictions


def aggregate(records: Sequence[Mapping[str, object]], method: str) -> Dict[str, float]:
    values = np.asarray([float(record[f"{method}_joint"]) for record in records])
    cubic = np.asarray([float(record["cubic_joint"]) for record in records])
    ratio = values / np.maximum(cubic, EPS)
    return {
        "median_joint_nrmse": float(np.median(values)),
        "geometric_mean_joint_nrmse": float(np.exp(np.mean(np.log(np.maximum(values, EPS))))),
        "geometric_mean_ratio_vs_cubic": float(np.exp(np.mean(np.log(np.maximum(ratio, EPS))))),
        "median_ratio_vs_cubic": float(np.median(ratio)),
        "win_fraction_vs_cubic": float(np.mean(ratio < 1.0)),
        "worst_ratio_vs_cubic": float(np.max(ratio)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol")
    parser.add_argument("--model")
    parser.add_argument("--split-protocol")
    parser.add_argument("--shared-module")
    parser.add_argument("--extraction-manifest")
    parser.add_argument("--data-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        if not args.shared_module:
            print("SELF_TEST_PASS_NO_IMPORT")
            return 0
        module = load_module(Path(args.shared_module))
        curve = module.Curve("test", "family", "confirmation", np.linspace(0, 1, 20), {"output": np.linspace(0, 1, 20) ** 2}, np.zeros(4), "synthetic")
        probes = module.select_probes(curve, 5)
        assert len(probes) == 5 and len(set(probes.tolist())) == 5
        print("SELF_TEST_PASS", probes.tolist())
        return 0

    protocol_path = Path(args.protocol)
    model_path = Path(args.model)
    split_protocol_path = Path(args.split_protocol)
    shared_path = Path(args.shared_module)
    extraction_manifest_path = Path(args.extraction_manifest)
    data_root = Path(args.data_root)
    protocol = json.loads(protocol_path.read_text())
    model = json.loads(model_path.read_text())
    split_protocol = json.loads(split_protocol_path.read_text())

    integrity = protocol["integrity"]
    checks = {
        model_path: integrity["frozen_model_sha256"],
        split_protocol_path: integrity["premeasurement_protocol_sha256"],
        shared_path: integrity["development_runner_sha256"],
        Path(__file__): integrity["frozen_runner_sha256"],
    }
    for path, expected in checks.items():
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"integrity mismatch {path}: {actual} != {expected}")

    extraction_manifest = json.loads(extraction_manifest_path.read_text())
    expected_families = {family["family_key"] for family in protocol["confirmation"]["families"]}
    observed_families = {str(item["family_key"]) for item in extraction_manifest}
    unexpected = observed_families - expected_families
    missing = expected_families - observed_families
    if unexpected:
        raise RuntimeError(f"non-confirmation family in extraction: {sorted(unexpected)}")
    if missing:
        raise RuntimeError(f"confirmation family missing from extraction: {sorted(missing)}")
    if any(str(item["split"]) != "confirmation" for item in extraction_manifest):
        raise RuntimeError("non-confirmation split in frozen extraction")

    module = load_module(shared_path)
    atlas = atlas_from_model(module, model)
    curves, parse_failures = module.extract_curves(split_protocol, extraction_manifest, data_root)
    invalid_curves = list(parse_failures)
    records = []
    for curve in curves:
        if curve.split != "confirmation":
            raise RuntimeError("development or validation curve reached frozen evaluator")
        supported = [output for output in atlas.output_coefficients if output in curve.outputs]
        if not supported:
            invalid_curves.append({"curve_id": curve.curve_id, "family_key": curve.family_key, "reason": "no_frozen_output_supported"})
            continue
        probes = module.select_probes(curve, int(protocol["binding"]["probe_count"]))
        probe_set = set(probes.tolist())
        scored = np.asarray([index for index in range(len(curve.x)) if index not in probe_set], int)
        if len(scored) < int(protocol["validity"]["minimum_scored_points"]):
            invalid_curves.append({"curve_id": curve.curve_id, "family_key": curve.family_key, "reason": "too_few_scored_points"})
            continue
        cubic = module.cubic_predict(curve, probes, supported)
        base_only = base_only_predict(module, atlas, curve, probes)
        raw_atlas, c_value = module.atlas_predict(atlas, curve, probes)
        safe_atlas, _, atlas_weight = module.safe_atlas_predict(atlas, curve, probes)
        methods = {
            "cubic": cubic,
            "base_only": base_only,
            "raw_atlas": raw_atlas,
            "safe_atlas": safe_atlas,
        }
        row: Dict[str, object] = {
            "curve_id": curve.curve_id,
            "family_key": curve.family_key,
            "source": curve.source,
            "points": len(curve.x),
            "outputs": ";".join(supported),
            "atlas_c": c_value,
            "atlas_weight": atlas_weight,
            "probe_indices": ";".join(str(index) for index in probes.tolist()),
        }
        for method, prediction in methods.items():
            joint, errors = module.prediction_error(atlas, curve, prediction, scored)
            row[f"{method}_joint"] = joint
            for output, error in errors.items():
                row[f"{method}_{output}"] = error
        records.append(row)

    valid_family_count = len({record["family_key"] for record in records})
    summaries = {method: aggregate(records, method) for method in ("cubic", "base_only", "raw_atlas", "safe_atlas")} if records else {}
    safe = summaries.get("safe_atlas", {})
    candidate = np.asarray([float(record["safe_atlas_joint"]) for record in records]) if records else np.asarray([])
    base = np.asarray([float(record["base_only_joint"]) for record in records]) if records else np.asarray([])
    base_ratio = float(np.exp(np.mean(np.log(np.maximum(candidate / np.maximum(base, EPS), EPS))))) if records else float("inf")
    base_win = float(np.mean(candidate < base)) if records else 0.0
    catastrophic = sum(
        not math.isfinite(float(record["safe_atlas_joint"]))
        or float(record["safe_atlas_joint"]) >= float(protocol["validity"]["catastrophic_joint_nrmse"])
        for record in records
    )
    gates = {
        "minimum_valid_families": {
            "value": valid_family_count,
            "threshold": protocol["validity"]["minimum_valid_families"],
            "pass": valid_family_count >= protocol["validity"]["minimum_valid_families"],
        },
        "minimum_valid_curves": {
            "value": len(records),
            "threshold": protocol["validity"]["minimum_valid_curves"],
            "pass": len(records) >= protocol["validity"]["minimum_valid_curves"],
        },
        "zero_catastrophic": {
            "value": catastrophic,
            "threshold": 0,
            "pass": catastrophic == 0,
        },
        "median_joint_nrmse": {
            "value": safe.get("median_joint_nrmse", float("inf")),
            "threshold": protocol["gates"]["median_joint_nrmse_lte"],
            "pass": safe.get("median_joint_nrmse", float("inf")) <= protocol["gates"]["median_joint_nrmse_lte"],
        },
        "geometric_mean_ratio_vs_cubic": {
            "value": safe.get("geometric_mean_ratio_vs_cubic", float("inf")),
            "threshold": protocol["gates"]["geometric_mean_ratio_vs_cubic_lte"],
            "pass": safe.get("geometric_mean_ratio_vs_cubic", float("inf")) <= protocol["gates"]["geometric_mean_ratio_vs_cubic_lte"],
        },
        "win_fraction_vs_cubic": {
            "value": safe.get("win_fraction_vs_cubic", 0.0),
            "threshold": protocol["gates"]["win_fraction_vs_cubic_gte"],
            "pass": safe.get("win_fraction_vs_cubic", 0.0) >= protocol["gates"]["win_fraction_vs_cubic_gte"],
        },
        "geometric_mean_ratio_vs_base_only": {
            "value": base_ratio,
            "threshold": protocol["gates"]["geometric_mean_ratio_vs_base_only_lte"],
            "pass": base_ratio <= protocol["gates"]["geometric_mean_ratio_vs_base_only_lte"],
        },
        "win_fraction_vs_base_only": {
            "value": base_win,
            "threshold": protocol["gates"]["win_fraction_vs_base_only_gte"],
            "pass": base_win >= protocol["gates"]["win_fraction_vs_base_only_gte"],
        },
    }
    result = {
        "protocol_id": protocol["protocol_id"],
        "domain": protocol["domain"],
        "primary_pass": all(gate["pass"] for gate in gates.values()),
        "summary": {
            "valid_family_count": valid_family_count,
            "valid_curve_count": len(records),
            "invalid_curve_count": len(invalid_curves),
            "methods": summaries,
            "safe_vs_base_only": {
                "geometric_mean_ratio": base_ratio,
                "win_fraction": base_win,
            },
            "median_atlas_c": float(np.median([record["atlas_c"] for record in records])) if records else None,
            "median_atlas_weight": float(np.median([record["atlas_weight"] for record in records])) if records else None,
        },
        "gates": gates,
        "invalid_curves": invalid_curves,
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if records:
        with (output / "per_curve.csv").open("w", newline="") as handle:
            fieldnames = sorted({key for record in records for key in record})
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
    verdict = [
        "# Causal Loom v0.21 frozen cross-domain confirmation",
        "",
        f"Domain: **{result['domain']}**",
        f"Primary pass: **{result['primary_pass']}**",
        f"Valid families: **{valid_family_count}**",
        f"Valid curves: **{len(records)}**",
        "",
    ]
    for key, gate in gates.items():
        verdict.append(f"- {key}: {gate['value']} (gate {gate['threshold']}) — {'PASS' if gate['pass'] else 'FAIL'}")
    (output / "verdict.md").write_text("\n".join(verdict) + "\n")
    print("V021_FROZEN_RESULT=" + json.dumps({
        "domain": result["domain"],
        "primary_pass": result["primary_pass"],
        "summary": result["summary"],
        "gates": gates,
    }, sort_keys=True))
    return 0 if result["primary_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
