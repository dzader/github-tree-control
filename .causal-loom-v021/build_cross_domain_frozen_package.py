#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-summary", required=True)
    parser.add_argument("--development-model", required=True)
    parser.add_argument("--premeasurement-protocol", required=True)
    parser.add_argument("--metadata-manifest", required=True)
    parser.add_argument("--development-runner", required=True)
    parser.add_argument("--frozen-runner", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    development_summary_path = Path(args.development_summary)
    development_model_path = Path(args.development_model)
    split_protocol_path = Path(args.premeasurement_protocol)
    metadata_path = Path(args.metadata_manifest)
    development_runner_path = Path(args.development_runner)
    frozen_runner_path = Path(args.frozen_runner)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    development = json.loads(development_summary_path.read_text())
    split_protocol = json.loads(split_protocol_path.read_text())
    metadata = json.loads(metadata_path.read_text())
    passed = bool(development.get("development_gate", {}).get("pass", False))
    confirmation_families = [family for family in split_protocol["families"] if family["split"] == "confirmation"]
    status = {
        "development_gate_pass": passed,
        "confirmation_family_count": len(confirmation_families),
        "confirmation_measurement_values_accessed": False,
        "development_summary_sha256": sha256(development_summary_path),
        "development_model_sha256": sha256(development_model_path),
        "premeasurement_protocol_sha256": sha256(split_protocol_path),
        "metadata_manifest_sha256": sha256(metadata_path),
    }
    (output / "PRE_FREEZE_STATUS.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    if not passed:
        print("V021_FREEZE_REFUSED_DEVELOPMENT_GATE_FALSE")
        return 3
    if len(confirmation_families) < 2:
        raise RuntimeError("fewer than two metadata-frozen confirmation families")
    if split_protocol.get("status") != "ready_for_development_access":
        raise RuntimeError("premeasurement protocol is not ready")

    model = json.loads(development_model_path.read_text())
    frozen_model_path = output / "V021_FROZEN_MODEL.json"
    frozen_model_path.write_text(json.dumps(model, indent=2, sort_keys=True) + "\n")

    protocol = {
        "protocol_id": "causal-loom-v021-cross-domain-projective-response-atlas-frozen-1",
        "scientific_status": "precommitted after sealed development/validation and before confirmation-family measurement access",
        "domain": split_protocol["domain"],
        "source_record_id": split_protocol["source_record_id"],
        "source_title": split_protocol["source_title"],
        "hypothesis": "A projective low-rank response atlas learned on development families transfers to unseen physical families with exactly five output-blind probes, outperforming a local cubic and a base-mechanism-only control.",
        "confirmation": {
            "families": confirmation_families,
            "family_count": len(confirmation_families),
            "measurement_values_previously_accessed": False,
            "metadata_manifest_sha256": sha256(metadata_path),
            "premeasurement_protocol_sha256": sha256(split_protocol_path),
        },
        "binding": {
            "probe_count": 5,
            "probe_selection": "deterministic maximin over the observed independent variable, output-blind",
            "candidate": "rank-2 projective response atlas with one chart coordinate shared across outputs and LOO convex cubic fallback",
            "controls": [
                "five-probe local cubic",
                "five-probe affine binding to the frozen base mechanism with chart coordinate fixed at zero"
            ],
        },
        "validity": {
            "minimum_points_per_curve": 12,
            "minimum_scored_points": 7,
            "minimum_valid_families": 2,
            "minimum_valid_curves": 3,
            "catastrophic_joint_nrmse": 1.0,
        },
        "gates": {
            "median_joint_nrmse_lte": 0.20,
            "geometric_mean_ratio_vs_cubic_lte": 0.90,
            "win_fraction_vs_cubic_gte": 0.60,
            "geometric_mean_ratio_vs_base_only_lte": 0.90,
            "win_fraction_vs_base_only_gte": 0.60,
            "zero_catastrophic": True,
        },
        "integrity": {
            "frozen_model_sha256": sha256(frozen_model_path),
            "development_summary_sha256": sha256(development_summary_path),
            "development_model_sha256": sha256(development_model_path),
            "development_runner_sha256": sha256(development_runner_path),
            "frozen_runner_sha256": sha256(frozen_runner_path),
            "metadata_manifest_sha256": sha256(metadata_path),
            "premeasurement_protocol_sha256": sha256(split_protocol_path),
        },
        "negative_result_policy": "Any missed gate is a permanent frozen failure. Invalid curves and families remain listed with reasons. No parser, probe, model, control, metric, or threshold may change after confirmation extraction begins.",
    }
    frozen_protocol_path = output / "V021_FROZEN_PROTOCOL.json"
    frozen_protocol_path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    (output / "READY_FOR_CONFIRMATION_ACCESS.txt").write_text("YES\n")
    (output / "CONFIRMATION_ACCESS_STATUS.txt").write_text("False — confirmation-family measurement files were not extracted while building this package.\n")
    manifest = {
        path.name: sha256(path)
        for path in (
            frozen_model_path,
            frozen_protocol_path,
            development_summary_path,
            development_model_path,
            split_protocol_path,
            metadata_path,
            development_runner_path,
            frozen_runner_path,
        )
    }
    (output / "FROZEN_INPUT_SHA256.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print("V021_FROZEN_PACKAGE=" + json.dumps({
        "domain": protocol["domain"],
        "confirmation_family_count": len(confirmation_families),
        "model_sha256": sha256(frozen_model_path),
        "protocol_sha256": sha256(frozen_protocol_path),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
