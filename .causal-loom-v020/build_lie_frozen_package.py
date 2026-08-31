#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Sequence

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


def norm(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def column_role(value: object) -> str | None:
    h = norm(value)
    roles = {
        "diameter": ("diameter", "propellerdiameter", "diametermm", "diameterin", "diameterinch"),
        "pitch": ("pitch", "propellerpitch", "pitchmm", "pitchin", "pitchinch"),
        "key": ("id", "name", "propeller", "propellername", "configuration", "config", "label", "type", "filename"),
        "material": ("material", "propmaterial"),
    }
    for role, options in roles.items():
        if h in options or any(len(option) >= 5 and option in h for option in options):
            return role
    return None


def number(value: object) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def dimensional_value(value: object, header: object) -> float | None:
    x = number(value)
    if x is None or x <= 0:
        return None
    h = str(header or "").lower()
    if "mm" in h or x > 40:
        return x / 25.4
    if "cm" in h or 20 < x <= 40:
        return x / 2.54
    if "meter" in h or (x < 2 and "in" not in h):
        return x / 0.0254
    return x


def workbook_geometry(metadata: Dict[str, object]) -> List[Dict[str, object]]:
    result: List[Dict[str, object]] = []
    for sheet in metadata.get("general_information_workbook", {}).get("sheets", []):
        rows = sheet.get("nonempty_rows", [])
        for header_index, row in enumerate(rows[:30]):
            mapping: Dict[str, int] = {}
            for column, cell in enumerate(row):
                role = column_role(cell)
                if role and role not in mapping:
                    mapping[role] = column
            if "diameter" not in mapping or "pitch" not in mapping:
                continue
            for data in rows[header_index + 1 :]:
                if max(mapping.values()) >= len(data):
                    continue
                diameter = dimensional_value(data[mapping["diameter"]], row[mapping["diameter"]])
                pitch = dimensional_value(data[mapping["pitch"]], row[mapping["pitch"]])
                if diameter is None or pitch is None or not (1 <= diameter <= 100 and 0.1 <= pitch <= 100):
                    continue
                key_column = mapping.get("key", mapping["diameter"])
                key = str(data[key_column] or "").strip() or f"{diameter:g}x{pitch:g}"
                material = None
                if "material" in mapping and mapping["material"] < len(data) and data[mapping["material"]] is not None:
                    material = str(data[mapping["material"]]).strip()
                aliases = [key, norm(key), f"{diameter:g}x{pitch:g}", f"{diameter:g} x {pitch:g}"]
                if material:
                    aliases.extend([material, f"{key}_{material}"])
                result.append({"key": key, "diameter_in": diameter, "pitch_in": pitch, "material": material, "aliases": sorted(set(aliases))})
            break
    return result


def path_geometry(paths: Sequence[str]) -> List[Dict[str, object]]:
    result: List[Dict[str, object]] = []
    patterns = [
        re.compile(r"(?<!\d)(\d+(?:[._]\d+)?)\s*[xX]\s*(\d+(?:[._]\d+)?)(?!\d)"),
        re.compile(r"(?<!\d)(\d{2,3})[-_](\d{1,3})(?!\d)"),
    ]
    for path in paths:
        for pattern in patterns:
            match = pattern.search(path)
            if not match:
                continue
            diameter = float(match.group(1).replace("_", "."))
            pitch = float(match.group(2).replace("_", "."))
            if diameter > 40 and float(match.group(1)).is_integer():
                diameter /= 10.0
            if pitch > 30 and float(match.group(2)).is_integer():
                pitch /= 10.0
            if not (1 <= diameter <= 40 and 0.1 <= pitch <= 30):
                continue
            parts = [part for part in Path(path).parts if part]
            containing = next((part for part in parts if match.group(0).lower() in part.lower()), Path(path).stem)
            aliases = [containing, Path(path).stem, match.group(0), f"{diameter:g}x{pitch:g}"]
            if parts:
                aliases.extend(parts[: min(3, len(parts))])
            result.append({"key": containing, "diameter_in": diameter, "pitch_in": pitch, "material": None, "aliases": sorted(set(aliases))})
            break
    return result


def merge_catalog(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    merged: Dict[tuple, Dict[str, object]] = {}
    for row in rows:
        signature = (norm(row["key"]), round(float(row["diameter_in"]), 4), round(float(row["pitch_in"]), 4))
        if signature not in merged:
            merged[signature] = dict(row)
        else:
            merged[signature]["aliases"] = sorted(set(merged[signature].get("aliases", []) + row.get("aliases", [])))
    return sorted(merged.values(), key=lambda row: (str(row["key"]), float(row["diameter_in"]), float(row["pitch_in"])))


def measurement_candidates(paths: Sequence[str]) -> List[str]:
    allowed = {".xlsx", ".xlsm", ".csv", ".txt", ".dat", ".tsv"}
    excluded = ("readme", "generalnfo", "generalinfo", "general_info", "license", "metadata")
    return sorted({path for path in paths if not path.endswith("/") and Path(path).suffix.lower() in allowed and not any(token in Path(path).name.lower() for token in excluded)})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-summary", required=True)
    parser.add_argument("--metadata-summary", required=True)
    parser.add_argument("--shared-module", required=True)
    parser.add_argument("--lowrank-module", required=True)
    parser.add_argument("--lie-module", required=True)
    parser.add_argument("--frozen-runner", required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    development_path = Path(args.development_summary)
    metadata_path = Path(args.metadata_summary)
    shared_path = Path(args.shared_module)
    lowrank_path = Path(args.lowrank_module)
    lie_path = Path(args.lie_module)
    runner_path = Path(args.frozen_runner)
    inventory_path = Path(args.inventory)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    development = json.loads(development_path.read_text())
    metadata = json.loads(metadata_path.read_text())
    gate = bool(development.get("development_gate", {}).get("pass", False))
    (output / "PRE_FREEZE_STATUS.json").write_text(json.dumps({
        "development_gate_pass": gate,
        "development_summary_sha256": sha256(development_path),
        "metadata_summary_sha256": sha256(metadata_path),
        "measurement_values_accessed": False,
    }, indent=2, sort_keys=True) + "\n")
    if not gate:
        print("LIE_FREEZE_REFUSED_DEVELOPMENT_GATE_FALSE")
        return 3

    shared = load_module(shared_path, "v020_lie_freeze_shared")
    lowrank = load_module(lowrank_path, "v020_lie_freeze_lowrank")
    lie = load_module(lie_path, "v020_lie_freeze_impl")
    families = shared.load_families(inventory_path, Path(args.data_dir))
    atlas, _, lowrank_atlas = lie.fit_atlas(shared, lowrank, families)

    model = {
        "model_id": "causal-loom-v020-projective-lie-transport-atlas-five-probe",
        "architecture": {
            "shape_coordinate": "z=J*R^-0.1",
            "context_coordinate": "r=ln(R)",
            "R": "(RPM/5000)*(D_in/10)^2",
            "flow": "h_o(z,r;c)=h0_o(z+c*v(z,r),r)",
            "shared_generator": "one learned vector field v and one family chart coordinate c shared by CT and CP",
            "write_binding": "output-specific affine writes from exactly five probes",
            "safe_fallback": "leave-one-probe-out convex blend with fixed-coordinate cubic",
        },
        "training": {
            "family_count": len(families),
            "families": sorted(family.name for family in families),
            "inventory_sha256": sha256(inventory_path),
            "shared_module_sha256": sha256(shared_path),
            "lowrank_module_sha256": sha256(lowrank_path),
            "lie_module_sha256": sha256(lie_path),
        },
        "atlas": {
            "bounds": atlas.bounds,
            "base_coefficients": {name: np.asarray(value).tolist() for name, value in atlas.base_coeffs.items()},
            "generator_coefficients": np.asarray(atlas.generator_coeffs).tolist(),
            "c_prior_beta": np.asarray(atlas.c_prior_beta).tolist(),
            "output_floors": atlas.output_floors,
            "c_lo": atlas.c_lo,
            "c_hi": atlas.c_hi,
            "prior_lambda": atlas.prior_lambda,
            "basis": {"base_z_degree": 6, "base_r_degree": 2, "generator_z_degree": 2, "generator_r_degree": 2},
        },
        "development": {
            "summary_sha256": sha256(development_path),
            "generator_alignment": development.get("generator_alignment"),
        },
    }
    model_path = output / "V020_LIE_ATLAS_MODEL.json"
    model_path.write_text(json.dumps(model, indent=2, sort_keys=True) + "\n")

    candidates = measurement_candidates(metadata["archive_paths"])
    catalog = merge_catalog(workbook_geometry(metadata) + path_geometry(candidates))
    if len(candidates) < 3 or len(catalog) < 6:
        raise RuntimeError(f"insufficient metadata inventory candidates={len(candidates)} catalog={len(catalog)}")

    protocol = {
        "protocol_id": "causal-loom-v020-enola-projective-lie-transport-atlas-frozen-1",
        "scientific_status": "precommitted before ENOLA performance-table access",
        "hypothesis": "A learned flow generator transports a reusable physical mechanism across systems, and five probes identify only the chart position and output writes.",
        "holdout": {
            "source": "ENOLA Propeller Database Zenodo record 20111572",
            "archive_sha256": metadata["archive_sha256"],
            "archive_bytes": metadata["archive_bytes"],
            "candidate_member_paths": candidates,
            "geometry_catalog": catalog,
            "metadata_summary_sha256": sha256(metadata_path),
            "performance_values_previously_read": False,
        },
        "physics": {"air_density_kg_m3": 1.225},
        "binding": {
            "probe_count": 5,
            "probe_selection": "output-blind deterministic maximin in normalized (z_fixed,r)",
            "candidate": "Projective Lie Transport Atlas with shared CT/CP flow coordinate and nested cubic fallback",
            "controls": ["fixed-coordinate cubic in J*R^-0.1", "plain-J cubic"],
            "outputs": ["CT", "CP"],
        },
        "validity": {"minimum_rows": 12, "minimum_scored_rows": 7, "minimum_output_range": 0.0001},
        "gates": {
            "minimum_valid_families": 6,
            "catastrophic_joint_nrmse": 1.0,
            "median_joint_nrmse_lte": 0.15,
            "geometric_mean_ratio_vs_fixed_lte": 0.90,
            "win_fraction_vs_fixed_gte": 0.60,
            "geometric_mean_ratio_vs_plain_j_lte": 0.83,
            "win_fraction_vs_plain_j_gte": 0.70,
        },
        "integrity": {
            "model_sha256": sha256(model_path),
            "shared_module_sha256": sha256(shared_path),
            "lowrank_module_sha256": sha256(lowrank_path),
            "lie_module_sha256": sha256(lie_path),
            "frozen_runner_sha256": sha256(runner_path),
            "development_summary_sha256": sha256(development_path),
            "metadata_summary_sha256": sha256(metadata_path),
        },
        "negative_result_policy": "Any missed gate is a permanent frozen failure. Invalid members and configurations remain in the evidence output with reasons.",
    }
    protocol_path = output / "V020_LIE_FROZEN_PROTOCOL.json"
    protocol_path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    (output / "READY_FOR_FIRST_MEASUREMENT_ACCESS.txt").write_text("YES\n")
    (output / "MEASUREMENT_ACCESS_STATUS.txt").write_text("No ENOLA performance member was extracted or parsed while building this package.\n")
    manifest = {path.name: sha256(path) for path in (model_path, protocol_path, shared_path, lowrank_path, lie_path, runner_path, development_path, metadata_path)}
    (output / "FROZEN_INPUT_SHA256.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print("V020_LIE_FROZEN_PACKAGE=" + json.dumps({
        "candidate_members": len(candidates),
        "catalog_entries": len(catalog),
        "model_sha256": sha256(model_path),
        "protocol_sha256": sha256(protocol_path),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
