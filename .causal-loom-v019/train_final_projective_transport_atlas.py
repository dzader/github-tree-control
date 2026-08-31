#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("v019dev", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load development module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--development-module", required=True)
    ap.add_argument("--inventory", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    module_path = Path(args.development_module)
    inventory_path = Path(args.inventory)
    m = load_module(module_path)
    families = m.load_families(inventory_path, Path(args.data_dir))
    atlas = m.fit_atlas(families)
    payload = {
        "model_id": "causal-loom-v019-projective-transport-atlas-five-probe",
        "architecture": {
            "shape_coordinate": "z_fixed = J * R^-0.1",
            "context_coordinate": "r = ln(R)",
            "R": "(RPM/5000)*(D_in/10)^2",
            "transport": "shared additive tau on z_fixed for CT and CP",
            "write_binding": "output-specific affine a_o+b_o*h_o from exactly five probes",
            "fallback": "leave-one-probe-out convex blend with fixed-coordinate cubic",
        },
        "training": {
            "family_count": len(families),
            "family_names": sorted(f.name for f in families),
            "inventory_sha256": sha256(inventory_path),
            "development_module_sha256": sha256(module_path),
        },
        "atlas": {
            "bounds": atlas.bounds,
            "coefficients": {k: np.asarray(v).tolist() for k, v in atlas.coeffs.items()},
            "tau_prior_beta": np.asarray(atlas.tau_prior_beta).tolist(),
            "tau_lo": atlas.tau_lo,
            "tau_hi": atlas.tau_hi,
            "output_floors": atlas.output_floors,
            "prior_lambda": atlas.prior_lambda,
            "basis": {"z_chebyshev_degree": 6, "r_chebyshev_degree": 2},
        },
        "frozen_binding": {
            "probe_count": 5,
            "probe_design": "output-blind deterministic maximin in normalized (z_fixed, r)",
            "fixed_control_degree": 3,
            "fixed_control_ridge": 0.002,
            "atlas_affine_ridge": 0.0002,
            "blend_grid": [round(float(x), 10) for x in np.linspace(0, 1, 11)],
            "blend_penalty_per_unit_atlas_weight": 0.008,
        },
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("FINAL_ATLAS_MODEL=" + json.dumps({"model_id": payload["model_id"], "family_count": len(families), "output": str(out)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
