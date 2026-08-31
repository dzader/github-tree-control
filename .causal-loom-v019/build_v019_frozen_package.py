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
from typing import Dict, Iterable, List, Sequence

import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def norm(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def load_module(path: Path):
    spec=importlib.util.spec_from_file_location("v019_freeze_shared",path)
    if spec is None or spec.loader is None: raise RuntimeError("cannot import shared module")
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module);return module


def column_role(value: object) -> str | None:
    h=norm(value)
    roles={
      "diameter":("diameter","propellerdiameter","diametermm","diameterin","diameterinch"),
      "pitch":("pitch","propellerpitch","pitchmm","pitchin","pitchinch"),
      "key":("id","name","propeller","propellername","configuration","config","label","type","filename"),
      "material":("material","propmaterial"),
    }
    for role,opts in roles.items():
        if h in opts or any(len(o)>=5 and o in h for o in opts): return role
    return None


def to_float(v: object) -> float | None:
    try: x=float(v)
    except (TypeError,ValueError): return None
    return x if math.isfinite(x) else None


def dimensional_value(value: object, header: object) -> float | None:
    x=to_float(value)
    if x is None or x<=0: return None
    h=str(header or "").lower()
    if "mm" in h or x>40: return x/25.4
    if "cm" in h or 20<x<=40: return x/2.54
    if "meter" in h or (x<2 and "in" not in h): return x/0.0254
    return x


def workbook_geometry(metadata: Dict[str,object]) -> List[Dict[str,object]]:
    out=[]
    wb=metadata.get("general_information_workbook",{})
    for sheet in wb.get("sheets",[]):
        rows=sheet.get("nonempty_rows",[])
        for i,row in enumerate(rows[:30]):
            mapping={}
            for j,cell in enumerate(row):
                role=column_role(cell)
                if role and role not in mapping: mapping[role]=j
            if "diameter" not in mapping or "pitch" not in mapping: continue
            for data in rows[i+1:]:
                if max(mapping.values())>=len(data): continue
                D=dimensional_value(data[mapping["diameter"]],row[mapping["diameter"]])
                P=dimensional_value(data[mapping["pitch"]],row[mapping["pitch"]])
                if D is None or P is None or not (1<=D<=100 and 0.1<=P<=100): continue
                key=str(data[mapping.get("key",mapping["diameter"]) ] or "").strip()
                if not key: key=f"{D:g}x{P:g}"
                material=str(data[mapping["material"]]).strip() if "material" in mapping and mapping["material"]<len(data) and data[mapping["material"]] is not None else None
                aliases=[key,norm(key),f"{D:g}x{P:g}",f"{D:g} x {P:g}"]
                if material: aliases.extend([material,f"{key}_{material}"])
                out.append({"key":key,"diameter_in":D,"pitch_in":P,"material":material,"aliases":sorted(set(a for a in aliases if a))})
            break
    return out


def path_geometry(paths: Sequence[str]) -> List[Dict[str,object]]:
    out=[]
    patterns=[
      re.compile(r"(?<!\d)(\d+(?:[._]\d+)?)\s*[xX]\s*(\d+(?:[._]\d+)?)(?!\d)"),
      re.compile(r"(?<!\d)(\d{2,3})[-_](\d{1,3})(?!\d)"),
    ]
    for path in paths:
        lower=path.lower()
        for pattern in patterns:
            m=pattern.search(path)
            if not m: continue
            D=float(m.group(1).replace("_",".") );P=float(m.group(2).replace("_","."))
            # Compact labels such as 125x75 conventionally mean 12.5x7.5.
            if D>40 and float(m.group(1)).is_integer(): D/=10.0
            if P>30 and float(m.group(2)).is_integer(): P/=10.0
            if not (1<=D<=40 and 0.1<=P<=30): continue
            parts=[p for p in Path(path).parts if p]
            containing=next((p for p in parts if m.group(0).lower() in p.lower()),Path(path).stem)
            key=containing
            aliases=[containing,Path(path).stem,m.group(0),f"{D:g}x{P:g}"]
            if parts: aliases.extend(parts[:min(3,len(parts))])
            out.append({"key":key,"diameter_in":D,"pitch_in":P,"material":None,"aliases":sorted(set(aliases))})
            break
    return out


def merge_catalog(rows: Sequence[Dict[str,object]]) -> List[Dict[str,object]]:
    merged={}
    for row in rows:
        sig=(norm(row["key"]),round(float(row["diameter_in"]),4),round(float(row["pitch_in"]),4))
        if sig not in merged: merged[sig]=dict(row)
        else: merged[sig]["aliases"]=sorted(set(merged[sig].get("aliases",[])+row.get("aliases",[])))
    # Prefer entries with richer aliases, but preserve every distinct labeled configuration.
    return sorted(merged.values(),key=lambda x:(str(x["key"]),float(x["diameter_in"]),float(x["pitch_in"])))


def measurement_candidates(paths: Sequence[str]) -> List[str]:
    allowed={".xlsx",".xlsm",".csv",".txt",".dat",".tsv"}
    excluded=("readme","generalnfo","generalinfo","general_info","license","metadata")
    out=[]
    for path in paths:
        if path.endswith("/"): continue
        base=Path(path).name.lower()
        if Path(base).suffix.lower() not in allowed: continue
        if any(x in base for x in excluded): continue
        out.append(path)
    return sorted(set(out))


def serialize_model(module, families, inventory: Path, shared: Path) -> Dict[str,object]:
    atlas=module.fit_atlas(families)
    return {
      "model_id":"causal-loom-v019-projective-transport-atlas-five-probe",
      "architecture":{
        "shape_coordinate":"z=J*R^-0.1","context_coordinate":"r=ln(R)","R":"(RPM/5000)*(D_in/10)^2",
        "shared_transport":"one family-local tau shifts z for both CT and CP",
        "write_binding":"independent affine output writes from exactly five probes",
        "safe_fallback":"LOO-selected convex blend with five-probe fixed-coordinate cubic",
      },
      "training":{"family_count":len(families),"families":sorted(f.name for f in families),"inventory_sha256":sha256(inventory),"shared_module_sha256":sha256(shared)},
      "atlas":{"bounds":atlas.bounds,"coefficients":{k:np.asarray(v).tolist() for k,v in atlas.coeffs.items()},"tau_prior_beta":np.asarray(atlas.tau_prior_beta).tolist(),"tau_lo":atlas.tau_lo,"tau_hi":atlas.tau_hi,"output_floors":atlas.output_floors,"prior_lambda":atlas.prior_lambda,"basis":{"z_degree":6,"r_degree":2}},
    }


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--development-summary",required=True)
    ap.add_argument("--metadata-summary",required=True)
    ap.add_argument("--shared-module",required=True)
    ap.add_argument("--frozen-runner",required=True)
    ap.add_argument("--inventory",required=True)
    ap.add_argument("--data-dir",required=True)
    ap.add_argument("--output-dir",required=True)
    args=ap.parse_args()
    dev_path=Path(args.development_summary);meta_path=Path(args.metadata_summary);shared_path=Path(args.shared_module);runner_path=Path(args.frozen_runner);inv_path=Path(args.inventory)
    dev=json.loads(dev_path.read_text());meta=json.loads(meta_path.read_text());out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    gate=bool(dev.get("development_gate",{}).get("pass",False))
    status={"development_gate_pass":gate,"development_summary_sha256":sha256(dev_path),"metadata_summary_sha256":sha256(meta_path),"measurement_values_accessed":False}
    (out/"PRE_FREEZE_STATUS.json").write_text(json.dumps(status,indent=2,sort_keys=True)+"\n")
    if not gate:
        print("FREEZE_REFUSED_DEVELOPMENT_GATE_FALSE")
        return 3
    paths=meta["archive_paths"]
    candidates=measurement_candidates(paths)
    catalog=merge_catalog(workbook_geometry(meta)+path_geometry(candidates))
    if len(candidates)<3: raise RuntimeError("too few metadata-selected candidate result files")
    if len(catalog)<6: raise RuntimeError(f"too few geometry catalog entries: {len(catalog)}")
    module=load_module(shared_path);families=module.load_families(inv_path,Path(args.data_dir));model=serialize_model(module,families,inv_path,shared_path)
    model_path=out/"V019_ATLAS_MODEL.json";model_path.write_text(json.dumps(model,indent=2,sort_keys=True)+"\n")
    protocol={
      "protocol_id":"causal-loom-v019-enola-projective-transport-atlas-frozen-1",
      "scientific_status":"precommitted before ENOLA performance-table access",
      "hypothesis":"A low-rank shared transport atlas plus output write binding transfers more efficiently than a scalar corrected coordinate fitted independently in each output.",
      "holdout":{"source":"ENOLA Propeller Database Zenodo record 20111572","archive_sha256":meta["archive_sha256"],"archive_bytes":meta["archive_bytes"],"candidate_member_paths":candidates,"geometry_catalog":catalog,"metadata_summary_sha256":sha256(meta_path),"performance_values_previously_read":False},
      "physics":{"air_density_kg_m3":1.225},
      "binding":{"probe_count":5,"probe_selection":"output-blind deterministic maximin in normalized (z_fixed,r)","candidate":"shared projective transport atlas with nested convex cubic fallback","controls":["fixed-coordinate cubic in J*R^-0.1","plain-J cubic"],"outputs":["CT","CP"]},
      "validity":{"minimum_rows":12,"minimum_scored_rows":7,"minimum_output_range":0.0001},
      "gates":{"minimum_valid_families":6,"catastrophic_joint_nrmse":1.0,"median_joint_nrmse_lte":0.15,"geometric_mean_ratio_vs_fixed_lte":0.92,"win_fraction_vs_fixed_gte":0.60,"geometric_mean_ratio_vs_plain_j_lte":0.85,"win_fraction_vs_plain_j_gte":0.70},
      "integrity":{"model_sha256":sha256(model_path),"shared_module_sha256":sha256(shared_path),"frozen_runner_sha256":sha256(runner_path),"development_summary_sha256":sha256(dev_path),"metadata_summary_sha256":sha256(meta_path)},
      "negative_result_policy":"Any missed gate is a permanent frozen failure. Invalid members and configurations remain in the evidence output with reasons.",
    }
    protocol_path=out/"V019_FROZEN_PROTOCOL.json";protocol_path.write_text(json.dumps(protocol,indent=2,sort_keys=True)+"\n")
    manifest={p.name:sha256(p) for p in (model_path,protocol_path,shared_path,runner_path,dev_path,meta_path)}
    (out/"FROZEN_INPUT_SHA256.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    (out/"READY_FOR_FIRST_MEASUREMENT_ACCESS.txt").write_text("YES\n")
    print("V019_FROZEN_PACKAGE="+json.dumps({"catalog_entries":len(catalog),"candidate_members":len(candidates),"model_sha256":sha256(model_path),"protocol_sha256":sha256(protocol_path)},sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
