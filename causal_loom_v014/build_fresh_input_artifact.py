#!/usr/bin/env python3
"""Retrieve exact frozen v0.14 public inputs and package them without modeling."""
from __future__ import annotations
import argparse, hashlib, json, shutil, subprocess
from pathlib import Path

REPO="https://github.com/syediu/nanobench-iros2026.git"
COMMIT="0ae7d689749db174dc779ab2f7c416cc12e14e2b"
FILES={
 "datasets/dataset/A1b_multisine_sysid_rep1.csv":("e549e1f3e036c8b4cb5e0b2b0c1aefe77e18fa5d",4852929),
 "datasets/dataset/B2_circle_slow_rep2.csv":("aa518943e36b192a01df651e99843479d9655ac9",2827544),
 "datasets/dataset/B2_circle_medium_rep2.csv":("e78e2325f8c8d9e0e947e7597848addf128e881c",2830853),
 "datasets/dataset/B2_circle_fast_rep3.csv":("26a375e5ed98485f8befc86090017c0e4135823c",2827695),
 "datasets/dataset/B5_helix_slow_rep2.csv":("fe4b7043da9923102655d6ac9429eefad875f28a",2823276),
 "datasets/dataset/B5_helix_medium_rep2.csv":("f2ef6fa8f5b7b5f4e848caa0c595a44bfa601d08",2823504),
 "datasets/dataset/B5_helix_fast_rep3.csv":("9f8fda14492d7a98c4fa7c0aff5d2accd101f009",2830323),
 "datasets/dataset/B7_oval_slow_rep2.csv":("f00fc9c1542441fec572492f17ea7f74206ff8a7",2827756),
 "datasets/dataset/B7_oval_medium_rep2.csv":("2c9ff5eb3d875d9bdc8cfa1fbef43021014e4642",2821697),
 "datasets/dataset/B7_oval_fast_rep3.csv":("7f80d81492ae0c123cd88fd0789dc5fcfffb3e34",2825069),
 "datasets/dataset/B8_star_slow_rep2.csv":("090436d106d2f87c5dffc9b6ac60d35d2c07d620",2800872),
 "datasets/dataset/B8_star_medium_rep2.csv":("e99c24c38944a50621d84810e55a0b6d592f3ee6",2803554),
 "datasets/dataset/B8_star_fast_rep3.csv":("ad9a7b9a33730c078e1f6b1af123304931afafdf",2811556),
 "datasets/dataset/B9_trefoil_slow_rep1.csv":("51a336455b6465ba50a1700305bb8ee90886cd7b",1814806),
 "datasets/dataset/B9_trefoil_medium_rep1.csv":("c34438ac7b43e68b9306d9ed283190bd4353b320",2799050),
 "datasets/dataset/B9_trefoil_fast_rep2.csv":("8be19dab1775e16aaaee4c26cb86582be77a6f6f",2883378),
}

def run(*args:str,cwd:Path|None=None)->str:
 return subprocess.check_output(args,cwd=cwd,text=True).strip()
def sha256(p:Path)->str:
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main()->int:
 ap=argparse.ArgumentParser();ap.add_argument('--work-dir',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True);a=ap.parse_args();a.work_dir.mkdir(parents=True,exist_ok=True);a.output_dir.mkdir(parents=True,exist_ok=True)
 repo=a.work_dir/'nanobench';run('git','clone','--filter=blob:none','--no-checkout',REPO,str(repo));run('git','sparse-checkout','init','--no-cone',cwd=repo);(repo/'.git/info/sparse-checkout').write_text('\n'.join(FILES)+'\n');run('git','checkout','--detach',COMMIT,cwd=repo)
 target=a.output_dir/'target';target.mkdir(parents=True,exist_ok=True);manifest={}
 for rel,(expected_blob,expected_size) in FILES.items():
  src=repo/rel;blob=run('git','hash-object',str(src),cwd=repo);size=src.stat().st_size
  if blob!=expected_blob or size!=expected_size:raise RuntimeError(f'mismatch {rel}: {blob}/{size}')
  dst=target/Path(rel).name;shutil.copy2(src,dst);manifest[dst.name]={'repository_path':rel,'git_blob_sha1':blob,'size_bytes':size,'sha256':sha256(dst)}
 out={'repository':REPO,'commit':COMMIT,'files':manifest,'bridge_script_sha256':sha256(Path(__file__).resolve())};(a.output_dir/'bridge_manifest.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps(out,indent=2,sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())
