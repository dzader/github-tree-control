#!/usr/bin/env python3
"""Retrieve and package frozen v0.13 public thrust-stand inputs without scoring."""
from __future__ import annotations
import argparse, hashlib, json, shutil, subprocess
from pathlib import Path

REPO = "https://github.com/alspitz/esc_test.git"
COMMIT = "4078609e3a84923b9cdbfdea8daf9814364fa9c8"
FILES = {
    "data/rcbench/Log_2019-11-05_171350.csv": "fe2653a7132921cc33127e6919d60888ef3b9f8c",
    "data/rcbench/Log_2019-11-07_174807.csv": "9c8bdfeb1743cb75185f904a91d770df9d15ea36",
    "data/rcbench/Log_2019-11-07_175456.csv": "daa2e3dfdc64f7dc9defbd7a3225472e465d53d0",
    "data/rcbench/Log_2019-11-07_184325.csv": "356f345ea5811764846555f1b3c147d22b4a0b00",
    "data/rcbench/Log_2019-11-07_211050.csv": "bb75a8280e068bec6336560055d57bf0896c704f",
    "data/rcbench/Log_2019-11-09_181347.csv": "4721722c9d3e8c3ce33e275f830a8954fe194cd3",
}

def run(*args: str, cwd: Path | None = None) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True).strip()

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def main() -> int:
    ap=argparse.ArgumentParser();ap.add_argument('--work-dir',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True);a=ap.parse_args()
    a.work_dir.mkdir(parents=True,exist_ok=True);a.output_dir.mkdir(parents=True,exist_ok=True)
    repo=a.work_dir/'repo'
    run('git','clone','--filter=blob:none','--no-checkout',REPO,str(repo))
    run('git','sparse-checkout','init','--no-cone',cwd=repo)
    (repo/'.git/info/sparse-checkout').write_text('\n'.join(FILES)+'\n')
    run('git','checkout','--detach',COMMIT,cwd=repo)
    out=a.output_dir/'target';out.mkdir(parents=True,exist_ok=True)
    manifest={}
    for rel,expected in FILES.items():
        src=repo/rel
        blob=run('git','hash-object',str(src),cwd=repo)
        if blob!=expected: raise RuntimeError(f'blob mismatch {rel}: {blob} != {expected}')
        dest=out/src.name;shutil.copy2(src,dest)
        manifest[dest.name]={'repository_path':rel,'git_blob_sha1':blob,'sha256':sha256(dest),'size_bytes':dest.stat().st_size}
    payload={'repository':REPO,'commit':COMMIT,'files':manifest,'bridge_script_sha256':sha256(Path(__file__).resolve())}
    (a.output_dir/'bridge_manifest.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
    print(json.dumps(payload,indent=2,sort_keys=True));return 0
if __name__=='__main__': raise SystemExit(main())
