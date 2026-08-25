#!/usr/bin/env python3
"""Retrieve the pinned UIUC archive and package only frozen v0.15 static files.

No numeric content is parsed or scored here. Selection is by pre-frozen archive
member token only; archive and extracted bytes are hashed into the manifest.
"""
from __future__ import annotations
import argparse, hashlib, json, shutil, subprocess, zipfile
from pathlib import Path

URLS=[
 'https://m-selig.ae.illinois.edu/props/download/UIUC-propDB.zip',
 'https://m-selig.web.engr.illinois.edu/props/download/UIUC-propDB.zip',
]
EXPECTED_MD5='a41e484f1fd0fb6ff80b76e27410808b'
TOKENS={
 'apc_4.2x4':'apc_4.2x4_static',
 'apc_4.5x4.1':'apc_4.5x4.1_static',
 'apc_5x3':'apc_5x3_static',
 'apc_5.25x6.25':'apc_5.25x6.25_static',
 'apc_5.5x4.5':'apc_5.5x4.5_static',
 'apc_6x4':'apc_6x4_static',
}

def digest(path:Path,name:str)->str:
 h=hashlib.new(name)
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def main()->int:
 ap=argparse.ArgumentParser();ap.add_argument('--work-dir',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True);a=ap.parse_args();a.work_dir.mkdir(parents=True,exist_ok=True);a.output_dir.mkdir(parents=True,exist_ok=True)
 archive=a.work_dir/'UIUC-propDB.zip'; used=None; errors=[]
 for url in URLS:
  try:
   subprocess.run(['curl','-L','--fail','--retry','3','--connect-timeout','30','--max-time','900','-A','Mozilla/5.0',url,'-o',str(archive)],check=True)
   if digest(archive,'md5').lower()!=EXPECTED_MD5: raise RuntimeError('MD5 mismatch')
   used=url; break
  except Exception as e:
   errors.append(f'{url}: {e}')
   if archive.exists(): archive.unlink()
 if used is None: raise RuntimeError('All official mirrors failed: '+' | '.join(errors))
 out=a.output_dir/'target';out.mkdir(parents=True,exist_ok=True);files={}
 with zipfile.ZipFile(archive) as z:
  names=[n for n in z.namelist() if not n.endswith('/')]
  for key,token in TOKENS.items():
   matches=[n for n in names if n.lower().endswith('.txt') and token in Path(n).stem.lower()]
   if len(matches)!=1: raise RuntimeError(f'{key}: expected exactly one archive member, found {matches}')
   member=matches[0]; dest=out/f'{key}.txt'
   with z.open(member) as src,dest.open('wb') as dst: shutil.copyfileobj(src,dst)
   files[dest.name]={'archive_member':member,'sha256':digest(dest,'sha256'),'size_bytes':dest.stat().st_size}
 payload={'archive_url':used,'archive_md5':EXPECTED_MD5,'archive_sha256':digest(archive,'sha256'),'bridge_script_sha256':digest(Path(__file__).resolve(),'sha256'),'files':files}
 (a.output_dir/'bridge_manifest.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n');print(json.dumps(payload,indent=2,sort_keys=True));return 0
if __name__=='__main__': raise SystemExit(main())
