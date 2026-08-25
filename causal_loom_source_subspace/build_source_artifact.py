#!/usr/bin/env python3
"""Retrieve and verify the exact source trajectories used by Causal Loom v0.10.

Development bridge only. No modeling or target-data access occurs here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

REPO = "https://github.com/idsia-robotics/nanodrone-sysid-benchmark.git"
COMMIT = "1038275426ba41135ac35afb1d8597c757b032b0"
FILES = {
    "data/train/square_20251017_run1.csv": "9b458d207d0a5e21933ba7ad16945bf505ab2b3abf6d151a1442cdd705d6c0cc",
    "data/train/square_20251017_run2.csv": "3b47b3712915aff4fd301fd0a2f989bbdce0a5ca1af852c923f472ec7d05d084",
    "data/train/random_20251017_run1.csv": "44b457f31ac71d6f18f8a680f69b909ac3cadbb270051430f2dd722acff4d890",
    "data/train/random_20251017_run2.csv": "64f382ee8c06f71397d6abc42b50b270dc4ba47df5f13ebc3515e43bbfaa0c75",
    "data/train/chirp_20251017_run1.csv": "00e19738869e223c295232c44771c195f8c2f8c499e5497e27e520c7fc60ca23",
    "data/train/chirp_20251017_run2.csv": "6e5587da5055fc620677aeeb8e016a8e1d8d219464d7b2b82f2667174038c979",
}


def run(*args: str, cwd: Path | None = None, env: dict[str, str] | None = None) -> str:
    return subprocess.check_output(args, cwd=cwd, env=env, text=True).strip()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    repo = args.work_dir / "source"
    env = dict(os.environ)
    env["GIT_LFS_SKIP_SMUDGE"] = "1"
    run("git", "clone", "--filter=blob:none", "--no-checkout", REPO, str(repo), env=env)
    run("git", "sparse-checkout", "init", "--no-cone", cwd=repo)
    (repo / ".git" / "info" / "sparse-checkout").write_text("\n".join(FILES) + "\n")
    run("git", "checkout", "--detach", COMMIT, cwd=repo, env=env)
    run("git", "lfs", "install", "--local", cwd=repo)
    run("git", "lfs", "pull", "--include=" + ",".join(FILES), cwd=repo)

    out = args.output_dir / "source"
    out.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {}
    for rel, expected in FILES.items():
        src = repo / rel
        actual = sha256(src)
        if actual != expected:
            raise RuntimeError(f"SHA256 mismatch for {rel}: {actual} != {expected}")
        dst = out / Path(rel).name
        shutil.copy2(src, dst)
        manifest[dst.name] = {
            "repository_path": rel,
            "sha256": actual,
            "size_bytes": dst.stat().st_size,
        }

    payload = {
        "repository": REPO,
        "commit": COMMIT,
        "files": manifest,
        "bridge_script_sha256": sha256(Path(__file__).resolve()),
    }
    (args.output_dir / "bridge_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
