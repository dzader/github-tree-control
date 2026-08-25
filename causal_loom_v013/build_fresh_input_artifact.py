#!/usr/bin/env python3
"""Retrieve only the frozen v0.13 calibration and fresh held-out target bytes."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

TARGET_REPO = "https://github.com/syediu/nanobench-iros2026.git"
TARGET_COMMIT = "0ae7d689749db174dc779ab2f7c416cc12e14e2b"
TARGET_FILES = {
    "datasets/dataset/A1b_multisine_sysid_rep1.csv": "e549e1f3e036c8b4cb5e0b2b0c1aefe77e18fa5d",
    "datasets/dataset/B2_circle_fast_rep2.csv": "85a6749006400b43a0049c9d0b0b5b6c6c70adac",
    "datasets/dataset/B3_figure8_fast_rep2.csv": "e94bd244d95bcf1eaf28cc64581a59c48c093a89",
    "datasets/dataset/B5_helix_fast_rep2.csv": "ed44edc416f32f3e8b5a4df031900ec25959eaed",
    "datasets/dataset/B6_linear_ramp_rep2.csv": "5da2c40aa8b85b9c596adb45cc27ea900d656ed4",
    "datasets/dataset/B7_oval_fast_rep2.csv": "eea070c2466842de9879e1b26c5b42973ecdb5bf",
    "datasets/dataset/B8_star_fast_rep2.csv": "7bc1edc9985601c87a93feed55dec49b01e7dbfb",
    "datasets/dataset/B9_trefoil_fast_rep1.csv": "9f1415945d3283ccf54859fb343f53785c23ed04",
}


def run(*args: str, cwd: Path | None = None) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


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
    repo = args.work_dir / "target"
    run("git", "clone", "--filter=blob:none", "--no-checkout", TARGET_REPO, str(repo))
    run("git", "sparse-checkout", "init", "--no-cone", cwd=repo)
    (repo / ".git" / "info" / "sparse-checkout").write_text("\n".join(TARGET_FILES) + "\n")
    run("git", "checkout", "--detach", TARGET_COMMIT, cwd=repo)

    target_out = args.output_dir / "target"
    target_out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "repository": TARGET_REPO,
        "commit": TARGET_COMMIT,
        "files": {},
        "bridge_script_sha256": sha256(Path(__file__).resolve()),
    }
    for rel, expected_blob in TARGET_FILES.items():
        src = repo / rel
        actual_blob = run("git", "hash-object", str(src), cwd=repo)
        if actual_blob != expected_blob:
            raise RuntimeError(f"Git blob mismatch for {rel}: {actual_blob} != {expected_blob}")
        dst = target_out / Path(rel).name
        shutil.copy2(src, dst)
        manifest["files"][dst.name] = {
            "repository_path": rel,
            "git_blob_sha1": actual_blob,
            "sha256": sha256(dst),
            "size_bytes": dst.stat().st_size,
        }
    (args.output_dir / "bridge_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
