#!/usr/bin/env python3
"""Retrieve and package the frozen Causal Loom v0.12 public inputs.

This bridge runs in GitHub Actions because the numerical container cannot
reliably retrieve large public GitHub blobs.  It performs no target fitting,
no target outcome summaries, and no scoring.  It checks out only the paths
frozen by the v0.12 protocol, records their immutable Git blob hashes and
SHA-256 digests, and reconstructs the exact v0.10 source motor normalization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

TARGET_REPO = "https://github.com/syediu/nanobench-iros2026.git"
TARGET_COMMIT = "0ae7d689749db174dc779ab2f7c416cc12e14e2b"
TARGET_FILES = [
    "datasets/dataset/A1b_multisine_sysid_rep1.csv",
    "datasets/dataset/B2_circle_slow_rep1.csv",
    "datasets/dataset/B2_circle_fast_rep1.csv",
    "datasets/dataset/B3_figure8_slow_rep1.csv",
    "datasets/dataset/B3_figure8_fast_rep1.csv",
    "datasets/dataset/B7_oval_slow_rep2.csv",
    "datasets/dataset/B7_oval_fast_rep1.csv",
    "datasets/dataset/B8_star_slow_rep1.csv",
    "datasets/dataset/B8_star_fast_rep1.csv",
]

SOURCE_REPO = "https://github.com/idsia-robotics/nanodrone-sysid-benchmark.git"
SOURCE_COMMIT = "1038275426ba41135ac35afb1d8597c757b032b0"
SOURCE_FILES = {
    "data/train/square_20251017_run1.csv": "9b458d207d0a5e21933ba7ad16945bf505ab2b3abf6d151a1442cdd705d6c0cc",
    "data/train/square_20251017_run2.csv": "3b47b3712915aff4fd301fd0a2f989bbdce0a5ca1af852c923f472ec7d05d084",
    "data/train/random_20251017_run1.csv": "44b457f31ac71d6f18f8a680f69b909ac3cadbb270051430f2dd722acff4d890",
    "data/train/random_20251017_run2.csv": "64f382ee8c06f71397d6abc42b50b270dc4ba47df5f13ebc3515e43bbfaa0c75",
    "data/train/chirp_20251017_run1.csv": "00e19738869e223c295232c44771c195f8c2f8c499e5497e27e520c7fc60ca23",
    "data/train/chirp_20251017_run2.csv": "6e5587da5055fc620677aeeb8e016a8e1d8d219464d7b2b82f2667174038c979",
}
MOTOR_COLUMNS = ["m1_rads", "m2_rads", "m3_rads", "m4_rads"]


def run(*args: str, cwd: Path | None = None, env: dict[str, str] | None = None) -> str:
    return subprocess.check_output(args, cwd=cwd, env=env, text=True).strip()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def clone_sparse(repo: str, commit: str, paths: list[str], destination: Path, *, lfs: bool) -> None:
    env = dict(os.environ)
    if lfs:
        env["GIT_LFS_SKIP_SMUDGE"] = "1"
    run("git", "clone", "--filter=blob:none", "--no-checkout", repo, str(destination), env=env)
    run("git", "sparse-checkout", "init", "--no-cone", cwd=destination, env=env)
    (destination / ".git" / "info" / "sparse-checkout").write_text("\n".join(paths) + "\n")
    run("git", "checkout", "--detach", commit, cwd=destination, env=env)
    if lfs:
        run("git", "lfs", "install", "--local", cwd=destination, env=env)
        run("git", "lfs", "pull", "--include=" + ",".join(paths), cwd=destination, env=env)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    target_dir = args.work_dir / "target"
    source_dir = args.work_dir / "source"
    clone_sparse(TARGET_REPO, TARGET_COMMIT, TARGET_FILES, target_dir, lfs=False)
    clone_sparse(SOURCE_REPO, SOURCE_COMMIT, list(SOURCE_FILES), source_dir, lfs=True)

    target_out = args.output_dir / "target"
    target_out.mkdir(parents=True, exist_ok=True)
    target_manifest: dict[str, object] = {}
    for rel in TARGET_FILES:
        path = target_dir / rel
        if not path.is_file():
            raise FileNotFoundError(path)
        blob = run("git", "hash-object", str(path), cwd=target_dir)
        dest = target_out / Path(rel).name
        shutil.copy2(path, dest)
        target_manifest[dest.name] = {
            "repository_path": rel,
            "git_blob_sha1": blob,
            "sha256": sha256(dest),
            "size_bytes": dest.stat().st_size,
        }

    motor_blocks: list[np.ndarray] = []
    source_manifest: list[dict[str, object]] = []
    for rel, expected_sha in SOURCE_FILES.items():
        path = source_dir / rel
        actual_sha = sha256(path)
        if actual_sha != expected_sha:
            raise RuntimeError(f"Source SHA-256 mismatch for {rel}: {actual_sha} != {expected_sha}")
        values = pd.read_csv(path, usecols=MOTOR_COLUMNS)[MOTOR_COLUMNS].to_numpy(np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Non-finite source motor values in {rel}")
        motor_blocks.append(values[:-1])
        source_manifest.append({
            "repository_path": rel,
            "sha256": actual_sha,
            "size_bytes": path.stat().st_size,
            "rows": int(len(values)),
        })

    motors = np.vstack(motor_blocks)
    stats = {
        "source_repository": SOURCE_REPO,
        "source_commit": SOURCE_COMMIT,
        "source_files": source_manifest,
        "motor_columns": MOTOR_COLUMNS,
        "rows_used": int(len(motors)),
        "last_row_excluded_per_file": True,
        "motor_mean_rads": motors.mean(axis=0).tolist(),
        "motor_scale_rads": motors.std(axis=0).tolist(),
    }
    stats_path = args.output_dir / "source_motor_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")

    manifest = {
        "target_repository": TARGET_REPO,
        "target_commit": TARGET_COMMIT,
        "target_files": target_manifest,
        "source_motor_stats_sha256": sha256(stats_path),
        "bridge_script_sha256": sha256(Path(__file__).resolve()),
    }
    manifest_path = args.output_dir / "bridge_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
