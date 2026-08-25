#!/usr/bin/env python3
"""Deterministically retrieve and package frozen v0.11 public inputs.

This script is designed for GitHub Actions because the local analysis container
cannot retrieve large public GitHub blobs. It does no model fitting and never
reads target state columns numerically; it only verifies/copies target bytes and
computes source motor normalization from the already-used v0.10 source files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

TARGET_REPO = "https://github.com/syediu/nanobench-iros2026.git"
TARGET_COMMIT = "0ae7d689749db174dc779ab2f7c416cc12e14e2b"
TARGET_FILES = {
    "datasets/dataset/A1b_multisine_sysid_rep1.csv": "e549e1f3e036c8b4cb5e0b2b0c1aefe77e18fa5d",
    "datasets/dataset/B10_lissajous_slow_rep1.csv": "da6198d6a2b89cfcbd6795e666f602eb234174d1",
    "datasets/dataset/B10_lissajous_fast_rep1.csv": "e5dbb137dbcb3b30f6b2a7f3e23fa70bcae5f2ab",
    "datasets/dataset/B11_random_waypoints_rep1.csv": "4211d514f607e9ff0adf145c224783b44989a3e5",
    "datasets/dataset/B11_random_waypoints_rep2.csv": "a383576238f56c5221b4ad4aa4a47bce1afc1f36",
    "datasets/dataset/B12_staircase_climb_rep1.csv": "144d9070911b4410b0f59f45f355857525fcbc28",
    "datasets/dataset/C4_battery_drain_rep1.csv": "901bce298ce717fca5b61ea2520b6f819fc0a432",
}
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
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def clone_sparse(repo: str, commit: str, paths: list[str], destination: Path, lfs: bool) -> None:
    env = None
    if lfs:
        import os
        env = dict(os.environ)
        env["GIT_LFS_SKIP_SMUDGE"] = "1"
    run("git", "clone", "--filter=blob:none", "--no-checkout", repo, str(destination), env=env)
    run("git", "sparse-checkout", "init", "--no-cone", cwd=destination)
    (destination / ".git" / "info" / "sparse-checkout").write_text("\n".join(paths) + "\n")
    run("git", "checkout", "--detach", commit, cwd=destination, env=env)
    if lfs:
        run("git", "lfs", "install", "--local", cwd=destination)
        run("git", "lfs", "pull", "--include=" + ",".join(paths), cwd=destination)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target_dir = args.work_dir / "target"
    source_dir = args.work_dir / "source"
    clone_sparse(TARGET_REPO, TARGET_COMMIT, list(TARGET_FILES), target_dir, lfs=False)
    clone_sparse(SOURCE_REPO, SOURCE_COMMIT, list(SOURCE_FILES), source_dir, lfs=True)

    target_out = args.output_dir / "target"
    target_out.mkdir(parents=True, exist_ok=True)
    target_manifest = {}
    for rel, expected_blob in TARGET_FILES.items():
        path = target_dir / rel
        actual_blob = run("git", "hash-object", str(path), cwd=target_dir)
        if actual_blob != expected_blob:
            raise RuntimeError(f"Target blob mismatch for {rel}: {actual_blob} != {expected_blob}")
        dest = target_out / Path(rel).name
        shutil.copy2(path, dest)
        target_manifest[dest.name] = {
            "repository_path": rel,
            "git_blob_sha1": actual_blob,
            "sha256": sha256(dest),
            "size_bytes": dest.stat().st_size,
        }

    motor_blocks = []
    source_manifest = []
    for rel, expected_sha in SOURCE_FILES.items():
        path = source_dir / rel
        actual_sha = sha256(path)
        if actual_sha != expected_sha:
            raise RuntimeError(f"Source SHA256 mismatch for {rel}: {actual_sha} != {expected_sha}")
        values = pd.read_csv(path, usecols=MOTOR_COLUMNS)[MOTOR_COLUMNS].to_numpy(np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Non-finite source motors in {rel}")
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
    (args.output_dir / "source_motor_stats.json").write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")
    manifest = {
        "target_repository": TARGET_REPO,
        "target_commit": TARGET_COMMIT,
        "target_files": target_manifest,
        "source_motor_stats_sha256": sha256(args.output_dir / "source_motor_stats.json"),
        "bridge_script_sha256": sha256(Path(__file__).resolve()),
    }
    (args.output_dir / "bridge_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
