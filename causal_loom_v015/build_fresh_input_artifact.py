#!/usr/bin/env python3
"""Retrieve exact frozen v0.15 public inputs without modeling them."""
from __future__ import annotations
import argparse, hashlib, json, shutil, subprocess
from pathlib import Path

REPO = "https://github.com/syediu/nanobench-iros2026.git"
COMMIT = "0ae7d689749db174dc779ab2f7c416cc12e14e2b"
FILES = {
    "datasets/dataset/A1b_multisine_sysid_rep1.csv": ("e549e1f3e036c8b4cb5e0b2b0c1aefe77e18fa5d", 4852929),
    "datasets/dataset/B2_circle_slow_rep3.csv": ("b95b6084cd24da6534541907adafd41cf763d538", 2828126),
    "datasets/dataset/B3_figure8_slow_rep3.csv": ("8587529559835ee698824e42fdddcad2dc1a026d", 2826585),
    "datasets/dataset/B5_helix_slow_rep1.csv": ("ac78639ef48d855c127066cb7811dc5fb2d95052", 2820787),
    "datasets/dataset/B7_oval_slow_rep3.csv": ("1117265ac4f3fbe183d2b053be74865fbd12b525", 2822384),
    "datasets/dataset/B8_star_slow_rep3.csv": ("80478bad333548fb26ea2632fdf71009402e6413", 2802746),
    "datasets/dataset/B9_trefoil_slow_rep3.csv": ("1d0c28391fbc2cb4017233ff1929cd36c234158a", 1820644),
    "datasets/dataset/B2_circle_medium_rep3.csv": ("e4c9a3d73094520ef6eafe93aaa7e611b61ddb63", 2841940),
    "datasets/dataset/B3_figure8_medium_rep3.csv": ("d49232fd802ea1dff8bf522a26d08c792fba44e6", 2808484),
    "datasets/dataset/B5_helix_medium_rep3.csv": ("9082e71746dd7f39bf531620d7f1c12bc600e818", 2817904),
    "datasets/dataset/B7_oval_medium_rep3.csv": ("28f4fdf42f42fac634ffe0df0ccb11ed961e23bf", 2827555),
    "datasets/dataset/B8_star_medium_rep3.csv": ("e3b9c11bc6eabc5e8d06fe122d57df791d4a954f", 2805025),
    "datasets/dataset/B9_trefoil_medium_rep2.csv": ("692d46d6bc772667d814f7db90c3e933e26ab6bf", 2803744),
    "datasets/dataset/B5_helix_fast_mell_rep1.csv": ("98c5e7fcb537584464c3ac2e5a215a75a3f88129", 2806034),
    "datasets/dataset/B9_trefoil_fast_rep3.csv": ("47b61d67529936b72de1a1d8d8f0643763792923", 2863581),
    "datasets/dataset/B9_trefoil_fast_rep4.csv": ("9b71d34f0c5b3a1eb38af7b5327193c1c82c86cd", 2814521),
    "datasets/dataset/B9_trefoil_fast_rep5.csv": ("7850bd0d7df78aa600349eac7cd8ce17062ae703", 2841186),
    "datasets/dataset/B9_trefoil_fast_rep6.csv": ("87b38c8870b13d27377b308f571d5c3b1f9918d9", 2806506),
    "datasets/dataset/B9_trefoil_fast_rep7.csv": ("a90b090936f047ef9b962b81d33768db4a8cfaa7", 2813205),
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
    repo = args.work_dir / "nanobench"
    run("git", "clone", "--filter=blob:none", "--no-checkout", REPO, str(repo))
    run("git", "sparse-checkout", "init", "--no-cone", cwd=repo)
    (repo / ".git/info/sparse-checkout").write_text("\n".join(FILES) + "\n")
    run("git", "checkout", "--detach", COMMIT, cwd=repo)
    target = args.output_dir / "target"
    target.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for rel, (expected_blob, expected_size) in FILES.items():
        src = repo / rel
        actual_blob = run("git", "hash-object", str(src), cwd=repo)
        actual_size = src.stat().st_size
        if actual_blob != expected_blob or actual_size != expected_size:
            raise RuntimeError(f"Mismatch {rel}: {actual_blob}/{actual_size}")
        dst = target / Path(rel).name
        shutil.copy2(src, dst)
        manifest[dst.name] = {
            "repository_path": rel,
            "git_blob_sha1": actual_blob,
            "size_bytes": actual_size,
            "sha256": sha256(dst),
        }
    result = {
        "repository": REPO,
        "commit": COMMIT,
        "files": manifest,
        "bridge_script_sha256": sha256(Path(__file__).resolve()),
    }
    (args.output_dir / "bridge_manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
