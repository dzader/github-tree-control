#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    source = Path(__file__).with_name("develop_lowrank_projective_atlas.py")
    spec = importlib.util.spec_from_file_location("v019_lowrank_v2_impl", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load low-rank implementation")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    def correct_projective_corr(a, b):
        a = np.asarray(a, float) - float(np.mean(a))
        b = np.asarray(b, float) - float(np.mean(b))
        if float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
            return 0.0
        return float(abs(np.corrcoef(a, b)[0, 1]))

    module.projective_corr = correct_projective_corr
    return int(module.main())


if __name__ == "__main__":
    raise SystemExit(main())
