#!/usr/bin/env python3
"""Backward-compatible entry point.

Runs Stage A pretrain only. Prefer the full pipeline:

  scripts/train_csencoder_pretrain.py
  scripts/mine_hard_examples.py
  scripts/finetune_csencoder.py
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "train_csencoder_pretrain.py"


def main() -> None:
    sys.argv[0] = str(SCRIPT)
    runpy.run_path(str(SCRIPT), run_name="__main__")


if __name__ == "__main__":
    main()
