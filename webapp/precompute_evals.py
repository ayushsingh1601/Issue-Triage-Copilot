#!/usr/bin/env python
"""Precompute the eval results cache used by the Evals tab (one-time per deploy)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from webapp.env import load_env_file
from webapp.evals_cache import precompute


def main() -> None:
    load_env_file()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="scikit-learn/scikit-learn")
    parser.add_argument("--held-out-limit", type=int, default=10)
    args = parser.parse_args()

    payload = precompute(args.repo, held_out_limit=args.held_out_limit)
    print(
        f"cached evals for {payload['repo']}: "
        f"corpus={payload['corpus_size']} held_out={payload['held_out_size']}"
    )


if __name__ == "__main__":
    main()
