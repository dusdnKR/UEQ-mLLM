"""Flatten the per-user result CSVs into the JSON record list scored by G-Eval.

Every (user, condition) pair becomes one record, so 80 users x 6 conditions
yields 480 records:

    {"test_num": "1", "test_type": "MultipleLLM",
     "user_info": "...", "user_status": "...", "system_output": "..."}

G-Eval itself was run separately against OpenAI's GPT-4 with n=10,
temperature=1, top_p=1; that scoring script is not part of this release. See the
README for the evaluation criteria and reported scores.

Usage:
    python -m ueq_mllm.export_for_geval
    python -m ueq_mllm.export_for_geval --results results --out results/geval_input.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from .workflow import CONDITIONS

TEST_NUM_PATTERN = re.compile(r"test_result_(\d+)\.csv$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results") / "questionnaires",
        help="directory holding test_result_<n>.csv files",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results") / "geval_input.json",
        help="destination JSON file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    records = []
    paths = sorted(
        args.results.glob("test_result_*.csv"),
        key=lambda p: int(TEST_NUM_PATTERN.search(p.name).group(1)),
    )
    if not paths:
        raise SystemExit(f"no test_result_*.csv found in {args.results}")

    for path in paths:
        test_num = TEST_NUM_PATTERN.search(path.name).group(1)
        frame = pd.read_csv(path)

        for _, row in frame.iterrows():
            for condition in CONDITIONS:
                records.append(
                    {
                        "test_num": test_num,
                        "test_type": condition,
                        "user_info": row["User_Info"],
                        "user_status": row["User_Status"],
                        "system_output": row[condition],
                    }
                )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as json_file:
        json.dump(records, json_file, indent=4)

    print(f"Wrote {len(records)} records from {len(paths)} files to {args.out}")


if __name__ == "__main__":
    main()
