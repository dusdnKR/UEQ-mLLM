"""Aggregate the sampled GPT-4 ratings into one score per user and condition.

Following G-Eval, a rating is the probability-weighted sum over the observed
rating values rather than a single sampled value:

    score = sum_v  p(v) * v          where p(v) = count(v) / n

With ratings drawn by sampling (n=10) rather than read off token log-probs,
that sum is the mean of the ten ratings; the weighted form is kept because it
is the definition the metric comes from.

Writes one CSV per criterion — 80 rows plus a final ``Average`` row — which is
what ``get_average.py`` and ``get_t_test.py`` read.

Usage:
    python evaluation/get_score.py
    python evaluation/get_score.py --raw results/geval/raw --out results/geval/scores
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

CRITERION_FILES = [
    "gpt4_con_detailed",
    "gpt4_flu_detailed",
    "gpt4_ind_detailed",
    "gpt4_rel_detailed",
]

TEST_TYPES = [
    "MultipleLLM",
    "WithoutL1",
    "WithoutL2",
    "WithoutL3",
    "WithoutL4",
    "SingleLLM",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw",
        type=Path,
        default=Path("results") / "geval" / "raw",
        help="directory holding gpt4_*_detailed.json from gpt4_eval.py",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results") / "geval" / "scores",
        help="directory that receives the per-criterion score CSVs",
    )
    return parser.parse_args()


def expected_score(responses: list) -> float:
    """Probability-weighted rating over the n samples drawn for one item."""
    values = list(map(float, responses))
    counter = Counter(values)
    return sum(count / len(values) * value for value, count in counter.items())


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    for file_name in CRITERION_FILES:
        raw_path = args.raw / f"{file_name}.json"
        records = json.loads(raw_path.read_text(encoding="utf-8"))

        scores: dict = {}
        for record in records:
            test_num = record["test_num"]
            scores.setdefault(test_num, {t: "" for t in TEST_TYPES})
            scores[test_num][record["test_type"]] = expected_score(
                record["all_responses"]
            )

        csv_path = args.out / f"{file_name}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["test_num"] + TEST_TYPES)
            writer.writeheader()

            for test_num, row in scores.items():
                writer.writerow({"test_num": test_num, **row})

            averages = {
                t: np.mean([row[t] for row in scores.values() if row[t] != ""])
                for t in TEST_TYPES
            }
            writer.writerow({"test_num": "Average", **averages})

        print(f"wrote {csv_path} ({len(scores)} users)")


if __name__ == "__main__":
    main()
