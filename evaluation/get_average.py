"""Combine the four criterion averages into the single 100-point score.

Fluency is rated 1-3 and the other three criteria 1-5, so each criterion is
normalized to 100 before averaging:

    Score = mean(con/5, flu/3, ind/5, rel/5) * 100

Usage:
    python evaluation/get_average.py
    python evaluation/get_average.py --scores results/geval/scores
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# criterion file stem -> (row label, maximum of the rating scale)
CRITERIA = {
    "gpt4_con_detailed": ("Consistency", 5),
    "gpt4_flu_detailed": ("Fluency", 3),
    "gpt4_ind_detailed": ("Individualization", 5),
    "gpt4_rel_detailed": ("Relevance", 5),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scores",
        type=Path,
        default=Path("results") / "geval" / "scores",
        help="directory holding the per-criterion CSVs from get_score.py",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="destination CSV (default: <scores>/combined_averages.csv)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = args.out or args.scores / "combined_averages.csv"

    rows, normalized = {}, {}

    for file_name, (label, scale) in CRITERIA.items():
        frame = pd.read_csv(args.scores / f"{file_name}.csv")
        # get_score.py appends the per-condition mean as the final "Average" row
        averages = frame.iloc[-1].drop("test_num").astype(float)
        rows[label] = averages
        normalized[label] = averages / scale

    rows["Score"] = sum(normalized.values()) / len(normalized) * 100

    result = pd.DataFrame(rows).transpose()
    result.index.name = "test_type"
    result.to_csv(out_path)

    print(result.round(4).to_string())
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
