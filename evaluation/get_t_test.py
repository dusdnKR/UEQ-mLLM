"""Compare UEQ-mLLM against the UEQ-sLLM baseline on each criterion.

Welch's t-test (unequal variances) over the per-user scores, followed by
Benjamini-Hochberg FDR correction across the four criteria.

Usage:
    python evaluation/get_t_test.py
    python evaluation/get_t_test.py --plots            # also write box plots
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

CRITERIA = {
    "gpt4_con_detailed": "Consistency",
    "gpt4_flu_detailed": "Fluency",
    "gpt4_ind_detailed": "Individualization",
    "gpt4_rel_detailed": "Relevance",
}

TREATMENT = "MultipleLLM"  # UEQ-mLLM
CONTROL = "SingleLLM"  # UEQ-sLLM


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
        default=Path("results") / "geval" / "t_test",
        help="directory that receives the t-test CSVs and plots",
    )
    parser.add_argument(
        "--plots", action="store_true", help="also render per-criterion box plots"
    )
    return parser.parse_args()


def load_pair(path: Path) -> tuple:
    """Return the treatment and control columns of one criterion's score CSV.

    get_score.py appends a trailing ``Average`` row; it is a summary of the
    per-user scores, not an observation, so it is dropped here.
    """
    frame = pd.read_csv(path)
    frame = frame[frame["test_num"] != "Average"]
    return (
        frame[TREATMENT].dropna().astype(float),
        frame[CONTROL].dropna().astype(float),
    )


def plot_box(treatment, control, label: str, path: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.ticker import MaxNLocator

    plt.figure(figsize=(12, 8))
    sns.set(style="whitegrid")

    axes = sns.boxplot(
        data=pd.DataFrame({"UEQ-mLLM": treatment, "UEQ-sLLM": control}),
        palette="Set2",
        linewidth=2.5,
        fliersize=8,
        width=0.5,
    )
    plt.title(label, fontsize=25, fontweight="bold")
    axes.tick_params(axis="both", labelsize=20)
    axes.yaxis.set_major_locator(MaxNLocator(integer=True))

    y_min, y_max = axes.get_ylim()
    if label == "Fluency":
        # fluency is a 1-3 scale, so the default limits crowd the boxes
        axes.set_ylim([y_min - 1, y_max])

    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    results = []

    for file_name, label in CRITERIA.items():
        treatment, control = load_pair(args.scores / f"{file_name}.csv")
        t_value, p_value = ttest_ind(treatment, control, equal_var=False)
        results.append({"col_name": label, "t_value": t_value, "p_value": p_value})

        if args.plots:
            plot_box(treatment, control, label, args.out / f"box_plot_{file_name}.png")

    frame = pd.DataFrame(results)
    frame.to_csv(args.out / "t_test_results.csv", index=False)

    reject, corrected, _, _ = multipletests(
        frame["p_value"], alpha=0.05, method="fdr_bh"
    )
    frame["p_value_corrected"] = corrected
    frame["significant"] = reject
    frame.to_csv(args.out / "t_test_results_fdr.csv", index=False)

    print(frame.to_string(index=False))
    print(f"\nwrote {args.out / 't_test_results_fdr.csv'}")


if __name__ == "__main__":
    main()
