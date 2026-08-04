"""Run the paper's experiment: 80 simulated users x 6 conditions.

For each simulated user the script runs UEQ-mLLM, the four single-agent
ablations, and the UEQ-sLLM baseline, then writes:

    <results>/questionnaires/test_result_<n>.csv   final questionnaire per condition
    <results>/logs/<condition>_<n>.csv             per-agent intermediate outputs
    <results>/user_data.csv                        the sampled user population

Usage:
    CUDA_VISIBLE_DEVICES=0,1 python -m ueq_mllm.run_experiment
    python -m ueq_mllm.run_experiment --start 1 --end 10 --results results/pilot
    python -m ueq_mllm.run_experiment --fixed-honeycomb honeycomb.txt
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from .llm import DEFAULT_MODEL, build_pipeline
from .simulate_users import CSV_HEADER, N_USERS, build_user
from .workflow import CONDITIONS

SEPARATOR = "=" * 81


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model id")
    parser.add_argument("--start", type=int, default=1, help="first test number")
    parser.add_argument("--end", type=int, default=N_USERS, help="last test number")
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results"),
        help="directory that receives the CSV outputs",
    )
    parser.add_argument(
        "--fixed-honeycomb",
        type=Path,
        default=None,
        help=(
            "text file holding one honeycomb questionnaire to reuse for every user, "
            "in place of running LLM-2. Freezing LLM-2 this way isolates the effect "
            "of the user-specific signal entering LLM-1, since it becomes the only "
            "thing that varies across users."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="seed for the user-state sampler"
    )
    parser.add_argument("--verbose", action="store_true", help="print every output")
    return parser.parse_args()


def write_row(path: Path, header: list, row: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerow(row)


def main() -> None:
    args = parse_args()

    fixed_honeycomb = (
        args.fixed_honeycomb.read_text(encoding="utf-8")
        if args.fixed_honeycomb
        else None
    )

    rng = np.random.default_rng(args.seed)
    pipeline = build_pipeline(args.model)

    questionnaire_dir = args.results / "questionnaires"
    log_dir = args.results / "logs"
    user_data_path = args.results / "user_data.csv"

    user_rows = []

    for test_num in range(args.start, args.end + 1):
        user = build_user(test_num, rng)
        user_rows.append(user.as_csv_row())
        print(f"\n### user {test_num}/{args.end}{user.user_info}{user.user_status}")

        final_outputs = [user.user_info, user.user_status]

        for name, condition in CONDITIONS.items():
            stages = condition(
                pipeline, user.user_info, user.user_status, fixed_honeycomb
            )
            final_outputs.append(list(stages.values())[-1])

            if args.verbose:
                for stage_name, text in stages.items():
                    print(f"--- {name} / {stage_name} ---\n{text}\n{SEPARATOR}")
            else:
                print(f"  {name}: done ({len(stages)} stage(s))")

            write_row(
                log_dir / f"{name}_{test_num}.csv",
                ["User_Info", "User_Status", *stages.keys()],
                [user.user_info, user.user_status, *stages.values()],
            )

        write_row(
            questionnaire_dir / f"test_result_{test_num}.csv",
            ["User_Info", "User_Status", *CONDITIONS.keys()],
            final_outputs,
        )

    user_data_path.parent.mkdir(parents=True, exist_ok=True)
    with open(user_data_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(CSV_HEADER)
        writer.writerows(user_rows)

    print(f"\nWrote {len(user_rows)} users to {args.results}")


if __name__ == "__main__":
    main()
