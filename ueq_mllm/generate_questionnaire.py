"""Generate one personalized UX questionnaire with the UEQ-mLLM workflow.

By default the user state is sampled at random, which is what the paper's
simulated population does. Pass the percentages explicitly to feed in a real
reading from the MIND dashboard.

Usage:
    python -m ueq_mllm.generate_questionnaire
    python -m ueq_mllm.generate_questionnaire --sex Female --age "30s'" \\
        --attention 71.4 18.2 10.4 --emotion 3 2 5 62 8 12 8
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .llm import DEFAULT_MODEL, build_pipeline
from .simulate_users import (
    AGE_LIST,
    ATTENTION_STATES,
    EMOTION_STATES,
    SEX_LIST,
    format_user_info,
    format_user_status,
    random_distribution,
)
from .workflow import run_ueq_mllm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model id")
    parser.add_argument("--sex", choices=SEX_LIST, default=None)
    parser.add_argument("--age", choices=AGE_LIST, default=None)
    parser.add_argument(
        "--attention",
        type=float,
        nargs=len(ATTENTION_STATES),
        metavar=tuple(s.upper() for s in ATTENTION_STATES),
        default=None,
        help="attention percentages from the MIND EEG classifier",
    )
    parser.add_argument(
        "--emotion",
        type=float,
        nargs=len(EMOTION_STATES),
        metavar=tuple(s.upper() for s in EMOTION_STATES),
        default=None,
        help="emotion percentages from the MIND facial expression module",
    )
    parser.add_argument(
        "--fixed-honeycomb",
        type=Path,
        default=None,
        help="text file holding a honeycomb questionnaire to reuse in place of LLM-2",
    )
    parser.add_argument(
        "--show-stages", action="store_true", help="print every agent's raw output"
    )
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    sex = args.sex or SEX_LIST[rng.integers(len(SEX_LIST))]
    age = args.age or AGE_LIST[rng.integers(len(AGE_LIST))]
    attention = (
        args.attention
        if args.attention is not None
        else random_distribution(len(ATTENTION_STATES), rng)
    )
    emotion = (
        args.emotion
        if args.emotion is not None
        else random_distribution(len(EMOTION_STATES), rng)
    )

    user_info = format_user_info(sex, age)
    user_status = format_user_status(attention, emotion)
    print(user_info)
    print(user_status)

    fixed_honeycomb = (
        args.fixed_honeycomb.read_text(encoding="utf-8")
        if args.fixed_honeycomb
        else None
    )

    pipeline = build_pipeline(args.model)
    stages = run_ueq_mllm(pipeline, user_info, user_status, fixed_honeycomb)

    if args.show_stages:
        for name, text in stages.items():
            print(f"\n--- {name} ---\n{text}")

    print("\n=== Personalized UX questionnaire ===\n")
    print(stages["Output_4"])


if __name__ == "__main__":
    main()
