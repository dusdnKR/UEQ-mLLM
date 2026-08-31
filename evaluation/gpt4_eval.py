"""Score generated questionnaires with G-Eval on GPT-4.

For each record produced by ``ueq_mllm.export_for_geval``, the criterion prompt
is filled in and sent to GPT-4 ``n`` times. The raw ratings are written back
alongside the record so that ``get_score.py`` can aggregate them.

Sampling parameters follow the paper: ``n=10``, ``temperature=1``, ``top_p=1``.
Drawing several ratings per item and averaging them is what makes the score
stable, since a single GPT-4 rating on a 1-5 scale is noisy.

The API key is read from the ``OPENAI_API_KEY`` environment variable so that it
never lands in shell history.

Usage:
    export OPENAI_API_KEY=...
    python evaluation/gpt4_eval.py --criterion con --input results/geval_input.json
    python evaluation/gpt4_eval.py --criterion ind --output results/geval/raw/gpt4_ind_detailed.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import tqdm
from openai import OpenAI

PROMPT_DIR = Path(__file__).parent / "prompts"

# criterion key -> (prompt file stem, rating scale)
CRITERIA = {
    "con": ("con_detailed", 5),  # Consistency
    "flu": ("flu_detailed", 3),  # Fluency
    "ind": ("ind_detailed", 5),  # Individualization (Personalization in the paper)
    "rel": ("rel_detailed", 5),  # Relevance
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--criterion", choices=sorted(CRITERIA), required=True, help="criterion to score"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results") / "geval_input.json",
        help="records from ueq_mllm.export_for_geval",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="destination JSON (default: results/geval/raw/gpt4_<criterion>_detailed.json)",
    )
    parser.add_argument("--model", default="gpt-4", help="OpenAI model")
    parser.add_argument("--n", type=int, default=10, help="ratings sampled per item")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--max-retries", type=int, default=5, help="retries per item before skipping"
    )
    return parser.parse_args()


def build_prompt(template: str, record: dict) -> str:
    """Fill a criterion template with one record's fields.

    Only ``ind_detailed`` uses the user placeholders; ``str.replace`` on an
    absent placeholder is a no-op, so one code path covers all four criteria.
    """
    return (
        template.replace("{{Questionnaire}}", record["system_output"])
        .replace("{{User_Information}}", record["user_info"])
        .replace("{{User_Status}}", record["user_status"])
    )


def main() -> None:
    args = parse_args()

    stem, _scale = CRITERIA[args.criterion]
    output_path = args.output or Path("results") / "geval" / "raw" / f"gpt4_{stem}.json"

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("set OPENAI_API_KEY before running")
    client = OpenAI(api_key=api_key)

    records = json.loads(args.input.read_text(encoding="utf-8"))
    template = (PROMPT_DIR / f"{stem}.txt").read_text(encoding="utf-8")

    scored, skipped = [], 0

    for record in tqdm.tqdm(records, desc=f"gpt4/{args.criterion}"):
        record["prompt"] = build_prompt(template, record)

        for attempt in range(args.max_retries):
            try:
                response = client.chat.completions.create(
                    model=args.model,
                    messages=[{"role": "system", "content": record["prompt"]}],
                    temperature=args.temperature,
                    max_tokens=5,
                    top_p=args.top_p,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=None,
                    n=args.n,
                )
                record["all_responses"] = [c.message.content for c in response.choices]
                scored.append(record)
                time.sleep(0.5)
                break
            except Exception as error:  # rate limits, transient API failures
                print(error)
                time.sleep(2 * (attempt + 1))
        else:
            skipped += 1
            print(f"skipped {record['test_num']}/{record['test_type']}")

    print(f"scored {len(scored)}, skipped {skipped}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as json_file:
        json.dump(scored, json_file, indent=4)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
