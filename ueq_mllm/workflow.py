"""The UEQ-mLLM agentic workflow, its ablations, and the UEQ-sLLM baseline.

Full workflow (Fig. 1(b) of the paper):

    user status ──> LLM-1 (personalized questions) ──┐
                                                     ├──> LLM-3 (assembly) ──> LLM-4 (QA) ──> questionnaire
    UX honeycomb ──> LLM-2 (honeycomb questions) ────┘

Each ``run_*`` function returns an ordered mapping from stage name to raw model
output. The last value is the questionnaire that condition produces, and is what
gets scored by G-Eval.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Callable, Dict, Optional

from . import prompts
from .llm import generate

# A condition takes (pipeline, user_info, user_status, fixed_honeycomb) and
# returns an ordered mapping of stage name -> output.
Condition = Callable[..., "OrderedDict[str, str]"]


def _llm1(pipeline, user_info: str, user_status: str) -> str:
    """LLM-1: personalized questions conditioned on the user's live state."""
    return generate(
        pipeline,
        prompts.SYSTEM_PROMPT_1,
        prompts.USER_QUERY_1.format(user_status=user_status, user_info=user_info),
    )


def _llm2(pipeline, fixed_honeycomb: Optional[str] = None) -> str:
    """LLM-2: generic questions covering the seven UX honeycomb facets.

    The paper's controlled experiment holds this output fixed to a single
    arbitrary honeycomb questionnaire so that the only thing varying across the
    80 simulated users is the user-specific signal entering LLM-1. Pass
    ``fixed_honeycomb=None`` to let LLM-2 generate the questions instead.
    """
    if fixed_honeycomb is not None:
        return fixed_honeycomb

    return generate(pipeline, prompts.SYSTEM_PROMPT_2, prompts.USER_QUERY_2)


def _llm3(pipeline, honeycomb_questions: str, personalized_questions: str) -> str:
    """LLM-3: file each personalized question under the right honeycomb facet."""
    return generate(
        pipeline,
        prompts.SYSTEM_PROMPT_3,
        prompts.USER_QUERY_3.format(
            honeycomb_questions=honeycomb_questions,
            personalized_questions=personalized_questions,
        ),
    )


def _llm4(pipeline, questionnaire: str) -> str:
    """LLM-4: grammar and style pass over the assembled questionnaire."""
    return generate(
        pipeline,
        prompts.SYSTEM_PROMPT_4,
        prompts.USER_QUERY_4.format(questionnaire=questionnaire),
    )


# --------------------------------------------------------------------------- #
# Full workflow
# --------------------------------------------------------------------------- #


def run_ueq_mllm(
    pipeline, user_info: str, user_status: str, fixed_honeycomb: Optional[str] = None
) -> "OrderedDict[str, str]":
    """UEQ-mLLM: LLM-1 + LLM-2 -> LLM-3 -> LLM-4."""
    output_1 = _llm1(pipeline, user_info, user_status)
    output_2 = _llm2(pipeline, fixed_honeycomb)
    output_3 = _llm3(pipeline, output_2, output_1)
    output_4 = _llm4(pipeline, output_3)

    return OrderedDict(
        [
            ("Output_1", output_1),
            ("Output_2", output_2),
            ("Output_3", output_3),
            ("Output_4", output_4),
        ]
    )


# --------------------------------------------------------------------------- #
# Ablations: one agent removed at a time
# --------------------------------------------------------------------------- #


def run_without_llm1(
    pipeline, user_info: str, user_status: str, fixed_honeycomb: Optional[str] = None
) -> "OrderedDict[str, str]":
    """Drop personalization: the honeycomb questionnaire alone reaches LLM-3."""
    output_2 = _llm2(pipeline, fixed_honeycomb)
    output_3 = _llm3(pipeline, output_2, "")
    output_4 = _llm4(pipeline, output_3)

    return OrderedDict(
        [("Output_2", output_2), ("Output_3", output_3), ("Output_4", output_4)]
    )


def run_without_llm2(
    pipeline, user_info: str, user_status: str, fixed_honeycomb: Optional[str] = None
) -> "OrderedDict[str, str]":
    """Drop honeycomb coverage: only personalized questions reach LLM-3."""
    output_1 = _llm1(pipeline, user_info, user_status)
    output_3 = _llm3(pipeline, "", output_1)
    output_4 = _llm4(pipeline, output_3)

    return OrderedDict(
        [("Output_1", output_1), ("Output_3", output_3), ("Output_4", output_4)]
    )


def run_without_llm3(
    pipeline, user_info: str, user_status: str, fixed_honeycomb: Optional[str] = None
) -> "OrderedDict[str, str]":
    """Drop assembly: the two question sets are concatenated instead of merged."""
    output_1 = _llm1(pipeline, user_info, user_status)
    output_2 = _llm2(pipeline, fixed_honeycomb)
    output_4 = _llm4(pipeline, output_1 + output_2)

    return OrderedDict(
        [("Output_1", output_1), ("Output_2", output_2), ("Output_4", output_4)]
    )


def run_without_llm4(
    pipeline, user_info: str, user_status: str, fixed_honeycomb: Optional[str] = None
) -> "OrderedDict[str, str]":
    """Drop refinement: the assembled questionnaire is returned unpolished."""
    output_1 = _llm1(pipeline, user_info, user_status)
    output_2 = _llm2(pipeline, fixed_honeycomb)
    output_3 = _llm3(pipeline, output_2, output_1)

    return OrderedDict(
        [("Output_1", output_1), ("Output_2", output_2), ("Output_3", output_3)]
    )


# --------------------------------------------------------------------------- #
# Baseline
# --------------------------------------------------------------------------- #


def run_ueq_sllm(
    pipeline, user_info: str, user_status: str, fixed_honeycomb: Optional[str] = None
) -> "OrderedDict[str, str]":
    """UEQ-sLLM: all four roles collapsed into one prompt and one call."""
    output = generate(
        pipeline,
        prompts.SYSTEM_PROMPT_SINGLE,
        prompts.USER_QUERY_SINGLE.format(
            user_status=user_status, user_info=user_info
        ),
    )

    return OrderedDict([("Output", output)])


# Column name in the result CSV -> condition. The order here is the column order
# used by ``run_experiment.py`` and by the released result files.
CONDITIONS: Dict[str, Condition] = OrderedDict(
    [
        ("MultipleLLM", run_ueq_mllm),
        ("WithoutL1", run_without_llm1),
        ("WithoutL2", run_without_llm2),
        ("WithoutL3", run_without_llm3),
        ("WithoutL4", run_without_llm4),
        ("SingleLLM", run_ueq_sllm),
    ]
)
