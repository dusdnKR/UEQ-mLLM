"""Synthetic user population used in the paper's controlled experiment.

Eighty hypothetical users are laid out on a fixed grid so that sex and age are
balanced by construction: users are assigned in blocks of ten, five male
followed by five female, and each block of ten advances one age bracket. Only
the attention and emotion distributions are sampled at random.

    test_num  1-5    -> Male,   10s'      test_num 41-45 -> Male,   50s'
    test_num  6-10   -> Female, 10s'      test_num 46-50 -> Female, 50s'
    ...                                   ...
    test_num 36-40   -> Female, 40s'      test_num 76-80 -> Female, 80s'
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

SEX_LIST = ["Male", "Female"]
AGE_LIST = ["10s'", "20s'", "30s'", "40s'", "50s'", "60s'", "70s'", "80s'"]

ATTENTION_STATES = ["focus", "unfocus", "drowsy"]
EMOTION_STATES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

N_USERS = len(AGE_LIST) * len(SEX_LIST) * 5  # 80


@dataclass
class SimulatedUser:
    """One synthetic participant, plus the prompt-ready renderings of its data."""

    test_num: int
    sex: str
    age: str
    attention: Sequence[float]  # percentages over ATTENTION_STATES
    emotion: Sequence[float]  # percentages over EMOTION_STATES

    @property
    def user_info(self) -> str:
        return format_user_info(self.sex, self.age)

    @property
    def user_status(self) -> str:
        return format_user_status(self.attention, self.emotion)

    def as_csv_row(self) -> list:
        return [self.test_num, self.sex, self.age] + [
            f"{value}%" for value in list(self.attention) + list(self.emotion)
        ]


def random_distribution(n: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Sample n non-negative percentages that sum to 100."""
    rng = rng or np.random.default_rng()
    values = rng.random(n)
    return values / np.sum(values) * 100


def build_user(test_num: int, rng: np.random.Generator | None = None) -> SimulatedUser:
    """Build the synthetic user for a 1-based test number (1-80)."""
    if not 1 <= test_num <= N_USERS:
        raise ValueError(f"test_num must be in [1, {N_USERS}], got {test_num}")

    return SimulatedUser(
        test_num=test_num,
        sex=SEX_LIST[((test_num - 1) % 10) // 5],
        age=AGE_LIST[(test_num - 1) // 10],
        attention=random_distribution(len(ATTENTION_STATES), rng),
        emotion=random_distribution(len(EMOTION_STATES), rng),
    )


def format_user_info(sex: str, age: str) -> str:
    """Render the '## User Information' block handed to the LLMs."""
    return f"""
## User Information
- Sex: {sex}
- Age: {age}
"""


def format_user_status(
    attention: Sequence[float], emotion: Sequence[float]
) -> str:
    """Render the '## Current User Status' block handed to the LLMs.

    In the deployed system these percentages come from the MIND dashboard: the
    attention triple from the EEG classifier and the emotion vector from the
    facial expression recognition module.
    """
    attention_block = "\n".join(
        f"- {state}: {value}%" for state, value in zip(ATTENTION_STATES, attention)
    )
    emotion_block = "\n".join(
        f"- {state}: {value}%" for state, value in zip(EMOTION_STATES, emotion)
    )

    return f"""
## Current User Status
- Attention:

{attention_block}

- Emotion:

{emotion_block}

"""


CSV_HEADER = ["test_num", "sex", "age"] + ATTENTION_STATES + EMOTION_STATES
