"""UEQ-mLLM: an agentic LLM workflow for personalized UX questionnaire generation.

Reference implementation for:

    Y. Kim, J. Lee, J. H. Han, M. Kim, H. Lee, and W. H. Lee,
    "Agentic LLM Workflows for Personalized User Experience Questionnaire
    Generation," IEEE ICCE-Asia 2024, pp. 1-4.
    https://doi.org/10.1109/ICCE-Asia63397.2024.10773955
"""

__version__ = "1.0.0"

from .workflow import (  # noqa: F401
    CONDITIONS,
    run_ueq_mllm,
    run_ueq_sllm,
    run_without_llm1,
    run_without_llm2,
    run_without_llm3,
    run_without_llm4,
)
