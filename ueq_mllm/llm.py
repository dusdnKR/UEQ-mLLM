"""Thin wrapper around the Hugging Face text-generation pipeline.

The paper uses Meta-Llama-3.1-8B-Instruct as the single foundation model behind
every agent in the workflow, so the four "LLMs" of UEQ-mLLM are four distinct
roles served by one shared pipeline rather than four different checkpoints.

Select GPUs through the environment, e.g. ``CUDA_VISIBLE_DEVICES=0,1``.
"""

from __future__ import annotations

import transformers
import torch

DEFAULT_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
DEFAULT_MAX_NEW_TOKENS = 1000


def build_pipeline(model_name: str = DEFAULT_MODEL):
    """Load a tokenizer and model, and return a text-generation pipeline."""
    print(f"model_name: {model_name}")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", trust_remote_code=True
    )

    return transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        pad_token_id=tokenizer.eos_token_id,
        model_kwargs={"torch_dtype": torch.float16},
    )


def generate(
    pipeline,
    system_prompt: str,
    user_query: str,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> str:
    """Run one chat completion and return the assistant message content."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
    ]

    outputs = pipeline(messages, max_new_tokens=max_new_tokens)

    return outputs[0]["generated_text"][-1]["content"]
