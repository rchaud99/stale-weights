"""Loss masking for the cloud trainer, as mlx-lm's ChatDataset does it with mask_prompt: the loss covers only the
tokens after the prompt (the assistant reply and the end-of-turn tokens). A pure function, so it is unit-tested
locally with the same tokenizer."""

from __future__ import annotations


def encode(tok, text: str) -> list[int]:
    """Chat-template text -> ids, with no extra special tokens (the template already has them)."""
    return list(tok(text, add_special_tokens=False)["input_ids"])


def mask_example(tok, messages: list[dict]) -> tuple[list[int], int]:
    """(token ids of the whole conversation, offset): the loss is on ids[offset:]."""
    full = encode(tok, tok.apply_chat_template(messages, tokenize=False))
    prompt = encode(tok, tok.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True))
    if full[:len(prompt)] != prompt:
        raise ValueError("the prompt is not a token prefix of the conversation")
    return full, len(prompt)
