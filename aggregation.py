"""
aggregation.py — Feature extraction strategy controlled by config.py.

Strategies (set via config.py):
  mean_pool      mean-pool all real tokens across specified layers
  last_token     last real token across specified layers
  response_pool  response-only max+mean pool (Sol. 4 default)

NOTE: use_geometric is accepted for interface compatibility with solution.py
but is intentionally ignored — geometric features are not used.
"""

from __future__ import annotations

import os
import warnings

import torch

from config import CFG

_AGG = CFG["aggregation"]

# ---------------------------------------------------------------------------
# Pre-compute prompt lengths at import time (only needed for response_pool).
# Loads dataset.csv then test.csv in the same order solution.py processes them.
# The global counter _call_idx maps sequential invocations to prompt lengths.
# WARNING: this mapping breaks if aggregation is called in a different order.
# ---------------------------------------------------------------------------

_prompt_lens: list[int] = []
_call_idx: int = 0


def _load_prompt_lens() -> list[int]:
    needs_boundary = _AGG.get("response_only", False) or _AGG.get("prompt_only", False)
    if not needs_boundary:
        return []
    try:
        import pandas as pd
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True
        )
        lens: list[int] = []
        for path in ["./data/dataset.csv", "./data/test.csv"]:
            if os.path.exists(path):
                df = pd.read_csv(path)
                for _, row in df.iterrows():
                    enc = tok(str(row["prompt"]))
                    lens.append(len(enc["input_ids"]))
        if not lens:
            raise RuntimeError("No prompt lengths loaded — data files not found.")
        return lens
    except Exception as e:
        raise RuntimeError(
            f"aggregation.py: failed to pre-tokenise prompts. "
            f"Response-only pooling will silently degrade to all-tokens (−9 pp AUROC). "
            f"Original error: {e}"
        ) from e


_prompt_lens = _load_prompt_lens()


def reset_call_counter() -> None:
    global _call_idx
    _call_idx = 0


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def _response_mask(real_mask: torch.Tensor, prompt_len: int | None) -> torch.Tensor:
    if prompt_len is None or prompt_len <= 0:
        return real_mask
    seq_len = real_mask.shape[0]
    start = min(prompt_len, seq_len - 1)
    resp = real_mask.clone()
    resp[:start] = False
    if not resp.any():
        warnings.warn(
            f"Response mask is empty after truncation "
            f"(prompt_len={prompt_len}, seq_len={seq_len}). "
            "Falling back to full sequence — response-only pooling inactive for this sample.",
            stacklevel=3,
        )
        return real_mask
    return resp


def _prompt_mask(real_mask: torch.Tensor, prompt_len: int | None) -> torch.Tensor:
    if prompt_len is None or prompt_len <= 0:
        return real_mask
    seq_len = real_mask.shape[0]
    end = min(prompt_len, seq_len)
    prompt = real_mask.clone()
    prompt[end:] = False
    if not prompt.any():
        warnings.warn(
            f"Prompt mask is empty (prompt_len={prompt_len}, seq_len={seq_len}). "
            "Falling back to full sequence.",
            stacklevel=3,
        )
        return real_mask
    return prompt


def _pool(h: torch.Tensor, mask: torch.Tensor, mode: str) -> torch.Tensor:
    tokens = h[mask]
    if mode == "max":
        return tokens.max(dim=0).values
    return tokens.mean(dim=0)


# ---------------------------------------------------------------------------
# Entry point called by solution.py
# ---------------------------------------------------------------------------


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,  # accepted for compatibility; intentionally unused
    prompt_len: int | None = None,
) -> torch.Tensor:
    global _call_idx

    hs   = hidden_states.cpu().float()
    mask = attention_mask.cpu().bool()
    parts: list[torch.Tensor] = []

    strategy = _AGG["strategy"]

    if strategy == "response_pool":
        if prompt_len is None:
            if _call_idx < len(_prompt_lens):
                prompt_len = _prompt_lens[_call_idx]
            else:
                warnings.warn(
                    f"_call_idx={_call_idx} exceeds pre-computed prompt_lens "
                    f"(n={len(_prompt_lens)}). prompt_len=None; falling back to full sequence.",
                    stacklevel=2,
                )
        _call_idx += 1

        if _AGG.get("prompt_only", False):
            resp = _prompt_mask(mask, prompt_len)
        elif _AGG.get("response_only", True):
            resp = _response_mask(mask, prompt_len)
        else:
            resp = mask

        for layer_idx in _AGG.get("max_pool_layers", []):
            parts.append(_pool(hs[layer_idx], resp, "max"))
        for layer_idx in _AGG.get("mean_pool_layers", []):
            parts.append(_pool(hs[layer_idx], resp, "mean"))

    elif strategy == "last_token":
        _call_idx += 1
        last_pos = int(mask.nonzero(as_tuple=False)[-1].item())
        for layer_idx in _AGG["layers"]:
            parts.append(hs[layer_idx][last_pos])

    elif strategy == "mean_pool":
        _call_idx += 1
        for layer_idx in _AGG["layers"]:
            parts.append(_pool(hs[layer_idx], mask, "mean"))

    return torch.cat(parts)


# Legacy entry points
def aggregate(hidden_states, attention_mask, prompt_len=None):
    return aggregation_and_feature_extraction(
        hidden_states, attention_mask, prompt_len=prompt_len
    )


def extract_geometric_features(hidden_states, attention_mask):
    return torch.empty(0)
