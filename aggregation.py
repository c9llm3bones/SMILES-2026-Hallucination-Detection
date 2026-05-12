"""
aggregation.py — Feature extraction strategy controlled by config.py.

Strategies (set via config.py):
  mean_pool      mean-pool all real tokens across specified layers
  last_token     last real token across specified layers
  response_pool  response-only max+mean pool (Sol. 4 default)
"""

from __future__ import annotations

import os

import torch

from config import CFG

_AGG = CFG["aggregation"]

# ---------------------------------------------------------------------------
# Pre-compute prompt lengths at import time (only needed for response_pool).
# ---------------------------------------------------------------------------

_prompt_lens: list[int] = []
_call_idx: int = 0


def _load_prompt_lens() -> list[int]:
    if not _AGG.get("response_only", False):
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
        return lens
    except Exception:
        return []


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
    return resp if resp.any() else real_mask


def _prompt_mask(real_mask: torch.Tensor, prompt_len: int | None) -> torch.Tensor:
    if prompt_len is None or prompt_len <= 0:
        return real_mask
    seq_len = real_mask.shape[0]
    end = min(prompt_len, seq_len)
    prompt = real_mask.clone()
    prompt[end:] = False
    return prompt if prompt.any() else real_mask


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
    use_geometric: bool = False,
    prompt_len: int | None = None,
) -> torch.Tensor:
    global _call_idx

    hs   = hidden_states.cpu().float()
    mask = attention_mask.cpu().bool()
    parts: list[torch.Tensor] = []

    strategy = _AGG["strategy"]

    if strategy == "response_pool":
        if prompt_len is None and _call_idx < len(_prompt_lens):
            prompt_len = _prompt_lens[_call_idx]
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
