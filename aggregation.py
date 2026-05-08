"""
aggregation.py — Token aggregation strategy and feature extraction.

Feature vector layout (USE_GEOMETRIC=True):
  Block A  [0        : 4480]  — mean pool of real tokens across 5 selected layers
  Block B  [4480     : 4504]  — inter-layer cosine similarity (24 transitions)
  Block C  [4504     : 4529]  — layer-wise mean L2 norm (25 layers)
  Block D  [4529     : 4534]  — scalar geometric features (5 values)
  Total: 4534-d vector

With USE_GEOMETRIC=False only Block A (4480-d) is returned.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# Transformer layer indices to use for Block A.
# Index 0 = token embeddings; 1-24 = transformer layers 1-24.
# Using all layers: PCA in probe.py finds the most discriminative directions
# across the full depth trajectory without manual layer selection bias.
_SELECTED_LAYERS = list(range(25))  # all 25 layers → 25*896 = 22400-d, PCA to 128


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Block A: last real token across selected transformer layers.

    Using the last real token rather than mean pool because:
    - In a decoder-only model the last token attends to the full context
    - ~60-70% of tokens belong to the (identical) prompt — mean pool is noisy
    - The final token state captures the model's committed conclusion

    Args:
        hidden_states:  (n_layers, seq_len, hidden_dim)
        attention_mask: (seq_len,) — 1 for real tokens, 0 for padding.

    Returns:
        (len(_SELECTED_LAYERS) * hidden_dim,) = (4480,)
    """
    real_positions = attention_mask.nonzero(as_tuple=False)
    last_pos = int(real_positions[-1].item())
    parts = []
    for layer_idx in _SELECTED_LAYERS:
        parts.append(hidden_states[layer_idx][last_pos])  # (896,)
    return torch.cat(parts)  # (4480,)


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Blocks B, C, D: geometric and statistical features of hidden states.

    Block B (24d): cosine similarity between consecutive layer representations.
        Low cosine = large representational shift at that layer boundary.

    Block C (25d): mean L2 norm of real-token hidden states per layer.
        Norms tend to grow through the network; degenerate outputs show
        anomalous growth patterns.

    Block D (5d): scalar summary statistics of the last-layer activations.
        norm_mean  — mean token norm in last layer
        norm_cv    — coefficient of variation (std/mean) of token norms;
                     low CV signals repetitive/degenerate generation
        norm_ratio — last-layer mean norm / first-transformer-layer mean norm
        min_cos    — minimum inter-layer cosine (= point of max drift)
        n_real     — number of real (non-padding) tokens (length proxy)

    Args:
        hidden_states:  (n_layers, seq_len, hidden_dim)
        attention_mask: (seq_len,)

    Returns:
        (24 + 25 + 5,) = (54,)
    """
    real_mask = attention_mask.bool()
    n_layers = hidden_states.shape[0]  # 25 for Qwen2.5-0.5B

    # Block B: inter-layer cosine similarity (24 values)
    block_b: list[float] = []
    for l in range(n_layers - 1):
        h_l  = hidden_states[l][real_mask].mean(0)      # (hidden_dim,)
        h_l1 = hidden_states[l + 1][real_mask].mean(0)  # (hidden_dim,)
        cos  = F.cosine_similarity(h_l.unsqueeze(0), h_l1.unsqueeze(0)).item()
        block_b.append(cos)

    # Block C: layer-wise mean L2 norm (25 values)
    block_c: list[float] = [
        hidden_states[l][real_mask].norm(dim=-1).mean().item()
        for l in range(n_layers)
    ]

    # Block D: scalars from last layer (5 values)
    last        = hidden_states[-1][real_mask]            # (n_real, hidden_dim)
    token_norms = last.norm(dim=-1)                       # (n_real,)

    norm_mean  = token_norms.mean().item()
    norm_cv    = (token_norms.std() / (token_norms.mean() + 1e-8)).item()
    first_norm = hidden_states[1][real_mask].norm(dim=-1).mean().item()
    norm_ratio = norm_mean / (first_norm + 1e-8)
    min_cos    = float(min(block_b))
    n_real     = float(real_mask.sum().item())

    block_d = [norm_mean, norm_cv, norm_ratio, min_cos, n_real]

    return torch.tensor(block_b + block_c + block_d, dtype=torch.float32)


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    """Combine Block A with optional geometric features (Blocks B, C, D).

    Args:
        hidden_states:  (n_layers, seq_len, hidden_dim)
        attention_mask: (seq_len,)
        use_geometric:  If True, append Blocks B+C+D to Block A.

    Returns:
        (4480,) when use_geometric=False, (4534,) when True.
    """
    hidden_states  = hidden_states.cpu()
    attention_mask = attention_mask.cpu()

    agg_features = aggregate(hidden_states, attention_mask)

    if use_geometric:
        geo_features = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg_features, geo_features], dim=0)

    return agg_features

