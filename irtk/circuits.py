"""Circuit analysis utilities for mechanistic interpretability.

Provides tools for analyzing attention head circuits:
- OV circuits: what information does a head write to the residual stream?
- QK circuits: what does a head attend to?
- Direct logit attribution: how much does each head's output affect the logits?
- Composition scores: how strongly do heads interact?
"""

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from irtk.hooked_transformer import HookedTransformer
from irtk.activation_cache import ActivationCache
from irtk.factored_matrix import FactoredMatrix


# ─── OV and QK Circuit Analysis ─────────────────────────────────────────────


def ov_circuit(model: HookedTransformer, layer: int, head: int) -> FactoredMatrix:
    """Get the OV circuit matrix for a specific head.

    The OV circuit determines what a head writes to the residual stream.
    OV = W_V @ W_O (as a FactoredMatrix: maps d_model -> d_model).

    The full OV circuit is W_E @ W_V @ W_O @ W_U, which tells you:
    "if the head attends to token X, how does that change the logits?"

    Args:
        model: HookedTransformer.
        layer: Layer index.
        head: Head index.

    Returns:
        FactoredMatrix of shape [d_model, d_model].
    """
    W_V = model.blocks[layer].attn.W_V[head]  # [d_model, d_head]
    W_O = model.blocks[layer].attn.W_O[head]  # [d_head, d_model]
    return FactoredMatrix(W_V, W_O)


def qk_circuit(model: HookedTransformer, layer: int, head: int) -> FactoredMatrix:
    """Get the QK circuit matrix for a specific head.

    The QK circuit determines attention patterns.
    QK = W_Q^T @ W_K (as a FactoredMatrix: maps d_model -> d_model).

    The full QK circuit is W_E^T @ W_Q^T @ W_K @ W_E, which tells you:
    "how much does source token X attend to destination token Y?"

    Args:
        model: HookedTransformer.
        layer: Layer index.
        head: Head index.

    Returns:
        FactoredMatrix of shape [d_model, d_model].
    """
    W_Q = model.blocks[layer].attn.W_Q[head]  # [d_model, d_head]
    W_K = model.blocks[layer].attn.W_K[head]  # [d_model, d_head]
    return FactoredMatrix(W_Q, W_K.T)


def full_ov_circuit(
    model: HookedTransformer, layer: int, head: int
) -> FactoredMatrix:
    """Get the full OV circuit: W_E @ W_V @ W_O @ W_U.

    Maps from input tokens to output logits through a specific head.
    Element [i, j] tells you how much attending to token i promotes token j.

    Args:
        model: HookedTransformer.
        layer: Layer index.
        head: Head index.

    Returns:
        FactoredMatrix of shape [d_vocab, d_vocab].
    """
    W_E = model.embed.W_E   # [d_vocab, d_model]
    W_V = model.blocks[layer].attn.W_V[head]  # [d_model, d_head]
    W_O = model.blocks[layer].attn.W_O[head]  # [d_head, d_model]
    W_U = model.unembed.W_U  # [d_model, d_vocab]

    # W_E @ W_V @ W_O @ W_U
    # Factor as (W_E @ W_V) @ (W_O @ W_U)
    left = W_E @ W_V    # [d_vocab, d_head]
    right = W_O @ W_U   # [d_head, d_vocab]
    return FactoredMatrix(left, right)


def full_qk_circuit(
    model: HookedTransformer, layer: int, head: int
) -> FactoredMatrix:
    """Get the full QK circuit: W_E^T @ W_Q^T @ W_K @ W_E.

    Maps from query tokens to key tokens through the attention pattern.
    Element [i, j] is the attention score from token i to token j (before softmax).

    Args:
        model: HookedTransformer.
        layer: Layer index.
        head: Head index.

    Returns:
        FactoredMatrix of shape [d_vocab, d_vocab].
    """
    W_E = model.embed.W_E   # [d_vocab, d_model]
    W_Q = model.blocks[layer].attn.W_Q[head]  # [d_model, d_head]
    W_K = model.blocks[layer].attn.W_K[head]  # [d_model, d_head]

    # (W_E @ W_Q) @ (W_E @ W_K)^T
    left = W_E @ W_Q    # [d_vocab, d_head]
    right = (W_E @ W_K).T  # [d_head, d_vocab]
    return FactoredMatrix(left, right)


# ─── Direct Logit Attribution ────────────────────────────────────────────────


def direct_logit_attribution(
    model: HookedTransformer,
    cache: ActivationCache,
    token: int,
    pos: int = -1,
) -> np.ndarray:
    """Compute each head's direct contribution to a specific output logit.

    Each head writes to the residual stream via z @ W_O. We project this
    through the unembedding to get the head's effect on the logits.

    Args:
        model: HookedTransformer.
        cache: ActivationCache from run_with_cache.
        token: Target token ID.
        pos: Position to analyze (-1 for last).

    Returns:
        [n_layers, n_heads] array of logit contributions for the target token.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    W_U = model.unembed.W_U  # [d_model, d_vocab]

    # Direction in d_model space for the target token
    logit_dir = W_U[:, token]  # [d_model]

    results = np.zeros((n_layers, n_heads))

    for layer in range(n_layers):
        z = cache[("z", layer)]  # [seq_len, n_heads, d_head]
        W_O = model.blocks[layer].attn.W_O  # [n_heads, d_head, d_model]

        for head in range(n_heads):
            # Head's output at the target position
            head_output = z[pos, head] @ W_O[head]  # [d_model]
            # Project onto target logit direction
            results[layer, head] = float(jnp.dot(head_output, logit_dir))

    return results


def residual_stream_attribution(
    model: HookedTransformer,
    cache: ActivationCache,
    token: int,
    pos: int = -1,
) -> dict[str, float]:
    """Decompose the logit for a token into contributions from each component.

    Returns the contribution of: embedding, each attention output, each MLP output.

    Args:
        model: HookedTransformer.
        cache: ActivationCache from run_with_cache.
        token: Target token ID.
        pos: Position to analyze.

    Returns:
        Dict mapping component name -> logit contribution.
    """
    W_U = model.unembed.W_U
    logit_dir = W_U[:, token]  # [d_model]

    contributions = {}

    # Embedding
    embed = cache.cache_dict.get("hook_embed", None)
    pos_embed = cache.cache_dict.get("hook_pos_embed", None)
    if embed is not None:
        e = embed[pos]
        if pos_embed is not None:
            e = e + pos_embed[pos]
        contributions["embed"] = float(jnp.dot(e, logit_dir))

    # Each layer's attention and MLP
    for layer in range(model.cfg.n_layers):
        attn_key = f"blocks.{layer}.hook_attn_out"
        mlp_key = f"blocks.{layer}.hook_mlp_out"

        if attn_key in cache.cache_dict:
            attn_out = cache.cache_dict[attn_key][pos]
            contributions[f"L{layer}_attn"] = float(jnp.dot(attn_out, logit_dir))

        if mlp_key in cache.cache_dict:
            mlp_out = cache.cache_dict[mlp_key][pos]
            contributions[f"L{layer}_mlp"] = float(jnp.dot(mlp_out, logit_dir))

    return contributions


# ─── Composition Scores ──────────────────────────────────────────────────────


def qk_composition_score(
    model: HookedTransformer,
    src_layer: int, src_head: int,
    dst_layer: int, dst_head: int,
) -> float:
    """Compute QK composition score between two heads.

    Measures how much the output of src_head (via its OV circuit) is used
    as a key by dst_head (via its QK circuit).

    The composition score is the Frobenius norm of
    W_O_src @ W_Q_dst (projected through W_K_dst).

    Higher score means stronger composition.

    Args:
        model: HookedTransformer.
        src_layer: Source head's layer.
        src_head: Source head index.
        dst_layer: Destination head's layer.
        dst_head: Destination head index.

    Returns:
        Scalar composition score.
    """
    if src_layer >= dst_layer:
        return 0.0

    W_O_src = model.blocks[src_layer].attn.W_O[src_head]  # [d_head, d_model]
    W_Q_dst = model.blocks[dst_layer].attn.W_Q[dst_head]  # [d_model, d_head]

    # Composition: W_O_src @ W_Q_dst -> [d_head_src, d_head_dst]
    composed = W_O_src @ W_Q_dst
    return float(jnp.linalg.norm(composed))


def ov_composition_score(
    model: HookedTransformer,
    src_layer: int, src_head: int,
    dst_layer: int, dst_head: int,
) -> float:
    """Compute OV composition score between two heads.

    Measures how much the output of src_head is used as a value by dst_head.
    This is W_O_src @ W_V_dst.

    Args:
        model: HookedTransformer.
        src_layer: Source head's layer.
        src_head: Source head index.
        dst_layer: Destination head's layer.
        dst_head: Destination head index.

    Returns:
        Scalar composition score.
    """
    if src_layer >= dst_layer:
        return 0.0

    W_O_src = model.blocks[src_layer].attn.W_O[src_head]  # [d_head, d_model]
    W_V_dst = model.blocks[dst_layer].attn.W_V[dst_head]  # [d_model, d_head]

    composed = W_O_src @ W_V_dst
    return float(jnp.linalg.norm(composed))


def all_composition_scores(
    model: HookedTransformer,
    composition_type: str = "qk",
) -> np.ndarray:
    """Compute composition scores between all pairs of heads.

    Args:
        model: HookedTransformer.
        composition_type: "qk" or "ov".

    Returns:
        [n_layers * n_heads, n_layers * n_heads] matrix of scores.
        Entry [i, j] is the composition score from head i to head j.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    total = n_layers * n_heads

    score_fn = qk_composition_score if composition_type == "qk" else ov_composition_score
    scores = np.zeros((total, total))

    for src_l in range(n_layers):
        for src_h in range(n_heads):
            src_idx = src_l * n_heads + src_h
            for dst_l in range(src_l + 1, n_layers):
                for dst_h in range(n_heads):
                    dst_idx = dst_l * n_heads + dst_h
                    scores[src_idx, dst_idx] = score_fn(
                        model, src_l, src_h, dst_l, dst_h
                    )

    return scores


# ─── Attention Pattern Analysis ──────────────────────────────────────────────


def attention_to_positions(
    cache: ActivationCache,
    layer: int,
    head: int,
    query_pos: int = -1,
) -> np.ndarray:
    """Get attention weights from a specific query position.

    Args:
        cache: ActivationCache from run_with_cache.
        layer: Layer index.
        head: Head index.
        query_pos: Position of the query token.

    Returns:
        [seq_len] array of attention weights from query_pos to all key positions.
    """
    pattern = cache[("pattern", layer)]  # [n_heads, q_pos, k_pos]
    return np.array(pattern[head, query_pos])


def prev_token_score(
    cache: ActivationCache,
    layer: int,
    head: int,
) -> float:
    """Score how much a head acts as a "previous token" head.

    A previous token head primarily attends to position i-1 when querying from
    position i. Score = average attention paid to the previous token.

    Args:
        cache: ActivationCache.
        layer: Layer index.
        head: Head index.

    Returns:
        Score between 0 and 1 (1 = perfect previous-token head).
    """
    pattern = np.array(cache[("pattern", layer)][head])  # [q, k]
    seq_len = pattern.shape[0]
    if seq_len < 2:
        return 0.0

    # For each position i >= 1, how much does it attend to i-1?
    scores = [pattern[i, i - 1] for i in range(1, seq_len)]
    return float(np.mean(scores))


def induction_score(
    cache: ActivationCache,
    layer: int,
    head: int,
    offset: int = 1,
) -> float:
    """Score how much a head acts as an induction head.

    An induction head attends to the token after a previous occurrence of the
    current token. For repeated sequences [A B C A B C], when querying from
    the second B, an induction head attends to the position after the first B
    (which is the first C).

    This computes the average diagonal offset in the attention pattern,
    which is high for induction heads on repeated sequences.

    Args:
        cache: ActivationCache from a repeated-sequence run.
        layer: Layer index.
        head: Head index.
        offset: Diagonal offset to check (1 for standard induction).

    Returns:
        Score between 0 and 1.
    """
    pattern = np.array(cache[("pattern", layer)][head])  # [q, k]
    seq_len = pattern.shape[0]
    half = seq_len // 2

    if half < 2:
        return 0.0

    # For the second half of a repeated sequence, induction heads attend
    # to position (current - half + offset)
    scores = []
    for q in range(half + offset, seq_len):
        k = q - half + offset
        if 0 <= k < seq_len:
            scores.append(pattern[q, k])

    return float(np.mean(scores)) if scores else 0.0
