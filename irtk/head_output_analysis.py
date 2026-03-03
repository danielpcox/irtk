"""Per-head output analysis for mechanistic interpretability.

Detailed analysis of individual attention head outputs: decomposition,
value-weighted analysis, output direction characterization, head
cooperation/competition, and output norm patterns.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Optional


def head_output_decomposition(
    model,
    tokens,
    layer: int = 0,
    pos: int = -1,
    top_k: int = 5,
) -> dict:
    """Decompose each head's output at a position.

    Args:
        model: HookedTransformer model.
        tokens: Input token array.
        layer: Layer.
        pos: Position.
        top_k: Top vocab tokens per head.

    Returns:
        Dict with per_head outputs, norms, top promoted/demoted tokens.
    """
    from irtk.hook_points import HookState

    cache = {}
    hs = HookState(hook_fns={}, cache=cache)
    model(tokens, hook_state=hs)

    z_key = f"blocks.{layer}.attn.hook_z"
    if z_key not in cache:
        return {"per_head": []}

    z = np.array(cache[z_key])  # [seq, n_heads, d_head]
    W_O = np.array(model.blocks[layer].attn.W_O)  # [n_heads, d_head, d_model]
    W_U = np.array(model.unembed.W_U)  # [d_model, d_vocab]
    n_heads = z.shape[1]

    per_head = []
    for h in range(n_heads):
        z_h = z[pos, h]  # [d_head]
        output = z_h @ W_O[h]  # [d_model]
        logit_effect = output @ W_U  # [d_vocab]

        promoted = np.argsort(logit_effect)[::-1][:top_k]
        demoted = np.argsort(logit_effect)[:top_k]

        per_head.append({
            "output_norm": float(np.linalg.norm(output)),
            "promoted": [(int(t), float(logit_effect[t])) for t in promoted],
            "demoted": [(int(t), float(logit_effect[t])) for t in demoted],
        })

    return {
        "per_head": per_head,
        "n_heads": n_heads,
    }


def value_weighted_analysis(
    model,
    tokens,
    layer: int = 0,
    head: int = 0,
    pos: int = -1,
) -> dict:
    """Analyze what values each source position contributes.

    Args:
        model: HookedTransformer model.
        tokens: Input token array.
        layer: Layer.
        head: Head.
        pos: Query position.

    Returns:
        Dict with per_source_contribution, attention_weights, dominant_source.
    """
    from irtk.hook_points import HookState

    cache = {}
    hs = HookState(hook_fns={}, cache=cache)
    model(tokens, hook_state=hs)

    pattern_key = f"blocks.{layer}.attn.hook_pattern"
    v_key = f"blocks.{layer}.attn.hook_v"

    if pattern_key not in cache or v_key not in cache:
        return {"per_source_contribution": [], "attention_weights": jnp.array([])}

    pattern = np.array(cache[pattern_key][head])  # [seq, seq]
    v = np.array(cache[v_key])[:, head, :]  # [seq, d_head]
    W_O = np.array(model.blocks[layer].attn.W_O[head])  # [d_head, d_model]

    attn_weights = pattern[pos]  # [seq]
    seq_len = len(attn_weights)

    per_source = []
    for src in range(seq_len):
        weighted_v = attn_weights[src] * v[src]  # [d_head]
        output = weighted_v @ W_O  # [d_model]
        per_source.append({
            "attention_weight": float(attn_weights[src]),
            "output_norm": float(np.linalg.norm(output)),
        })

    dominant = int(np.argmax([s["output_norm"] for s in per_source]))

    return {
        "per_source_contribution": per_source,
        "attention_weights": jnp.array(attn_weights),
        "dominant_source": dominant,
        "n_sources": seq_len,
    }


def output_direction_characterization(
    model,
    tokens,
    layer: int = 0,
    pos: int = -1,
) -> dict:
    """Characterize the direction of each head's output.

    Args:
        model: HookedTransformer model.
        tokens: Input token array.
        layer: Layer.
        pos: Position.

    Returns:
        Dict with head_directions, pairwise_cosines, diversity_score.
    """
    from irtk.hook_points import HookState

    cache = {}
    hs = HookState(hook_fns={}, cache=cache)
    model(tokens, hook_state=hs)

    z_key = f"blocks.{layer}.attn.hook_z"
    if z_key not in cache:
        return {"head_directions": jnp.array([]), "pairwise_cosines": jnp.array([])}

    z = np.array(cache[z_key])
    W_O = np.array(model.blocks[layer].attn.W_O)
    n_heads = z.shape[1]

    outputs = []
    for h in range(n_heads):
        out = z[pos, h] @ W_O[h]
        outputs.append(out / (np.linalg.norm(out) + 1e-10))
    outputs = np.stack(outputs)

    # Pairwise cosines
    cosines = outputs @ outputs.T

    # Diversity: mean absolute off-diagonal cosine
    off_diag = []
    for i in range(n_heads):
        for j in range(i + 1, n_heads):
            off_diag.append(abs(cosines[i, j]))
    diversity = 1.0 - (float(np.mean(off_diag)) if off_diag else 0.0)

    return {
        "head_directions": jnp.array(outputs),
        "pairwise_cosines": jnp.array(cosines),
        "diversity_score": diversity,
        "n_heads": n_heads,
    }


def head_cooperation_competition(
    model,
    tokens,
    layer: int = 0,
    pos: int = -1,
) -> dict:
    """Measure cooperation and competition between heads.

    Args:
        model: HookedTransformer model.
        tokens: Input token array.
        layer: Layer.
        pos: Position.

    Returns:
        Dict with cooperation_matrix, cooperating_pairs, competing_pairs.
    """
    from irtk.hook_points import HookState

    cache = {}
    hs = HookState(hook_fns={}, cache=cache)
    model(tokens, hook_state=hs)

    z_key = f"blocks.{layer}.attn.hook_z"
    if z_key not in cache:
        return {"cooperation_matrix": jnp.array([[]]), "cooperating_pairs": [], "competing_pairs": []}

    z = np.array(cache[z_key])
    W_O = np.array(model.blocks[layer].attn.W_O)
    W_U = np.array(model.unembed.W_U)
    n_heads = z.shape[1]

    # Each head's logit effect
    logit_effects = []
    for h in range(n_heads):
        out = z[pos, h] @ W_O[h]
        effect = out @ W_U  # [d_vocab]
        logit_effects.append(effect)
    logit_effects = np.stack(logit_effects)

    # Cooperation = positive correlation of logit effects
    cooperation = np.corrcoef(logit_effects)

    cooperating = []
    competing = []
    for i in range(n_heads):
        for j in range(i + 1, n_heads):
            c = float(cooperation[i, j])
            if c > 0.1:
                cooperating.append((i, j, c))
            elif c < -0.1:
                competing.append((i, j, c))

    return {
        "cooperation_matrix": jnp.array(cooperation),
        "cooperating_pairs": sorted(cooperating, key=lambda x: -x[2]),
        "competing_pairs": sorted(competing, key=lambda x: x[2]),
        "n_heads": n_heads,
    }


def output_norm_analysis(
    model,
    tokens,
    layer: int = 0,
) -> dict:
    """Analyze output norm patterns across positions and heads.

    Args:
        model: HookedTransformer model.
        tokens: Input token array.
        layer: Layer.

    Returns:
        Dict with norm_matrix (heads x positions), head_mean_norms,
        position_mean_norms, max_norm_head.
    """
    from irtk.hook_points import HookState

    cache = {}
    hs = HookState(hook_fns={}, cache=cache)
    model(tokens, hook_state=hs)

    z_key = f"blocks.{layer}.attn.hook_z"
    if z_key not in cache:
        return {"norm_matrix": jnp.array([[]]), "head_mean_norms": jnp.array([])}

    z = np.array(cache[z_key])  # [seq, n_heads, d_head]
    W_O = np.array(model.blocks[layer].attn.W_O)
    seq_len, n_heads = z.shape[0], z.shape[1]

    norms = np.zeros((n_heads, seq_len))
    for h in range(n_heads):
        for s in range(seq_len):
            out = z[s, h] @ W_O[h]
            norms[h, s] = float(np.linalg.norm(out))

    head_means = np.mean(norms, axis=1)
    pos_means = np.mean(norms, axis=0)

    return {
        "norm_matrix": jnp.array(norms),
        "head_mean_norms": jnp.array(head_means),
        "position_mean_norms": jnp.array(pos_means),
        "max_norm_head": int(np.argmax(head_means)),
        "max_norm_position": int(np.argmax(pos_means)),
    }
