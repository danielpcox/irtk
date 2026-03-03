"""Position encoding analysis.

Tools for understanding how positional information is encoded and used:
- Position embedding structure
- Positional vs content separation
- Position information persistence
- Position-dependent attention patterns
- Position encoding capacity
"""

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from irtk.hooked_transformer import HookedTransformer


def position_embedding_structure(
    model: HookedTransformer,
) -> dict:
    """Analyze the structure of position embeddings.

    Examines distances, periodicity, and dimensionality of W_pos.

    Args:
        model: HookedTransformer.

    Returns:
        Dict with position embedding analysis.
    """
    if model.pos_embed is None:
        return {'has_position_embeddings': False}

    W_pos = np.array(model.pos_embed.W_pos)  # [n_ctx, d_model]
    n_ctx, d_model = W_pos.shape

    # Norms
    norms = np.linalg.norm(W_pos, axis=1)

    # Distance matrix (cosine similarity between positions)
    normed = W_pos / (norms[:, None] + 1e-10)
    cos_matrix = normed @ normed.T  # [n_ctx, n_ctx]

    # Nearest neighbor distance
    np.fill_diagonal(cos_matrix, -2)
    nearest = np.argmax(cos_matrix, axis=1)

    # Effective rank
    U, s, Vt = np.linalg.svd(W_pos, full_matrices=False)
    s_norm = s / np.sum(s)
    entropy = -float(np.sum(s_norm * np.log(s_norm + 1e-10)))
    eff_rank = float(np.exp(entropy))

    return {
        'has_position_embeddings': True,
        'n_positions': n_ctx,
        'd_model': d_model,
        'mean_norm': round(float(np.mean(norms)), 4),
        'norm_std': round(float(np.std(norms)), 4),
        'effective_rank': round(eff_rank, 2),
        'mean_adjacent_similarity': round(float(np.mean([cos_matrix[i, i+1] for i in range(n_ctx - 1)])), 4),
    }


def position_content_separation(
    model: HookedTransformer,
    tokens: jnp.ndarray,
) -> dict:
    """Measure how well position and content are separated in the residual stream.

    Compares variance explained by position vs content in the residual.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] token IDs.

    Returns:
        Dict with position/content separation metrics.
    """
    _, cache = model.run_with_cache(tokens)
    seq_len = len(tokens)

    per_layer = []
    for l in range(model.cfg.n_layers):
        resid = np.array(cache[f'blocks.{l}.hook_resid_post'])  # [seq, d_model]

        # Position-dependent component: project residuals onto position directions
        if model.pos_embed is not None:
            W_pos = np.array(model.pos_embed.W_pos[:seq_len])
            # Component along position direction
            pos_norms = np.linalg.norm(W_pos, axis=1, keepdims=True)
            pos_dirs = W_pos / (pos_norms + 1e-10)
            pos_component = np.sum(resid * pos_dirs, axis=1)  # [seq]
            pos_variance = float(np.var(pos_component))
        else:
            pos_variance = 0.0

        # Total variance
        total_variance = float(np.mean(np.var(resid, axis=0)))

        per_layer.append({
            'layer': l,
            'total_variance': round(total_variance, 6),
            'position_variance': round(pos_variance, 6),
            'position_fraction': round(pos_variance / total_variance, 4) if total_variance > 0 else 0.0,
        })

    return {
        'per_layer': per_layer,
        'has_position_embeddings': model.pos_embed is not None,
    }


def position_info_persistence(
    model: HookedTransformer,
    tokens: jnp.ndarray,
) -> dict:
    """Track how well position can be decoded from the residual at each layer.

    Uses a simple linear probe approach: project residuals onto position
    embedding directions and measure separability.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] token IDs.

    Returns:
        Dict with per-layer position decodability.
    """
    _, cache = model.run_with_cache(tokens)
    seq_len = len(tokens)

    if model.pos_embed is None:
        return {'has_position_embeddings': False, 'per_layer': []}

    W_pos = np.array(model.pos_embed.W_pos[:seq_len])

    per_layer = []

    # At each layer, measure cosine similarity between resid and position embeddings
    for l in range(model.cfg.n_layers):
        resid = np.array(cache[f'blocks.{l}.hook_resid_post'])  # [seq, d_model]

        # For each position, compute cosine with its position embedding
        correct_cosines = []
        for p in range(seq_len):
            rn = float(np.linalg.norm(resid[p]))
            pn = float(np.linalg.norm(W_pos[p]))
            if rn > 1e-10 and pn > 1e-10:
                cos = float(np.dot(resid[p], W_pos[p]) / (rn * pn))
                correct_cosines.append(cos)

        mean_correct_cos = float(np.mean(correct_cosines)) if correct_cosines else 0.0

        per_layer.append({
            'layer': l,
            'mean_position_cosine': round(mean_correct_cos, 4),
        })

    return {
        'has_position_embeddings': True,
        'per_layer': per_layer,
    }


def position_attention_pattern(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    layers: Optional[list[int]] = None,
) -> dict:
    """Analyze position-dependent attention biases.

    Checks if attention patterns are determined more by relative or
    absolute position.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] token IDs.
        layers: Layers to analyze (default: all).

    Returns:
        Dict with position-dependent attention metrics.
    """
    _, cache = model.run_with_cache(tokens)
    seq_len = len(tokens)
    if layers is None:
        layers = list(range(model.cfg.n_layers))

    per_head = []
    for l in layers:
        pattern = np.array(cache[f'blocks.{l}.attn.hook_pattern'])  # [n_heads, seq, seq]

        for h in range(model.cfg.n_heads):
            p = pattern[h]

            # Relative position bias: average attention by relative distance
            rel_attn = {}
            for q in range(seq_len):
                for k in range(q + 1):
                    dist = q - k
                    if dist not in rel_attn:
                        rel_attn[dist] = []
                    rel_attn[dist].append(float(p[q, k]))

            rel_profile = {}
            for dist, vals in sorted(rel_attn.items()):
                rel_profile[dist] = round(float(np.mean(vals)), 4)

            # Positional bias score: variance explained by relative position
            all_vals = [v for vals in rel_attn.values() for v in vals]
            total_var = float(np.var(all_vals)) if len(all_vals) > 1 else 0.0

            group_means = [np.mean(vals) for vals in rel_attn.values()]
            between_var = float(np.var(group_means) * len(group_means) / max(len(all_vals), 1))

            pos_bias = between_var / total_var if total_var > 0 else 0.0

            per_head.append({
                'layer': l,
                'head': h,
                'positional_bias': round(pos_bias, 4),
                'relative_profile': rel_profile,
            })

    return {'per_head': per_head}


def position_encoding_capacity(
    model: HookedTransformer,
    n_positions: Optional[int] = None,
) -> dict:
    """Measure how much capacity position embeddings use.

    Computes the fraction of d_model capacity used by position info.

    Args:
        model: HookedTransformer.
        n_positions: Number of positions to analyze (default: min(n_ctx, 64)).

    Returns:
        Dict with capacity metrics.
    """
    if model.pos_embed is None:
        return {'has_position_embeddings': False}

    W_pos = np.array(model.pos_embed.W_pos)
    n_ctx, d_model = W_pos.shape

    if n_positions is None:
        n_positions = min(n_ctx, 64)
    W_pos = W_pos[:n_positions]

    # SVD of position embeddings
    U, s, Vt = np.linalg.svd(W_pos, full_matrices=False)

    # How many dimensions needed for 90% of variance
    total_var = float(np.sum(s ** 2))
    cumulative = np.cumsum(s ** 2) / total_var

    dims_90 = int(np.searchsorted(cumulative, 0.9)) + 1
    dims_95 = int(np.searchsorted(cumulative, 0.95)) + 1
    dims_99 = int(np.searchsorted(cumulative, 0.99)) + 1

    # Position norm relative to token embedding norm
    pos_norms = np.linalg.norm(W_pos, axis=1)

    W_E = np.array(model.embed.W_E)
    token_norms = np.linalg.norm(W_E, axis=1)

    return {
        'has_position_embeddings': True,
        'n_positions': n_positions,
        'd_model': d_model,
        'dims_for_90pct': dims_90,
        'dims_for_95pct': dims_95,
        'dims_for_99pct': dims_99,
        'capacity_fraction_90': round(dims_90 / d_model, 4),
        'mean_position_norm': round(float(np.mean(pos_norms)), 4),
        'mean_token_norm': round(float(np.mean(token_norms)), 4),
        'pos_to_token_ratio': round(float(np.mean(pos_norms) / np.mean(token_norms)), 4) if float(np.mean(token_norms)) > 0 else 0.0,
    }
