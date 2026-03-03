"""Attention QKV decomposition.

Decompose attention computations through Q, K, V matrices: eigenspectrum
analysis, positional vs content contributions, value composition, and
cross-head QKV alignment.
"""

import jax
import jax.numpy as jnp


def qk_eigenspectrum(model, layer, head):
    """Compute eigenvalues of W_Q^T @ W_K for a head.

    Args:
        model: HookedTransformer
        layer: layer index
        head: head index

    Returns:
        dict with eigenspectrum analysis.
    """
    W_Q = model.blocks[layer].attn.W_Q[head]  # [d_model, d_head]
    W_K = model.blocks[layer].attn.W_K[head]  # [d_model, d_head]

    # QK circuit: W_Q^T @ W_K -> [d_head, d_model] @ [d_model, d_head] is wrong
    # Actually: d_model x d_model via Q^T K
    qk = W_Q @ W_K.T  # [d_model, d_model]

    eigenvalues = jnp.linalg.eigvalsh(qk)
    eigenvalues = jnp.sort(jnp.abs(eigenvalues))[::-1]

    total = float(jnp.sum(eigenvalues))
    cumulative = jnp.cumsum(eigenvalues) / (total + 1e-10)

    # Effective rank
    normed = eigenvalues / (total + 1e-10)
    entropy = -float(jnp.sum(normed * jnp.log(normed + 1e-10)))
    eff_rank = float(jnp.exp(entropy))

    return {
        'layer': layer,
        'head': head,
        'eigenvalues': eigenvalues,
        'effective_rank': eff_rank,
        'top_eigenvalue': float(eigenvalues[0]),
        'spectral_norm': float(eigenvalues[0]),
        'rank_90': int(jnp.searchsorted(cumulative, 0.9)) + 1,
    }


def ov_eigenspectrum(model, layer, head):
    """Compute singular values of W_V @ W_O (the OV circuit).

    Args:
        model: HookedTransformer
        layer: layer index
        head: head index

    Returns:
        dict with OV eigenspectrum.
    """
    W_V = model.blocks[layer].attn.W_V[head]  # [d_model, d_head]
    W_O = model.blocks[layer].attn.W_O[head]  # [d_head, d_model]

    ov = W_V @ W_O  # [d_model, d_model]
    _, S, _ = jnp.linalg.svd(ov)

    total = float(jnp.sum(S))
    normed = S / (total + 1e-10)
    entropy = -float(jnp.sum(normed * jnp.log(normed + 1e-10)))
    eff_rank = float(jnp.exp(entropy))

    cumulative = jnp.cumsum(S) / (total + 1e-10)

    return {
        'layer': layer,
        'head': head,
        'singular_values': S,
        'effective_rank': eff_rank,
        'top_singular_value': float(S[0]),
        'rank_90': int(jnp.searchsorted(cumulative, 0.9)) + 1,
    }


def positional_vs_content_qk(model, tokens, layer, head):
    """Decompose QK scores into positional and content contributions.

    Args:
        model: HookedTransformer
        tokens: input token IDs
        layer: layer index
        head: head index

    Returns:
        dict with positional/content decomposition.
    """
    _, cache = model.run_with_cache(tokens)
    pattern = cache[f'blocks.{layer}.attn.hook_pattern']  # [n_heads, seq, seq]
    q = cache[f'blocks.{layer}.attn.hook_q']  # [seq, n_heads, d_head]
    k = cache[f'blocks.{layer}.attn.hook_k']  # [seq, n_heads, d_head]

    q_h = q[:, head, :]  # [seq, d_head]
    k_h = k[:, head, :]  # [seq, d_head]

    # Compute raw QK scores
    scores = q_h @ k_h.T  # [seq, seq]

    # Positional contribution: how much does position alone determine the pattern?
    # Use positional encoding difference
    seq_len = len(tokens)
    pos_indices = jnp.arange(seq_len)
    pos_diff = pos_indices[:, None] - pos_indices[None, :]  # [seq, seq]

    # Correlation between scores and position difference
    scores_flat = scores.reshape(-1)
    pos_flat = pos_diff.reshape(-1).astype(jnp.float32)
    corr_num = jnp.sum((scores_flat - jnp.mean(scores_flat)) * (pos_flat - jnp.mean(pos_flat)))
    corr_denom = jnp.sqrt(jnp.sum((scores_flat - jnp.mean(scores_flat))**2) *
                          jnp.sum((pos_flat - jnp.mean(pos_flat))**2) + 1e-10)
    pos_correlation = float(corr_num / corr_denom)

    return {
        'layer': layer,
        'head': head,
        'positional_correlation': pos_correlation,
        'is_positional': abs(pos_correlation) > 0.5,
        'mean_score': float(jnp.mean(scores)),
        'score_std': float(jnp.std(scores)),
    }


def value_composition_profile(model, tokens, layer, head):
    """Analyze what information the value vectors contribute.

    Args:
        model: HookedTransformer
        tokens: input token IDs
        layer: layer index
        head: head index

    Returns:
        dict with value vector analysis.
    """
    _, cache = model.run_with_cache(tokens)
    v = cache[f'blocks.{layer}.attn.hook_v']  # [seq, n_heads, d_head]
    z = cache[f'blocks.{layer}.attn.hook_z']  # [seq, n_heads, d_head]
    pattern = cache[f'blocks.{layer}.attn.hook_pattern']  # [n_heads, seq, seq]

    v_h = v[:, head, :]  # [seq, d_head]
    z_h = z[:, head, :]  # [seq, d_head]
    W_O = model.blocks[layer].attn.W_O[head]  # [d_head, d_model]

    # Output per position
    output = z_h @ W_O  # [seq, d_model]
    output_norms = jnp.linalg.norm(output, axis=-1)

    # Value diversity: how different are value vectors across positions?
    v_norms = jnp.linalg.norm(v_h, axis=-1)
    v_normalized = v_h / (v_norms[:, None] + 1e-10)
    v_similarity = v_normalized @ v_normalized.T
    mean_v_similarity = float(jnp.mean(v_similarity) - 1.0 / len(tokens))

    return {
        'layer': layer,
        'head': head,
        'mean_output_norm': float(jnp.mean(output_norms)),
        'output_norm_std': float(jnp.std(output_norms)),
        'mean_value_norm': float(jnp.mean(v_norms)),
        'value_diversity': 1.0 - mean_v_similarity,
    }


def cross_head_qkv_alignment(model, layer):
    """Measure how aligned QKV matrices are across heads in a layer.

    Args:
        model: HookedTransformer
        layer: layer index

    Returns:
        dict with cross-head alignment analysis.
    """
    n_heads = model.cfg.n_heads

    q_alignments = []
    k_alignments = []
    v_alignments = []
    ov_alignments = []

    for h1 in range(n_heads):
        for h2 in range(h1 + 1, n_heads):
            # Q alignment
            W_Q1 = model.blocks[layer].attn.W_Q[h1].reshape(-1)
            W_Q2 = model.blocks[layer].attn.W_Q[h2].reshape(-1)
            q_cos = float(jnp.sum(W_Q1 * W_Q2) / (jnp.linalg.norm(W_Q1) * jnp.linalg.norm(W_Q2) + 1e-10))
            q_alignments.append({'h1': h1, 'h2': h2, 'cosine': q_cos})

            # K alignment
            W_K1 = model.blocks[layer].attn.W_K[h1].reshape(-1)
            W_K2 = model.blocks[layer].attn.W_K[h2].reshape(-1)
            k_cos = float(jnp.sum(W_K1 * W_K2) / (jnp.linalg.norm(W_K1) * jnp.linalg.norm(W_K2) + 1e-10))
            k_alignments.append({'h1': h1, 'h2': h2, 'cosine': k_cos})

            # V alignment
            W_V1 = model.blocks[layer].attn.W_V[h1].reshape(-1)
            W_V2 = model.blocks[layer].attn.W_V[h2].reshape(-1)
            v_cos = float(jnp.sum(W_V1 * W_V2) / (jnp.linalg.norm(W_V1) * jnp.linalg.norm(W_V2) + 1e-10))
            v_alignments.append({'h1': h1, 'h2': h2, 'cosine': v_cos})

            # OV circuit alignment
            ov1 = (model.blocks[layer].attn.W_V[h1] @ model.blocks[layer].attn.W_O[h1]).reshape(-1)
            ov2 = (model.blocks[layer].attn.W_V[h2] @ model.blocks[layer].attn.W_O[h2]).reshape(-1)
            ov_cos = float(jnp.sum(ov1 * ov2) / (jnp.linalg.norm(ov1) * jnp.linalg.norm(ov2) + 1e-10))
            ov_alignments.append({'h1': h1, 'h2': h2, 'cosine': ov_cos})

    return {
        'layer': layer,
        'q_alignments': q_alignments,
        'k_alignments': k_alignments,
        'v_alignments': v_alignments,
        'ov_alignments': ov_alignments,
        'mean_q_alignment': float(jnp.mean(jnp.array([a['cosine'] for a in q_alignments]))) if q_alignments else 0.0,
        'mean_ov_alignment': float(jnp.mean(jnp.array([a['cosine'] for a in ov_alignments]))) if ov_alignments else 0.0,
    }
