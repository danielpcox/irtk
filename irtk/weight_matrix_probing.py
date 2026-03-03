"""Weight matrix probing: finding interpretable structure in weights.

Analyze weight matrices for interpretable patterns:
- Singular value spectrum
- Effective rank per weight matrix
- Weight direction alignment
- Weight matrix symmetry
- Low-rank structure detection
"""

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from irtk.hooked_transformer import HookedTransformer


def weight_spectrum(
    model: HookedTransformer,
    layer: int = 0,
    component: str = 'attn',
) -> dict:
    """Analyze the singular value spectrum of weight matrices.

    Args:
        model: HookedTransformer.
        layer: Layer to analyze.
        component: 'attn' or 'mlp'.

    Returns:
        Dict with SVD spectrum analysis.
    """
    results = {}

    if component == 'attn':
        matrices = {
            'W_Q': np.array(model.blocks[layer].attn.W_Q).reshape(-1, model.cfg.d_model).T,
            'W_K': np.array(model.blocks[layer].attn.W_K).reshape(-1, model.cfg.d_model).T,
            'W_V': np.array(model.blocks[layer].attn.W_V).reshape(-1, model.cfg.d_model).T,
            'W_O': np.array(model.blocks[layer].attn.W_O).reshape(-1, model.cfg.d_model),
        }
    else:
        matrices = {
            'W_in': np.array(model.blocks[layer].mlp.W_in),
            'W_out': np.array(model.blocks[layer].mlp.W_out),
        }

    for name, W in matrices.items():
        U, s, Vt = np.linalg.svd(W, full_matrices=False)

        # Effective rank
        s_norm = s / np.sum(s)
        entropy = -float(np.sum(s_norm * np.log(s_norm + 1e-10)))
        eff_rank = float(np.exp(entropy))

        # Concentration: fraction in top-k singular values
        total = float(np.sum(s ** 2))
        top_1 = float(s[0] ** 2) / total if total > 0 else 0.0
        top_5 = float(np.sum(s[:5] ** 2)) / total if total > 0 else 0.0

        results[name] = {
            'shape': list(W.shape),
            'effective_rank': round(eff_rank, 2),
            'max_singular_value': round(float(s[0]), 4),
            'min_singular_value': round(float(s[-1]), 6),
            'condition_number': round(float(s[0] / (s[-1] + 1e-10)), 2),
            'top1_variance_fraction': round(top_1, 4),
            'top5_variance_fraction': round(top_5, 4),
        }

    return {
        'layer': layer,
        'component': component,
        'matrices': results,
    }


def weight_alignment(
    model: HookedTransformer,
    layer: int = 0,
) -> dict:
    """Measure alignment between attention weight matrices.

    Checks QK and OV alignment to understand circuit structure.

    Args:
        model: HookedTransformer.
        layer: Layer to analyze.

    Returns:
        Dict with weight alignment metrics.
    """
    W_Q = np.array(model.blocks[layer].attn.W_Q)  # [n_heads, d_model, d_head]
    W_K = np.array(model.blocks[layer].attn.W_K)
    W_V = np.array(model.blocks[layer].attn.W_V)
    W_O = np.array(model.blocks[layer].attn.W_O)  # [n_heads, d_head, d_model]

    per_head = []
    for h in range(model.cfg.n_heads):
        # QK alignment: do Q and K project into similar subspaces?
        qk_prod = W_Q[h].T @ W_K[h]  # [d_head, d_head]
        qk_norm = float(np.linalg.norm(qk_prod, ord='fro'))

        # OV alignment: does the OV circuit preserve information?
        ov_prod = W_V[h] @ W_O[h]  # [d_model, d_model]
        ov_norm = float(np.linalg.norm(ov_prod, ord='fro'))

        # SVD of OV to check effective rank
        _, s_ov, _ = np.linalg.svd(ov_prod, full_matrices=False)
        s_ov_norm = s_ov / (np.sum(s_ov) + 1e-10)
        ov_entropy = -float(np.sum(s_ov_norm * np.log(s_ov_norm + 1e-10)))
        ov_eff_rank = float(np.exp(ov_entropy))

        per_head.append({
            'head': h,
            'qk_norm': round(qk_norm, 4),
            'ov_norm': round(ov_norm, 4),
            'ov_effective_rank': round(ov_eff_rank, 2),
        })

    return {
        'layer': layer,
        'per_head': per_head,
    }


def weight_norms_profile(
    model: HookedTransformer,
) -> dict:
    """Profile weight matrix norms across all layers.

    Args:
        model: HookedTransformer.

    Returns:
        Dict with per-layer weight norms.
    """
    per_layer = []
    for l in range(model.cfg.n_layers):
        W_Q = np.array(model.blocks[l].attn.W_Q)
        W_K = np.array(model.blocks[l].attn.W_K)
        W_V = np.array(model.blocks[l].attn.W_V)
        W_O = np.array(model.blocks[l].attn.W_O)
        W_in = np.array(model.blocks[l].mlp.W_in)
        W_out = np.array(model.blocks[l].mlp.W_out)

        per_layer.append({
            'layer': l,
            'W_Q_norm': round(float(np.linalg.norm(W_Q)), 4),
            'W_K_norm': round(float(np.linalg.norm(W_K)), 4),
            'W_V_norm': round(float(np.linalg.norm(W_V)), 4),
            'W_O_norm': round(float(np.linalg.norm(W_O)), 4),
            'W_in_norm': round(float(np.linalg.norm(W_in)), 4),
            'W_out_norm': round(float(np.linalg.norm(W_out)), 4),
        })

    return {'per_layer': per_layer}


def low_rank_structure(
    model: HookedTransformer,
    layer: int = 0,
    rank_threshold: float = 0.95,
) -> dict:
    """Detect low-rank structure in weight matrices.

    If a small number of singular values explain most of the variance,
    the weight matrix has low-rank structure.

    Args:
        model: HookedTransformer.
        layer: Layer to analyze.
        rank_threshold: Variance fraction to target.

    Returns:
        Dict with low-rank analysis per weight matrix.
    """
    matrices = {
        'W_Q': np.array(model.blocks[layer].attn.W_Q).reshape(-1, model.cfg.d_model).T,
        'W_K': np.array(model.blocks[layer].attn.W_K).reshape(-1, model.cfg.d_model).T,
        'W_V': np.array(model.blocks[layer].attn.W_V).reshape(-1, model.cfg.d_model).T,
        'W_O': np.array(model.blocks[layer].attn.W_O).reshape(-1, model.cfg.d_model),
        'W_in': np.array(model.blocks[layer].mlp.W_in),
        'W_out': np.array(model.blocks[layer].mlp.W_out),
    }

    results = {}
    for name, W in matrices.items():
        _, s, _ = np.linalg.svd(W, full_matrices=False)
        total = float(np.sum(s ** 2))
        cumulative = np.cumsum(s ** 2) / total

        rank_needed = int(np.searchsorted(cumulative, rank_threshold)) + 1
        full_rank = min(W.shape)

        results[name] = {
            'full_rank': full_rank,
            'rank_for_threshold': rank_needed,
            'compression_ratio': round(rank_needed / full_rank, 4),
            'is_low_rank': rank_needed < full_rank * 0.5,
        }

    return {
        'layer': layer,
        'threshold': rank_threshold,
        'matrices': results,
        'n_low_rank': sum(1 for v in results.values() if v['is_low_rank']),
    }


def embed_unembed_alignment(
    model: HookedTransformer,
    top_k: int = 10,
) -> dict:
    """Measure alignment between embedding and unembedding matrices.

    High alignment means the model uses similar representations for
    reading and writing tokens.

    Args:
        model: HookedTransformer.
        top_k: Number of most/least aligned tokens to return.

    Returns:
        Dict with embedding-unembedding alignment analysis.
    """
    W_E = np.array(model.embed.W_E)    # [d_vocab, d_model]
    W_U = np.array(model.unembed.W_U)  # [d_model, d_vocab]

    d_vocab = W_E.shape[0]

    # Per-token alignment: cosine between embedding and unembedding
    alignments = []
    for tok in range(d_vocab):
        e = W_E[tok]
        u = W_U[:, tok]
        en = float(np.linalg.norm(e))
        un = float(np.linalg.norm(u))
        if en > 1e-10 and un > 1e-10:
            cos = float(np.dot(e, u) / (en * un))
        else:
            cos = 0.0
        alignments.append(cos)

    alignments = np.array(alignments)
    sorted_idx = np.argsort(alignments)

    most_aligned = [{'token': int(sorted_idx[-i-1]),
                      'cosine': round(float(alignments[sorted_idx[-i-1]]), 4)}
                     for i in range(min(top_k, d_vocab))]

    least_aligned = [{'token': int(sorted_idx[i]),
                       'cosine': round(float(alignments[sorted_idx[i]]), 4)}
                      for i in range(min(top_k, d_vocab))]

    return {
        'mean_alignment': round(float(np.mean(alignments)), 4),
        'std_alignment': round(float(np.std(alignments)), 4),
        'most_aligned': most_aligned,
        'least_aligned': least_aligned,
    }
