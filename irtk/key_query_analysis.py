"""Key-query analysis: how keys and queries interact to form attention patterns."""

import jax
import jax.numpy as jnp
from irtk import HookedTransformer


def key_query_alignment(model: HookedTransformer, tokens: jnp.ndarray, layer: int) -> dict:
    """How well do keys and queries align at each position?

    High alignment = strong self-attention or predictable patterns.
    """
    _, cache = model.run_with_cache(tokens)
    n_heads = model.cfg.n_heads
    seq_len = tokens.shape[0]

    q = cache[f'blocks.{layer}.attn.hook_q']  # [seq, n_heads, d_head]
    k = cache[f'blocks.{layer}.attn.hook_k']  # [seq, n_heads, d_head]

    per_head = []
    for h in range(n_heads):
        q_h = q[:, h, :]  # [seq, d_head]
        k_h = k[:, h, :]  # [seq, d_head]

        q_normed = q_h / (jnp.linalg.norm(q_h, axis=-1, keepdims=True) + 1e-10)
        k_normed = k_h / (jnp.linalg.norm(k_h, axis=-1, keepdims=True) + 1e-10)

        # Self-alignment: cos(q_i, k_i)
        self_align = jnp.sum(q_normed * k_normed, axis=-1)  # [seq]
        mean_self = float(jnp.mean(self_align))

        # Cross-alignment: mean cos(q_i, k_j) for j <= i
        scores = q_normed @ k_normed.T  # [seq, seq]
        mask = jnp.tril(jnp.ones((seq_len, seq_len)))
        cross_align = float(jnp.sum(scores * mask) / jnp.sum(mask))

        per_head.append({
            'head': h,
            'mean_self_alignment': mean_self,
            'mean_cross_alignment': cross_align,
            'q_mean_norm': float(jnp.mean(jnp.linalg.norm(q_h, axis=-1))),
            'k_mean_norm': float(jnp.mean(jnp.linalg.norm(k_h, axis=-1))),
        })

    return {
        'layer': layer,
        'per_head': per_head,
    }


def key_query_subspace(model: HookedTransformer, tokens: jnp.ndarray, layer: int) -> dict:
    """Effective dimensionality of key and query subspaces.

    Low rank = keys/queries live in a constrained subspace.
    """
    _, cache = model.run_with_cache(tokens)
    n_heads = model.cfg.n_heads

    q = cache[f'blocks.{layer}.attn.hook_q']  # [seq, n_heads, d_head]
    k = cache[f'blocks.{layer}.attn.hook_k']  # [seq, n_heads, d_head]

    per_head = []
    for h in range(n_heads):
        q_h = q[:, h, :]
        k_h = k[:, h, :]

        q_sv = jnp.linalg.svd(q_h, compute_uv=False)
        k_sv = jnp.linalg.svd(k_h, compute_uv=False)

        def eff_rank(sv):
            sv_norm = sv / (jnp.sum(sv) + 1e-10)
            entropy = -jnp.sum(sv_norm * jnp.log(sv_norm + 1e-10))
            return float(jnp.exp(entropy))

        per_head.append({
            'head': h,
            'q_effective_rank': eff_rank(q_sv),
            'k_effective_rank': eff_rank(k_sv),
            'q_top_sv_fraction': float(q_sv[0] / (jnp.sum(q_sv) + 1e-10)),
            'k_top_sv_fraction': float(k_sv[0] / (jnp.sum(k_sv) + 1e-10)),
        })

    return {
        'layer': layer,
        'per_head': per_head,
    }


def key_query_position_dependence(model: HookedTransformer, tokens: jnp.ndarray, layer: int) -> dict:
    """How much do keys and queries depend on position vs. content?

    High variation across positions = position-dependent computation.
    """
    _, cache = model.run_with_cache(tokens)
    n_heads = model.cfg.n_heads

    q = cache[f'blocks.{layer}.attn.hook_q']  # [seq, n_heads, d_head]
    k = cache[f'blocks.{layer}.attn.hook_k']  # [seq, n_heads, d_head]

    per_head = []
    for h in range(n_heads):
        q_h = q[:, h, :]
        k_h = k[:, h, :]

        # Direction variation
        q_norms = jnp.linalg.norm(q_h, axis=-1)
        k_norms = jnp.linalg.norm(k_h, axis=-1)
        q_normed = q_h / (q_norms[:, None] + 1e-10)
        k_normed = k_h / (k_norms[:, None] + 1e-10)

        q_sim = q_normed @ q_normed.T
        k_sim = k_normed @ k_normed.T
        q_variation = 1.0 - float(jnp.mean(q_sim))
        k_variation = 1.0 - float(jnp.mean(k_sim))

        # Norm variation
        q_cv = float(jnp.std(q_norms) / (jnp.mean(q_norms) + 1e-10))
        k_cv = float(jnp.std(k_norms) / (jnp.mean(k_norms) + 1e-10))

        per_head.append({
            'head': h,
            'q_direction_variation': q_variation,
            'k_direction_variation': k_variation,
            'q_norm_cv': q_cv,
            'k_norm_cv': k_cv,
            'is_position_dependent': bool(q_variation > 0.3 or k_variation > 0.3),
        })

    return {
        'layer': layer,
        'per_head': per_head,
        'n_position_dependent': sum(1 for h in per_head if h['is_position_dependent']),
    }


def key_query_match_profile(model: HookedTransformer, tokens: jnp.ndarray, layer: int, head: int, position: int = -1) -> dict:
    """Detailed profile of which keys a specific query matches.

    Shows the pre-softmax scores and what drives the attention pattern.
    """
    _, cache = model.run_with_cache(tokens)
    seq_len = tokens.shape[0]
    pos = position if position >= 0 else seq_len + position

    q = cache[f'blocks.{layer}.attn.hook_q']  # [seq, n_heads, d_head]
    k = cache[f'blocks.{layer}.attn.hook_k']  # [seq, n_heads, d_head]

    q_vec = q[pos, head, :]  # [d_head]
    d_head = model.cfg.d_head

    per_key = []
    for s in range(pos + 1):
        k_vec = k[s, head, :]  # [d_head]
        score = float(jnp.dot(q_vec, k_vec) / jnp.sqrt(d_head))
        q_norm = float(jnp.linalg.norm(q_vec))
        k_norm = float(jnp.linalg.norm(k_vec))
        cos = float(jnp.dot(q_vec, k_vec) / (q_norm * k_norm + 1e-10))

        per_key.append({
            'key_position': s,
            'key_token': int(tokens[s]),
            'score': score,
            'cosine': cos,
            'key_norm': k_norm,
        })

    per_key.sort(key=lambda x: x['score'], reverse=True)

    return {
        'layer': layer,
        'head': head,
        'query_position': pos,
        'query_token': int(tokens[pos]),
        'query_norm': float(jnp.linalg.norm(q_vec)),
        'per_key': per_key,
    }


def key_query_weight_decomposition(model: HookedTransformer, layer: int, head: int) -> dict:
    """Decompose the QK weight circuit.

    Analyzes W_Q^T W_K to understand what the head looks for.
    """
    W_Q = model.blocks[layer].attn.W_Q[head]  # [d_model, d_head]
    W_K = model.blocks[layer].attn.W_K[head]  # [d_model, d_head]

    QK = W_Q @ W_K.T  # [d_model, d_model]

    sv = jnp.linalg.svd(QK, compute_uv=False)
    total = jnp.sum(sv)
    sv_norm = sv / (total + 1e-10)
    entropy = -float(jnp.sum(sv_norm * jnp.log(sv_norm + 1e-10)))
    eff_rank = float(jnp.exp(entropy))

    # Symmetry: how symmetric is QK?
    sym = (QK + QK.T) / 2
    asym = (QK - QK.T) / 2
    sym_frac = float(jnp.linalg.norm(sym) / (jnp.linalg.norm(QK) + 1e-10))

    return {
        'layer': layer,
        'head': head,
        'spectral_norm': float(sv[0]),
        'effective_rank': eff_rank,
        'top_sv_fraction': float(sv[0] / (total + 1e-10)),
        'symmetry_fraction': sym_frac,
        'frobenius_norm': float(jnp.linalg.norm(QK)),
    }
