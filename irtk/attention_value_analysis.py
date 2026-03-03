"""Attention value analysis: analyze the value vectors and their role in computation."""

import jax
import jax.numpy as jnp
from irtk import HookedTransformer


def value_vector_profile(model: HookedTransformer, tokens: jnp.ndarray, layer: int) -> dict:
    """Profile the value vectors at a given layer.

    Norm, direction, and similarity of V vectors.
    """
    _, cache = model.run_with_cache(tokens)
    n_heads = model.cfg.n_heads

    v = cache[f'blocks.{layer}.attn.hook_v']  # [seq, n_heads, d_head]
    seq_len = tokens.shape[0]

    per_head = []
    for h in range(n_heads):
        v_h = v[:, h, :]  # [seq, d_head]
        norms = jnp.linalg.norm(v_h, axis=-1)
        mean_norm = float(jnp.mean(norms))
        std_norm = float(jnp.std(norms))

        # Direction consistency
        normed = v_h / (norms[:, None] + 1e-10)
        mean_dir = jnp.mean(normed, axis=0)
        mean_dir = mean_dir / (jnp.linalg.norm(mean_dir) + 1e-10)
        consistency = float(jnp.mean(normed @ mean_dir))

        per_head.append({
            'head': h,
            'mean_norm': mean_norm,
            'std_norm': std_norm,
            'direction_consistency': consistency,
        })

    return {
        'layer': layer,
        'per_head': per_head,
    }


def value_weighted_output(model: HookedTransformer, tokens: jnp.ndarray, layer: int, head: int, position: int = -1) -> dict:
    """What does the attention-weighted value produce at a position?

    Decomposes the output by source contribution.
    """
    _, cache = model.run_with_cache(tokens)
    seq_len = tokens.shape[0]
    pos = position if position >= 0 else seq_len + position

    v = cache[f'blocks.{layer}.attn.hook_v']  # [seq, n_heads, d_head]
    patterns = cache[f'blocks.{layer}.attn.hook_pattern']  # [n_heads, seq, seq]

    attn_weights = patterns[head, pos, :pos + 1]  # [pos+1]
    values = v[:pos + 1, head, :]  # [pos+1, d_head]

    weighted_sum = jnp.sum(attn_weights[:, None] * values, axis=0)  # [d_head]
    weighted_norm = float(jnp.linalg.norm(weighted_sum))

    per_source = []
    for s in range(pos + 1):
        contrib = attn_weights[s] * values[s]
        contrib_norm = float(jnp.linalg.norm(contrib))
        per_source.append({
            'source': s,
            'attention_weight': float(attn_weights[s]),
            'value_norm': float(jnp.linalg.norm(values[s])),
            'contribution_norm': contrib_norm,
            'fraction': contrib_norm / (weighted_norm + 1e-10),
        })

    per_source.sort(key=lambda x: x['contribution_norm'], reverse=True)

    return {
        'layer': layer,
        'head': head,
        'position': pos,
        'weighted_output_norm': weighted_norm,
        'per_source': per_source,
    }


def value_rank_analysis(model: HookedTransformer, tokens: jnp.ndarray, layer: int) -> dict:
    """Effective rank of the value vectors.

    Low rank = values are constrained to a low-dimensional subspace.
    """
    _, cache = model.run_with_cache(tokens)
    n_heads = model.cfg.n_heads

    v = cache[f'blocks.{layer}.attn.hook_v']  # [seq, n_heads, d_head]

    per_head = []
    for h in range(n_heads):
        v_h = v[:, h, :]  # [seq, d_head]
        U, S, Vt = jnp.linalg.svd(v_h, full_matrices=False)
        S_norm = S / (jnp.sum(S) + 1e-10)
        entropy = -float(jnp.sum(S_norm * jnp.log(S_norm + 1e-10)))
        eff_rank = float(jnp.exp(entropy))

        per_head.append({
            'head': h,
            'effective_rank': eff_rank,
            'top_sv_fraction': float(S[0] / (jnp.sum(S) + 1e-10)),
            'is_low_rank': eff_rank < 2.0,
        })

    return {
        'layer': layer,
        'per_head': per_head,
        'mean_rank': sum(h['effective_rank'] for h in per_head) / len(per_head),
    }


def value_position_variation(model: HookedTransformer, tokens: jnp.ndarray, layer: int) -> dict:
    """How much do value vectors vary across positions?

    High variation = position-dependent value computation.
    """
    _, cache = model.run_with_cache(tokens)
    n_heads = model.cfg.n_heads
    seq_len = tokens.shape[0]

    v = cache[f'blocks.{layer}.attn.hook_v']  # [seq, n_heads, d_head]

    per_head = []
    for h in range(n_heads):
        v_h = v[:, h, :]  # [seq, d_head]
        norms = jnp.linalg.norm(v_h, axis=-1)
        normed = v_h / (norms[:, None] + 1e-10)

        # Pairwise cosine similarity
        sim_matrix = normed @ normed.T
        mean_sim = float(jnp.mean(sim_matrix))
        variation = 1.0 - mean_sim

        # Coefficient of variation of norms
        cv = float(jnp.std(norms) / (jnp.mean(norms) + 1e-10))

        per_head.append({
            'head': h,
            'direction_variation': variation,
            'norm_cv': cv,
            'is_position_dependent': variation > 0.3,
        })

    return {
        'layer': layer,
        'per_head': per_head,
        'n_position_dependent': sum(1 for h in per_head if h['is_position_dependent']),
    }


def value_unembed_projection(model: HookedTransformer, tokens: jnp.ndarray, layer: int, head: int, position: int = -1) -> dict:
    """Project value vectors through OV and unembedding.

    Shows what tokens the value vectors would promote.
    """
    _, cache = model.run_with_cache(tokens)
    seq_len = tokens.shape[0]
    pos = position if position >= 0 else seq_len + position

    v = cache[f'blocks.{layer}.attn.hook_v']  # [seq, n_heads, d_head]
    W_O = model.blocks[layer].attn.W_O[head]  # [d_head, d_model]
    W_U = model.unembed.W_U  # [d_model, d_vocab]

    per_source = []
    for s in range(pos + 1):
        v_s = v[s, head, :]  # [d_head]
        output = v_s @ W_O  # [d_model]
        logits = output @ W_U  # [d_vocab]

        top_3 = jnp.argsort(logits)[-3:][::-1]
        bottom_3 = jnp.argsort(logits)[:3]

        per_source.append({
            'source': s,
            'source_token': int(tokens[s]),
            'output_norm': float(jnp.linalg.norm(output)),
            'top_tokens': [{'token': int(t), 'logit': float(logits[t])} for t in top_3],
            'bottom_tokens': [{'token': int(t), 'logit': float(logits[t])} for t in bottom_3],
        })

    return {
        'layer': layer,
        'head': head,
        'position': pos,
        'per_source': per_source,
    }
