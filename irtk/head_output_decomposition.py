"""Head output decomposition: decompose each head's contribution to the residual stream."""

import jax
import jax.numpy as jnp
from irtk import HookedTransformer


def head_residual_contribution(model: HookedTransformer, tokens: jnp.ndarray, layer: int) -> dict:
    """How much does each head contribute to the residual stream at this layer?

    Measures norm and direction of each head's output.
    """
    _, cache = model.run_with_cache(tokens)
    n_heads = model.cfg.n_heads

    z = cache[f'blocks.{layer}.attn.hook_z']  # [seq, n_heads, d_head]
    W_O = model.blocks[layer].attn.W_O  # [n_heads, d_head, d_model]

    resid_pre = cache[f'blocks.{layer}.hook_resid_pre']  # [seq, d_model]
    resid_norm = float(jnp.mean(jnp.linalg.norm(resid_pre, axis=-1)))

    per_head = []
    for h in range(n_heads):
        head_out = jnp.einsum('sh,hm->sm', z[:, h, :], W_O[h])  # [seq, d_model]
        norm = float(jnp.mean(jnp.linalg.norm(head_out, axis=-1)))
        fraction = norm / (resid_norm + 1e-10)

        # Alignment with residual stream
        cos = float(jnp.mean(
            jnp.sum(head_out * resid_pre, axis=-1) /
            (jnp.linalg.norm(head_out, axis=-1) * jnp.linalg.norm(resid_pre, axis=-1) + 1e-10)
        ))

        per_head.append({
            'head': h,
            'mean_norm': norm,
            'fraction_of_residual': fraction,
            'alignment_to_residual': cos,
            'is_constructive': cos > 0,
        })

    per_head.sort(key=lambda x: x['mean_norm'], reverse=True)

    return {
        'layer': layer,
        'per_head': per_head,
        'residual_norm': resid_norm,
    }


def head_logit_projection(model: HookedTransformer, tokens: jnp.ndarray, layer: int, position: int = -1) -> dict:
    """Project each head's output through the unembedding to get logit contributions.

    Shows which tokens each head promotes/suppresses.
    """
    _, cache = model.run_with_cache(tokens)
    seq_len = tokens.shape[0]
    pos = position if position >= 0 else seq_len + position
    n_heads = model.cfg.n_heads

    z = cache[f'blocks.{layer}.attn.hook_z']  # [seq, n_heads, d_head]
    W_O = model.blocks[layer].attn.W_O  # [n_heads, d_head, d_model]
    W_U = model.unembed.W_U  # [d_model, d_vocab]

    per_head = []
    for h in range(n_heads):
        head_out = jnp.einsum('h,hm->m', z[pos, h, :], W_O[h])  # [d_model]
        logit_contrib = head_out @ W_U  # [d_vocab]

        top_5 = jnp.argsort(logit_contrib)[-5:][::-1]
        bottom_5 = jnp.argsort(logit_contrib)[:5]

        per_head.append({
            'head': h,
            'logit_norm': float(jnp.linalg.norm(logit_contrib)),
            'mean_logit': float(jnp.mean(logit_contrib)),
            'top_promoted': [{'token': int(t), 'logit': float(logit_contrib[t])} for t in top_5],
            'top_suppressed': [{'token': int(t), 'logit': float(logit_contrib[t])} for t in bottom_5],
        })

    per_head.sort(key=lambda x: x['logit_norm'], reverse=True)

    return {
        'layer': layer,
        'position': pos,
        'per_head': per_head,
    }


def head_value_decomposition(model: HookedTransformer, tokens: jnp.ndarray, layer: int, head: int) -> dict:
    """Decompose a head's output by source position contribution.

    How much does each source position contribute to the head's output?
    """
    _, cache = model.run_with_cache(tokens)
    seq_len = tokens.shape[0]

    patterns = cache[f'blocks.{layer}.attn.hook_pattern']  # [n_heads, seq, seq]
    v = cache[f'blocks.{layer}.attn.hook_v']  # [seq, n_heads, d_head]
    W_O = model.blocks[layer].attn.W_O  # [n_heads, d_head, d_model]

    per_query = []
    for q in range(seq_len):
        per_source = []
        for s in range(q + 1):
            # Contribution: attn_weight * value * W_O
            attn_weight = float(patterns[head, q, s])
            value_out = v[s, head, :] @ W_O[head]  # [d_model]
            contrib = attn_weight * value_out
            contrib_norm = float(jnp.linalg.norm(contrib))

            per_source.append({
                'source': s,
                'attention_weight': attn_weight,
                'contribution_norm': contrib_norm,
            })

        per_source.sort(key=lambda x: x['contribution_norm'], reverse=True)
        per_query.append({
            'query_position': q,
            'per_source': per_source,
            'top_source': per_source[0]['source'] if per_source else 0,
        })

    return {
        'layer': layer,
        'head': head,
        'per_query': per_query,
    }


def head_output_interference(model: HookedTransformer, tokens: jnp.ndarray, layer: int, position: int = -1) -> dict:
    """Measure interference (constructive/destructive) between head outputs.

    Do heads cooperate or compete at this position?
    """
    _, cache = model.run_with_cache(tokens)
    seq_len = tokens.shape[0]
    pos = position if position >= 0 else seq_len + position
    n_heads = model.cfg.n_heads

    z = cache[f'blocks.{layer}.attn.hook_z']  # [seq, n_heads, d_head]
    W_O = model.blocks[layer].attn.W_O  # [n_heads, d_head, d_model]

    head_outputs = []
    for h in range(n_heads):
        out = jnp.einsum('h,hm->m', z[pos, h, :], W_O[h])
        head_outputs.append(out)

    head_outputs = jnp.stack(head_outputs)  # [n_heads, d_model]
    combined = jnp.sum(head_outputs, axis=0)
    combined_norm = float(jnp.linalg.norm(combined))
    sum_norms = float(jnp.sum(jnp.linalg.norm(head_outputs, axis=-1)))

    # Interference ratio: 1.0 = perfect constructive, <1 = some destructive
    interference_ratio = combined_norm / (sum_norms + 1e-10)

    # Pairwise cosine similarities
    normed = head_outputs / (jnp.linalg.norm(head_outputs, axis=-1, keepdims=True) + 1e-10)
    pairwise = normed @ normed.T

    pairs = []
    for i in range(n_heads):
        for j in range(i + 1, n_heads):
            pairs.append({
                'head_a': i,
                'head_b': j,
                'cosine_similarity': float(pairwise[i, j]),
                'is_constructive': float(pairwise[i, j]) > 0,
            })

    n_constructive = sum(1 for p in pairs if p['is_constructive'])

    return {
        'layer': layer,
        'position': pos,
        'combined_norm': combined_norm,
        'sum_individual_norms': sum_norms,
        'interference_ratio': interference_ratio,
        'is_mostly_constructive': interference_ratio > 0.7,
        'pairs': pairs,
        'n_constructive_pairs': n_constructive,
    }


def head_output_rank_analysis(model: HookedTransformer, tokens: jnp.ndarray, layer: int) -> dict:
    """Analyze the effective rank of head outputs.

    Low rank = head produces simple, low-dimensional output.
    """
    _, cache = model.run_with_cache(tokens)
    n_heads = model.cfg.n_heads

    z = cache[f'blocks.{layer}.attn.hook_z']  # [seq, n_heads, d_head]
    W_O = model.blocks[layer].attn.W_O  # [n_heads, d_head, d_model]

    per_head = []
    for h in range(n_heads):
        head_out = jnp.einsum('sh,hm->sm', z[:, h, :], W_O[h])  # [seq, d_model]
        # SVD of output matrix
        U, S, Vt = jnp.linalg.svd(head_out, full_matrices=False)
        S_normalized = S / (jnp.sum(S) + 1e-10)

        # Effective rank (exponential entropy of singular values)
        entropy = -float(jnp.sum(S_normalized * jnp.log(S_normalized + 1e-10)))
        effective_rank = float(jnp.exp(entropy))

        # Top singular value fraction
        top_sv_fraction = float(S[0] / (jnp.sum(S) + 1e-10))

        per_head.append({
            'head': h,
            'effective_rank': effective_rank,
            'top_sv_fraction': top_sv_fraction,
            'is_low_rank': effective_rank < 2.0,
        })

    return {
        'layer': layer,
        'per_head': per_head,
        'mean_effective_rank': sum(h['effective_rank'] for h in per_head) / len(per_head),
    }
