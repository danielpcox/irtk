"""Head value analysis: what information value vectors carry.

Analyze what each attention head writes to the residual stream by
examining value vectors, output projections, and token-specific contributions.
"""

import jax
import jax.numpy as jnp
from irtk.hook_points import HookState


def _run_and_cache(model, tokens):
    """Run model and return activation cache."""
    hook_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=hook_state)
    return hook_state.cache


def value_vector_analysis(model, tokens, layer=0, pos=-1):
    """Analyze what value vectors encode at each position.

    Returns:
        dict with per_head list containing:
        - head: head index
        - value_norm: norm of the value vector
        - value_direction: dominant direction (top PCA component alignment)
        - position_variation: how much values vary across positions
    """
    cache = _run_and_cache(model, tokens)

    v_key = f'blocks.{layer}.attn.hook_v'
    if v_key not in cache:
        return {'per_head': []}

    v = cache[v_key]  # [seq, n_heads, d_head]
    n_heads = v.shape[1]
    seq_len = v.shape[0]

    results = []
    for h in range(n_heads):
        v_h = v[:, h, :]  # [seq, d_head]

        # Norms
        norms = jnp.linalg.norm(v_h, axis=-1)  # [seq]
        mean_norm = float(jnp.mean(norms))

        # Position variation (std of value vectors)
        pos_var = float(jnp.mean(jnp.std(v_h, axis=0)))

        # How concentrated values are (cosine similarity to mean)
        mean_v = jnp.mean(v_h, axis=0)  # [d_head]
        mean_v_norm = jnp.linalg.norm(mean_v)
        cos_to_mean = []
        for p in range(seq_len):
            vn = jnp.linalg.norm(v_h[p])
            cos = float(jnp.dot(v_h[p], mean_v) / jnp.maximum(vn * mean_v_norm, 1e-10))
            cos_to_mean.append(cos)

        results.append({
            'head': h,
            'value_norm': mean_norm,
            'position_variation': pos_var,
            'mean_cosine_to_centroid': float(jnp.mean(jnp.array(cos_to_mean))),
            'value_norms': [float(n) for n in norms],
        })

    return {
        'layer': layer,
        'per_head': results,
    }


def head_output_decomposition(model, tokens, layer=0, pos=-1):
    """Decompose each head's output into its contribution to the residual.

    Returns:
        dict with per_head list containing:
        - head: head index
        - output_norm: norm of head's contribution to residual
        - logit_contribution: how much this head's output affects top logit
        - unembed_alignment: cosine similarity with top token's unembed direction
    """
    cache = _run_and_cache(model, tokens)

    z_key = f'blocks.{layer}.attn.hook_z'
    if z_key not in cache:
        return {'per_head': []}

    z = cache[z_key]  # [seq, n_heads, d_head]
    W_O = model.blocks[layer].attn.W_O  # [n_heads, d_head, d_model]
    n_heads = z.shape[1]

    # Get top predicted token
    logits = model(tokens)
    top_token = int(jnp.argmax(logits[pos]))
    W_U_col = model.unembed.W_U[:, top_token]  # [d_model]

    results = []
    for h in range(n_heads):
        # Head output: z[pos, h] @ W_O[h]
        head_out = jnp.einsum('h,hm->m', z[pos, h], W_O[h])  # [d_model]

        out_norm = float(jnp.linalg.norm(head_out))

        # Logit contribution
        logit_contrib = float(jnp.dot(head_out, W_U_col))

        # Alignment with unembed
        head_norm = jnp.linalg.norm(head_out)
        wu_norm = jnp.linalg.norm(W_U_col)
        alignment = float(jnp.dot(head_out, W_U_col) / jnp.maximum(head_norm * wu_norm, 1e-10))

        results.append({
            'head': h,
            'output_norm': out_norm,
            'logit_contribution': logit_contrib,
            'unembed_alignment': alignment,
        })

    return {
        'layer': layer,
        'top_token': top_token,
        'per_head': results,
    }


def value_weighted_attention(model, tokens, layer=0, pos=-1):
    """Analyze what each position contributes through attention-weighted values.

    Returns:
        dict with per_head list containing:
        - head: head index
        - source_contributions: list of (position, contribution_norm) pairs
        - dominant_source: position that contributes most
        - concentration: fraction of value from top source
    """
    cache = _run_and_cache(model, tokens)

    pattern_key = f'blocks.{layer}.attn.hook_pattern'
    v_key = f'blocks.{layer}.attn.hook_v'

    if pattern_key not in cache or v_key not in cache:
        return {'per_head': []}

    pattern = cache[pattern_key]  # [n_heads, seq_q, seq_k]
    v = cache[v_key]  # [seq, n_heads, d_head]
    n_heads = pattern.shape[0]
    seq_len = v.shape[0]

    if pos < 0:
        pos = seq_len + pos

    results = []
    for h in range(n_heads):
        # Attention weights for this position
        attn_weights = pattern[h, pos]  # [seq_k]

        # Value-weighted contributions
        source_contribs = []
        for s in range(seq_len):
            weighted_v = attn_weights[s] * v[s, h]  # [d_head]
            contrib_norm = float(jnp.linalg.norm(weighted_v))
            source_contribs.append({
                'position': s,
                'attention_weight': float(attn_weights[s]),
                'contribution_norm': contrib_norm,
            })

        # Sort by contribution
        source_contribs.sort(key=lambda x: -x['contribution_norm'])

        total_contrib = sum(s['contribution_norm'] for s in source_contribs)
        top_source = source_contribs[0] if source_contribs else None
        concentration = top_source['contribution_norm'] / max(total_contrib, 1e-10) if top_source else 0.0

        results.append({
            'head': h,
            'source_contributions': source_contribs,
            'dominant_source': top_source['position'] if top_source else 0,
            'concentration': concentration,
        })

    return {
        'layer': layer,
        'query_position': pos,
        'per_head': results,
    }


def value_rank_analysis(model, layer=0):
    """Analyze the effective rank of value computations (W_V @ W_O).

    Low rank = head writes in a low-dimensional subspace.

    Returns:
        dict with per_head list containing:
        - head: head index
        - effective_rank: effective rank of OV matrix
        - top_singular_value: largest singular value
        - condition_number: condition number
    """
    W_V = model.blocks[layer].attn.W_V  # [n_heads, d_model, d_head]
    W_O = model.blocks[layer].attn.W_O  # [n_heads, d_head, d_model]
    n_heads = W_V.shape[0]

    results = []
    for h in range(n_heads):
        # OV matrix: W_V[h] @ W_O[h] = [d_model, d_head] @ [d_head, d_model] = [d_model, d_model]
        OV = jnp.einsum('mh,hd->md', W_V[h], W_O[h])  # [d_model, d_model]

        S = jnp.linalg.svd(OV, compute_uv=False)

        # Effective rank
        S_norm = S / jnp.maximum(jnp.sum(S), 1e-10)
        S_norm = jnp.maximum(S_norm, 1e-10)
        entropy = -jnp.sum(S_norm * jnp.log(S_norm))
        eff_rank = float(jnp.exp(entropy))

        results.append({
            'head': h,
            'effective_rank': eff_rank,
            'top_singular_value': float(S[0]),
            'condition_number': float(S[0] / jnp.maximum(S[-1], 1e-10)),
        })

    return {
        'layer': layer,
        'per_head': results,
    }


def head_writing_direction(model, tokens, layer=0, pos=-1, top_k=3):
    """What directions in residual space does each head write in?

    Returns:
        dict with per_head list containing:
        - head: head index
        - writing_direction: primary writing direction (d_model vector)
        - top_token_alignments: which token embeddings align with writing direction
    """
    cache = _run_and_cache(model, tokens)

    z_key = f'blocks.{layer}.attn.hook_z'
    if z_key not in cache:
        return {'per_head': []}

    z = cache[z_key]  # [seq, n_heads, d_head]
    W_O = model.blocks[layer].attn.W_O  # [n_heads, d_head, d_model]
    W_U = model.unembed.W_U  # [d_model, d_vocab]
    n_heads = z.shape[1]

    results = []
    for h in range(n_heads):
        # Head output direction
        head_out = jnp.einsum('h,hm->m', z[pos, h], W_O[h])  # [d_model]
        out_norm = jnp.linalg.norm(head_out)
        direction = head_out / jnp.maximum(out_norm, 1e-10)

        # Project onto unembed to see which tokens this promotes
        logit_contrib = direction @ W_U  # [d_vocab]
        top_tokens = jnp.argsort(-logit_contrib)[:top_k]

        alignments = []
        for t_idx in top_tokens:
            alignments.append({
                'token': int(t_idx),
                'logit_contribution': float(logit_contrib[t_idx]),
            })

        results.append({
            'head': h,
            'output_norm': float(out_norm),
            'top_token_alignments': alignments,
        })

    return {
        'layer': layer,
        'per_head': results,
    }
