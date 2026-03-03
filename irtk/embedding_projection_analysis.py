"""Embedding projection analysis: how embeddings project into model spaces."""

import jax
import jax.numpy as jnp
from irtk import HookedTransformer


def embedding_to_attention_projection(model: HookedTransformer, tokens: jnp.ndarray, layer: int) -> dict:
    """How do token embeddings project into Q/K/V spaces?

    Shows which embedding dimensions contribute most to attention computation.
    """
    _, cache = model.run_with_cache(tokens)
    seq_len = tokens.shape[0]
    n_heads = model.cfg.n_heads

    embed_key = 'hook_embed'
    embed = cache[embed_key]  # [seq, d_model]
    if 'hook_pos_embed' in cache:
        embed = embed + cache['hook_pos_embed']

    W_Q = model.blocks[layer].attn.W_Q  # [n_heads, d_model, d_head]
    W_K = model.blocks[layer].attn.W_K
    W_V = model.blocks[layer].attn.W_V

    per_head = []
    for h in range(n_heads):
        q_proj = embed @ W_Q[h]  # [seq, d_head]
        k_proj = embed @ W_K[h]
        v_proj = embed @ W_V[h]

        q_norm = float(jnp.mean(jnp.linalg.norm(q_proj, axis=-1)))
        k_norm = float(jnp.mean(jnp.linalg.norm(k_proj, axis=-1)))
        v_norm = float(jnp.mean(jnp.linalg.norm(v_proj, axis=-1)))

        per_head.append({
            'head': h,
            'q_projection_norm': q_norm,
            'k_projection_norm': k_norm,
            'v_projection_norm': v_norm,
            'qk_ratio': q_norm / (k_norm + 1e-10),
        })

    return {
        'layer': layer,
        'per_head': per_head,
        'mean_q_norm': sum(h['q_projection_norm'] for h in per_head) / n_heads,
        'mean_v_norm': sum(h['v_projection_norm'] for h in per_head) / n_heads,
    }


def embedding_to_mlp_projection(model: HookedTransformer, tokens: jnp.ndarray, layer: int) -> dict:
    """How do embeddings project into MLP input space?

    Shows which neurons the embeddings activate.
    """
    _, cache = model.run_with_cache(tokens)

    resid_key = f'blocks.{layer}.hook_resid_pre'
    if resid_key not in cache:
        resid_key = f'blocks.{layer}.hook_resid_mid'
    resid = cache[resid_key]  # [seq, d_model]

    W_in = model.blocks[layer].mlp.W_in  # [d_model, d_mlp]
    pre_acts = resid @ W_in  # [seq, d_mlp]
    if hasattr(model.blocks[layer].mlp, 'b_in') and model.blocks[layer].mlp.b_in is not None:
        pre_acts = pre_acts + model.blocks[layer].mlp.b_in

    seq_len = pre_acts.shape[0]
    d_mlp = pre_acts.shape[1]

    # Per-position activation stats
    per_position = []
    for pos in range(seq_len):
        acts = pre_acts[pos]
        n_positive = int(jnp.sum(acts > 0))
        mean_act = float(jnp.mean(jnp.abs(acts)))
        max_act = float(jnp.max(jnp.abs(acts)))

        per_position.append({
            'position': pos,
            'n_activated': n_positive,
            'activation_fraction': n_positive / d_mlp,
            'mean_activation': mean_act,
            'max_activation': max_act,
        })

    return {
        'layer': layer,
        'per_position': per_position,
        'mean_activation_fraction': sum(p['activation_fraction'] for p in per_position) / seq_len,
    }


def embedding_unembed_circuit(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Analyze the direct embedding-to-unembedding circuit (skip connection path).

    What does the model predict from embeddings alone (no transformer layers)?
    """
    _, cache = model.run_with_cache(tokens)
    seq_len = tokens.shape[0]

    embed = cache['hook_embed']  # [seq, d_model]
    if 'hook_pos_embed' in cache:
        embed = embed + cache['hook_pos_embed']

    W_U = model.unembed.W_U
    b_U = model.unembed.b_U if hasattr(model.unembed, 'b_U') and model.unembed.b_U is not None else None

    direct_logits = embed @ W_U
    if b_U is not None:
        direct_logits = direct_logits + b_U

    per_position = []
    for pos in range(seq_len):
        probs = jax.nn.softmax(direct_logits[pos])
        top_token = int(jnp.argmax(probs))
        confidence = float(probs[top_token])
        entropy = float(-jnp.sum(probs * jnp.log(probs + 1e-10)))

        # Does direct path predict the input token?
        input_prob = float(probs[tokens[pos]])

        per_position.append({
            'position': pos,
            'input_token': int(tokens[pos]),
            'predicted_token': top_token,
            'confidence': confidence,
            'entropy': entropy,
            'input_token_probability': input_prob,
            'predicts_self': top_token == int(tokens[pos]),
        })

    n_self_predict = sum(1 for p in per_position if p['predicts_self'])

    return {
        'per_position': per_position,
        'n_self_predictions': n_self_predict,
        'self_prediction_rate': n_self_predict / seq_len,
    }


def embedding_subspace_utilization(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """How much of the embedding space do the input tokens use?

    Measures effective dimensionality of the embedded tokens.
    """
    _, cache = model.run_with_cache(tokens)

    embed = cache['hook_embed']  # [seq, d_model]
    if 'hook_pos_embed' in cache:
        embed = embed + cache['hook_pos_embed']

    seq_len, d_model = embed.shape

    # SVD to measure effective dimensionality
    centered = embed - jnp.mean(embed, axis=0)
    svd_vals = jnp.linalg.svd(centered, compute_uv=False)
    svd_normalized = svd_vals / (jnp.sum(svd_vals) + 1e-10)
    effective_rank = float(jnp.exp(-jnp.sum(svd_normalized * jnp.log(svd_normalized + 1e-10))))

    # Fraction of variance explained by top-k dims
    variance_explained = jnp.cumsum(svd_vals ** 2) / (jnp.sum(svd_vals ** 2) + 1e-10)
    dims_for_90 = int(jnp.searchsorted(variance_explained, 0.9) + 1)
    dims_for_95 = int(jnp.searchsorted(variance_explained, 0.95) + 1)

    # Per-position norms
    norms = jnp.linalg.norm(embed, axis=-1)

    return {
        'd_model': d_model,
        'seq_len': seq_len,
        'effective_rank': effective_rank,
        'utilization': effective_rank / d_model,
        'dims_for_90_pct': dims_for_90,
        'dims_for_95_pct': dims_for_95,
        'mean_norm': float(jnp.mean(norms)),
        'norm_std': float(jnp.std(norms)),
    }


def token_embedding_similarity_structure(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Analyze similarity structure of embedded input tokens.

    Which tokens have similar embeddings? How does positional encoding
    affect token similarity?
    """
    _, cache = model.run_with_cache(tokens)
    seq_len = tokens.shape[0]

    # Content embeddings only
    content_embed = cache['hook_embed']  # [seq, d_model]
    content_norms = jnp.linalg.norm(content_embed, axis=-1, keepdims=True) + 1e-10
    content_normed = content_embed / content_norms
    content_sim = content_normed @ content_normed.T

    # With position
    full_embed = content_embed
    if 'hook_pos_embed' in cache:
        full_embed = full_embed + cache['hook_pos_embed']
    full_norms = jnp.linalg.norm(full_embed, axis=-1, keepdims=True) + 1e-10
    full_normed = full_embed / full_norms
    full_sim = full_normed @ full_normed.T

    # Average similarities
    triu_idx = jnp.triu_indices(seq_len, k=1)
    mean_content_sim = float(jnp.mean(content_sim[triu_idx]))
    mean_full_sim = float(jnp.mean(full_sim[triu_idx]))

    # Most/least similar pairs
    pairs = []
    for i in range(seq_len):
        for j in range(i+1, seq_len):
            pairs.append({
                'pos_a': i,
                'pos_b': j,
                'content_similarity': float(content_sim[i, j]),
                'full_similarity': float(full_sim[i, j]),
            })
    pairs.sort(key=lambda x: x['content_similarity'], reverse=True)

    return {
        'mean_content_similarity': mean_content_sim,
        'mean_full_similarity': mean_full_sim,
        'position_effect': mean_full_sim - mean_content_sim,
        'most_similar_pair': pairs[0] if pairs else None,
        'least_similar_pair': pairs[-1] if pairs else None,
        'n_pairs': len(pairs),
    }
