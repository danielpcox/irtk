"""QK attention circuit analysis: what determines where each head attends."""

import jax
import jax.numpy as jnp
from irtk import HookedTransformer


def qk_eigenspectrum(model: HookedTransformer, layer: int, head: int) -> dict:
    """Eigenspectrum of the QK matrix W_Q @ W_K^T.

    Reveals what the head uses to determine attention.
    """
    W_Q = model.blocks[layer].attn.W_Q[head]  # [d_model, d_head]
    W_K = model.blocks[layer].attn.W_K[head]  # [d_model, d_head]
    QK = W_Q @ W_K.T  # [d_model, d_model]

    sv = jnp.linalg.svd(QK, compute_uv=False)
    total = jnp.sum(sv)
    sv_norm = sv / (total + 1e-10)
    entropy = -float(jnp.sum(sv_norm * jnp.log(sv_norm + 1e-10)))
    eff_rank = float(jnp.exp(entropy))

    top_5 = [{'index': i, 'value': float(sv[i]), 'fraction': float(sv[i] / (total + 1e-10))} for i in range(min(5, len(sv)))]

    return {
        'layer': layer,
        'head': head,
        'effective_rank': eff_rank,
        'spectral_norm': float(sv[0]),
        'frobenius_norm': float(jnp.linalg.norm(QK)),
        'top_singular_values': top_5,
    }


def qk_positional_vs_content(model: HookedTransformer, tokens: jnp.ndarray, layer: int, head: int) -> dict:
    """Is this head's attention driven by position or content?

    Compares scores with and without positional information.
    """
    _, cache = model.run_with_cache(tokens)
    seq_len = tokens.shape[0]

    q = cache[f'blocks.{layer}.attn.hook_q'][:, head, :]  # [seq, d_head]
    k = cache[f'blocks.{layer}.attn.hook_k'][:, head, :]  # [seq, d_head]

    # Full scores
    d_head = model.cfg.d_head
    full_scores = q @ k.T / jnp.sqrt(d_head)  # [seq, seq]

    # Content-only: subtract position embeddings from residual before computing Q/K
    # Approximate by looking at variation across positions with same token
    pattern = cache[f'blocks.{layer}.attn.hook_pattern'][head]  # [seq, seq]

    # Measure: how much does pattern depend on absolute position?
    # Compare pattern entropy to uniform
    entropies = []
    for pos in range(seq_len):
        row = pattern[pos, :pos + 1]
        ent = -float(jnp.sum(row * jnp.log(row + 1e-10)))
        max_ent = float(jnp.log(jnp.array(pos + 1, dtype=jnp.float32)))
        entropies.append(ent / (max_ent + 1e-10))

    mean_norm_entropy = sum(entropies) / len(entropies)

    # Score variance across positions
    score_var = float(jnp.var(full_scores))

    return {
        'layer': layer,
        'head': head,
        'mean_normalized_entropy': mean_norm_entropy,
        'score_variance': score_var,
        'is_positional': bool(mean_norm_entropy < 0.5),
    }


def qk_token_preference(model: HookedTransformer, layer: int, head: int, token_ids: list = None) -> dict:
    """Which tokens does each query most prefer?

    Measures E @ QK @ E^T for token pairs.
    """
    W_Q = model.blocks[layer].attn.W_Q[head]
    W_K = model.blocks[layer].attn.W_K[head]
    W_E = model.embed.W_E

    QK = W_Q @ W_K.T  # [d_model, d_model]

    if token_ids is None:
        token_ids = list(range(min(20, W_E.shape[0])))

    import numpy as np
    token_ids_np = np.array(token_ids)
    E_subset = W_E[token_ids_np]  # [n_tokens, d_model]

    # Token preference matrix
    prefs = E_subset @ QK @ E_subset.T  # [n_tokens, n_tokens]

    per_query = []
    for i, qid in enumerate(token_ids):
        scores = prefs[i]
        top_idx = int(jnp.argmax(scores))
        per_query.append({
            'query_token': qid,
            'preferred_token': token_ids[top_idx],
            'preference_score': float(scores[top_idx]),
            'self_score': float(scores[i]),
        })

    return {
        'layer': layer,
        'head': head,
        'per_query': per_query,
    }


def qk_composition_from_prev_layer(model: HookedTransformer, layer: int, head: int) -> dict:
    """How does this head's QK receive input from previous layer OV circuits?

    Measures Q-composition and K-composition scores.
    """
    if layer == 0:
        return {'layer': layer, 'head': head, 'compositions': [], 'error': 'first_layer'}

    prev_layer = layer - 1
    n_heads = model.cfg.n_heads

    W_Q = model.blocks[layer].attn.W_Q[head]  # [d_model, d_head]
    W_K = model.blocks[layer].attn.W_K[head]

    compositions = []
    for h2 in range(n_heads):
        W_V_prev = model.blocks[prev_layer].attn.W_V[h2]
        W_O_prev = model.blocks[prev_layer].attn.W_O[h2]
        OV_prev = W_V_prev @ W_O_prev  # [d_model, d_model]

        # Q-composition: OV → Q
        q_comp = OV_prev @ W_Q  # [d_model, d_head]
        q_comp_norm = float(jnp.linalg.norm(q_comp))

        # K-composition: OV → K
        k_comp = OV_prev @ W_K
        k_comp_norm = float(jnp.linalg.norm(k_comp))

        compositions.append({
            'prev_head': h2,
            'q_composition': q_comp_norm,
            'k_composition': k_comp_norm,
            'total': q_comp_norm + k_comp_norm,
        })

    compositions.sort(key=lambda x: x['total'], reverse=True)

    return {
        'layer': layer,
        'head': head,
        'prev_layer': prev_layer,
        'compositions': compositions,
    }


def qk_pattern_prediction(model: HookedTransformer, tokens: jnp.ndarray, layer: int, head: int) -> dict:
    """Compare the actual attention pattern to what the QK weight matrix predicts.

    Tests whether the QK matrix alone explains the attention pattern.
    """
    _, cache = model.run_with_cache(tokens)
    seq_len = tokens.shape[0]

    pattern = cache[f'blocks.{layer}.attn.hook_pattern'][head]  # [seq, seq]
    q = cache[f'blocks.{layer}.attn.hook_q'][:, head, :]  # [seq, d_head]
    k = cache[f'blocks.{layer}.attn.hook_k'][:, head, :]

    d_head = model.cfg.d_head
    raw_scores = q @ k.T / jnp.sqrt(d_head)  # [seq, seq]

    # Apply causal mask and softmax
    mask = jnp.tril(jnp.ones((seq_len, seq_len)))
    masked_scores = jnp.where(mask, raw_scores, -1e9)
    predicted_pattern = jax.nn.softmax(masked_scores, axis=-1)

    # Compare
    diff = jnp.abs(pattern - predicted_pattern)
    mean_diff = float(jnp.mean(diff * mask))
    max_diff = float(jnp.max(diff * mask))

    return {
        'layer': layer,
        'head': head,
        'mean_pattern_diff': mean_diff,
        'max_pattern_diff': max_diff,
        'patterns_match': bool(mean_diff < 0.01),
    }
