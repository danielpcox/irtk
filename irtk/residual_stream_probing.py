"""Residual stream probing.

Probe the residual stream for specific types of information:
token identity, position, next-token prediction quality, and
directional probes.
"""

import jax
import jax.numpy as jnp
from irtk.hook_points import HookState


def _run_and_cache(model, tokens):
    hook_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=hook_state)
    return hook_state.cache


def token_identity_probe(model, tokens, layer=0, pos=0):
    """How well can the input token identity be recovered from the residual?

    Probes by projecting onto the embedding matrix.

    Returns:
        dict with:
        - true_token: the actual input token
        - predicted_token: most-similar embedding to residual
        - true_token_rank: rank of true token among all embeddings
        - true_token_similarity: cosine similarity to true token's embedding
    """
    cache = _run_and_cache(model, tokens)
    key = f'blocks.{layer}.hook_resid_post' if layer >= 0 else 'blocks.0.hook_resid_pre'
    if key not in cache:
        return {}

    resid = cache[key][pos]  # [d_model]
    W_E = model.embed.W_E  # [d_vocab, d_model]

    # Cosine similarity with all embeddings
    resid_norm = jnp.linalg.norm(resid)
    embed_norms = jnp.linalg.norm(W_E, axis=-1)
    similarities = (W_E @ resid) / jnp.maximum(embed_norms * resid_norm, 1e-10)

    predicted = int(jnp.argmax(similarities))
    true_token = int(tokens[pos])
    true_sim = float(similarities[true_token])

    # Rank of true token
    rank = int(jnp.sum(similarities > similarities[true_token]))

    return {
        'true_token': true_token,
        'predicted_token': predicted,
        'true_token_rank': rank,
        'true_token_similarity': true_sim,
        'predicted_similarity': float(similarities[predicted]),
    }


def next_token_probe(model, tokens, layer=0, pos=-1):
    """How well does the residual stream predict the next token at this layer?

    Projects through the unembedding matrix.

    Returns:
        dict with:
        - top_prediction: most likely next token from this layer
        - top_logit: logit of top prediction
        - entropy: entropy of the logit distribution
    """
    cache = _run_and_cache(model, tokens)
    seq_len = len(tokens)
    if pos < 0:
        pos = seq_len + pos

    key = f'blocks.{layer}.hook_resid_post'
    if key not in cache:
        return {}

    resid = cache[key][pos]  # [d_model]
    W_U = model.unembed.W_U  # [d_model, d_vocab]
    b_U = model.unembed.b_U  # [d_vocab]

    logits = resid @ W_U + b_U  # [d_vocab]
    probs = jax.nn.softmax(logits)

    top_token = int(jnp.argmax(logits))
    top_logit = float(logits[top_token])

    # Entropy
    probs_safe = jnp.maximum(probs, 1e-10)
    entropy = -float(jnp.sum(probs * jnp.log(probs_safe)))

    return {
        'layer': layer,
        'top_prediction': top_token,
        'top_logit': top_logit,
        'top_probability': float(probs[top_token]),
        'entropy': entropy,
    }


def directional_probe(model, tokens, direction, label='custom'):
    """Track a specific direction in the residual stream across layers.

    Args:
        direction: [d_model] vector to track
        label: name for this direction

    Returns:
        dict with per_layer projection magnitudes.
    """
    cache = _run_and_cache(model, tokens)
    n_layers = model.cfg.n_layers

    direction = direction / jnp.maximum(jnp.linalg.norm(direction), 1e-10)

    results = []
    for l in range(n_layers):
        key = f'blocks.{l}.hook_resid_post'
        if key not in cache:
            continue

        resid = cache[key]  # [seq, d_model]
        projs = resid @ direction  # [seq]

        results.append({
            'layer': l,
            'mean_projection': float(jnp.mean(projs)),
            'max_projection': float(jnp.max(projs)),
            'min_projection': float(jnp.min(projs)),
            'std': float(jnp.std(projs)),
        })

    return {'direction_label': label, 'per_layer': results}


def layer_prediction_quality(model, tokens):
    """How good are predictions at each layer (via unembed)?

    Returns:
        dict with per_layer prediction quality metrics.
    """
    cache = _run_and_cache(model, tokens)
    n_layers = model.cfg.n_layers
    seq_len = len(tokens)

    # Final predictions
    final_logits = model(tokens)
    final_top = int(jnp.argmax(final_logits[-1]))

    results = []
    for l in range(n_layers):
        key = f'blocks.{l}.hook_resid_post'
        if key not in cache:
            continue

        resid = cache[key][-1]  # last position, [d_model]
        logits = resid @ model.unembed.W_U + model.unembed.b_U

        top_token = int(jnp.argmax(logits))
        final_token_logit = float(logits[final_top])

        # Rank of final prediction at this layer
        rank = int(jnp.sum(logits > logits[final_top]))

        probs = jax.nn.softmax(logits)
        entropy = -float(jnp.sum(jnp.maximum(probs, 1e-10) * jnp.log(jnp.maximum(probs, 1e-10))))

        results.append({
            'layer': l,
            'top_prediction': top_token,
            'matches_final': top_token == final_top,
            'final_token_rank': rank,
            'final_token_logit': final_token_logit,
            'entropy': entropy,
        })

    return {
        'final_prediction': final_top,
        'per_layer': results,
    }


def residual_information_content(model, tokens, pos=-1):
    """Measure information content of the residual stream at each layer.

    Uses effective dimensionality and norm to quantify information.

    Returns:
        dict with per_layer information metrics.
    """
    cache = _run_and_cache(model, tokens)
    n_layers = model.cfg.n_layers
    seq_len = len(tokens)
    if pos < 0:
        pos = seq_len + pos

    results = []

    for l in range(n_layers):
        key = f'blocks.{l}.hook_resid_post'
        if key not in cache:
            continue

        resid = cache[key]  # [seq, d_model]

        # Norm at target position
        norm = float(jnp.linalg.norm(resid[pos]))

        # Effective dimensionality (from SVD of all positions)
        U, S, Vt = jnp.linalg.svd(resid, full_matrices=False)
        S_norm = S / jnp.maximum(jnp.sum(S), 1e-10)
        S_safe = jnp.maximum(S_norm, 1e-10)
        entropy = -float(jnp.sum(S_safe * jnp.log(S_safe)))
        eff_dim = float(jnp.exp(jnp.array(entropy)))

        # Sparsity: how concentrated is the representation?
        r = resid[pos]
        r_abs = jnp.abs(r)
        l1 = float(jnp.sum(r_abs))
        l2 = float(jnp.linalg.norm(r))
        sparsity = l1 / jnp.maximum(l2 * jnp.sqrt(jnp.array(len(r), dtype=jnp.float32)), 1e-10)
        sparsity = float(sparsity)

        results.append({
            'layer': l,
            'norm': norm,
            'effective_dimensionality': eff_dim,
            'sparsity': sparsity,
        })

    return {'per_layer': results}
