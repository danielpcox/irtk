"""Token representation tracking: follow token representations through layers."""

import jax
import jax.numpy as jnp
from irtk import HookedTransformer


def token_identity_trajectory(model: HookedTransformer, tokens: jnp.ndarray, position: int) -> dict:
    """Track a token's representation through all layers.

    Measures norm, prediction, and similarity to embedding at each layer.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    seq_len = tokens.shape[0]
    pos = position if position >= 0 else seq_len + position

    W_U = model.unembed.W_U
    b_U = model.unembed.b_U if hasattr(model.unembed, 'b_U') and model.unembed.b_U is not None else None

    # Get embedding
    embed_key = 'hook_embed'
    embed = cache[embed_key][pos] if embed_key in cache else None

    per_layer = []
    for layer in range(n_layers):
        resid_key = f'blocks.{layer}.hook_resid_post'
        if resid_key not in cache:
            continue
        resid = cache[resid_key][pos]  # [d_model]

        norm = float(jnp.linalg.norm(resid))
        logits = resid @ W_U
        if b_U is not None:
            logits = logits + b_U
        top_token = int(jnp.argmax(logits))
        confidence = float(jax.nn.softmax(logits)[top_token])

        # Similarity to initial embedding
        if embed is not None:
            embed_sim = float(jnp.dot(resid, embed) / (norm * jnp.linalg.norm(embed) + 1e-10))
        else:
            embed_sim = 0.0

        per_layer.append({
            'layer': layer,
            'norm': norm,
            'top_token': top_token,
            'confidence': confidence,
            'embed_similarity': embed_sim,
        })

    # How fast does identity decay?
    if len(per_layer) >= 2 and per_layer[0]['embed_similarity'] != 0:
        identity_decay = per_layer[0]['embed_similarity'] - per_layer[-1]['embed_similarity']
    else:
        identity_decay = 0.0

    return {
        'position': pos,
        'original_token': int(tokens[pos]),
        'per_layer': per_layer,
        'identity_decay': identity_decay,
        'final_prediction': per_layer[-1]['top_token'] if per_layer else -1,
    }


def position_representation_divergence(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """How much do different positions' representations diverge through layers?

    Measures mean pairwise distance at each layer.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    seq_len = tokens.shape[0]

    per_layer = []
    for layer in range(n_layers):
        resid_key = f'blocks.{layer}.hook_resid_post'
        if resid_key not in cache:
            continue
        resid = cache[resid_key]  # [seq, d_model]

        # Mean pairwise cosine distance
        norms = jnp.linalg.norm(resid, axis=-1, keepdims=True) + 1e-10
        normed = resid / norms
        sim_matrix = normed @ normed.T  # [seq, seq]
        mean_sim = float(jnp.mean(sim_matrix[jnp.triu_indices(seq_len, k=1)]))

        # Mean pairwise L2 distance
        diffs = resid[:, None, :] - resid[None, :, :]  # [seq, seq, d_model]
        l2_dists = jnp.linalg.norm(diffs, axis=-1)
        mean_l2 = float(jnp.mean(l2_dists[jnp.triu_indices(seq_len, k=1)]))

        per_layer.append({
            'layer': layer,
            'mean_cosine_similarity': mean_sim,
            'mean_l2_distance': mean_l2,
            'divergence': 1.0 - mean_sim,
        })

    # Divergence trend
    if len(per_layer) >= 2:
        trend = per_layer[-1]['divergence'] - per_layer[0]['divergence']
    else:
        trend = 0.0

    return {
        'per_layer': per_layer,
        'divergence_trend': trend,
        'representations_diverge': trend > 0,
    }


def token_mixing_rate(model: HookedTransformer, tokens: jnp.ndarray, position: int) -> dict:
    """How fast does a token's representation mix with others?

    Tracks the cosine similarity of position's representation with its own
    embedding vs the mean representation.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    seq_len = tokens.shape[0]
    pos = position if position >= 0 else seq_len + position

    per_layer = []
    for layer in range(n_layers):
        resid_key = f'blocks.{layer}.hook_resid_post'
        if resid_key not in cache:
            continue
        resid = cache[resid_key]  # [seq, d_model]

        # Self-retention: similarity to own position
        self_vec = resid[pos]
        mean_vec = jnp.mean(resid, axis=0)

        self_norm = jnp.linalg.norm(self_vec) + 1e-10
        mean_norm = jnp.linalg.norm(mean_vec) + 1e-10

        # How much does this position's repr look like the mean?
        alignment_to_mean = float(jnp.dot(self_vec, mean_vec) / (self_norm * mean_norm))

        # How unique is this position? (distance from mean)
        uniqueness = float(jnp.linalg.norm(self_vec - mean_vec) / self_norm)

        per_layer.append({
            'layer': layer,
            'alignment_to_mean': alignment_to_mean,
            'uniqueness': uniqueness,
            'norm': float(self_norm),
        })

    # Mixing rate: how fast does alignment_to_mean increase?
    if len(per_layer) >= 2:
        mixing_rate = per_layer[-1]['alignment_to_mean'] - per_layer[0]['alignment_to_mean']
    else:
        mixing_rate = 0.0

    return {
        'position': pos,
        'per_layer': per_layer,
        'mixing_rate': mixing_rate,
    }


def representation_velocity(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """How fast do representations change between layers?

    Velocity = L2 distance between consecutive layer representations.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    seq_len = tokens.shape[0]

    per_layer = []
    prev_resid = None

    for layer in range(n_layers):
        resid_key = f'blocks.{layer}.hook_resid_post'
        if resid_key not in cache:
            continue
        resid = cache[resid_key]  # [seq, d_model]

        if prev_resid is not None:
            velocity = jnp.linalg.norm(resid - prev_resid, axis=-1)  # [seq]
            mean_vel = float(jnp.mean(velocity))
            max_vel = float(jnp.max(velocity))
            max_vel_pos = int(jnp.argmax(velocity))
        else:
            mean_vel = 0.0
            max_vel = 0.0
            max_vel_pos = 0

        per_layer.append({
            'layer': layer,
            'mean_velocity': mean_vel,
            'max_velocity': max_vel,
            'fastest_position': max_vel_pos,
        })
        prev_resid = resid

    velocities = [p['mean_velocity'] for p in per_layer[1:]]
    mean_total = sum(velocities) / len(velocities) if velocities else 0.0

    return {
        'per_layer': per_layer,
        'mean_velocity': mean_total,
        'peak_layer': max(per_layer[1:], key=lambda x: x['mean_velocity'])['layer'] if len(per_layer) > 1 else 0,
    }


def representation_convergence(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Do representations converge or diverge at the last layer?

    Compares similarity structure at early vs late layers.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    seq_len = tokens.shape[0]

    def layer_similarity(layer):
        resid_key = f'blocks.{layer}.hook_resid_post'
        resid = cache[resid_key]
        norms = jnp.linalg.norm(resid, axis=-1, keepdims=True) + 1e-10
        normed = resid / norms
        sim = normed @ normed.T
        return float(jnp.mean(sim[jnp.triu_indices(seq_len, k=1)]))

    early_sim = layer_similarity(0)
    late_sim = layer_similarity(n_layers - 1)

    per_layer = []
    for layer in range(n_layers):
        sim = layer_similarity(layer)
        per_layer.append({
            'layer': layer,
            'mean_pairwise_similarity': sim,
        })

    convergence = late_sim - early_sim

    return {
        'per_layer': per_layer,
        'early_similarity': early_sim,
        'late_similarity': late_sim,
        'convergence': convergence,
        'representations_converge': convergence > 0,
    }
