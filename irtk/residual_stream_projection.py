"""Residual stream projection: project onto meaningful directions."""

import jax
import jax.numpy as jnp
from irtk import HookedTransformer


def project_to_token_direction(model: HookedTransformer, tokens: jnp.ndarray, target_token: int) -> dict:
    """Project the residual stream onto a target token's unembedding direction.

    Tracks how strongly each layer's residual points toward the target.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    seq_len = tokens.shape[0]

    W_U = model.unembed.W_U
    direction = W_U[:, target_token]
    direction = direction / (jnp.linalg.norm(direction) + 1e-10)

    per_layer = []
    for layer in range(n_layers):
        resid_key = f'blocks.{layer}.hook_resid_post'
        if resid_key not in cache:
            continue
        resid = cache[resid_key]  # [seq, d_model]

        projections = resid @ direction  # [seq]
        per_position = [float(projections[pos]) for pos in range(seq_len)]
        mean_proj = float(jnp.mean(projections))

        per_layer.append({
            'layer': layer,
            'mean_projection': mean_proj,
            'last_position_projection': float(projections[-1]),
            'per_position': per_position,
        })

    return {
        'target_token': target_token,
        'per_layer': per_layer,
        'final_projection': per_layer[-1]['last_position_projection'] if per_layer else 0.0,
    }


def project_to_difference_direction(model: HookedTransformer, tokens: jnp.ndarray, token_a: int, token_b: int) -> dict:
    """Project residual stream onto the direction between two tokens.

    Shows which token the model "prefers" at each layer.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers

    W_U = model.unembed.W_U
    dir_a = W_U[:, token_a]
    dir_b = W_U[:, token_b]
    diff_dir = dir_a - dir_b
    diff_dir = diff_dir / (jnp.linalg.norm(diff_dir) + 1e-10)

    per_layer = []
    for layer in range(n_layers):
        resid_key = f'blocks.{layer}.hook_resid_post'
        if resid_key not in cache:
            continue
        resid = cache[resid_key][-1]  # last position

        proj = float(jnp.dot(resid, diff_dir))
        per_layer.append({
            'layer': layer,
            'projection': proj,
            'favors_a': proj > 0,
        })

    return {
        'token_a': token_a,
        'token_b': token_b,
        'per_layer': per_layer,
        'final_preference': 'a' if per_layer and per_layer[-1]['favors_a'] else 'b',
    }


def residual_pca_projection(model: HookedTransformer, tokens: jnp.ndarray, layer: int, n_components: int = 3) -> dict:
    """Project residual stream onto its principal components.

    Shows the dominant directions of variation across positions.
    """
    _, cache = model.run_with_cache(tokens)

    resid_key = f'blocks.{layer}.hook_resid_post'
    resid = cache[resid_key]  # [seq, d_model]
    seq_len = resid.shape[0]

    centered = resid - jnp.mean(resid, axis=0)
    U, S, Vh = jnp.linalg.svd(centered, full_matrices=False)

    n_comp = min(n_components, len(S))
    variance_explained = (S[:n_comp] ** 2) / (jnp.sum(S ** 2) + 1e-10)

    projections = centered @ Vh[:n_comp].T  # [seq, n_comp]

    per_component = []
    for c in range(n_comp):
        per_component.append({
            'component': c,
            'variance_explained': float(variance_explained[c]),
            'singular_value': float(S[c]),
            'projections': [float(projections[pos, c]) for pos in range(seq_len)],
        })

    return {
        'layer': layer,
        'per_component': per_component,
        'total_variance_explained': float(jnp.sum(variance_explained)),
    }


def project_to_embedding_subspace(model: HookedTransformer, tokens: jnp.ndarray, layer: int) -> dict:
    """Project residual stream into/out of the embedding subspace.

    Shows how much of the representation lies in the original embedding space.
    """
    _, cache = model.run_with_cache(tokens)

    embed = cache['hook_embed']  # [seq, d_model]
    if 'hook_pos_embed' in cache:
        embed = embed + cache['hook_pos_embed']

    resid = cache[f'blocks.{layer}.hook_resid_post']  # [seq, d_model]
    seq_len = resid.shape[0]

    # Get embedding subspace via SVD
    U, S, Vh = jnp.linalg.svd(embed, full_matrices=False)
    n_dims = min(embed.shape)

    # Project each position
    per_position = []
    for pos in range(seq_len):
        r = resid[pos]
        r_norm = float(jnp.linalg.norm(r))

        # Projection into embedding subspace
        proj_in_embed = Vh.T @ (Vh @ r)  # project and unproject
        proj_norm = float(jnp.linalg.norm(proj_in_embed))
        fraction_in_embed = proj_norm / (r_norm + 1e-10)

        per_position.append({
            'position': pos,
            'residual_norm': r_norm,
            'in_embed_norm': proj_norm,
            'fraction_in_embed': fraction_in_embed,
            'fraction_orthogonal': 1.0 - fraction_in_embed,
        })

    mean_in_embed = sum(p['fraction_in_embed'] for p in per_position) / len(per_position)

    return {
        'layer': layer,
        'per_position': per_position,
        'mean_fraction_in_embed': mean_in_embed,
        'moved_beyond_embed': mean_in_embed < 0.5,
    }


def multi_direction_projection(model: HookedTransformer, tokens: jnp.ndarray, target_tokens: list, position: int = -1) -> dict:
    """Project residual stream at each layer onto multiple token directions.

    Shows how the representation moves between multiple targets.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    seq_len = tokens.shape[0]
    pos = position if position >= 0 else seq_len + position

    W_U = model.unembed.W_U

    directions = {}
    for t in target_tokens:
        d = W_U[:, t]
        directions[t] = d / (jnp.linalg.norm(d) + 1e-10)

    per_layer = []
    for layer in range(n_layers):
        resid_key = f'blocks.{layer}.hook_resid_post'
        if resid_key not in cache:
            continue
        resid = cache[resid_key][pos]

        projections = {}
        for t, d in directions.items():
            projections[t] = float(jnp.dot(resid, d))

        winner = max(projections, key=projections.get)
        per_layer.append({
            'layer': layer,
            'projections': projections,
            'winner': winner,
        })

    return {
        'position': pos,
        'target_tokens': target_tokens,
        'per_layer': per_layer,
        'final_winner': per_layer[-1]['winner'] if per_layer else target_tokens[0],
    }
