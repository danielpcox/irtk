"""Residual stream decomposition: break down the residual stream by source.

Decompose the residual stream at each layer into contributions from
embedding, attention, and MLP components, tracking their relative importance.
"""

import jax.numpy as jnp


def component_contribution_norms(model, tokens, position=-1):
    """Norm of each component's contribution at each layer.

    Returns:
        dict with 'per_layer' list of component norm dicts.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = len(model.blocks)
    per_layer = []
    for layer in range(n_layers):
        attn_out = cache[("attn_out", layer)][position]  # [d_model]
        mlp_out = cache[("mlp_out", layer)][position]  # [d_model]
        resid_post = cache[("resid_post", layer)][position]
        attn_norm = float(jnp.linalg.norm(attn_out))
        mlp_norm = float(jnp.linalg.norm(mlp_out))
        resid_norm = float(jnp.linalg.norm(resid_post))
        per_layer.append({
            "layer": layer,
            "attn_norm": attn_norm,
            "mlp_norm": mlp_norm,
            "residual_norm": resid_norm,
        })
    return {"per_layer": per_layer}


def cumulative_component_balance(model, tokens, position=-1):
    """Cumulative contribution of attn vs MLP vs embedding to the residual.

    Returns:
        dict with 'per_layer' list tracking cumulative norms and fractions.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = len(model.blocks)
    cum_attn = 0.0
    cum_mlp = 0.0
    embed_norm = float(jnp.linalg.norm(cache[("resid_pre", 0)][position]))
    per_layer = []
    for layer in range(n_layers):
        attn_out = cache[("attn_out", layer)][position]
        mlp_out = cache[("mlp_out", layer)][position]
        cum_attn += float(jnp.linalg.norm(attn_out))
        cum_mlp += float(jnp.linalg.norm(mlp_out))
        total = embed_norm + cum_attn + cum_mlp
        per_layer.append({
            "layer": layer,
            "embed_fraction": embed_norm / (total + 1e-10),
            "attn_fraction": cum_attn / (total + 1e-10),
            "mlp_fraction": cum_mlp / (total + 1e-10),
        })
    return {"per_layer": per_layer, "embed_norm": embed_norm}


def residual_component_cosines(model, tokens, position=-1):
    """Cosine similarity between each component's output and the residual.

    Returns:
        dict with 'per_layer' list of cosine dicts.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = len(model.blocks)
    per_layer = []
    for layer in range(n_layers):
        attn_out = cache[("attn_out", layer)][position]
        mlp_out = cache[("mlp_out", layer)][position]
        resid = cache[("resid_post", layer)][position]
        resid_norm = jnp.linalg.norm(resid) + 1e-10
        attn_cos = float(jnp.dot(attn_out, resid) / (jnp.linalg.norm(attn_out) * resid_norm + 1e-10))
        mlp_cos = float(jnp.dot(mlp_out, resid) / (jnp.linalg.norm(mlp_out) * resid_norm + 1e-10))
        per_layer.append({
            "layer": layer,
            "attn_cosine": attn_cos,
            "mlp_cosine": mlp_cos,
        })
    return {"per_layer": per_layer}


def residual_projection_decomposition(model, tokens, position=-1, target_token=None):
    """Decompose residual stream into unembed direction and orthogonal complement.

    If target_token is None, uses the predicted token.

    Returns:
        dict with 'per_layer' list of projection dicts.
    """
    _, cache = model.run_with_cache(tokens)
    W_U = model.unembed.W_U  # [d_model, d_vocab]
    n_layers = len(model.blocks)
    if target_token is None:
        final_resid = cache[("resid_post", n_layers - 1)][position]
        target_token = int(jnp.argmax(final_resid @ W_U))
    unembed_dir = W_U[:, target_token]
    unembed_dir = unembed_dir / (jnp.linalg.norm(unembed_dir) + 1e-10)
    per_layer = []
    for layer in range(n_layers):
        resid = cache[("resid_post", layer)][position]
        proj = float(jnp.dot(resid, unembed_dir))
        resid_norm = float(jnp.linalg.norm(resid))
        per_layer.append({
            "layer": layer,
            "unembed_projection": proj,
            "residual_norm": resid_norm,
            "projection_fraction": abs(proj) / (resid_norm + 1e-10),
        })
    return {
        "per_layer": per_layer,
        "target_token": int(target_token),
    }


def residual_stream_decomposition_summary(model, tokens, position=-1):
    """Summary of residual stream decomposition.

    Returns:
        dict with 'per_layer' list of summary dicts.
    """
    norms = component_contribution_norms(model, tokens, position=position)
    cosines = residual_component_cosines(model, tokens, position=position)
    per_layer = []
    for i, (n, c) in enumerate(zip(norms["per_layer"], cosines["per_layer"])):
        per_layer.append({
            "layer": n["layer"],
            "attn_norm": n["attn_norm"],
            "mlp_norm": n["mlp_norm"],
            "attn_cosine": c["attn_cosine"],
            "mlp_cosine": c["mlp_cosine"],
        })
    return {"per_layer": per_layer}
