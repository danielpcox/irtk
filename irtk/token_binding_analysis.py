"""Token binding analysis: how tokens become bound to roles in context.

Detect when token representations become role-specific, measure binding
strength, track binding emergence, and analyze binding competition.
"""

import jax
import jax.numpy as jnp


def token_role_binding(model, tokens, position=-1):
    """Measure how much a token's representation diverges from its static embedding.

    Large divergence = strong contextual binding; small = still close to embedding.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    if position < 0:
        position = len(tokens) + position

    W_E = model.embed.W_E  # [vocab, d_model]
    token_id = int(tokens[position])
    static_embed = W_E[token_id]
    static_norm = jnp.linalg.norm(static_embed) + 1e-10

    per_layer = []
    for layer in range(n_layers):
        key = f"blocks.{layer}.hook_resid_post"
        if key not in cache:
            key = f"blocks.{layer}.hook_resid_pre"
        rep = cache[key][position]
        rep_norm = jnp.linalg.norm(rep) + 1e-10

        cosine_to_embed = float(jnp.dot(rep, static_embed) / (rep_norm * static_norm))
        divergence = float(jnp.linalg.norm(rep - static_embed))

        per_layer.append({
            "layer": layer,
            "cosine_to_embed": cosine_to_embed,
            "divergence": divergence,
            "rep_norm": float(rep_norm),
        })

    total_divergence = per_layer[-1]["divergence"] if per_layer else 0.0

    return {
        "per_layer": per_layer,
        "position": position,
        "token_id": token_id,
        "total_divergence": total_divergence,
        "is_strongly_bound": per_layer[-1]["cosine_to_embed"] < 0.5 if per_layer else False,
    }


def binding_strength_comparison(model, tokens):
    """Compare binding strength across all positions.

    Shows which tokens get the most contextual modification.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    seq_len = len(tokens)
    W_E = model.embed.W_E

    per_position = []
    for pos in range(seq_len):
        token_id = int(tokens[pos])
        static = W_E[token_id]
        static_norm = jnp.linalg.norm(static) + 1e-10

        # Final layer representation
        key = f"blocks.{n_layers - 1}.hook_resid_post"
        if key not in cache:
            key = f"blocks.{n_layers - 1}.hook_resid_pre"
        rep = cache[key][pos]
        rep_norm = jnp.linalg.norm(rep) + 1e-10

        cosine = float(jnp.dot(rep, static) / (rep_norm * static_norm))
        divergence = float(jnp.linalg.norm(rep - static))

        per_position.append({
            "position": pos,
            "token_id": token_id,
            "cosine_to_embed": cosine,
            "divergence": divergence,
        })

    per_position.sort(key=lambda p: p["divergence"], reverse=True)

    return {
        "per_position": per_position,
        "most_bound_position": per_position[0]["position"] if per_position else 0,
        "least_bound_position": per_position[-1]["position"] if per_position else 0,
    }


def binding_source_attribution(model, tokens, position=-1):
    """Attribute token binding to attention vs MLP components.

    Which components cause the most divergence from the static embedding?
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    if position < 0:
        position = len(tokens) + position

    W_E = model.embed.W_E
    token_id = int(tokens[position])
    static = W_E[token_id]
    static_norm = jnp.linalg.norm(static) + 1e-10
    unit_static = static / static_norm

    per_layer = []
    for layer in range(n_layers):
        attn_key = f"blocks.{layer}.hook_attn_out"
        mlp_key = f"blocks.{layer}.hook_mlp_out"

        # How much does each component push away from embedding?
        attn_divergence = 0.0
        mlp_divergence = 0.0
        attn_perp = 0.0
        mlp_perp = 0.0

        if attn_key in cache:
            attn_out = cache[attn_key][position]
            # Component perpendicular to embedding direction
            proj = jnp.dot(attn_out, unit_static) * unit_static
            perp = attn_out - proj
            attn_perp = float(jnp.linalg.norm(perp))
            attn_divergence = float(jnp.linalg.norm(attn_out))

        if mlp_key in cache:
            mlp_out = cache[mlp_key][position]
            proj = jnp.dot(mlp_out, unit_static) * unit_static
            perp = mlp_out - proj
            mlp_perp = float(jnp.linalg.norm(perp))
            mlp_divergence = float(jnp.linalg.norm(mlp_out))

        per_layer.append({
            "layer": layer,
            "attn_divergence": attn_divergence,
            "mlp_divergence": mlp_divergence,
            "attn_perpendicular": attn_perp,
            "mlp_perpendicular": mlp_perp,
            "binding_source": "attention" if attn_perp > mlp_perp else "mlp",
        })

    return {
        "per_layer": per_layer,
        "position": position,
        "token_id": token_id,
    }


def binding_competition(model, tokens, position=-1, top_k=5):
    """Analyze which other tokens' embeddings compete for a position's representation.

    Shows if the representation is pulled toward specific other token identities.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    if position < 0:
        position = len(tokens) + position

    W_E = model.embed.W_E  # [vocab, d_model]
    vocab_size = W_E.shape[0]

    # Normalize embeddings
    E_norms = jnp.linalg.norm(W_E, axis=-1, keepdims=True) + 1e-10
    E_normed = W_E / E_norms

    per_layer = []
    for layer in range(n_layers):
        key = f"blocks.{layer}.hook_resid_post"
        if key not in cache:
            key = f"blocks.{layer}.hook_resid_pre"
        rep = cache[key][position]
        rep_norm = jnp.linalg.norm(rep) + 1e-10
        rep_normed = rep / rep_norm

        # Cosine with all embeddings
        cos_scores = E_normed @ rep_normed  # [vocab]
        top_indices = jnp.argsort(cos_scores)[-top_k:][::-1]

        competitors = []
        for idx in top_indices:
            competitors.append({
                "token_id": int(idx),
                "cosine": float(cos_scores[int(idx)]),
            })

        per_layer.append({
            "layer": layer,
            "top_competitor": int(top_indices[0]),
            "top_cosine": float(cos_scores[int(top_indices[0])]),
            "competitors": competitors,
        })

    return {
        "per_layer": per_layer,
        "position": position,
        "original_token": int(tokens[position]),
    }


def token_binding_summary(model, tokens):
    """Cross-position summary of token binding patterns.

    Tracks binding strength, divergence, and contextual modification.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    seq_len = len(tokens)
    W_E = model.embed.W_E

    per_position = []
    for pos in range(seq_len):
        token_id = int(tokens[pos])
        static = W_E[token_id]
        static_norm = jnp.linalg.norm(static) + 1e-10

        # Track across layers
        layer_cosines = []
        for layer in range(n_layers):
            key = f"blocks.{layer}.hook_resid_post"
            if key not in cache:
                key = f"blocks.{layer}.hook_resid_pre"
            rep = cache[key][pos]
            rep_norm = jnp.linalg.norm(rep) + 1e-10
            cos = float(jnp.dot(rep, static) / (rep_norm * static_norm))
            layer_cosines.append(cos)

        per_position.append({
            "position": pos,
            "token_id": token_id,
            "initial_cosine": layer_cosines[0] if layer_cosines else 0.0,
            "final_cosine": layer_cosines[-1] if layer_cosines else 0.0,
            "binding_rate": (layer_cosines[0] - layer_cosines[-1]) if layer_cosines else 0.0,
        })

    return {
        "per_position": per_position,
        "mean_binding_rate": sum(p["binding_rate"] for p in per_position) / max(len(per_position), 1),
        "n_positions": seq_len,
    }
