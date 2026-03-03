"""Residual attribution.

Fine-grained attribution of residual stream changes: per-component
contribution tracking, cumulative vs incremental effects, directional
attribution, and interference analysis.

References:
    Elhage et al. (2021) "A Mathematical Framework for Transformer Circuits"
    Nanda (2022) "Attribution Patching: Activation Patching At Industrial Scale"
"""

import jax
import jax.numpy as jnp
import numpy as np


def per_component_residual_contribution(model, tokens, pos=-1):
    """Track each component's contribution to the residual stream.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        pos: Position to analyze.

    Returns:
        dict with:
            embed_contribution: [d_model] from embedding
            attn_contributions: [n_layers, d_model] from each attention layer
            mlp_contributions: [n_layers, d_model] from each MLP
            contribution_norms: dict mapping component name -> float
            dominant_component: str
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model

    cache_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=cache_state)
    cache = cache_state.cache

    # Embedding contribution
    resid_pre = cache.get("blocks.0.hook_resid_pre")
    embed_contrib = np.array(resid_pre[pos]) if resid_pre is not None else np.zeros(d_model)

    attn_contribs = np.zeros((n_layers, d_model))
    mlp_contribs = np.zeros((n_layers, d_model))

    for layer in range(n_layers):
        attn_out = cache.get(f"blocks.{layer}.hook_attn_out")
        if attn_out is not None:
            attn_contribs[layer] = np.array(attn_out[pos])

        mlp_out = cache.get(f"blocks.{layer}.hook_mlp_out")
        if mlp_out is not None:
            mlp_contribs[layer] = np.array(mlp_out[pos])

    # Norms
    norms = {"embed": float(np.linalg.norm(embed_contrib))}
    best_name = "embed"
    best_norm = norms["embed"]
    for l in range(n_layers):
        an = float(np.linalg.norm(attn_contribs[l]))
        mn = float(np.linalg.norm(mlp_contribs[l]))
        norms[f"attn_L{l}"] = an
        norms[f"mlp_L{l}"] = mn
        if an > best_norm:
            best_norm = an
            best_name = f"attn_L{l}"
        if mn > best_norm:
            best_norm = mn
            best_name = f"mlp_L{l}"

    return {
        "embed_contribution": embed_contrib,
        "attn_contributions": attn_contribs,
        "mlp_contributions": mlp_contribs,
        "contribution_norms": norms,
        "dominant_component": best_name,
    }


def cumulative_residual_buildup(model, tokens, pos=-1):
    """Track how the residual stream builds up layer by layer.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        pos: Position to analyze.

    Returns:
        dict with:
            residual_norms: [n_layers+1] norm at each stage
            residual_directions: [n_layers+1, d_model] direction at each stage
            direction_changes: [n_layers] angle between consecutive residuals
            growth_rates: [n_layers] relative norm increase per layer
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model

    cache_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=cache_state)
    cache = cache_state.cache

    residuals = []
    # Layer 0 input
    r = cache.get("blocks.0.hook_resid_pre")
    if r is not None:
        residuals.append(np.array(r[pos]))
    else:
        residuals.append(np.zeros(d_model))

    for layer in range(n_layers):
        r = cache.get(f"blocks.{layer}.hook_resid_post")
        if r is not None:
            residuals.append(np.array(r[pos]))
        else:
            residuals.append(np.zeros(d_model))

    norms = np.array([np.linalg.norm(r) for r in residuals])
    directions = np.array(residuals)

    # Direction changes (angle between consecutive)
    angles = np.zeros(n_layers)
    for l in range(n_layers):
        n1 = np.linalg.norm(residuals[l]) + 1e-10
        n2 = np.linalg.norm(residuals[l + 1]) + 1e-10
        cos = np.dot(residuals[l], residuals[l + 1]) / (n1 * n2)
        angles[l] = float(np.arccos(np.clip(cos, -1, 1)))

    # Growth rates
    growth = np.zeros(n_layers)
    for l in range(n_layers):
        if norms[l] > 1e-10:
            growth[l] = (norms[l + 1] - norms[l]) / norms[l]

    return {
        "residual_norms": norms,
        "residual_directions": directions,
        "direction_changes": angles,
        "growth_rates": growth,
    }


def directional_attribution(model, tokens, target_direction, pos=-1):
    """Attribute the residual's component along a target direction to each component.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        target_direction: Direction to project onto [d_model].
        pos: Position.

    Returns:
        dict with:
            embed_attribution: float, embedding's contribution along direction
            attn_attributions: [n_layers] attention contributions
            mlp_attributions: [n_layers] MLP contributions
            total_attribution: float, total along direction
            attribution_fractions: dict of component -> fraction of total
    """
    from irtk.hook_points import HookState

    target = np.array(target_direction)
    target_unit = target / (np.linalg.norm(target) + 1e-10)

    n_layers = model.cfg.n_layers

    cache_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=cache_state)
    cache = cache_state.cache

    # Embedding
    resid_pre = cache.get("blocks.0.hook_resid_pre")
    embed_attr = 0.0
    if resid_pre is not None:
        embed_attr = float(np.dot(np.array(resid_pre[pos]), target_unit))

    attn_attrs = np.zeros(n_layers)
    mlp_attrs = np.zeros(n_layers)

    for layer in range(n_layers):
        attn_out = cache.get(f"blocks.{layer}.hook_attn_out")
        if attn_out is not None:
            attn_attrs[layer] = float(np.dot(np.array(attn_out[pos]), target_unit))

        mlp_out = cache.get(f"blocks.{layer}.hook_mlp_out")
        if mlp_out is not None:
            mlp_attrs[layer] = float(np.dot(np.array(mlp_out[pos]), target_unit))

    total = embed_attr + float(np.sum(attn_attrs)) + float(np.sum(mlp_attrs))

    fractions = {}
    denom = abs(total) + 1e-10
    fractions["embed"] = embed_attr / denom
    for l in range(n_layers):
        fractions[f"attn_L{l}"] = float(attn_attrs[l]) / denom
        fractions[f"mlp_L{l}"] = float(mlp_attrs[l]) / denom

    return {
        "embed_attribution": embed_attr,
        "attn_attributions": attn_attrs,
        "mlp_attributions": mlp_attrs,
        "total_attribution": float(total),
        "attribution_fractions": fractions,
    }


def component_interference_analysis(model, tokens, pos=-1):
    """Analyze constructive/destructive interference between components.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        pos: Position.

    Returns:
        dict with:
            pairwise_alignment: dict of (comp_a, comp_b) -> cosine similarity
            constructive_pairs: list of (comp_a, comp_b) with positive alignment
            destructive_pairs: list with negative alignment
            net_alignment: float, overall constructive vs destructive
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers

    cache_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=cache_state)
    cache = cache_state.cache

    # Collect all component outputs
    components = {}
    resid_pre = cache.get("blocks.0.hook_resid_pre")
    if resid_pre is not None:
        components["embed"] = np.array(resid_pre[pos])

    for layer in range(n_layers):
        attn = cache.get(f"blocks.{layer}.hook_attn_out")
        if attn is not None:
            components[f"attn_L{layer}"] = np.array(attn[pos])
        mlp = cache.get(f"blocks.{layer}.hook_mlp_out")
        if mlp is not None:
            components[f"mlp_L{layer}"] = np.array(mlp[pos])

    names = list(components.keys())
    alignment = {}
    constructive = []
    destructive = []

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = components[names[i]]
            b = components[names[j]]
            na = np.linalg.norm(a) + 1e-10
            nb = np.linalg.norm(b) + 1e-10
            cos = float(np.dot(a, b) / (na * nb))
            pair = (names[i], names[j])
            alignment[pair] = cos
            if cos > 0.1:
                constructive.append(pair)
            elif cos < -0.1:
                destructive.append(pair)

    net = sum(alignment.values()) / max(1, len(alignment))

    return {
        "pairwise_alignment": alignment,
        "constructive_pairs": constructive,
        "destructive_pairs": destructive,
        "net_alignment": float(net),
    }


def residual_decomposition_at_unembed(model, tokens, pos=-1, top_k=5):
    """Decompose final residual into per-component logit contributions.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        pos: Position.
        top_k: Number of top tokens per component.

    Returns:
        dict with:
            component_logits: dict of component -> [d_vocab] logit contribution
            top_tokens_per_component: dict of component -> list of (token, logit)
            total_logits: [d_vocab]
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers

    cache_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=cache_state)
    cache = cache_state.cache

    W_U = np.array(model.unembed.W_U)  # [d_model, d_vocab]

    comp_logits = {}
    top_tokens = {}

    # Embedding
    resid_pre = cache.get("blocks.0.hook_resid_pre")
    if resid_pre is not None:
        embed_vec = np.array(resid_pre[pos])
        logits = embed_vec @ W_U
        comp_logits["embed"] = logits
        top_idx = np.argsort(logits)[::-1][:top_k]
        top_tokens["embed"] = [(int(i), float(logits[i])) for i in top_idx]

    for layer in range(n_layers):
        for comp_type, hook_key in [("attn", f"blocks.{layer}.hook_attn_out"),
                                     ("mlp", f"blocks.{layer}.hook_mlp_out")]:
            out = cache.get(hook_key)
            if out is not None:
                vec = np.array(out[pos])
                logits = vec @ W_U
                name = f"{comp_type}_L{layer}"
                comp_logits[name] = logits
                top_idx = np.argsort(logits)[::-1][:top_k]
                top_tokens[name] = [(int(i), float(logits[i])) for i in top_idx]

    # Total
    total = sum(comp_logits.values())

    return {
        "component_logits": comp_logits,
        "top_tokens_per_component": top_tokens,
        "total_logits": total,
    }
