"""MLP knowledge storage: what knowledge is stored in MLP weights.

Analyze what tokens/concepts MLP neurons promote and suppress,
the structure of stored associations, and knowledge localization.
"""

import jax.numpy as jnp


def neuron_logit_effect(model, layer=0, top_k=5):
    """What tokens does each neuron promote/suppress?

    Computes W_out[neuron] @ W_U to get each neuron's effect on logits.

    Returns:
        dict with 'per_neuron' list (top-k) with promoted/suppressed tokens.
    """
    W_out = model.blocks[layer].mlp.W_out  # [d_mlp, d_model]
    W_U = model.unembed.W_U  # [d_model, d_vocab]
    logit_effect = W_out @ W_U  # [d_mlp, d_vocab]
    d_mlp = logit_effect.shape[0]
    per_neuron = []
    for n in range(min(top_k, d_mlp)):
        effect = logit_effect[n]
        top_idx = jnp.argsort(-effect)[:3]
        bot_idx = jnp.argsort(effect)[:3]
        promoted = [(int(i), float(effect[i])) for i in top_idx]
        suppressed = [(int(i), float(effect[i])) for i in bot_idx]
        per_neuron.append({
            "neuron": n,
            "promoted": promoted,
            "suppressed": suppressed,
            "effect_range": float(jnp.max(effect) - jnp.min(effect)),
        })
    return {"per_neuron": per_neuron}


def mlp_layer_logit_effect(model, tokens, layer=0, position=-1, top_k=5):
    """Net logit effect of the entire MLP layer.

    Returns:
        dict with 'promoted' and 'suppressed' token lists, 'total_effect_norm'.
    """
    _, cache = model.run_with_cache(tokens)
    mlp_out = cache[("mlp_out", layer)][position]  # [d_model]
    W_U = model.unembed.W_U  # [d_model, d_vocab]
    logit_effect = mlp_out @ W_U  # [d_vocab]
    top_idx = jnp.argsort(-logit_effect)[:top_k]
    bot_idx = jnp.argsort(logit_effect)[:top_k]
    promoted = [(int(i), float(logit_effect[i])) for i in top_idx]
    suppressed = [(int(i), float(logit_effect[i])) for i in bot_idx]
    return {
        "promoted": promoted,
        "suppressed": suppressed,
        "total_effect_norm": float(jnp.linalg.norm(logit_effect)),
    }


def knowledge_localization(model, tokens, target_token, position=-1):
    """How localized is the knowledge for a specific token across MLP layers?

    Returns:
        dict with 'per_layer' list of logit contributions toward target_token.
    """
    _, cache = model.run_with_cache(tokens)
    W_U = model.unembed.W_U  # [d_model, d_vocab]
    unembed_dir = W_U[:, target_token]
    n_layers = len(model.blocks)
    per_layer = []
    total_contribution = 0.0
    for layer in range(n_layers):
        mlp_out = cache[("mlp_out", layer)][position]  # [d_model]
        contribution = float(jnp.dot(mlp_out, unembed_dir))
        total_contribution += abs(contribution)
        per_layer.append({
            "layer": layer,
            "contribution": contribution,
        })
    for p in per_layer:
        p["fraction"] = abs(p["contribution"]) / (total_contribution + 1e-10)
    most_important = max(per_layer, key=lambda x: abs(x["contribution"]))
    return {
        "per_layer": per_layer,
        "most_important_layer": most_important["layer"],
        "target_token": int(target_token),
    }


def mlp_association_structure(model, layer=0, top_k=5):
    """Analyze the association structure of the MLP via W_in @ W_out.

    The product maps residual -> residual and reveals stored associations.

    Returns:
        dict with 'effective_rank', 'top_singular_values', 'condition_number'.
    """
    W_in = model.blocks[layer].mlp.W_in  # [d_model, d_mlp]
    W_out = model.blocks[layer].mlp.W_out  # [d_mlp, d_model]
    product = W_in @ W_out  # [d_model, d_model]
    s = jnp.linalg.svd(product, compute_uv=False)
    s_norm = s / (jnp.sum(s) + 1e-10)
    s_safe = jnp.where(s_norm > 1e-10, s_norm, 1e-10)
    eff_rank = float(jnp.exp(-jnp.sum(s_safe * jnp.log(s_safe))))
    k = min(top_k, len(s))
    top_sv = [float(v) for v in s[:k]]
    cond = float(s[0] / (s[-1] + 1e-10))
    return {
        "effective_rank": eff_rank,
        "top_singular_values": top_sv,
        "condition_number": cond,
    }


def mlp_knowledge_summary(model, tokens, position=-1):
    """Summary of MLP knowledge storage across layers.

    Returns:
        dict with 'per_layer' list of summary dicts.
    """
    n_layers = len(model.blocks)
    per_layer = []
    for layer in range(n_layers):
        effect = mlp_layer_logit_effect(model, tokens, layer=layer, position=position, top_k=1)
        assoc = mlp_association_structure(model, layer=layer, top_k=1)
        per_layer.append({
            "layer": layer,
            "top_promoted_token": effect["promoted"][0][0] if effect["promoted"] else None,
            "effect_norm": effect["total_effect_norm"],
            "effective_rank": assoc["effective_rank"],
        })
    return {"per_layer": per_layer}
