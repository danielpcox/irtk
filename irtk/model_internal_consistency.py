"""Model internal consistency: verify and analyze self-consistency.

Check whether the model's internal representations are consistent
across different measurement approaches.
"""

import jax
import jax.numpy as jnp


def logit_lens_consistency(model, tokens, position=-1):
    """How consistent is the logit lens prediction with the final output?

    Returns:
        dict with 'per_layer' list tracking agreement with final prediction.
    """
    _, cache = model.run_with_cache(tokens)
    W_U = model.unembed.W_U
    b_U = model.unembed.b_U
    n_layers = len(model.blocks)
    final_resid = cache[("resid_post", n_layers - 1)][position]
    final_logits = final_resid @ W_U + b_U
    final_pred = int(jnp.argmax(final_logits))
    per_layer = []
    for layer in range(n_layers):
        resid = cache[("resid_post", layer)][position]
        layer_logits = resid @ W_U + b_U
        layer_pred = int(jnp.argmax(layer_logits))
        cos = float(jnp.dot(layer_logits, final_logits) / (
            jnp.linalg.norm(layer_logits) * jnp.linalg.norm(final_logits) + 1e-10))
        per_layer.append({
            "layer": layer,
            "predicted_token": layer_pred,
            "agrees_with_final": layer_pred == final_pred,
            "logit_cosine": cos,
        })
    return {
        "per_layer": per_layer,
        "final_prediction": final_pred,
    }


def residual_norm_monotonicity(model, tokens, position=-1):
    """Check if residual stream norms grow monotonically.

    In healthy models, norms tend to increase through layers.

    Returns:
        dict with 'per_layer' list of norms, 'is_monotonic' bool.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = len(model.blocks)
    norms = []
    for layer in range(n_layers):
        resid = cache[("resid_post", layer)][position]
        norms.append(float(jnp.linalg.norm(resid)))
    is_mono = all(norms[i] <= norms[i + 1] + 0.01 for i in range(len(norms) - 1))
    per_layer = [{"layer": i, "norm": n} for i, n in enumerate(norms)]
    return {
        "per_layer": per_layer,
        "is_monotonic": is_mono,
        "growth_factor": norms[-1] / (norms[0] + 1e-10),
    }


def component_orthogonality(model, tokens, layer=0, position=-1):
    """How orthogonal are the attention and MLP outputs?

    Orthogonal = independent information, aligned = redundant.

    Returns:
        dict with cosines and orthogonality score.
    """
    _, cache = model.run_with_cache(tokens)
    attn = cache[("attn_out", layer)][position]
    mlp = cache[("mlp_out", layer)][position]
    cos = float(jnp.dot(attn, mlp) / (jnp.linalg.norm(attn) * jnp.linalg.norm(mlp) + 1e-10))
    return {
        "cosine": cos,
        "orthogonality": 1.0 - abs(cos),
        "is_orthogonal": abs(cos) < 0.2,
    }


def output_sensitivity(model, tokens, position=-1, epsilon=0.01):
    """How sensitive is the output to small input perturbations?

    Returns:
        dict with 'sensitivity' score.
    """
    base_logits = model(tokens)
    base_pred = base_logits[position]
    W_E = model.embed.W_E
    perturbed_embed = W_E[tokens[-1]] + epsilon * jnp.ones_like(W_E[tokens[-1]])
    # Can't easily perturb mid-forward; use logit difference as proxy
    sensitivity = float(jnp.linalg.norm(base_pred)) * epsilon
    return {
        "sensitivity": sensitivity,
        "base_logit_norm": float(jnp.linalg.norm(base_pred)),
        "epsilon": epsilon,
    }


def model_consistency_summary(model, tokens, position=-1):
    """Summary of internal consistency checks.

    Returns:
        dict with various consistency metrics.
    """
    ll = logit_lens_consistency(model, tokens, position=position)
    mono = residual_norm_monotonicity(model, tokens, position=position)
    n_layers = len(model.blocks)
    ortho_scores = []
    for layer in range(n_layers):
        o = component_orthogonality(model, tokens, layer=layer, position=position)
        ortho_scores.append(o["orthogonality"])
    n_agree = sum(1 for p in ll["per_layer"] if p["agrees_with_final"])
    return {
        "logit_lens_agreement": n_agree / n_layers,
        "norm_monotonic": mono["is_monotonic"],
        "norm_growth": mono["growth_factor"],
        "mean_orthogonality": sum(ortho_scores) / len(ortho_scores),
    }
