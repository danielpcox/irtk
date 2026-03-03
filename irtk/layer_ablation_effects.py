"""Layer ablation effects: measure the impact of removing each layer.

Analyze what happens to model predictions when individual layers
or components are ablated (zeroed out).
"""

import jax
import jax.numpy as jnp
from irtk.hook_points import HookState


def layer_zero_ablation(model, tokens, position=-1):
    """Measure prediction change when each layer's output is zeroed.

    Uses hooks to zero out attn+mlp at each layer.

    Returns:
        dict with 'per_layer' list of KL divergence and pred change.
    """
    _, cache = model.run_with_cache(tokens)
    W_U = model.unembed.W_U
    b_U = model.unembed.b_U
    n_layers = len(model.blocks)
    final_resid = cache[("resid_post", n_layers - 1)][position]
    base_logits = final_resid @ W_U + b_U
    base_probs = jax.nn.softmax(base_logits)
    base_pred = int(jnp.argmax(base_logits))

    per_layer = []
    for ablate_layer in range(n_layers):
        hook_fns = {
            f"blocks.{ablate_layer}.hook_attn_out": lambda x, name: jnp.zeros_like(x),
            f"blocks.{ablate_layer}.hook_mlp_out": lambda x, name: jnp.zeros_like(x),
        }
        hook_state = HookState(hook_fns=hook_fns)
        ablated_logits = model(tokens, hook_state=hook_state)
        abl_logits_pos = ablated_logits[position]
        abl_probs = jax.nn.softmax(abl_logits_pos)
        abl_pred = int(jnp.argmax(abl_logits_pos))
        kl = float(jnp.sum(base_probs * (jnp.log(base_probs + 1e-10) - jnp.log(abl_probs + 1e-10))))
        per_layer.append({
            "layer": ablate_layer,
            "kl_divergence": kl,
            "prediction_changed": abl_pred != base_pred,
            "ablated_prediction": abl_pred,
        })
    return {"per_layer": per_layer, "base_prediction": base_pred}


def component_ablation(model, tokens, layer=0, position=-1):
    """Compare ablating attention vs MLP at a specific layer.

    Returns:
        dict with 'attn_ablation' and 'mlp_ablation' KL divergences.
    """
    base_logits = model(tokens)
    base_probs = jax.nn.softmax(base_logits[position])

    attn_hook = {f"blocks.{layer}.hook_attn_out": lambda x, name: jnp.zeros_like(x)}
    hook_state_attn = HookState(hook_fns=attn_hook)
    attn_abl_logits = model(tokens, hook_state=hook_state_attn)[position]
    attn_probs = jax.nn.softmax(attn_abl_logits)
    attn_kl = float(jnp.sum(base_probs * (jnp.log(base_probs + 1e-10) - jnp.log(attn_probs + 1e-10))))

    mlp_hook = {f"blocks.{layer}.hook_mlp_out": lambda x, name: jnp.zeros_like(x)}
    hook_state_mlp = HookState(hook_fns=mlp_hook)
    mlp_abl_logits = model(tokens, hook_state=hook_state_mlp)[position]
    mlp_probs = jax.nn.softmax(mlp_abl_logits)
    mlp_kl = float(jnp.sum(base_probs * (jnp.log(base_probs + 1e-10) - jnp.log(mlp_probs + 1e-10))))

    return {
        "attn_kl": attn_kl,
        "mlp_kl": mlp_kl,
        "more_important": "attn" if attn_kl > mlp_kl else "mlp",
    }


def mean_ablation(model, tokens, layer=0, position=-1):
    """Replace layer output with its mean (instead of zero).

    Returns:
        dict with 'kl_divergence', 'prediction_changed'.
    """
    _, cache = model.run_with_cache(tokens)
    attn_mean = jnp.mean(cache[("attn_out", layer)], axis=0, keepdims=True)
    mlp_mean = jnp.mean(cache[("mlp_out", layer)], axis=0, keepdims=True)

    def attn_hook(x, name, _mean=attn_mean):
        return jnp.broadcast_to(_mean, x.shape)

    def mlp_hook(x, name, _mean=mlp_mean):
        return jnp.broadcast_to(_mean, x.shape)

    hook_fns = {
        f"blocks.{layer}.hook_attn_out": attn_hook,
        f"blocks.{layer}.hook_mlp_out": mlp_hook,
    }
    base_logits = model(tokens)
    base_probs = jax.nn.softmax(base_logits[position])

    hook_state = HookState(hook_fns=hook_fns)
    abl_logits = model(tokens, hook_state=hook_state)[position]
    abl_probs = jax.nn.softmax(abl_logits)
    kl = float(jnp.sum(base_probs * (jnp.log(base_probs + 1e-10) - jnp.log(abl_probs + 1e-10))))
    return {
        "kl_divergence": kl,
        "prediction_changed": int(jnp.argmax(abl_logits)) != int(jnp.argmax(base_logits[position])),
    }


def cumulative_ablation(model, tokens, position=-1):
    """Ablate layers cumulatively from the top to find minimum needed.

    Returns:
        dict with 'per_n_layers_removed' list.
    """
    base_logits = model(tokens)
    base_pred = int(jnp.argmax(base_logits[position]))
    base_probs = jax.nn.softmax(base_logits[position])
    n_layers = len(model.blocks)
    per_removed = []
    for n_remove in range(1, n_layers + 1):
        hook_fns = {}
        for l in range(n_layers - n_remove, n_layers):
            hook_fns[f"blocks.{l}.hook_attn_out"] = lambda x, name: jnp.zeros_like(x)
            hook_fns[f"blocks.{l}.hook_mlp_out"] = lambda x, name: jnp.zeros_like(x)
        hook_state = HookState(hook_fns=hook_fns)
        abl_logits = model(tokens, hook_state=hook_state)[position]
        abl_pred = int(jnp.argmax(abl_logits))
        abl_probs = jax.nn.softmax(abl_logits)
        kl = float(jnp.sum(base_probs * (jnp.log(base_probs + 1e-10) - jnp.log(abl_probs + 1e-10))))
        per_removed.append({
            "n_layers_removed": n_remove,
            "kl_divergence": kl,
            "prediction_changed": abl_pred != base_pred,
        })
    return {"per_n_layers_removed": per_removed, "base_prediction": base_pred}


def layer_ablation_summary(model, tokens, position=-1):
    """Summary of layer ablation effects.

    Returns:
        dict with 'per_layer' list.
    """
    zero = layer_zero_ablation(model, tokens, position=position)
    per_layer = []
    for p in zero["per_layer"]:
        comp = component_ablation(model, tokens, layer=p["layer"], position=position)
        per_layer.append({
            "layer": p["layer"],
            "kl_divergence": p["kl_divergence"],
            "more_important": comp["more_important"],
        })
    return {"per_layer": per_layer}
