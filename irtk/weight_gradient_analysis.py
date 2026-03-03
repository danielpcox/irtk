"""Weight gradient analysis: which weights matter most for specific predictions."""

import jax
import jax.numpy as jnp
from irtk.hooked_transformer import HookedTransformer


def weight_sensitivity_profile(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Sensitivity of the loss to each weight matrix.

    Higher gradient norm = more influence on the prediction.
    """
    def loss_fn(m):
        logits = m(tokens)
        # Cross-entropy on next-token prediction
        targets = jnp.concatenate([tokens[1:], jnp.array([0])])
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        return -jnp.mean(log_probs[jnp.arange(len(targets)), targets])

    grads = jax.grad(loss_fn)(model)
    leaves, treedef = jax.tree.flatten(grads)
    orig_leaves, _ = jax.tree.flatten(model)

    per_param = []
    for i, (grad_leaf, param_leaf) in enumerate(zip(leaves, orig_leaves)):
        if isinstance(grad_leaf, jnp.ndarray) and grad_leaf.dtype == jnp.float32:
            grad_norm = float(jnp.sqrt(jnp.sum(grad_leaf ** 2)))
            param_norm = float(jnp.sqrt(jnp.sum(param_leaf ** 2)).clip(1e-8))
            per_param.append({
                "param_index": i,
                "grad_norm": grad_norm,
                "param_norm": param_norm,
                "relative_sensitivity": grad_norm / param_norm,
            })

    per_param.sort(key=lambda x: x["grad_norm"], reverse=True)
    return {
        "n_params": len(per_param),
        "top_sensitive": per_param[:10],
        "mean_grad_norm": sum(p["grad_norm"] for p in per_param) / max(len(per_param), 1),
    }


def layer_weight_gradient_norms(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Gradient norms organized by layer.

    Shows which layers have the strongest gradients.
    """
    def loss_fn(m):
        logits = m(tokens)
        targets = jnp.concatenate([tokens[1:], jnp.array([0])])
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        return -jnp.mean(log_probs[jnp.arange(len(targets)), targets])

    grads = jax.grad(loss_fn)(model)

    per_layer = []
    for layer in range(model.cfg.n_layers):
        block_grads = grads.blocks[layer]
        block_leaves = jax.tree.leaves(block_grads)
        total_grad_norm = 0.0
        n_params = 0
        for leaf in block_leaves:
            if isinstance(leaf, jnp.ndarray) and leaf.dtype == jnp.float32:
                total_grad_norm += float(jnp.sum(leaf ** 2))
                n_params += leaf.size
        per_layer.append({
            "layer": layer,
            "grad_norm": float(jnp.sqrt(total_grad_norm)),
            "n_params": n_params,
        })

    norms = [p["grad_norm"] for p in per_layer]
    total = sum(norms)
    for p, n in zip(per_layer, norms):
        p["fraction"] = n / max(total, 1e-8)

    return {
        "per_layer": per_layer,
        "most_sensitive_layer": max(range(len(per_layer)),
                                     key=lambda i: per_layer[i]["grad_norm"]),
    }


def attention_weight_gradients(model: HookedTransformer, tokens: jnp.ndarray,
                                 layer: int = 0) -> dict:
    """Gradient norms for attention weight matrices (Q, K, V, O).

    Shows which projections are most important for the prediction.
    """
    def loss_fn(m):
        logits = m(tokens)
        targets = jnp.concatenate([tokens[1:], jnp.array([0])])
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        return -jnp.mean(log_probs[jnp.arange(len(targets)), targets])

    grads = jax.grad(loss_fn)(model)
    attn = grads.blocks[layer].attn

    results = {}
    for name in ["W_Q", "W_K", "W_V", "W_O"]:
        w_grad = getattr(attn, name)
        results[name] = float(jnp.sqrt(jnp.sum(w_grad ** 2)))

    total = sum(results.values())
    fractions = {k: v / max(total, 1e-8) for k, v in results.items()}

    return {
        "layer": layer,
        "grad_norms": results,
        "fractions": fractions,
        "dominant_matrix": max(results, key=results.get),
    }


def mlp_weight_gradients(model: HookedTransformer, tokens: jnp.ndarray,
                           layer: int = 0) -> dict:
    """Gradient norms for MLP weight matrices.

    Shows which MLP weights are most important.
    """
    def loss_fn(m):
        logits = m(tokens)
        targets = jnp.concatenate([tokens[1:], jnp.array([0])])
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        return -jnp.mean(log_probs[jnp.arange(len(targets)), targets])

    grads = jax.grad(loss_fn)(model)
    mlp = grads.blocks[layer].mlp

    results = {}
    for name in ["W_in", "W_out"]:
        w_grad = getattr(mlp, name)
        results[name] = float(jnp.sqrt(jnp.sum(w_grad ** 2)))

    total = sum(results.values())
    fractions = {k: v / max(total, 1e-8) for k, v in results.items()}

    return {
        "layer": layer,
        "grad_norms": results,
        "fractions": fractions,
        "dominant_matrix": max(results, key=results.get),
    }


def weight_gradient_summary(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Cross-layer weight gradient summary."""
    layer_grads = layer_weight_gradient_norms(model, tokens)
    per_layer = []
    for p in layer_grads["per_layer"]:
        per_layer.append({
            "layer": p["layer"],
            "grad_norm": p["grad_norm"],
            "fraction": p["fraction"],
        })
    return {
        "per_layer": per_layer,
        "most_sensitive_layer": layer_grads["most_sensitive_layer"],
    }
