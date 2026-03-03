"""MLP gradient flow: how gradients propagate through MLP layers."""

import jax
import jax.numpy as jnp
from irtk.hooked_transformer import HookedTransformer


def _get_loss(model, tokens):
    logits = model(tokens)
    targets = tokens[1:]
    log_probs = jax.nn.log_softmax(logits[:-1], axis=-1)
    loss = -jnp.mean(log_probs[jnp.arange(len(targets)), targets])
    return loss


def mlp_input_gradient(model: HookedTransformer, tokens: jnp.ndarray,
                          layer: int = 0) -> dict:
    """Gradient of loss with respect to MLP input (pre-activation).

    Shows which dimensions of MLP input are most important.
    """
    _, cache = model.run_with_cache(tokens)
    pre = cache[("pre", layer)]  # [seq, d_mlp]

    def loss_fn(pre_val):
        # Forward with modified pre
        _, c = model.run_with_cache(tokens)
        post = jax.nn.gelu(pre_val)
        out = post @ model.blocks[layer].mlp.W_out + model.blocks[layer].mlp.b_out
        return jnp.mean(out ** 2)  # proxy loss

    grad = jax.grad(loss_fn)(pre)
    grad_norms = jnp.sqrt(jnp.sum(grad ** 2, axis=-1))  # [seq]

    per_position = []
    for pos in range(len(tokens)):
        per_position.append({
            "position": pos,
            "grad_norm": float(grad_norms[pos]),
        })
    return {
        "layer": layer,
        "per_position": per_position,
        "mean_grad_norm": float(jnp.mean(grad_norms)),
    }


def mlp_weight_gradient_norms(model: HookedTransformer, tokens: jnp.ndarray,
                                 layer: int = 0) -> dict:
    """Gradient norms for MLP weight matrices."""
    grads = jax.grad(_get_loss)(model, tokens)
    mlp_grads = grads.blocks[layer].mlp

    w_in_grad = float(jnp.sqrt(jnp.sum(mlp_grads.W_in ** 2)))
    w_out_grad = float(jnp.sqrt(jnp.sum(mlp_grads.W_out ** 2)))
    b_in_grad = float(jnp.sqrt(jnp.sum(mlp_grads.b_in ** 2)))
    b_out_grad = float(jnp.sqrt(jnp.sum(mlp_grads.b_out ** 2)))

    return {
        "layer": layer,
        "W_in_grad_norm": w_in_grad,
        "W_out_grad_norm": w_out_grad,
        "b_in_grad_norm": b_in_grad,
        "b_out_grad_norm": b_out_grad,
        "total_grad_norm": w_in_grad + w_out_grad + b_in_grad + b_out_grad,
    }


def mlp_neuron_gradient_profile(model: HookedTransformer, tokens: jnp.ndarray,
                                   layer: int = 0, top_k: int = 10) -> dict:
    """Per-neuron gradient magnitude through W_out columns."""
    grads = jax.grad(_get_loss)(model, tokens)
    W_out_grad = grads.blocks[layer].mlp.W_out  # [d_mlp, d_model]

    neuron_grads = jnp.sqrt(jnp.sum(W_out_grad ** 2, axis=-1))  # [d_mlp]
    top_indices = jnp.argsort(-neuron_grads)[:top_k]

    top_neurons = []
    for idx in top_indices:
        idx_int = int(idx)
        top_neurons.append({
            "neuron": idx_int,
            "grad_norm": float(neuron_grads[idx_int]),
        })
    return {
        "layer": layer,
        "top_neurons": top_neurons,
        "mean_neuron_grad": float(jnp.mean(neuron_grads)),
        "max_neuron_grad": float(jnp.max(neuron_grads)),
    }


def mlp_gradient_sparsity(model: HookedTransformer, tokens: jnp.ndarray,
                             layer: int = 0, threshold: float = 0.01) -> dict:
    """Sparsity of MLP gradients: fraction of neurons with significant gradients."""
    grads = jax.grad(_get_loss)(model, tokens)
    W_out_grad = grads.blocks[layer].mlp.W_out

    neuron_grads = jnp.sqrt(jnp.sum(W_out_grad ** 2, axis=-1))
    max_grad = float(jnp.max(neuron_grads))
    thresh = threshold * max_grad
    n_significant = int(jnp.sum(neuron_grads > thresh))
    total = neuron_grads.shape[0]

    return {
        "layer": layer,
        "n_significant": n_significant,
        "total_neurons": total,
        "gradient_sparsity": 1.0 - n_significant / total,
        "threshold_used": float(thresh),
    }


def mlp_gradient_flow_summary(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Cross-layer MLP gradient flow summary."""
    per_layer = []
    for layer in range(model.cfg.n_layers):
        wg = mlp_weight_gradient_norms(model, tokens, layer)
        sp = mlp_gradient_sparsity(model, tokens, layer)
        per_layer.append({
            "layer": layer,
            "total_grad_norm": wg["total_grad_norm"],
            "gradient_sparsity": sp["gradient_sparsity"],
        })
    return {"per_layer": per_layer}
