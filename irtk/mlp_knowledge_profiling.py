"""MLP knowledge profiling: profile what knowledge MLP neurons encode."""

import jax
import jax.numpy as jnp
from irtk import HookedTransformer


def neuron_vocabulary_profile(model: HookedTransformer, layer: int,
                               top_k: int = 10) -> dict:
    """Which vocabulary tokens does each neuron most promote/suppress?"""
    W_out = model.blocks[layer].mlp.W_out  # [d_mlp, d_model]
    W_U = model.unembed.W_U  # [d_model, d_vocab]
    d_mlp = W_out.shape[0]

    # Find top-norm neurons
    out_norms = jnp.linalg.norm(W_out, axis=1)
    top_neurons = jnp.argsort(out_norms)[-top_k:][::-1]

    per_neuron = []
    for idx in top_neurons:
        idx = int(idx)
        logits = W_out[idx] @ W_U  # [d_vocab]
        top_token = int(jnp.argmax(logits))
        bottom_token = int(jnp.argmin(logits))

        per_neuron.append({
            'neuron': idx,
            'output_norm': float(out_norms[idx]),
            'top_promoted': top_token,
            'top_promoted_logit': float(logits[top_token]),
            'top_suppressed': bottom_token,
            'top_suppressed_logit': float(logits[bottom_token]),
            'logit_range': float(logits[top_token] - logits[bottom_token]),
        })

    return {
        'layer': layer,
        'per_neuron': per_neuron,
    }


def neuron_selectivity_profile(model: HookedTransformer, tokens: jnp.ndarray,
                                layer: int, top_k: int = 10) -> dict:
    """Which neurons activate selectively (on few tokens)?"""
    _, cache = model.run_with_cache(tokens)
    mlp_post = cache[f'blocks.{layer}.mlp.hook_post']  # [seq, d_mlp]
    d_mlp = mlp_post.shape[1]
    seq_len = tokens.shape[0]

    # For each neuron: what fraction of positions is it active?
    active = (mlp_post > 0).astype(jnp.float32)  # [seq, d_mlp]
    activation_rate = jnp.mean(active, axis=0)  # [d_mlp]
    mean_activation = jnp.mean(mlp_post, axis=0)  # [d_mlp]

    # Most selective (lowest activation rate among active neurons)
    # Find neurons that fire at least once
    ever_active = activation_rate > 0
    selectivity = jnp.where(ever_active, 1 - activation_rate, 2.0)  # high = selective
    top_indices = jnp.argsort(selectivity)[-top_k:][::-1]

    per_neuron = []
    for idx in top_indices:
        idx = int(idx)
        if float(selectivity[idx]) > 1.5:
            continue  # never active
        per_neuron.append({
            'neuron': idx,
            'activation_rate': float(activation_rate[idx]),
            'mean_activation': float(mean_activation[idx]),
            'selectivity': float(selectivity[idx]),
            'is_selective': float(activation_rate[idx]) < 0.3,
        })

    return {
        'layer': layer,
        'n_ever_active': int(jnp.sum(ever_active)),
        'n_dead': d_mlp - int(jnp.sum(ever_active)),
        'per_neuron': per_neuron,
    }


def neuron_position_specificity(model: HookedTransformer, tokens: jnp.ndarray,
                                 layer: int) -> dict:
    """Do neurons activate differently by position?"""
    _, cache = model.run_with_cache(tokens)
    mlp_post = cache[f'blocks.{layer}.mlp.hook_post']  # [seq, d_mlp]
    seq_len = tokens.shape[0]
    d_mlp = mlp_post.shape[1]

    per_position = []
    for pos in range(seq_len):
        activations = mlp_post[pos]  # [d_mlp]
        n_active = int(jnp.sum(activations > 0))
        mean_act = float(jnp.mean(activations))
        max_act = float(jnp.max(activations))
        top_neuron = int(jnp.argmax(activations))

        per_position.append({
            'position': pos,
            'n_active': n_active,
            'sparsity': 1 - n_active / d_mlp,
            'mean_activation': mean_act,
            'max_activation': max_act,
            'top_neuron': top_neuron,
        })

    return {
        'layer': layer,
        'per_position': per_position,
    }


def neuron_cooperation_profile(model: HookedTransformer, tokens: jnp.ndarray,
                                layer: int, sample_size: int = 20) -> dict:
    """Do certain neurons consistently co-activate?"""
    _, cache = model.run_with_cache(tokens)
    mlp_post = cache[f'blocks.{layer}.mlp.hook_post']  # [seq, d_mlp]
    d_mlp = mlp_post.shape[1]
    n = min(sample_size, d_mlp)

    # Binary activation pattern
    active = (mlp_post[:, :n] > 0).astype(jnp.float32)  # [seq, n]

    # Co-activation rate
    coact = (active.T @ active) / tokens.shape[0]  # [n, n]
    diag = jnp.diag(coact)

    # Normalize to get Jaccard-like similarity
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            union = float(diag[i] + diag[j] - coact[i, j])
            if union > 0:
                jaccard = float(coact[i, j]) / union
            else:
                jaccard = 0.0
            if jaccard > 0.3:
                pairs.append({
                    'neuron_a': i,
                    'neuron_b': j,
                    'coactivation_rate': float(coact[i, j]),
                    'jaccard': jaccard,
                })

    return {
        'layer': layer,
        'n_sampled': n,
        'n_cooperative_pairs': len(pairs),
        'pairs': pairs[:20],  # limit output
    }


def layer_knowledge_summary(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Cross-layer summary of MLP knowledge encoding."""
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    W_U = model.unembed.W_U

    per_layer = []
    for layer in range(n_layers):
        mlp_post = cache[f'blocks.{layer}.mlp.hook_post']  # [seq, d_mlp]
        W_out = model.blocks[layer].mlp.W_out

        n_active = int(jnp.mean(jnp.sum(mlp_post > 0, axis=1)))
        mean_act = float(jnp.mean(jnp.abs(mlp_post)))

        # Logit impact: mean absolute logit from MLP output
        mlp_out = cache[f'blocks.{layer}.hook_mlp_out']  # [seq, d_model]
        logit_impact = float(jnp.mean(jnp.abs(mlp_out @ W_U)))

        per_layer.append({
            'layer': layer,
            'mean_active_neurons': n_active,
            'mean_activation_magnitude': mean_act,
            'logit_impact': logit_impact,
        })

    return {
        'per_layer': per_layer,
    }
