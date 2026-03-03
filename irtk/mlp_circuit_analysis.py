"""MLP circuit analysis.

Analyze MLP circuits: neuron-to-logit paths, gating patterns,
feature extraction vs knowledge storage, and MLP contribution
decomposition.
"""

import jax
import jax.numpy as jnp


def neuron_to_logit_paths(model, tokens, layer, top_k=5):
    """Trace paths from MLP neurons to specific logit outputs.

    Args:
        model: HookedTransformer
        tokens: input token IDs
        layer: layer index
        top_k: top neurons to analyze

    Returns:
        dict with neuron-to-logit path analysis.
    """
    _, cache = model.run_with_cache(tokens)
    post = cache[f'blocks.{layer}.mlp.hook_post']  # [seq, d_mlp]
    W_out = model.blocks[layer].mlp.W_out  # [d_mlp, d_model]
    W_U = model.unembed.W_U  # [d_model, d_vocab]

    # Logit contribution per neuron = post * (W_out @ W_U)
    logit_weights = W_out @ W_U  # [d_mlp, d_vocab]

    # For last position
    pos = -1
    activations = post[pos]  # [d_mlp]
    logit_contribs = activations[:, None] * logit_weights  # [d_mlp, d_vocab]

    # Top neurons by absolute logit contribution
    total_abs = jnp.sum(jnp.abs(logit_contribs), axis=-1)  # [d_mlp]
    top_neurons = jnp.argsort(-total_abs)[:top_k]

    results = []
    for n in top_neurons:
        n = int(n)
        contrib = logit_contribs[n]  # [d_vocab]
        top_promoted = jnp.argsort(-contrib)[:3]
        top_demoted = jnp.argsort(contrib)[:3]
        results.append({
            'neuron': n,
            'activation': float(activations[n]),
            'total_abs_logit': float(total_abs[n]),
            'top_promoted': [{'token': int(t), 'logit': float(contrib[t])} for t in top_promoted],
            'top_demoted': [{'token': int(t), 'logit': float(contrib[t])} for t in top_demoted],
        })

    return {'layer': layer, 'position': pos, 'neurons': results}


def mlp_knowledge_vs_feature(model, tokens, layer):
    """Classify MLP neurons as knowledge-storing vs feature-extracting.

    Knowledge neurons: activate sparsely and contribute to specific tokens.
    Feature neurons: activate broadly and carry abstract features.

    Args:
        model: HookedTransformer
        tokens: input token IDs
        layer: layer index

    Returns:
        dict with neuron classification.
    """
    _, cache = model.run_with_cache(tokens)
    post = cache[f'blocks.{layer}.mlp.hook_post']  # [seq, d_mlp]
    W_out = model.blocks[layer].mlp.W_out  # [d_mlp, d_model]
    W_U = model.unembed.W_U  # [d_model, d_vocab]

    logit_weights = W_out @ W_U  # [d_mlp, d_vocab]
    d_mlp = post.shape[1]

    knowledge_neurons = []
    feature_neurons = []

    for n in range(d_mlp):
        acts = post[:, n]  # [seq]
        # Sparsity: fraction of positions with non-trivial activation
        active_frac = float(jnp.mean(jnp.abs(acts) > 0.01))

        # Logit specificity: how concentrated is the logit effect?
        logit_effect = logit_weights[n]  # [d_vocab]
        logit_entropy = -float(jnp.sum(
            jax.nn.softmax(jnp.abs(logit_effect)) *
            jnp.log(jnp.maximum(jax.nn.softmax(jnp.abs(logit_effect)), 1e-10))))

        info = {
            'neuron': n,
            'active_fraction': active_frac,
            'logit_entropy': logit_entropy,
            'mean_activation': float(jnp.mean(jnp.abs(acts))),
        }

        # Knowledge: sparse activation + focused logit effect
        if active_frac < 0.5 and logit_entropy < jnp.log(jnp.array(10.0)):
            knowledge_neurons.append(info)
        else:
            feature_neurons.append(info)

    return {
        'layer': layer,
        'n_knowledge': len(knowledge_neurons),
        'n_feature': len(feature_neurons),
        'knowledge_fraction': len(knowledge_neurons) / max(d_mlp, 1),
        'knowledge_neurons': knowledge_neurons[:10],
        'feature_neurons': feature_neurons[:10],
    }


def mlp_contribution_decomposition(model, tokens, layer):
    """Decompose MLP contribution into per-neuron terms.

    Args:
        model: HookedTransformer
        tokens: input token IDs
        layer: layer index

    Returns:
        dict with per-neuron contribution to the residual stream.
    """
    _, cache = model.run_with_cache(tokens)
    post = cache[f'blocks.{layer}.mlp.hook_post']  # [seq, d_mlp]
    W_out = model.blocks[layer].mlp.W_out  # [d_mlp, d_model]
    d_mlp = post.shape[1]

    # Per-neuron contribution: activation * W_out_row
    neuron_norms = []
    for n in range(d_mlp):
        act = post[:, n]  # [seq]
        w = W_out[n]  # [d_model]
        contrib_norm = float(jnp.mean(jnp.abs(act)) * jnp.linalg.norm(w))
        neuron_norms.append(contrib_norm)

    neuron_norms = jnp.array(neuron_norms)
    total = float(jnp.sum(neuron_norms))

    # Top contributors
    top_indices = jnp.argsort(-neuron_norms)[:10]
    top_neurons = [{'neuron': int(i), 'contribution': float(neuron_norms[i]),
                    'fraction': float(neuron_norms[i] / max(total, 1e-10))}
                   for i in top_indices]

    return {
        'layer': layer,
        'top_neurons': top_neurons,
        'total_contribution': total,
        'n_neurons': d_mlp,
        'gini': float(_gini(neuron_norms)),
    }


def _gini(values):
    sorted_vals = jnp.sort(values)
    n = len(sorted_vals)
    index = jnp.arange(1, n + 1, dtype=jnp.float32)
    return float((2 * jnp.sum(index * sorted_vals) / (n * jnp.sum(sorted_vals) + 1e-10)) - (n + 1) / n)


def mlp_nonlinearity_effect(model, tokens, layer):
    """Measure the effect of the nonlinearity in the MLP.

    Compares pre-activation and post-activation to quantify
    how much the activation function changes the output.

    Args:
        model: HookedTransformer
        tokens: input token IDs
        layer: layer index

    Returns:
        dict with nonlinearity effect analysis.
    """
    _, cache = model.run_with_cache(tokens)
    pre = cache[f'blocks.{layer}.mlp.hook_pre']  # [seq, d_mlp]
    post = cache[f'blocks.{layer}.mlp.hook_post']  # [seq, d_mlp]

    # How much does the nonlinearity change things?
    pre_norm = float(jnp.mean(jnp.linalg.norm(pre, axis=-1)))
    post_norm = float(jnp.mean(jnp.linalg.norm(post, axis=-1)))
    diff_norm = float(jnp.mean(jnp.linalg.norm(post - pre, axis=-1)))

    # Cosine between pre and post
    cos = jnp.sum(pre * post, axis=-1) / jnp.maximum(
        jnp.linalg.norm(pre, axis=-1) * jnp.linalg.norm(post, axis=-1), 1e-10)
    mean_cos = float(jnp.mean(cos))

    # Sparsity induced by nonlinearity
    pre_zero = float(jnp.mean(jnp.abs(pre) < 0.01))
    post_zero = float(jnp.mean(jnp.abs(post) < 0.01))

    return {
        'layer': layer,
        'pre_norm': pre_norm,
        'post_norm': post_norm,
        'change_norm': diff_norm,
        'pre_post_cosine': mean_cos,
        'pre_near_zero': pre_zero,
        'post_near_zero': post_zero,
        'sparsity_increase': post_zero - pre_zero,
    }


def mlp_layer_comparison(model, tokens):
    """Compare MLP behavior across all layers.

    Returns:
        dict with per-layer MLP statistics.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers

    results = []
    for l in range(n_layers):
        mlp_out = cache[f'blocks.{l}.hook_mlp_out']  # [seq, d_model]
        post = cache[f'blocks.{l}.mlp.hook_post']

        output_norm = float(jnp.mean(jnp.linalg.norm(mlp_out, axis=-1)))
        activation_sparsity = float(jnp.mean(jnp.abs(post) < 0.01))
        mean_activation = float(jnp.mean(jnp.abs(post)))

        results.append({
            'layer': l,
            'output_norm': output_norm,
            'activation_sparsity': activation_sparsity,
            'mean_activation': mean_activation,
        })

    return {
        'per_layer': results,
        'most_active_layer': max(results, key=lambda r: r['output_norm'])['layer'],
    }
