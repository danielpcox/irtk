"""MLP contribution profiling: analyze how MLPs transform the residual stream."""

import jax
import jax.numpy as jnp
from irtk import HookedTransformer


def mlp_residual_contribution(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """How much does each MLP contribute to the residual stream?

    Measures norm and direction of MLP output relative to residual.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers

    per_layer = []
    for layer in range(n_layers):
        mlp_out = cache[f'blocks.{layer}.hook_mlp_out']  # [seq, d_model]
        resid_pre = cache[f'blocks.{layer}.hook_resid_mid']  # [seq, d_model]

        mlp_norm = float(jnp.mean(jnp.linalg.norm(mlp_out, axis=-1)))
        resid_norm = float(jnp.mean(jnp.linalg.norm(resid_pre, axis=-1)))
        fraction = mlp_norm / (resid_norm + 1e-10)

        # Alignment
        cos = float(jnp.mean(
            jnp.sum(mlp_out * resid_pre, axis=-1) /
            (jnp.linalg.norm(mlp_out, axis=-1) * jnp.linalg.norm(resid_pre, axis=-1) + 1e-10)
        ))

        per_layer.append({
            'layer': layer,
            'mlp_norm': mlp_norm,
            'residual_norm': resid_norm,
            'fraction_of_residual': fraction,
            'alignment_to_residual': cos,
            'is_constructive': cos > 0,
        })

    return {
        'per_layer': per_layer,
        'n_constructive': sum(1 for p in per_layer if p['is_constructive']),
    }


def mlp_logit_effect(model: HookedTransformer, tokens: jnp.ndarray, position: int = -1) -> dict:
    """What tokens does each MLP layer promote/suppress?

    Projects MLP output through the unembedding.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    seq_len = tokens.shape[0]
    pos = position if position >= 0 else seq_len + position

    W_U = model.unembed.W_U

    per_layer = []
    for layer in range(n_layers):
        mlp_out = cache[f'blocks.{layer}.hook_mlp_out'][pos]  # [d_model]
        logit_contrib = mlp_out @ W_U  # [d_vocab]

        top_5 = jnp.argsort(logit_contrib)[-5:][::-1]
        bottom_5 = jnp.argsort(logit_contrib)[:5]

        per_layer.append({
            'layer': layer,
            'logit_norm': float(jnp.linalg.norm(logit_contrib)),
            'top_promoted': [{'token': int(t), 'logit': float(logit_contrib[t])} for t in top_5],
            'top_suppressed': [{'token': int(t), 'logit': float(logit_contrib[t])} for t in bottom_5],
        })

    per_layer.sort(key=lambda x: x['logit_norm'], reverse=True)

    return {
        'position': pos,
        'per_layer': per_layer,
    }


def mlp_position_profile(model: HookedTransformer, tokens: jnp.ndarray, layer: int) -> dict:
    """How does the MLP treat different positions?

    Measures output norm and direction per position.
    """
    _, cache = model.run_with_cache(tokens)
    seq_len = tokens.shape[0]

    mlp_out = cache[f'blocks.{layer}.hook_mlp_out']  # [seq, d_model]

    # Mean direction
    mean_out = jnp.mean(mlp_out, axis=0)
    mean_dir = mean_out / (jnp.linalg.norm(mean_out) + 1e-10)

    per_position = []
    for pos in range(seq_len):
        norm = float(jnp.linalg.norm(mlp_out[pos]))
        direction = mlp_out[pos] / (jnp.linalg.norm(mlp_out[pos]) + 1e-10)
        align_to_mean = float(jnp.dot(direction, mean_dir))

        per_position.append({
            'position': pos,
            'norm': norm,
            'alignment_to_mean': align_to_mean,
            'is_aligned': align_to_mean > 0.5,
        })

    norms = [p['norm'] for p in per_position]
    cv = (sum((n - sum(norms)/len(norms))**2 for n in norms) / len(norms))**0.5 / (sum(norms)/len(norms) + 1e-10)

    return {
        'layer': layer,
        'per_position': per_position,
        'norm_cv': cv,
        'is_position_uniform': cv < 0.3,
    }


def mlp_layer_comparison(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Compare MLP behavior across layers.

    Shows how MLP outputs evolve: norm trends, direction shifts.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers

    mean_dirs = []
    per_layer = []

    for layer in range(n_layers):
        mlp_out = cache[f'blocks.{layer}.hook_mlp_out']
        mean_out = jnp.mean(mlp_out, axis=0)
        mean_dir = mean_out / (jnp.linalg.norm(mean_out) + 1e-10)
        mean_dirs.append(mean_dir)

        norm = float(jnp.mean(jnp.linalg.norm(mlp_out, axis=-1)))
        per_layer.append({
            'layer': layer,
            'mean_norm': norm,
        })

    # Pairwise direction similarity
    for i in range(n_layers):
        sims = []
        for j in range(n_layers):
            if i != j:
                sims.append(float(jnp.dot(mean_dirs[i], mean_dirs[j])))
        per_layer[i]['mean_similarity_to_others'] = sum(sims) / len(sims) if sims else 0
        per_layer[i]['is_unique'] = per_layer[i]['mean_similarity_to_others'] < 0.3

    return {
        'per_layer': per_layer,
        'n_unique': sum(1 for p in per_layer if p['is_unique']),
    }


def mlp_neuron_contribution_ranking(model: HookedTransformer, tokens: jnp.ndarray, layer: int, position: int = -1, top_k: int = 10) -> dict:
    """Rank neurons by their contribution to the output at a position.

    Uses post-activation magnitude times output weight norm.
    """
    _, cache = model.run_with_cache(tokens)
    seq_len = tokens.shape[0]
    pos = position if position >= 0 else seq_len + position

    post_key = f'blocks.{layer}.mlp.hook_post'
    if post_key not in cache:
        return {'layer': layer, 'position': pos, 'per_neuron': [], 'n_active': 0}

    post = cache[post_key]  # [seq, d_mlp]
    W_out = model.blocks[layer].mlp.W_out  # [d_mlp, d_model]

    activations = post[pos]  # [d_mlp]
    n_neurons = activations.shape[0]

    # Contribution = activation * output weight norm
    out_norms = jnp.linalg.norm(W_out, axis=-1)  # [d_mlp]
    contributions = jnp.abs(activations) * out_norms

    top_indices = jnp.argsort(contributions)[-top_k:][::-1]

    per_neuron = []
    for idx in top_indices:
        idx_int = int(idx)
        per_neuron.append({
            'neuron': idx_int,
            'activation': float(activations[idx_int]),
            'output_norm': float(out_norms[idx_int]),
            'contribution': float(contributions[idx_int]),
            'is_active': float(activations[idx_int]) > 0,
        })

    n_active = int(jnp.sum(activations > 0))

    return {
        'layer': layer,
        'position': pos,
        'per_neuron': per_neuron,
        'n_active': n_active,
        'total_neurons': n_neurons,
    }
