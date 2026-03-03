"""Layer functional profiling: characterize what each layer does."""

import jax
import jax.numpy as jnp
from irtk import HookedTransformer


def layer_prediction_impact(model: HookedTransformer, tokens: jnp.ndarray,
                             position: int = -1) -> dict:
    """How much does each layer change the prediction?"""
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    if position < 0:
        position = tokens.shape[0] + position

    W_U = model.unembed.W_U  # [d_model, d_vocab]

    per_layer = []
    for layer in range(n_layers):
        resid_pre = cache[f'blocks.{layer}.hook_resid_pre'][position] if layer > 0 else (
            cache['hook_embed'][position] + cache['hook_pos_embed'][position]
        )
        resid_post = cache[f'blocks.{layer}.hook_resid_post'][position]

        logits_pre = resid_pre @ W_U
        logits_post = resid_post @ W_U

        top_pre = int(jnp.argmax(logits_pre))
        top_post = int(jnp.argmax(logits_post))

        # KL-like divergence (simplified)
        probs_pre = jax.nn.softmax(logits_pre)
        probs_post = jax.nn.softmax(logits_post)
        kl = float(jnp.sum(probs_post * jnp.log((probs_post + 1e-10) / (probs_pre + 1e-10))))

        per_layer.append({
            'layer': layer,
            'top_token_before': top_pre,
            'top_token_after': top_post,
            'prediction_changed': top_pre != top_post,
            'kl_divergence': kl,
        })

    return {
        'position': position,
        'per_layer': per_layer,
    }


def layer_computation_type(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Characterize each layer: attention-dominated vs MLP-dominated."""
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers

    per_layer = []
    for layer in range(n_layers):
        attn_out = cache[f'blocks.{layer}.hook_attn_out']  # [seq, d_model]
        mlp_out = cache[f'blocks.{layer}.hook_mlp_out']  # [seq, d_model]

        attn_norm = float(jnp.mean(jnp.linalg.norm(attn_out, axis=1)))
        mlp_norm = float(jnp.mean(jnp.linalg.norm(mlp_out, axis=1)))
        total = attn_norm + mlp_norm + 1e-10

        if attn_norm > 2 * mlp_norm:
            comp_type = 'attention_dominated'
        elif mlp_norm > 2 * attn_norm:
            comp_type = 'mlp_dominated'
        else:
            comp_type = 'balanced'

        per_layer.append({
            'layer': layer,
            'attn_norm': attn_norm,
            'mlp_norm': mlp_norm,
            'attn_fraction': attn_norm / total,
            'computation_type': comp_type,
        })

    return {
        'per_layer': per_layer,
    }


def layer_information_gain(model: HookedTransformer, tokens: jnp.ndarray,
                            position: int = -1) -> dict:
    """How much new information does each layer add?"""
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    if position < 0:
        position = tokens.shape[0] + position

    embed = cache['hook_embed'][position] + cache['hook_pos_embed'][position]
    prev_dir = embed / (jnp.linalg.norm(embed) + 1e-10)

    per_layer = []
    for layer in range(n_layers):
        resid = cache[f'blocks.{layer}.hook_resid_post'][position]
        curr_dir = resid / (jnp.linalg.norm(resid) + 1e-10)

        # New info = component orthogonal to previous direction
        parallel = float(jnp.dot(resid, prev_dir))
        perp = float(jnp.sqrt(jnp.maximum(jnp.dot(resid, resid) - parallel ** 2, 0)))

        per_layer.append({
            'layer': layer,
            'parallel_component': parallel,
            'perpendicular_component': perp,
            'new_info_fraction': perp / (float(jnp.linalg.norm(resid)) + 1e-10),
        })

        prev_dir = curr_dir

    return {
        'position': position,
        'per_layer': per_layer,
    }


def layer_redundancy_analysis(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Which layers produce similar outputs to other layers?"""
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers

    # Collect per-layer outputs (attn + mlp)
    layer_outputs = []
    for layer in range(n_layers):
        attn = cache[f'blocks.{layer}.hook_attn_out']  # [seq, d_model]
        mlp = cache[f'blocks.{layer}.hook_mlp_out']
        combined = attn + mlp  # [seq, d_model]
        flat = combined.reshape(-1)
        layer_outputs.append(flat / (jnp.linalg.norm(flat) + 1e-10))

    layer_outputs = jnp.stack(layer_outputs)
    sim_matrix = layer_outputs @ layer_outputs.T

    pairs = []
    for i in range(n_layers):
        for j in range(i + 1, n_layers):
            pairs.append({
                'layer_a': i,
                'layer_b': j,
                'similarity': float(sim_matrix[i, j]),
                'is_redundant': float(sim_matrix[i, j]) > 0.8,
            })

    return {
        'pairs': pairs,
        'n_redundant_pairs': sum(1 for p in pairs if p['is_redundant']),
    }


def layer_functional_summary(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """One-line functional summary of each layer."""
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    W_U = model.unembed.W_U

    per_layer = []
    for layer in range(n_layers):
        attn_out = cache[f'blocks.{layer}.hook_attn_out']
        mlp_out = cache[f'blocks.{layer}.hook_mlp_out']
        attn_norm = float(jnp.mean(jnp.linalg.norm(attn_out, axis=1)))
        mlp_norm = float(jnp.mean(jnp.linalg.norm(mlp_out, axis=1)))

        # Attention entropy
        pattern = cache[f'blocks.{layer}.attn.hook_pattern']
        ents = []
        for h in range(n_heads):
            ent = -jnp.sum(pattern[h] * jnp.log(pattern[h] + 1e-10), axis=1)
            ents.append(float(jnp.mean(ent)))
        mean_entropy = sum(ents) / len(ents)

        # Logit impact
        combined = attn_out + mlp_out
        logit_impact = float(jnp.mean(jnp.abs(combined @ W_U)))

        per_layer.append({
            'layer': layer,
            'attn_magnitude': attn_norm,
            'mlp_magnitude': mlp_norm,
            'mean_attn_entropy': mean_entropy,
            'logit_impact': logit_impact,
        })

    return {
        'per_layer': per_layer,
    }
