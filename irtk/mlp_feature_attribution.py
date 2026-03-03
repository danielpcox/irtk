"""MLP feature attribution: what MLP neurons contribute to predictions.

Fine-grained analysis of how individual MLP neurons and neuron groups
contribute to model output:
- Per-neuron logit attribution
- Feature direction analysis
- Active neuron profiling
- Layer-wise MLP attribution
- Neuron cooperation patterns
"""

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from irtk.hooked_transformer import HookedTransformer


def neuron_logit_attribution(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    layer: int = 0,
    pos: int = -1,
    top_k: int = 10,
    target_token: Optional[int] = None,
) -> dict:
    """Attribute the logit of a target token to individual MLP neurons.

    Each neuron's contribution = activation * W_out row projected onto
    the unembedding direction.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] token IDs.
        layer: MLP layer to analyze.
        pos: Position to analyze.
        top_k: Number of top neurons to return.
        target_token: Token to attribute (default: top prediction).

    Returns:
        Dict with per-neuron logit attributions.
    """
    logits = model(tokens)
    _, cache = model.run_with_cache(tokens)

    if target_token is None:
        target_token = int(jnp.argmax(logits[pos]))

    # MLP post-activation
    post = np.array(cache[f'blocks.{layer}.mlp.hook_post'][pos])  # [d_mlp]
    W_out = np.array(model.blocks[layer].mlp.W_out)  # [d_mlp, d_model]
    W_U = np.array(model.unembed.W_U[:, target_token])  # [d_model]

    # Per-neuron contribution = activation * (W_out row dot W_U)
    logit_directions = W_out @ W_U  # [d_mlp]
    contributions = post * logit_directions  # [d_mlp]

    sorted_neurons = np.argsort(np.abs(contributions))[::-1]

    promoting = []
    suppressing = []
    for n in sorted_neurons[:top_k]:
        entry = {
            'neuron': int(n),
            'activation': round(float(post[n]), 4),
            'logit_contribution': round(float(contributions[n]), 4),
            'direction_alignment': round(float(logit_directions[n]), 4),
        }
        if contributions[n] > 0:
            promoting.append(entry)
        else:
            suppressing.append(entry)

    total_mlp_logit = float(np.sum(contributions))
    top_k_logit = float(np.sum(contributions[sorted_neurons[:top_k]]))

    return {
        'layer': layer,
        'target_token': target_token,
        'promoting': promoting,
        'suppressing': suppressing,
        'total_mlp_logit': round(total_mlp_logit, 4),
        'top_k_fraction': round(abs(top_k_logit / total_mlp_logit), 4) if abs(total_mlp_logit) > 1e-10 else 0.0,
    }


def active_neuron_profile(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    layer: int = 0,
    activation_threshold: float = 0.1,
) -> dict:
    """Profile which neurons are active and their statistics.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] token IDs.
        layer: MLP layer to analyze.
        activation_threshold: Minimum absolute activation to count as active.

    Returns:
        Dict with active neuron statistics.
    """
    _, cache = model.run_with_cache(tokens)

    post = np.array(cache[f'blocks.{layer}.mlp.hook_post'])  # [seq, d_mlp]
    seq_len, d_mlp = post.shape

    # Per-position active counts
    active_mask = np.abs(post) > activation_threshold
    n_active_per_pos = np.sum(active_mask, axis=1)
    n_active_per_neuron = np.sum(active_mask, axis=0)

    # Neurons that are active everywhere vs nowhere
    always_active = int(np.sum(n_active_per_neuron == seq_len))
    never_active = int(np.sum(n_active_per_neuron == 0))

    # Mean and max activations for active neurons
    abs_acts = np.abs(post)
    mean_activation = float(np.mean(abs_acts[active_mask])) if np.any(active_mask) else 0.0
    max_activation = float(np.max(abs_acts))

    return {
        'layer': layer,
        'd_mlp': d_mlp,
        'mean_active_per_position': round(float(np.mean(n_active_per_pos)), 1),
        'sparsity': round(1.0 - float(np.mean(active_mask)), 4),
        'always_active': always_active,
        'never_active': never_active,
        'mean_activation': round(mean_activation, 4),
        'max_activation': round(max_activation, 4),
    }


def mlp_layer_attribution(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    pos: int = -1,
    target_token: Optional[int] = None,
) -> dict:
    """Attribute the target logit to each MLP layer's total contribution.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] token IDs.
        pos: Position to analyze.
        target_token: Token to attribute (default: top prediction).

    Returns:
        Dict with per-layer MLP logit contributions.
    """
    logits = model(tokens)
    _, cache = model.run_with_cache(tokens)

    if target_token is None:
        target_token = int(jnp.argmax(logits[pos]))

    W_U = np.array(model.unembed.W_U[:, target_token])

    per_layer = []
    for l in range(model.cfg.n_layers):
        mlp_out = np.array(cache[f'blocks.{l}.hook_mlp_out'][pos])
        logit_contrib = float(np.dot(mlp_out, W_U))
        norm = float(np.linalg.norm(mlp_out))

        per_layer.append({
            'layer': l,
            'logit_contribution': round(logit_contrib, 4),
            'output_norm': round(norm, 4),
            'promotes': logit_contrib > 0,
        })

    total = sum(p['logit_contribution'] for p in per_layer)
    return {
        'target_token': target_token,
        'per_layer': per_layer,
        'total_mlp_logit': round(total, 4),
        'most_promoting_layer': max(per_layer, key=lambda x: x['logit_contribution'])['layer'],
        'most_suppressing_layer': min(per_layer, key=lambda x: x['logit_contribution'])['layer'],
    }


def neuron_feature_directions(
    model: HookedTransformer,
    layer: int = 0,
    top_k: int = 5,
) -> dict:
    """Analyze the output directions (features) that MLP neurons write.

    Each neuron writes a direction in d_model space via W_out.
    This function characterizes those directions.

    Args:
        model: HookedTransformer.
        layer: MLP layer to analyze.
        top_k: Number of top/bottom vocab tokens per neuron.

    Returns:
        Dict with neuron feature direction analysis.
    """
    W_out = np.array(model.blocks[layer].mlp.W_out)  # [d_mlp, d_model]
    W_U = np.array(model.unembed.W_U)  # [d_model, d_vocab]

    d_mlp = W_out.shape[0]

    # Neuron output norms
    norms = np.linalg.norm(W_out, axis=1)  # [d_mlp]

    # Project each neuron's output direction through unembedding
    # to see what vocabulary items it promotes
    projections = W_out @ W_U  # [d_mlp, d_vocab]

    per_neuron = []
    sorted_by_norm = np.argsort(norms)[::-1]
    for n in sorted_by_norm[:top_k]:
        proj = projections[int(n)]
        top_tokens = np.argsort(proj)[::-1][:top_k]
        bottom_tokens = np.argsort(proj)[:top_k]

        per_neuron.append({
            'neuron': int(n),
            'output_norm': round(float(norms[n]), 4),
            'top_promoted_tokens': [int(t) for t in top_tokens],
            'top_suppressed_tokens': [int(t) for t in bottom_tokens],
            'max_promotion': round(float(np.max(proj)), 4),
            'max_suppression': round(float(np.min(proj)), 4),
        })

    # Pairwise cosine similarity of top neurons
    top_neurons = sorted_by_norm[:top_k]
    neuron_dirs = W_out[top_neurons]
    neuron_norms = norms[top_neurons]

    return {
        'layer': layer,
        'd_mlp': d_mlp,
        'per_neuron': per_neuron,
        'mean_output_norm': round(float(np.mean(norms)), 4),
        'max_output_norm': round(float(np.max(norms)), 4),
    }


def neuron_cooperation(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    layer: int = 0,
    pos: int = -1,
    top_k: int = 5,
) -> dict:
    """Find neurons that cooperate (fire together and write in same direction).

    Args:
        model: HookedTransformer.
        tokens: [seq_len] token IDs.
        layer: MLP layer to analyze.
        pos: Position to analyze.
        top_k: Number of top cooperating pairs.

    Returns:
        Dict with neuron cooperation patterns.
    """
    _, cache = model.run_with_cache(tokens)

    post = np.array(cache[f'blocks.{layer}.mlp.hook_post'][pos])  # [d_mlp]
    W_out = np.array(model.blocks[layer].mlp.W_out)  # [d_mlp, d_model]

    # Find active neurons
    active = np.where(np.abs(post) > 0.01)[0]
    if len(active) < 2:
        return {
            'layer': layer,
            'n_active': len(active),
            'cooperating_pairs': [],
            'competing_pairs': [],
        }

    # Compute output directions for active neurons
    outputs = {}
    for n in active:
        out = post[int(n)] * W_out[int(n)]  # Scaled by activation
        outputs[int(n)] = out

    # Pairwise cosine of actual outputs
    cooperating = []
    competing = []
    active_list = list(outputs.keys())

    for i in range(min(len(active_list), 50)):  # Limit for efficiency
        for j in range(i + 1, min(len(active_list), 50)):
            n1, n2 = active_list[i], active_list[j]
            v1, v2 = outputs[n1], outputs[n2]
            norm1 = float(np.linalg.norm(v1))
            norm2 = float(np.linalg.norm(v2))
            if norm1 > 1e-10 and norm2 > 1e-10:
                cos = float(np.dot(v1, v2) / (norm1 * norm2))
                entry = {
                    'neuron_a': n1,
                    'neuron_b': n2,
                    'cosine_similarity': round(cos, 4),
                    'combined_norm': round(norm1 + norm2, 4),
                }
                if cos > 0.5:
                    cooperating.append(entry)
                elif cos < -0.5:
                    competing.append(entry)

    cooperating.sort(key=lambda x: -x['cosine_similarity'])
    competing.sort(key=lambda x: x['cosine_similarity'])

    return {
        'layer': layer,
        'n_active': len(active),
        'cooperating_pairs': cooperating[:top_k],
        'competing_pairs': competing[:top_k],
    }
