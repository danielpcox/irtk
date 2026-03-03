"""Component specialization analysis.

Tools for understanding what each model component specializes in:
- Head function profiling
- MLP specialization detection
- Component consistency across inputs
- Specialization vs generalization spectrum
- Component redundancy identification
"""

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from irtk.hooked_transformer import HookedTransformer


def head_function_profile(
    model: HookedTransformer,
    tokens: jnp.ndarray,
) -> dict:
    """Profile what each attention head does functionally.

    Computes multiple metrics for each head to characterize its behavior:
    self-attention, previous-token, first-token, uniform, and copying.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] token IDs.

    Returns:
        Dict with per-head function profiles.
    """
    _, cache = model.run_with_cache(tokens)
    seq_len = len(tokens)

    per_head = []
    for l in range(model.cfg.n_layers):
        pattern = cache[f'blocks.{l}.attn.hook_pattern']  # [n_heads, seq, seq]

        for h in range(model.cfg.n_heads):
            p = np.array(pattern[h])  # [seq, seq]

            # Self-attention: diagonal mass
            self_attn = float(np.mean(np.diag(p)))

            # Previous-token: off-diagonal by 1
            prev_token = 0.0
            count = 0
            for i in range(1, seq_len):
                prev_token += p[i, i - 1]
                count += 1
            prev_token = prev_token / count if count > 0 else 0.0

            # First-token: column 0
            first_token = float(np.mean(p[1:, 0])) if seq_len > 1 else 0.0

            # Entropy (uniformity)
            entropies = []
            for q in range(seq_len):
                row = p[q, :q + 1]
                row = row + 1e-10
                ent = -float(np.sum(row * np.log(row)))
                entropies.append(ent)
            mean_entropy = float(np.mean(entropies))
            max_entropy = float(np.log(seq_len))
            uniformity = mean_entropy / max_entropy if max_entropy > 0 else 0.0

            # Determine dominant function
            scores = {
                'self': self_attn,
                'previous_token': prev_token,
                'first_token': first_token,
            }
            dominant = max(scores, key=scores.get)

            per_head.append({
                'layer': l,
                'head': h,
                'self_attention': round(self_attn, 4),
                'previous_token': round(prev_token, 4),
                'first_token': round(first_token, 4),
                'uniformity': round(uniformity, 4),
                'dominant_function': dominant,
            })

    return {
        'per_head': per_head,
        'function_distribution': {
            fn: sum(1 for h in per_head if h['dominant_function'] == fn)
            for fn in ['self', 'previous_token', 'first_token']
        },
    }


def mlp_specialization(
    model: HookedTransformer,
    tokens_list: list[jnp.ndarray],
    layer: int = 0,
    top_k: int = 10,
) -> dict:
    """Measure how specialized each MLP neuron is.

    A specialized neuron fires selectively for specific inputs.
    A generalist neuron fires for many/all inputs.

    Args:
        model: HookedTransformer.
        tokens_list: List of token sequences.
        layer: Layer to analyze.
        top_k: Number of most/least specialized neurons to return.

    Returns:
        Dict with per-neuron specialization scores.
    """
    all_acts = []
    for tokens in tokens_list:
        _, cache = model.run_with_cache(tokens)
        post = np.array(cache[f'blocks.{layer}.mlp.hook_post'])  # [seq, d_mlp]
        # Use last position
        all_acts.append(post[-1])

    acts = np.stack(all_acts)  # [n_examples, d_mlp]
    n_examples = acts.shape[0]
    d_mlp = acts.shape[1]

    # Specialization = fraction of inputs where neuron is active
    active = (np.abs(acts) > 0.01).astype(float)
    activation_freq = np.mean(active, axis=0)  # [d_mlp]

    # Specialization score: 1 - activation_frequency (high = specialized)
    specialization = 1.0 - activation_freq

    # Sort by specialization
    sorted_neurons = np.argsort(specialization)[::-1]

    most_specialized = []
    for n in sorted_neurons[:top_k]:
        most_specialized.append({
            'neuron': int(n),
            'specialization': round(float(specialization[n]), 4),
            'activation_frequency': round(float(activation_freq[n]), 4),
        })

    least_specialized = []
    for n in sorted_neurons[-top_k:]:
        least_specialized.append({
            'neuron': int(n),
            'specialization': round(float(specialization[n]), 4),
            'activation_frequency': round(float(activation_freq[n]), 4),
        })

    return {
        'layer': layer,
        'most_specialized': most_specialized,
        'least_specialized': least_specialized,
        'mean_specialization': round(float(np.mean(specialization)), 4),
        'n_dead': int(np.sum(activation_freq == 0)),
        'n_always_active': int(np.sum(activation_freq == 1.0)),
    }


def component_consistency(
    model: HookedTransformer,
    tokens_list: list[jnp.ndarray],
    pos: int = -1,
) -> dict:
    """Measure how consistent each component's output is across inputs.

    A consistent component produces similar outputs regardless of input.
    An input-sensitive component produces very different outputs.

    Args:
        model: HookedTransformer.
        tokens_list: List of token sequences.
        pos: Position to analyze.

    Returns:
        Dict with per-component consistency scores.
    """
    component_outputs = {}

    for tokens in tokens_list:
        _, cache = model.run_with_cache(tokens)

        for l in range(model.cfg.n_layers):
            for comp in ['hook_attn_out', 'hook_mlp_out']:
                name = f'blocks.{l}.{comp}'
                act = np.array(cache[name][pos])
                if name not in component_outputs:
                    component_outputs[name] = []
                component_outputs[name].append(act)

    per_component = []
    for name, outputs in component_outputs.items():
        outputs_arr = np.stack(outputs)  # [n_examples, d_model]
        mean_output = np.mean(outputs_arr, axis=0)
        # Variance from mean
        deviations = np.linalg.norm(outputs_arr - mean_output, axis=1)
        mean_norm = float(np.mean(np.linalg.norm(outputs_arr, axis=1)))

        consistency = 1.0 - float(np.mean(deviations)) / mean_norm if mean_norm > 1e-10 else 1.0
        consistency = max(0.0, consistency)

        short = name.replace('blocks.', 'L').replace('.hook_', ' ')
        per_component.append({
            'component': short,
            'consistency': round(consistency, 4),
            'mean_deviation': round(float(np.mean(deviations)), 4),
            'mean_norm': round(mean_norm, 4),
        })

    per_component.sort(key=lambda x: -x['consistency'])
    return {
        'per_component': per_component,
        'most_consistent': per_component[0]['component'] if per_component else None,
        'least_consistent': per_component[-1]['component'] if per_component else None,
    }


def specialization_spectrum(
    model: HookedTransformer,
    tokens_list: list[jnp.ndarray],
    pos: int = -1,
) -> dict:
    """Place each component on a specialization-generalization spectrum.

    Combines output consistency with output diversity to classify
    components as specialists or generalists.

    Args:
        model: HookedTransformer.
        tokens_list: List of token sequences.
        pos: Position to analyze.

    Returns:
        Dict with per-component specialization scores.
    """
    component_data = {}

    for tokens in tokens_list:
        _, cache = model.run_with_cache(tokens)
        for l in range(model.cfg.n_layers):
            for comp, short in [('hook_attn_out', f'L{l}_attn'), ('hook_mlp_out', f'L{l}_mlp')]:
                act = np.array(cache[f'blocks.{l}.{comp}'][pos])
                if short not in component_data:
                    component_data[short] = []
                component_data[short].append(act)

    per_component = []
    for name, outputs in component_data.items():
        arr = np.stack(outputs)
        mean = np.mean(arr, axis=0)
        # How diverse are the outputs?
        norms = np.linalg.norm(arr - mean, axis=1)
        diversity = float(np.mean(norms))
        mean_norm = float(np.mean(np.linalg.norm(arr, axis=1)))

        # High diversity relative to norm = specialist
        # Low diversity = generalist
        if mean_norm > 1e-10:
            spec_score = diversity / mean_norm
        else:
            spec_score = 0.0

        per_component.append({
            'component': name,
            'specialization_score': round(spec_score, 4),
            'output_diversity': round(diversity, 4),
            'mean_output_norm': round(mean_norm, 4),
            'classification': 'specialist' if spec_score > 0.5 else 'generalist',
        })

    per_component.sort(key=lambda x: -x['specialization_score'])

    return {
        'per_component': per_component,
        'n_specialists': sum(1 for c in per_component if c['classification'] == 'specialist'),
        'n_generalists': sum(1 for c in per_component if c['classification'] == 'generalist'),
    }


def component_redundancy(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    pos: int = -1,
) -> dict:
    """Identify redundant components (ones that produce similar outputs).

    Args:
        model: HookedTransformer.
        tokens: [seq_len] token IDs.
        pos: Position to analyze.

    Returns:
        Dict with pairwise similarity between components.
    """
    _, cache = model.run_with_cache(tokens)

    components = []
    names = []
    for l in range(model.cfg.n_layers):
        for comp, short in [('hook_attn_out', f'L{l}_attn'), ('hook_mlp_out', f'L{l}_mlp')]:
            act = np.array(cache[f'blocks.{l}.{comp}'][pos])
            components.append(act)
            names.append(short)

    # Pairwise cosine similarity
    n = len(components)
    redundant_pairs = []

    for i in range(n):
        for j in range(i + 1, n):
            ni = float(np.linalg.norm(components[i]))
            nj = float(np.linalg.norm(components[j]))
            if ni > 1e-10 and nj > 1e-10:
                cos = float(np.dot(components[i], components[j]) / (ni * nj))
            else:
                cos = 0.0

            if abs(cos) > 0.8:
                redundant_pairs.append({
                    'component_a': names[i],
                    'component_b': names[j],
                    'cosine_similarity': round(cos, 4),
                })

    redundant_pairs.sort(key=lambda x: -abs(x['cosine_similarity']))

    return {
        'n_redundant_pairs': len(redundant_pairs),
        'redundant_pairs': redundant_pairs,
        'n_components': n,
    }
