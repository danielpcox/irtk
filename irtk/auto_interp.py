"""Automated interpretability tools for model components.

Heuristic methods to automatically characterize what attention heads
and neurons do, without requiring human inspection of every component.
"""

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from irtk.hooked_transformer import HookedTransformer


def auto_label_head(
    model: HookedTransformer,
    layer: int,
    head: int,
    token_sequences: list,
) -> dict:
    """Automatically characterize an attention head's behavior.

    Analyzes attention patterns across multiple inputs to detect common
    head types: previous-token, induction, copy, positional, etc.

    Args:
        model: HookedTransformer.
        layer: Layer index.
        head: Head index.
        token_sequences: List of token arrays to analyze.

    Returns:
        Dict with:
        - "label": best-matching head type string
        - "confidence": confidence score (0-1)
        - "scores": dict of type_name -> match score
        - "mean_entropy": average attention entropy
    """
    patterns = []
    for tokens in token_sequences:
        tokens = jnp.array(tokens)
        _, cache = model.run_with_cache(tokens)
        hook = f"blocks.{layer}.attn.hook_pattern"
        if hook in cache.cache_dict:
            pattern = np.array(cache.cache_dict[hook][head])  # [seq, seq]
            patterns.append(pattern)

    if not patterns:
        return {"label": "unknown", "confidence": 0.0, "scores": {}, "mean_entropy": 0.0}

    scores = {}

    # Previous-token score: attention on position i-1
    prev_scores = []
    for pat in patterns:
        seq = pat.shape[0]
        if seq < 2:
            continue
        diag = np.array([pat[i, i - 1] for i in range(1, seq)])
        prev_scores.append(float(np.mean(diag)))
    scores["previous_token"] = float(np.mean(prev_scores)) if prev_scores else 0.0

    # Current-token score: attention on diagonal
    curr_scores = []
    for pat in patterns:
        seq = pat.shape[0]
        diag = np.array([pat[i, i] for i in range(seq)])
        curr_scores.append(float(np.mean(diag)))
    scores["current_token"] = float(np.mean(curr_scores)) if curr_scores else 0.0

    # BOS/first-token score: attention on position 0
    bos_scores = []
    for pat in patterns:
        seq = pat.shape[0]
        if seq < 2:
            continue
        bos_scores.append(float(np.mean(pat[1:, 0])))
    scores["bos_attending"] = float(np.mean(bos_scores)) if bos_scores else 0.0

    # Positional/local score: attention concentrated near diagonal
    local_scores = []
    for pat in patterns:
        seq = pat.shape[0]
        if seq < 3:
            continue
        local = 0.0
        for i in range(seq):
            for j in range(max(0, i - 2), min(seq, i + 3)):
                local += pat[i, j]
        local /= seq
        local_scores.append(float(local))
    scores["local_attention"] = float(np.mean(local_scores)) if local_scores else 0.0

    # Entropy of attention (low = focused, high = diffuse)
    entropies = []
    for pat in patterns:
        for i in range(pat.shape[0]):
            p = pat[i]
            p = p[p > 1e-10]
            entropies.append(float(-np.sum(p * np.log(p))))
    mean_entropy = float(np.mean(entropies)) if entropies else 0.0

    # Induction score: match between offset patterns
    induction_scores = []
    for pat in patterns:
        seq = pat.shape[0]
        if seq < 4:
            continue
        # Induction heads attend to token after previous occurrence
        # Simple proxy: correlation between pat[i,j] and pat[i-1,j-1]
        if seq > 3:
            diag_shift = [pat[i, i - 1] for i in range(2, seq)]
            induction_scores.append(float(np.mean(diag_shift)))
    scores["induction"] = float(np.mean(induction_scores)) if induction_scores else 0.0

    # Find best label
    best_label = max(scores, key=scores.get)
    best_score = scores[best_label]

    return {
        "label": best_label,
        "confidence": min(best_score, 1.0),
        "scores": scores,
        "mean_entropy": mean_entropy,
    }


def auto_label_neuron(
    model: HookedTransformer,
    layer: int,
    neuron_idx: int,
    token_sequences: list,
    k: int = 10,
) -> dict:
    """Automatically characterize a neuron by its top activating contexts.

    Args:
        model: HookedTransformer.
        layer: Layer index.
        neuron_idx: Neuron index within the MLP.
        token_sequences: List of token arrays.
        k: Number of top activations to collect.

    Returns:
        Dict with:
        - "top_activations": list of (activation, position, prompt_idx) tuples
        - "mean_activation": mean activation across all inputs
        - "firing_rate": fraction of positions where neuron fires (> 0)
        - "max_activation": maximum activation observed
    """
    hook = f"blocks.{layer}.hook_mlp_out"

    all_acts = []
    top_examples = []

    for pi, tokens in enumerate(token_sequences):
        tokens = jnp.array(tokens)
        _, cache = model.run_with_cache(tokens)

        # Get MLP pre-activation if available, else use mlp_out
        pre_hook = f"blocks.{layer}.mlp.hook_post"
        if pre_hook in cache.cache_dict:
            acts = np.array(cache.cache_dict[pre_hook])
        elif hook in cache.cache_dict:
            acts = np.array(cache.cache_dict[hook])
        else:
            continue

        if neuron_idx < acts.shape[-1]:
            neuron_acts = acts[..., neuron_idx] if acts.ndim > 1 else acts
            if neuron_acts.ndim == 1:
                for pos in range(len(neuron_acts)):
                    top_examples.append((float(neuron_acts[pos]), pos, pi))
                all_acts.extend(neuron_acts.tolist())
            else:
                all_acts.append(float(neuron_acts))

    if not all_acts:
        return {"top_activations": [], "mean_activation": 0.0,
                "firing_rate": 0.0, "max_activation": 0.0}

    all_acts = np.array(all_acts)
    top_examples.sort(key=lambda x: x[0], reverse=True)

    return {
        "top_activations": top_examples[:k],
        "mean_activation": float(np.mean(all_acts)),
        "firing_rate": float(np.mean(all_acts > 0)),
        "max_activation": float(np.max(all_acts)),
    }


def feature_summary_stats(
    model: HookedTransformer,
    hook_name: str,
    token_sequences: list,
) -> dict:
    """Compute summary statistics for all features at a hook point.

    Args:
        model: HookedTransformer.
        hook_name: Hook point to analyze.
        token_sequences: List of token arrays.

    Returns:
        Dict with:
        - "mean_activations": [d] mean activation per dimension
        - "std_activations": [d] std per dimension
        - "sparsity": [d] fraction of zero activations per dimension
        - "kurtosis": [d] kurtosis per dimension (peakedness)
    """
    all_acts = []

    for tokens in token_sequences:
        tokens = jnp.array(tokens)
        _, cache = model.run_with_cache(tokens)
        if hook_name in cache.cache_dict:
            acts = np.array(cache.cache_dict[hook_name])
            if acts.ndim > 1:
                all_acts.append(acts.reshape(-1, acts.shape[-1]))

    if not all_acts:
        return {"mean_activations": np.array([]),
                "std_activations": np.array([]),
                "sparsity": np.array([]),
                "kurtosis": np.array([])}

    combined = np.concatenate(all_acts, axis=0)  # [n_samples, d]
    means = np.mean(combined, axis=0)
    stds = np.std(combined, axis=0)
    sparsity = np.mean(np.abs(combined) < 1e-6, axis=0)

    # Kurtosis
    centered = combined - means[None, :]
    m4 = np.mean(centered ** 4, axis=0)
    m2 = np.mean(centered ** 2, axis=0)
    kurtosis = m4 / np.maximum(m2 ** 2, 1e-10) - 3.0  # excess kurtosis

    return {
        "mean_activations": means,
        "std_activations": stds,
        "sparsity": sparsity,
        "kurtosis": kurtosis,
    }


def head_type_classifier(
    model: HookedTransformer,
    token_sequences: list,
) -> dict:
    """Classify all heads in the model by type.

    Args:
        model: HookedTransformer.
        token_sequences: List of token arrays.

    Returns:
        Dict with:
        - "classifications": dict of "L{l}H{h}" -> label
        - "type_counts": dict of label -> count
        - "confidence_matrix": [n_layers, n_heads] confidence scores
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    classifications = {}
    confidence_matrix = np.zeros((n_layers, n_heads))

    for l in range(n_layers):
        for h in range(n_heads):
            result = auto_label_head(model, l, h, token_sequences)
            name = f"L{l}H{h}"
            classifications[name] = result["label"]
            confidence_matrix[l, h] = result["confidence"]

    # Count types
    type_counts = {}
    for label in classifications.values():
        type_counts[label] = type_counts.get(label, 0) + 1

    return {
        "classifications": classifications,
        "type_counts": type_counts,
        "confidence_matrix": confidence_matrix,
    }


def component_report(
    model: HookedTransformer,
    token_sequences: list,
) -> dict:
    """Generate a summary report of model component roles.

    Classifies all heads and computes per-layer statistics to give
    an overview of what each part of the model does.

    Args:
        model: HookedTransformer.
        token_sequences: List of token arrays.

    Returns:
        Dict with:
        - "head_classifications": dict of "L{l}H{h}" -> label
        - "layer_summary": list of per-layer summary strings
        - "n_layers": number of layers
        - "n_heads": number of heads
    """
    head_result = head_type_classifier(model, token_sequences)

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    layer_summaries = []
    for l in range(n_layers):
        types = []
        for h in range(n_heads):
            name = f"L{l}H{h}"
            types.append(head_result["classifications"][name])
        type_counts = {}
        for t in types:
            type_counts[t] = type_counts.get(t, 0) + 1
        summary = ", ".join(f"{count}x {t}" for t, count in sorted(type_counts.items(), key=lambda x: -x[1]))
        layer_summaries.append(f"Layer {l}: {summary}")

    return {
        "head_classifications": head_result["classifications"],
        "layer_summary": layer_summaries,
        "n_layers": n_layers,
        "n_heads": n_heads,
    }
