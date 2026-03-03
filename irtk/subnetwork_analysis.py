"""Subnetwork extraction and evaluation.

Identifies minimal circuits within the model that reproduce a given behavior.
Evaluates faithfulness, minimality, and compares different subnetworks.

References:
    Conmy et al. (2023) "Towards Automated Circuit Discovery for Mechanistic Interpretability"
    Wang et al. (2023) "Interpretability in the Wild"
"""

import jax
import jax.numpy as jnp
import numpy as np


def extract_important_components(model, tokens, metric_fn, threshold=0.1):
    """Identify important components via ablation.

    Ablates each component and keeps those whose removal significantly
    affects the metric.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        metric_fn: Function from logits -> scalar.
        threshold: Minimum effect fraction to be considered important.

    Returns:
        dict with:
            important_heads: list of (layer, head) tuples
            important_mlps: list of layer indices
            ablation_effects: dict mapping component -> effect
            n_important: int
            fraction_important: float
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    baseline_logits = model(tokens)
    baseline = metric_fn(baseline_logits)

    effects = {}
    important_heads = []
    important_mlps = []

    for layer in range(n_layers):
        # Test each head
        z_key = f"blocks.{layer}.attn.hook_z"
        for h in range(n_heads):
            def make_zero_head(head_idx):
                def fn(x, name):
                    return x.at[:, head_idx, :].set(0.0)
                return fn

            state = HookState(hook_fns={z_key: make_zero_head(h)}, cache={})
            logits = model(tokens, hook_state=state)
            effect = abs(baseline - metric_fn(logits))
            effects[("attn", layer, h)] = float(effect)

            if abs(baseline) > 1e-10 and effect / abs(baseline) >= threshold:
                important_heads.append((layer, h))

        # Test MLP
        mlp_key = f"blocks.{layer}.hook_mlp_out"
        def zero_fn(x, name):
            return jnp.zeros_like(x)

        state = HookState(hook_fns={mlp_key: zero_fn}, cache={})
        logits = model(tokens, hook_state=state)
        effect = abs(baseline - metric_fn(logits))
        effects[("mlp", layer)] = float(effect)

        if abs(baseline) > 1e-10 and effect / abs(baseline) >= threshold:
            important_mlps.append(layer)

    total_components = n_layers * n_heads + n_layers
    n_important = len(important_heads) + len(important_mlps)

    return {
        "important_heads": important_heads,
        "important_mlps": important_mlps,
        "ablation_effects": effects,
        "n_important": n_important,
        "fraction_important": n_important / total_components if total_components > 0 else 0.0,
    }


def subnetwork_faithfulness(model, tokens, metric_fn, heads, mlps):
    """Evaluate faithfulness of a subnetwork.

    Keeps only the specified components active (zeros out everything else)
    and measures how well the metric is reproduced.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        metric_fn: Function from logits -> scalar.
        heads: List of (layer, head) tuples to keep.
        mlps: List of layer indices to keep.

    Returns:
        dict with:
            full_metric: float
            subnetwork_metric: float
            faithfulness: float, ratio of subnetwork metric to full metric
            absolute_error: float
            relative_error: float
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    full_logits = model(tokens)
    full_metric = metric_fn(full_logits)

    heads_set = set(heads)
    mlps_set = set(mlps)

    hooks = {}
    for layer in range(n_layers):
        # Zero out non-included heads
        z_key = f"blocks.{layer}.attn.hook_z"
        heads_to_zero = [h for h in range(n_heads) if (layer, h) not in heads_set]
        if heads_to_zero:
            def make_zero_heads(to_zero):
                def fn(x, name):
                    for h in to_zero:
                        x = x.at[:, h, :].set(0.0)
                    return x
                return fn
            hooks[z_key] = make_zero_heads(heads_to_zero)

        # Zero out non-included MLPs
        if layer not in mlps_set:
            mlp_key = f"blocks.{layer}.hook_mlp_out"
            def zero_fn(x, name):
                return jnp.zeros_like(x)
            hooks[mlp_key] = zero_fn

    state = HookState(hook_fns=hooks, cache={})
    sub_logits = model(tokens, hook_state=state)
    sub_metric = metric_fn(sub_logits)

    abs_error = abs(full_metric - sub_metric)
    rel_error = abs_error / (abs(full_metric) + 1e-10)
    faithfulness = sub_metric / (full_metric + 1e-10) if abs(full_metric) > 1e-10 else 0.0

    return {
        "full_metric": float(full_metric),
        "subnetwork_metric": float(sub_metric),
        "faithfulness": float(faithfulness),
        "absolute_error": float(abs_error),
        "relative_error": float(rel_error),
    }


def subnetwork_minimality(model, tokens, metric_fn, heads, mlps, threshold=0.9):
    """Test minimality: can any component be removed without hurting faithfulness?

    For each component in the subnetwork, removes it and checks if the
    subnetwork still achieves above-threshold faithfulness.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        metric_fn: Function from logits -> scalar.
        heads: List of (layer, head) tuples.
        mlps: List of layer indices.
        threshold: Minimum faithfulness to maintain.

    Returns:
        dict with:
            removable_heads: list of (layer, head) that can be removed
            removable_mlps: list of layer indices that can be removed
            n_removable: int
            is_minimal: bool, True if nothing can be removed
            component_necessity: dict mapping component -> bool (True = necessary)
    """
    full_logits = model(tokens)
    full_metric = metric_fn(full_logits)

    removable_heads = []
    removable_mlps = []
    necessity = {}

    # Test each head
    for head in heads:
        reduced_heads = [h for h in heads if h != head]
        faith = subnetwork_faithfulness(model, tokens, metric_fn, reduced_heads, mlps)
        is_necessary = abs(faith["faithfulness"]) < threshold
        necessity[("attn",) + head] = is_necessary
        if not is_necessary:
            removable_heads.append(head)

    # Test each MLP
    for mlp_layer in mlps:
        reduced_mlps = [m for m in mlps if m != mlp_layer]
        faith = subnetwork_faithfulness(model, tokens, metric_fn, heads, reduced_mlps)
        is_necessary = abs(faith["faithfulness"]) < threshold
        necessity[("mlp", mlp_layer)] = is_necessary
        if not is_necessary:
            removable_mlps.append(mlp_layer)

    n_removable = len(removable_heads) + len(removable_mlps)

    return {
        "removable_heads": removable_heads,
        "removable_mlps": removable_mlps,
        "n_removable": n_removable,
        "is_minimal": n_removable == 0,
        "component_necessity": necessity,
    }


def compare_subnetworks(model, tokens, metric_fn, subnetwork_a, subnetwork_b):
    """Compare two subnetworks on the same input.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        metric_fn: Function from logits -> scalar.
        subnetwork_a: dict with 'heads' and 'mlps' lists.
        subnetwork_b: dict with 'heads' and 'mlps' lists.

    Returns:
        dict with:
            faithfulness_a: float
            faithfulness_b: float
            overlap_heads: list of shared heads
            overlap_mlps: list of shared MLPs
            jaccard_similarity: float, Jaccard index of component sets
            size_a: int
            size_b: int
    """
    faith_a = subnetwork_faithfulness(
        model, tokens, metric_fn, subnetwork_a["heads"], subnetwork_a["mlps"]
    )
    faith_b = subnetwork_faithfulness(
        model, tokens, metric_fn, subnetwork_b["heads"], subnetwork_b["mlps"]
    )

    set_a_heads = set(map(tuple, subnetwork_a["heads"]))
    set_b_heads = set(map(tuple, subnetwork_b["heads"]))
    set_a_mlps = set(subnetwork_a["mlps"])
    set_b_mlps = set(subnetwork_b["mlps"])

    overlap_heads = list(set_a_heads & set_b_heads)
    overlap_mlps = list(set_a_mlps & set_b_mlps)

    union_size = len(set_a_heads | set_b_heads) + len(set_a_mlps | set_b_mlps)
    inter_size = len(overlap_heads) + len(overlap_mlps)
    jaccard = inter_size / union_size if union_size > 0 else 0.0

    return {
        "faithfulness_a": faith_a["faithfulness"],
        "faithfulness_b": faith_b["faithfulness"],
        "overlap_heads": overlap_heads,
        "overlap_mlps": overlap_mlps,
        "jaccard_similarity": float(jaccard),
        "size_a": len(subnetwork_a["heads"]) + len(subnetwork_a["mlps"]),
        "size_b": len(subnetwork_b["heads"]) + len(subnetwork_b["mlps"]),
    }


def greedy_subnetwork_search(model, tokens, metric_fn, target_faithfulness=0.9):
    """Greedily build a minimal subnetwork that achieves target faithfulness.

    Iteratively adds the most important component until the target is reached.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        metric_fn: Function from logits -> scalar.
        target_faithfulness: Target metric fraction.

    Returns:
        dict with:
            selected_heads: list of (layer, head)
            selected_mlps: list of layer indices
            faithfulness_trajectory: array of faithfulness after each addition
            n_components_needed: int
            final_faithfulness: float
    """
    important = extract_important_components(model, tokens, metric_fn, threshold=0.0)
    effects = important["ablation_effects"]

    # Sort by effect (most important first)
    sorted_components = sorted(effects.items(), key=lambda x: x[1], reverse=True)

    full_logits = model(tokens)
    full_metric = metric_fn(full_logits)

    selected_heads = []
    selected_mlps = []
    trajectory = []

    for comp, _ in sorted_components:
        if comp[0] == "attn":
            selected_heads.append((comp[1], comp[2]))
        else:
            selected_mlps.append(comp[1])

        faith = subnetwork_faithfulness(model, tokens, metric_fn, selected_heads, selected_mlps)
        trajectory.append(faith["faithfulness"])

        if abs(faith["faithfulness"]) >= target_faithfulness:
            break

    return {
        "selected_heads": selected_heads,
        "selected_mlps": selected_mlps,
        "faithfulness_trajectory": np.array(trajectory),
        "n_components_needed": len(selected_heads) + len(selected_mlps),
        "final_faithfulness": float(trajectory[-1]) if trajectory else 0.0,
    }
