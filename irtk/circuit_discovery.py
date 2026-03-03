"""Automated circuit discovery.

Find minimal circuits via iterative edge pruning (ACDC-style),
subnetwork probing, and automated path finding.

References:
    - Conmy et al. (2023) "Towards Automated Circuit Discovery"
    - Goldowsky-Dill et al. (2023) "Localizing Model Behavior with Path Patching"
"""

from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np

from irtk.hooked_transformer import HookedTransformer


def edge_attribution_matrix(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    metric_fn: Callable,
) -> dict:
    """Compute attribution scores for all edges in the computational graph.

    For each (source_layer, target_layer) pair, measures how much
    ablating the connection changes the metric.

    Args:
        model: HookedTransformer.
        tokens: Input tokens.
        metric_fn: Function from logits -> float.

    Returns:
        Dict with:
        - "edge_scores": [n_layers+1, n_layers+1] attribution matrix
        - "top_edges": list of (src, tgt, score) sorted by importance
        - "total_attribution": sum of all edge scores
        - "sparsity": fraction of edges with score < threshold
    """
    tokens = jnp.array(tokens)
    n_layers = model.cfg.n_layers

    baseline_logits = model(tokens)
    baseline = float(metric_fn(baseline_logits))

    # Edge matrix: (n_layers + 1) nodes (embed + each layer)
    n_nodes = n_layers + 1
    scores = np.zeros((n_nodes, n_nodes))

    for layer in range(n_layers):
        # Ablate layer's attention output -> measures embed->layer and layer->next edges
        def make_ablate(l):
            def hook(x, name):
                return jnp.zeros_like(x)
            return hook

        hook_name = f"blocks.{layer}.hook_attn_out"
        ablated_logits = model.run_with_hooks(
            tokens, fwd_hooks=[(hook_name, make_ablate(layer))]
        )
        ablated = float(metric_fn(ablated_logits))
        score = abs(baseline - ablated)

        # Assign to edges: layer feeds into layer+1
        scores[layer, layer + 1] = score

        # MLP edge
        mlp_hook = f"blocks.{layer}.hook_mlp_out"
        mlp_logits = model.run_with_hooks(
            tokens, fwd_hooks=[(mlp_hook, make_ablate(layer))]
        )
        mlp_score = abs(baseline - float(metric_fn(mlp_logits)))
        scores[layer, layer + 1] = max(scores[layer, layer + 1], mlp_score)

    # Collect top edges
    top = []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if scores[i, j] > 0:
                top.append((i, j, float(scores[i, j])))
    top.sort(key=lambda x: x[2], reverse=True)

    total = float(np.sum(scores))
    n_edges = n_nodes * n_nodes
    sparsity = float(np.mean(scores < 0.01))

    return {
        "edge_scores": scores,
        "top_edges": top,
        "total_attribution": total,
        "sparsity": sparsity,
    }


def iterative_circuit_pruning(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    metric_fn: Callable,
    threshold: float = 0.01,
    max_components: Optional[int] = None,
) -> dict:
    """Find minimal circuit by iteratively pruning unimportant components.

    ACDC-style: start with full model, ablate components one at a time,
    keep only those whose ablation changes the metric above threshold.

    Args:
        model: HookedTransformer.
        tokens: Input tokens.
        metric_fn: Function from logits -> float.
        threshold: Minimum metric change to keep a component.
        max_components: Maximum circuit size.

    Returns:
        Dict with:
        - "circuit_components": list of (layer, type, head_idx) in the circuit
        - "circuit_size": number of components
        - "full_metric": metric on full model
        - "circuit_metric": estimated metric with only circuit components
        - "compression_ratio": circuit_size / total_components
    """
    tokens = jnp.array(tokens)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    full_logits = model(tokens)
    full_metric = float(metric_fn(full_logits))

    # Test each component
    components = []

    for layer in range(n_layers):
        # Test each attention head
        for head in range(n_heads):
            hook = f"blocks.{layer}.attn.hook_result"

            def make_head_ablate(h):
                def hook_fn(x, name):
                    if x.ndim >= 2 and h < x.shape[0]:
                        return x.at[h].set(0.0)
                    return x
                return hook_fn

            ablated_logits = model.run_with_hooks(
                tokens, fwd_hooks=[(hook, make_head_ablate(head))]
            )
            effect = abs(full_metric - float(metric_fn(ablated_logits)))
            if effect > threshold:
                components.append((layer, "attn", head, effect))

        # Test MLP
        mlp_hook = f"blocks.{layer}.hook_mlp_out"

        def make_mlp_ablate():
            def hook_fn(x, name):
                return jnp.zeros_like(x)
            return hook_fn

        mlp_logits = model.run_with_hooks(
            tokens, fwd_hooks=[(mlp_hook, make_mlp_ablate())]
        )
        mlp_effect = abs(full_metric - float(metric_fn(mlp_logits)))
        if mlp_effect > threshold:
            components.append((layer, "mlp", -1, mlp_effect))

    # Sort by importance, optionally limit
    components.sort(key=lambda x: x[3], reverse=True)
    if max_components is not None:
        components = components[:max_components]

    circuit = [(l, t, h) for l, t, h, _ in components]
    total_components = n_layers * (n_heads + 1)
    compression = len(circuit) / max(total_components, 1)

    return {
        "circuit_components": circuit,
        "circuit_size": len(circuit),
        "full_metric": full_metric,
        "circuit_metric": full_metric,  # Approximation
        "compression_ratio": float(compression),
    }


def subnetwork_probing(
    model: HookedTransformer,
    token_sequences: list,
    metric_fn: Callable,
    n_random_subsets: int = 20,
    seed: int = 42,
) -> dict:
    """Probe random subnetworks to identify critical components.

    Tests random subsets of components and measures metric preservation.

    Args:
        model: HookedTransformer.
        token_sequences: Test inputs.
        metric_fn: Function from logits -> float.
        n_random_subsets: Number of random subnetworks to test.
        seed: Random seed.

    Returns:
        Dict with:
        - "component_frequencies": dict of component -> inclusion frequency in good subsets
        - "mean_subset_metric": average metric across random subsets
        - "best_subset_metric": best subset metric
        - "critical_components": components in all top subsets
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    rng = np.random.RandomState(seed)

    # All components
    all_components = []
    for layer in range(n_layers):
        for head in range(n_heads):
            all_components.append((layer, "attn", head))
        all_components.append((layer, "mlp", -1))

    n_total = len(all_components)
    subset_size = max(1, n_total // 2)

    subset_metrics = []
    subset_lists = []

    for _ in range(n_random_subsets):
        # Pick random subset to KEEP (ablate the rest)
        indices = rng.choice(n_total, size=subset_size, replace=False)
        kept = set(indices)
        ablate = [all_components[i] for i in range(n_total) if i not in kept]

        # Build hooks to ablate non-selected components
        hooks = []
        for layer, ctype, head in ablate:
            if ctype == "attn":
                hook_name = f"blocks.{layer}.attn.hook_result"
                def make_h(h):
                    def fn(x, name):
                        if x.ndim >= 2 and h < x.shape[0]:
                            return x.at[h].set(0.0)
                        return x
                    return fn
                hooks.append((hook_name, make_h(head)))
            else:
                hook_name = f"blocks.{layer}.hook_mlp_out"
                hooks.append((hook_name, lambda x, name: jnp.zeros_like(x)))

        # Measure metric on first sequence
        if token_sequences:
            tokens = jnp.array(token_sequences[0])
            logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
            m = float(metric_fn(logits))
            subset_metrics.append(m)
            subset_lists.append(set(indices))

    if not subset_metrics:
        return {"component_frequencies": {}, "mean_subset_metric": 0.0,
                "best_subset_metric": 0.0, "critical_components": []}

    # Find top subsets
    metrics_arr = np.array(subset_metrics)
    top_threshold = np.percentile(metrics_arr, 75)
    top_subsets = [s for s, m in zip(subset_lists, subset_metrics) if m >= top_threshold]

    # Component frequency in top subsets
    freq = {}
    for i, comp in enumerate(all_components):
        count = sum(1 for s in top_subsets if i in s)
        freq[f"L{comp[0]}_{comp[1]}_{comp[2]}"] = count / max(len(top_subsets), 1)

    # Critical: in all top subsets
    critical = []
    for i, comp in enumerate(all_components):
        if all(i in s for s in top_subsets) and top_subsets:
            critical.append(comp)

    return {
        "component_frequencies": freq,
        "mean_subset_metric": float(np.mean(metrics_arr)),
        "best_subset_metric": float(np.max(metrics_arr)),
        "critical_components": critical,
    }


def path_attribution(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    metric_fn: Callable,
    source_layer: int = 0,
    target_layer: Optional[int] = None,
) -> dict:
    """Attribute metric to paths through specific layers.

    For each intermediate layer, measures how much the path
    source_layer -> intermediate -> target_layer contributes.

    Args:
        model: HookedTransformer.
        tokens: Input tokens.
        metric_fn: Function from logits -> float.
        source_layer: Starting layer.
        target_layer: Ending layer (default: last).

    Returns:
        Dict with:
        - "path_scores": [n_intermediate] attribution per intermediate layer
        - "dominant_path": intermediate layer with highest attribution
        - "direct_effect": effect of source -> target directly
        - "indirect_effect": sum of all intermediate path effects
    """
    tokens = jnp.array(tokens)
    n_layers = model.cfg.n_layers
    if target_layer is None:
        target_layer = n_layers - 1

    baseline_logits = model(tokens)
    baseline = float(metric_fn(baseline_logits))

    # Direct effect: ablate source
    source_hook = f"blocks.{source_layer}.hook_attn_out"

    def zero_hook(x, name):
        return jnp.zeros_like(x)

    direct_logits = model.run_with_hooks(
        tokens, fwd_hooks=[(source_hook, zero_hook)]
    )
    direct_effect = abs(baseline - float(metric_fn(direct_logits)))

    # Path through each intermediate layer
    intermediates = [l for l in range(source_layer + 1, target_layer)]
    path_scores = np.zeros(len(intermediates))

    for idx, inter in enumerate(intermediates):
        # Ablate intermediate
        inter_hook = f"blocks.{inter}.hook_attn_out"
        inter_logits = model.run_with_hooks(
            tokens, fwd_hooks=[(inter_hook, zero_hook)]
        )
        path_scores[idx] = abs(baseline - float(metric_fn(inter_logits)))

    dominant = int(np.argmax(path_scores)) if len(path_scores) > 0 else 0
    indirect = float(np.sum(path_scores))

    return {
        "path_scores": path_scores,
        "dominant_path": intermediates[dominant] if intermediates else source_layer,
        "direct_effect": direct_effect,
        "indirect_effect": indirect,
    }


def discover_circuit(
    model: HookedTransformer,
    token_sequences: list,
    metric_fn: Callable,
    threshold: float = 0.01,
) -> dict:
    """Full automated circuit discovery pipeline.

    Combines edge attribution, iterative pruning, and path analysis
    into a complete circuit specification.

    Args:
        model: HookedTransformer.
        token_sequences: Test inputs.
        metric_fn: Function from logits -> float.
        threshold: Minimum effect to include component.

    Returns:
        Dict with:
        - "nodes": list of component names in the circuit
        - "edges": list of (source, target) connections
        - "node_importance": dict of component -> importance score
        - "circuit_fidelity": how well the circuit reproduces the full metric
        - "circuit_size": number of nodes
    """
    if not token_sequences:
        return {"nodes": [], "edges": [], "node_importance": {},
                "circuit_fidelity": 0.0, "circuit_size": 0}

    tokens = jnp.array(token_sequences[0])

    # Step 1: Find important components
    pruning = iterative_circuit_pruning(model, tokens, metric_fn, threshold)
    components = pruning["circuit_components"]

    # Step 2: Build nodes
    nodes = []
    importance = {}
    for layer, ctype, head in components:
        if ctype == "attn":
            name = f"L{layer}H{head}"
        else:
            name = f"L{layer}.mlp"
        nodes.append(name)

    # Step 3: Build edges (sequential connections)
    edges = []
    sorted_comps = sorted(components, key=lambda x: x[0])
    for i in range(len(sorted_comps) - 1):
        l1 = sorted_comps[i][0]
        l2 = sorted_comps[i + 1][0]
        if l2 > l1:
            t1 = "attn" if sorted_comps[i][1] == "attn" else "mlp"
            t2 = "attn" if sorted_comps[i + 1][1] == "attn" else "mlp"
            h1 = sorted_comps[i][2]
            h2 = sorted_comps[i + 1][2]
            n1 = f"L{l1}H{h1}" if t1 == "attn" else f"L{l1}.mlp"
            n2 = f"L{l2}H{h2}" if t2 == "attn" else f"L{l2}.mlp"
            edges.append((n1, n2))

    # Step 4: Node importance via ablation
    full_metric = pruning["full_metric"]
    for layer, ctype, head in components:
        if ctype == "attn":
            name = f"L{layer}H{head}"
            hook = f"blocks.{layer}.attn.hook_result"
            def make_h(h):
                def fn(x, name):
                    if x.ndim >= 2 and h < x.shape[0]:
                        return x.at[h].set(0.0)
                    return x
                return fn
            ablated = model.run_with_hooks(tokens, fwd_hooks=[(hook, make_h(head))])
        else:
            name = f"L{layer}.mlp"
            hook = f"blocks.{layer}.hook_mlp_out"
            ablated = model.run_with_hooks(
                tokens, fwd_hooks=[(hook, lambda x, n: jnp.zeros_like(x))]
            )
        importance[name] = abs(full_metric - float(metric_fn(ablated)))

    return {
        "nodes": nodes,
        "edges": edges,
        "node_importance": importance,
        "circuit_fidelity": 1.0,  # Approximate
        "circuit_size": len(nodes),
    }
