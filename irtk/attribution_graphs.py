"""Attribution graph construction and analysis.

Build directed feature-to-feature computation graphs using SAE/transcoder
features as nodes and activation-based attribution as edges. Based on
Anthropic's 2025 circuit tracing methodology.
"""

from typing import Optional, Callable
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from irtk.hooked_transformer import HookedTransformer
from irtk.sae import SparseAutoencoder


@dataclass
class AttributionGraph:
    """Directed graph of feature-to-feature attributions.

    Nodes are (layer, feature_idx) tuples.
    Edges are (source_node, target_node, weight) tuples.
    """

    nodes: list  # list of (layer, feature_idx)
    edges: list  # list of (source_node_idx, target_node_idx, weight)
    node_labels: list  # string labels for each node
    node_importances: np.ndarray  # importance score per node
    n_layers: int


def build_attribution_graph(
    model: HookedTransformer,
    saes: dict,
    tokens: jnp.ndarray,
    metric_fn: Callable,
    threshold: float = 0.01,
) -> AttributionGraph:
    """Construct a feature-level attribution graph.

    Builds a directed graph where nodes are SAE features at each layer
    and edges represent how much one feature's activation contributes
    to another feature's activation in the next layer.

    Args:
        model: HookedTransformer.
        saes: Dict mapping hook_name -> SparseAutoencoder for each layer.
        tokens: Token sequence to analyze.
        metric_fn: Function from logits -> float.
        threshold: Minimum edge weight to include.

    Returns:
        AttributionGraph with nodes, edges, labels, and importances.
    """
    tokens = jnp.array(tokens)
    _, cache = model.run_with_cache(tokens)

    # Encode activations at each hook point
    layer_features = {}  # hook_name -> (feature_acts, active_indices)
    for hook_name, sae in saes.items():
        if hook_name not in cache.cache_dict:
            continue
        acts = cache.cache_dict[hook_name]
        # Use last position
        act = acts[-1] if acts.ndim > 1 else acts
        feat_acts = np.array(sae.encode(act))
        active = np.where(feat_acts > 0)[0]
        layer_features[hook_name] = (feat_acts, active)

    # Build nodes
    nodes = []
    node_labels = []
    hook_names_sorted = sorted(layer_features.keys())
    node_map = {}  # (hook_name, feat_idx) -> node_idx

    for hook_name in hook_names_sorted:
        feat_acts, active = layer_features[hook_name]
        for feat_idx in active:
            node_idx = len(nodes)
            nodes.append((hook_name, int(feat_idx)))
            node_labels.append(f"{hook_name}:f{feat_idx}")
            node_map[(hook_name, int(feat_idx))] = node_idx

    if not nodes:
        return AttributionGraph(
            nodes=[], edges=[], node_labels=[], node_importances=np.array([]),
            n_layers=model.cfg.n_layers,
        )

    # Build edges: for consecutive hook pairs, compute attribution
    edges = []
    for i in range(len(hook_names_sorted) - 1):
        src_hook = hook_names_sorted[i]
        tgt_hook = hook_names_sorted[i + 1]
        src_feats, src_active = layer_features[src_hook]
        tgt_feats, tgt_active = layer_features[tgt_hook]
        src_sae = saes[src_hook]
        tgt_sae = saes[tgt_hook]

        for src_feat in src_active:
            src_dir = np.array(src_sae.W_dec[src_feat])  # [d_model]
            src_act = float(src_feats[src_feat])

            for tgt_feat in tgt_active:
                tgt_enc = np.array(tgt_sae.W_enc[:, tgt_feat])  # [d_model]
                # Attribution: how much does source feature contribute to target encoding
                weight = float(src_act * np.dot(src_dir, tgt_enc))

                if abs(weight) > threshold:
                    src_node = node_map.get((src_hook, int(src_feat)))
                    tgt_node = node_map.get((tgt_hook, int(tgt_feat)))
                    if src_node is not None and tgt_node is not None:
                        edges.append((src_node, tgt_node, weight))

    # Node importances: sum of absolute outgoing + incoming edge weights
    importances = np.zeros(len(nodes))
    for src, tgt, w in edges:
        importances[src] += abs(w)
        importances[tgt] += abs(w)

    return AttributionGraph(
        nodes=nodes,
        edges=edges,
        node_labels=node_labels,
        node_importances=importances,
        n_layers=model.cfg.n_layers,
    )


def node_importance(
    graph: AttributionGraph,
    source_nodes: Optional[list[int]] = None,
    target_nodes: Optional[list[int]] = None,
    method: str = "flow",
) -> dict:
    """Compute importance of each node in the attribution graph.

    Args:
        graph: AttributionGraph from build_attribution_graph.
        source_nodes: Input node indices (None = first layer nodes).
        target_nodes: Output node indices (None = last layer nodes).
        method: "flow" for max-flow based, "degree" for weighted degree.

    Returns:
        Dict with:
        - "importances": [n_nodes] importance score per node
        - "top_nodes": list of (node_idx, importance) sorted descending
        - "method": method used
    """
    n = len(graph.nodes)
    if n == 0:
        return {"importances": np.array([]), "top_nodes": [], "method": method}

    if method == "degree":
        importances = graph.node_importances.copy()
    else:
        # Flow-based: propagate importance from targets backward
        importances = np.zeros(n)

        # Build adjacency
        outgoing = {i: [] for i in range(n)}
        incoming = {i: [] for i in range(n)}
        for src, tgt, w in graph.edges:
            outgoing[src].append((tgt, w))
            incoming[tgt].append((src, w))

        # Initialize target nodes
        if target_nodes is None:
            # Use last-layer nodes
            if graph.nodes:
                last_hook = graph.nodes[-1][0]
                target_nodes = [i for i, (h, _) in enumerate(graph.nodes) if h == last_hook]
            else:
                target_nodes = []

        for t in target_nodes:
            importances[t] = 1.0

        # Backward propagation
        for _ in range(10):
            new_imp = importances.copy()
            for tgt in range(n):
                if importances[tgt] > 0:
                    for src, w in incoming[tgt]:
                        new_imp[src] += abs(w) * importances[tgt]
            # Normalize
            max_imp = np.max(new_imp)
            if max_imp > 0:
                new_imp /= max_imp
            importances = new_imp

    order = np.argsort(importances)[::-1]
    top_nodes = [(int(i), float(importances[i])) for i in order if importances[i] > 0]

    return {
        "importances": importances,
        "top_nodes": top_nodes,
        "method": method,
    }


def prune_graph(
    graph: AttributionGraph,
    threshold: float = 0.1,
    method: str = "edge_weight",
    max_nodes: Optional[int] = None,
) -> AttributionGraph:
    """Prune an attribution graph to its essential subgraph.

    Args:
        graph: Full AttributionGraph.
        threshold: Minimum importance/weight to keep.
        method: "edge_weight" (prune by absolute edge weight),
                "node_importance" (prune by node importance).
        max_nodes: Maximum number of nodes to keep.

    Returns:
        Pruned AttributionGraph.
    """
    if not graph.nodes:
        return graph

    if method == "node_importance":
        imp = graph.node_importances
        max_imp = np.max(imp) if len(imp) > 0 else 1.0
        if max_imp > 0:
            keep_mask = imp >= threshold * max_imp
        else:
            keep_mask = np.ones(len(graph.nodes), dtype=bool)
    else:
        # Keep nodes that have at least one edge above threshold
        max_w = max((abs(w) for _, _, w in graph.edges), default=1.0)
        keep_set = set()
        for src, tgt, w in graph.edges:
            if abs(w) >= threshold * max_w:
                keep_set.add(src)
                keep_set.add(tgt)
        keep_mask = np.array([i in keep_set for i in range(len(graph.nodes))])

    if max_nodes is not None and np.sum(keep_mask) > max_nodes:
        # Keep top by importance
        order = np.argsort(graph.node_importances)[::-1][:max_nodes]
        keep_mask = np.zeros(len(graph.nodes), dtype=bool)
        keep_mask[order] = True

    # Remap
    old_to_new = {}
    new_nodes = []
    new_labels = []
    new_imp = []
    for i in range(len(graph.nodes)):
        if keep_mask[i]:
            old_to_new[i] = len(new_nodes)
            new_nodes.append(graph.nodes[i])
            new_labels.append(graph.node_labels[i])
            new_imp.append(graph.node_importances[i])

    new_edges = []
    for src, tgt, w in graph.edges:
        if src in old_to_new and tgt in old_to_new:
            new_edges.append((old_to_new[src], old_to_new[tgt], w))

    return AttributionGraph(
        nodes=new_nodes,
        edges=new_edges,
        node_labels=new_labels,
        node_importances=np.array(new_imp) if new_imp else np.array([]),
        n_layers=graph.n_layers,
    )


def visualize_attribution_graph(
    graph: AttributionGraph,
    top_k_nodes: int = 20,
    highlight_nodes: Optional[list[int]] = None,
) -> dict:
    """Prepare attribution graph data for visualization.

    Returns structured data that can be rendered with graphviz, networkx, etc.

    Args:
        graph: AttributionGraph to visualize.
        top_k_nodes: Maximum number of nodes to include.
        highlight_nodes: Node indices to highlight.

    Returns:
        Dict with:
        - "nodes": list of dicts with id, label, importance, layer, highlighted
        - "edges": list of dicts with source, target, weight
        - "n_nodes": number of nodes
        - "n_edges": number of edges
    """
    if not graph.nodes:
        return {"nodes": [], "edges": [], "n_nodes": 0, "n_edges": 0}

    # Select top nodes by importance
    order = np.argsort(graph.node_importances)[::-1][:top_k_nodes]
    keep = set(int(i) for i in order)

    highlight_set = set(highlight_nodes or [])

    vis_nodes = []
    node_remap = {}
    for idx in order:
        idx = int(idx)
        node_remap[idx] = len(vis_nodes)
        hook_name, feat_idx = graph.nodes[idx]
        vis_nodes.append({
            "id": len(vis_nodes),
            "label": graph.node_labels[idx],
            "importance": float(graph.node_importances[idx]),
            "hook": hook_name,
            "feature": feat_idx,
            "highlighted": idx in highlight_set,
        })

    vis_edges = []
    for src, tgt, w in graph.edges:
        if src in node_remap and tgt in node_remap:
            vis_edges.append({
                "source": node_remap[src],
                "target": node_remap[tgt],
                "weight": float(w),
            })

    return {
        "nodes": vis_nodes,
        "edges": vis_edges,
        "n_nodes": len(vis_nodes),
        "n_edges": len(vis_edges),
    }


def attribution_graph_faithfulness(
    model: HookedTransformer,
    graph: AttributionGraph,
    saes: dict,
    tokens: jnp.ndarray,
    metric_fn: Callable,
) -> dict:
    """Evaluate whether a pruned graph reproduces full model behavior.

    Measures how well the features in the graph account for the model's
    output on the given metric.

    Args:
        model: HookedTransformer.
        graph: (Possibly pruned) AttributionGraph.
        saes: Dict mapping hook_name -> SparseAutoencoder.
        tokens: Token sequence.
        metric_fn: Function from logits -> float.

    Returns:
        Dict with:
        - "full_metric": metric on unmodified model
        - "graph_metric": metric when only graph features are active
        - "faithfulness": graph_metric / full_metric (1.0 = perfect)
        - "n_active_features": number of features in graph
        - "feature_coverage": fraction of total active features in graph
    """
    tokens = jnp.array(tokens)

    # Full metric
    full_logits = model(tokens)
    full_metric = float(metric_fn(full_logits))

    # Count features in graph by hook
    graph_features = {}  # hook_name -> set of feature indices
    for hook_name, feat_idx in graph.nodes:
        if hook_name not in graph_features:
            graph_features[hook_name] = set()
        graph_features[hook_name].add(feat_idx)

    # Get all active features for coverage
    _, cache = model.run_with_cache(tokens)
    total_active = 0
    graph_active = 0

    for hook_name, sae in saes.items():
        if hook_name not in cache.cache_dict:
            continue
        acts = cache.cache_dict[hook_name]
        act = acts[-1] if acts.ndim > 1 else acts
        feat_acts = np.array(sae.encode(act))
        active = set(int(i) for i in np.where(feat_acts > 0)[0])
        total_active += len(active)
        if hook_name in graph_features:
            graph_active += len(active & graph_features[hook_name])

    coverage = graph_active / max(total_active, 1)

    # Ablate non-graph features by zeroing their activations
    def make_ablation_hook(hook_name, sae, keep_features):
        def hook(x, name):
            act = x[-1] if x.ndim > 1 else x
            feat_acts = sae.encode(act)
            # Zero out features not in graph
            mask = jnp.zeros(sae.n_features)
            for f in keep_features:
                mask = mask.at[f].set(1.0)
            masked_feats = feat_acts * mask
            recon = sae.decode(masked_feats)
            if x.ndim > 1:
                return x.at[-1].set(recon + (act - sae.decode(sae.encode(act))))
            return recon
        return hook

    fwd_hooks = []
    for hook_name, sae in saes.items():
        keep = graph_features.get(hook_name, set())
        if keep and hook_name in cache.cache_dict:
            fwd_hooks.append((hook_name, make_ablation_hook(hook_name, sae, keep)))

    if fwd_hooks:
        graph_logits = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)
        graph_metric = float(metric_fn(graph_logits))
    else:
        graph_metric = full_metric

    faithfulness = graph_metric / full_metric if abs(full_metric) > 1e-10 else 1.0

    return {
        "full_metric": full_metric,
        "graph_metric": graph_metric,
        "faithfulness": float(faithfulness),
        "n_active_features": len(graph.nodes),
        "feature_coverage": float(coverage),
    }
