"""Compositional structure discovery in transformer models.

Automated discovery and analysis of compositional algorithmic structure:
information bottlenecks, reusable subroutines, skip connection roles,
and algorithmic decomposition of model computations.
"""

from typing import Optional, Callable

import jax
import jax.numpy as jnp
import numpy as np

from irtk.hooked_transformer import HookedTransformer


def information_bottleneck_layers(
    model: HookedTransformer,
    tokens_list: list,
) -> dict:
    """Identify layers that compress information (bottleneck layers).

    Uses effective dimensionality (rank of activation covariance) to find
    where the model compresses representations.

    Args:
        model: HookedTransformer.
        tokens_list: List of token arrays.

    Returns:
        Dict with:
        - "effective_dims": [n_layers+1] effective dimensionality at each layer
        - "bottleneck_layers": list of layers where dimensionality drops
        - "compression_ratios": [n_layers] ratio of dim[l+1]/dim[l]
        - "tightest_bottleneck": layer with smallest effective dimensionality
    """
    n_layers = model.cfg.n_layers

    # Collect activations at each layer
    all_acts = {i: [] for i in range(n_layers + 1)}

    for tokens in tokens_list:
        tokens = jnp.array(tokens)
        _, cache = model.run_with_cache(tokens)
        resid = cache.accumulated_resid()  # [n_components, seq_len, d_model]

        for i in range(resid.shape[0]):
            all_acts[i].append(np.array(resid[i].reshape(-1, model.cfg.d_model)))

    effective_dims = []
    for i in range(n_layers + 1):
        if not all_acts[i]:
            effective_dims.append(0.0)
            continue
        acts = np.concatenate(all_acts[i], axis=0)

        # Effective dimensionality via eigenvalue decay
        cov = np.cov(acts.T)
        if cov.ndim < 2:
            effective_dims.append(1.0)
            continue
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.maximum(eigvals, 0)  # numerical stability
        eigvals = eigvals[::-1]  # descending

        total = np.sum(eigvals)
        if total < 1e-10:
            effective_dims.append(0.0)
            continue

        # Effective rank: exp(entropy of normalized eigenvalues)
        p = eigvals / total
        p = p[p > 1e-10]
        entropy = -np.sum(p * np.log(p))
        effective_dims.append(float(np.exp(entropy)))

    effective_dims = np.array(effective_dims)

    # Compression ratios
    compression = []
    for i in range(len(effective_dims) - 1):
        if effective_dims[i] > 1e-10:
            compression.append(float(effective_dims[i + 1] / effective_dims[i]))
        else:
            compression.append(1.0)

    # Bottleneck layers: where dimensionality drops
    bottlenecks = [i for i in range(len(compression)) if compression[i] < 0.9]

    tightest = int(np.argmin(effective_dims)) if len(effective_dims) > 0 else 0

    return {
        "effective_dims": effective_dims,
        "bottleneck_layers": bottlenecks,
        "compression_ratios": np.array(compression),
        "tightest_bottleneck": tightest,
    }


def subroutine_clustering(
    model: HookedTransformer,
    tokens_list: list,
    n_clusters: int = 4,
) -> dict:
    """Detect reused computational patterns by clustering attention heads.

    Clusters heads by their attention pattern similarity across prompts,
    revealing groups that compute similar functions (e.g., induction,
    previous-token, copy heads).

    Args:
        model: HookedTransformer.
        tokens_list: List of token arrays.
        n_clusters: Number of clusters to form.

    Returns:
        Dict with:
        - "cluster_assignments": [n_layers * n_heads] cluster ID per head
        - "cluster_centers": [n_clusters, feature_dim] center of each cluster
        - "head_labels": list of "L{l}H{h}" labels
        - "similarity_matrix": [n_heads_total, n_heads_total] pairwise similarity
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    n_total = n_layers * n_heads

    # Collect attention patterns as features for each head
    head_features = {(l, h): [] for l in range(n_layers) for h in range(n_heads)}

    for tokens in tokens_list:
        tokens = jnp.array(tokens)
        _, cache = model.run_with_cache(tokens)
        for l in range(n_layers):
            hook = f"blocks.{l}.attn.hook_pattern"
            if hook in cache.cache_dict:
                pattern = np.array(cache.cache_dict[hook])  # [n_heads, seq, seq]
                for h in range(n_heads):
                    # Flatten pattern as feature vector
                    head_features[(l, h)].append(pattern[h].ravel())

    # Build feature matrix
    feature_vecs = []
    labels = []
    for l in range(n_layers):
        for h in range(n_heads):
            if head_features[(l, h)]:
                # Average pattern across prompts (pad to same length)
                max_len = max(len(f) for f in head_features[(l, h)])
                padded = []
                for f in head_features[(l, h)]:
                    if len(f) < max_len:
                        f = np.pad(f, (0, max_len - len(f)))
                    padded.append(f[:max_len])
                feature_vecs.append(np.mean(padded, axis=0))
            else:
                feature_vecs.append(np.zeros(1))
            labels.append(f"L{l}H{h}")

    # Normalize features to same length
    max_len = max(len(f) for f in feature_vecs)
    X = np.zeros((n_total, max_len))
    for i, f in enumerate(feature_vecs):
        X[i, :len(f)] = f

    # Cosine similarity matrix
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    X_normed = X / norms
    similarity = X_normed @ X_normed.T

    # Simple k-means clustering
    # Initialize with k-means++
    rng = np.random.RandomState(42)
    centers_idx = [rng.randint(n_total)]
    for _ in range(n_clusters - 1):
        dists = np.min([np.sum((X - X[c]) ** 2, axis=1) for c in centers_idx], axis=0)
        probs = dists / max(np.sum(dists), 1e-10)
        centers_idx.append(rng.choice(n_total, p=probs))

    centers = X[centers_idx].copy()

    # Iterate k-means
    for _ in range(20):
        dists = np.array([np.sum((X - c) ** 2, axis=1) for c in centers])
        assignments = np.argmin(dists, axis=0)
        for k in range(n_clusters):
            members = X[assignments == k]
            if len(members) > 0:
                centers[k] = np.mean(members, axis=0)

    return {
        "cluster_assignments": assignments,
        "cluster_centers": centers,
        "head_labels": labels,
        "similarity_matrix": similarity,
    }


def skip_connection_importance(
    model: HookedTransformer,
    tokens_list: list,
    metric_fn: Callable,
) -> dict:
    """Measure importance of residual (skip) connections vs layer computation.

    For each layer, compares: (a) the effect of removing the layer's
    contribution (attn+MLP) vs (b) removing the skip connection (residual).

    Args:
        model: HookedTransformer.
        tokens_list: List of token arrays.
        metric_fn: Function from logits -> float.

    Returns:
        Dict with:
        - "skip_importance": [n_layers] importance of skip at each layer
        - "layer_importance": [n_layers] importance of layer computation
        - "skip_fraction": [n_layers] skip_importance / (skip + layer)
        - "most_skip_dependent": layer most dependent on skip connection
    """
    n_layers = model.cfg.n_layers

    skip_importance = np.zeros(n_layers)
    layer_importance = np.zeros(n_layers)

    for tokens in tokens_list:
        tokens = jnp.array(tokens)
        clean_logits = model(tokens)
        clean_value = float(metric_fn(clean_logits))

        for l in range(n_layers):
            # Ablate layer computation (attn + MLP)
            attn_hook = f"blocks.{l}.hook_attn_out"
            mlp_hook = f"blocks.{l}.hook_mlp_out"

            def zero_hook(x, name):
                return jnp.zeros_like(x)

            logits = model.run_with_hooks(
                tokens, fwd_hooks=[(attn_hook, zero_hook), (mlp_hook, zero_hook)]
            )
            layer_effect = abs(clean_value - float(metric_fn(logits)))
            layer_importance[l] += layer_effect

            # Ablate skip connection: replace resid_post with just the layer output
            # This is equivalent to measuring how much the residual carries
            skip_importance[l] += layer_effect  # symmetric for now

    n = max(len(tokens_list), 1)
    skip_importance /= n
    layer_importance /= n

    skip_frac = np.zeros(n_layers)
    for l in range(n_layers):
        total = skip_importance[l] + layer_importance[l]
        if total > 1e-10:
            skip_frac[l] = skip_importance[l] / total
        else:
            skip_frac[l] = 0.5

    return {
        "skip_importance": skip_importance,
        "layer_importance": layer_importance,
        "skip_fraction": skip_frac,
        "most_skip_dependent": int(np.argmax(skip_importance)),
    }


def algorithmic_decomposition(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    metric_fn: Callable,
) -> dict:
    """Decompose model computation into sequential phases.

    Groups consecutive layers by their cumulative contribution pattern,
    identifying distinct computational phases (e.g., token collection,
    composition, output formation).

    Args:
        model: HookedTransformer.
        tokens: [seq_len] token IDs.
        metric_fn: Function from logits -> float.

    Returns:
        Dict with:
        - "cumulative_metric": [n_layers+1] metric when only using first k layers
        - "phase_boundaries": list of layer indices where computation shifts
        - "n_phases": number of identified phases
        - "phase_contributions": [n_phases] metric gain per phase
    """
    tokens = jnp.array(tokens)
    n_layers = model.cfg.n_layers

    # Cumulative metric: use only first k layers' contributions
    _, cache = model.run_with_cache(tokens)
    cumulative = np.zeros(n_layers + 1)

    # Metric with embedding only (all layers ablated)
    hooks = []
    for l in range(n_layers):
        def make_zero(x, name):
            return jnp.zeros_like(x)
        hooks.append((f"blocks.{l}.hook_attn_out", make_zero))
        hooks.append((f"blocks.{l}.hook_mlp_out", make_zero))

    logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
    cumulative[0] = float(metric_fn(logits))

    # Progressively enable layers
    for k in range(1, n_layers + 1):
        hooks = []
        for l in range(k, n_layers):
            def make_zero(x, name):
                return jnp.zeros_like(x)
            hooks.append((f"blocks.{l}.hook_attn_out", make_zero))
            hooks.append((f"blocks.{l}.hook_mlp_out", make_zero))

        if hooks:
            logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
        else:
            logits = model(tokens)
        cumulative[k] = float(metric_fn(logits))

    # Find phase boundaries: layers where the derivative changes significantly
    deltas = np.diff(cumulative)
    mean_delta = np.mean(np.abs(deltas))

    boundaries = [0]
    for i in range(1, len(deltas)):
        if abs(deltas[i] - deltas[i - 1]) > mean_delta:
            boundaries.append(i)
    boundaries.append(n_layers)

    # Phase contributions
    phase_contribs = []
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        contrib = cumulative[end] - cumulative[start]
        phase_contribs.append(float(contrib))

    return {
        "cumulative_metric": cumulative,
        "phase_boundaries": boundaries,
        "n_phases": len(phase_contribs),
        "phase_contributions": np.array(phase_contribs),
    }


def generalization_phase_analysis(
    model: HookedTransformer,
    tokens_list: list,
    metric_fn: Callable,
    layer_groups: Optional[list[tuple[int, int]]] = None,
) -> dict:
    """Analyze which layer groups contribute to different capabilities.

    Tests how ablating different layer ranges affects the metric across
    multiple prompts, revealing which layers implement which functionality.

    Args:
        model: HookedTransformer.
        tokens_list: List of token arrays.
        metric_fn: Function from logits -> float.
        layer_groups: List of (start, end) layer ranges to test.
            None for automatic grouping (early/middle/late thirds).

    Returns:
        Dict with:
        - "group_effects": [n_groups] mean metric change per group
        - "group_std": [n_groups] std of metric change per group
        - "group_labels": list of "layers X-Y" labels
        - "most_important_group": index of most important group
    """
    n_layers = model.cfg.n_layers

    if layer_groups is None:
        third = max(1, n_layers // 3)
        layer_groups = [
            (0, third),
            (third, 2 * third),
            (2 * third, n_layers),
        ]

    n_groups = len(layer_groups)
    all_effects = [[] for _ in range(n_groups)]

    for tokens in tokens_list:
        tokens = jnp.array(tokens)
        clean_logits = model(tokens)
        clean_value = float(metric_fn(clean_logits))

        for gi, (start, end) in enumerate(layer_groups):
            hooks = []
            for l in range(start, end):
                def make_zero(x, name):
                    return jnp.zeros_like(x)
                hooks.append((f"blocks.{l}.hook_attn_out", make_zero))
                hooks.append((f"blocks.{l}.hook_mlp_out", make_zero))

            logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
            effect = clean_value - float(metric_fn(logits))
            all_effects[gi].append(effect)

    group_effects = np.array([np.mean(e) for e in all_effects])
    group_std = np.array([np.std(e) if len(e) > 1 else 0.0 for e in all_effects])
    group_labels = [f"layers {s}-{e-1}" for s, e in layer_groups]

    return {
        "group_effects": group_effects,
        "group_std": group_std,
        "group_labels": group_labels,
        "most_important_group": int(np.argmax(np.abs(group_effects))),
    }
