"""Attention pattern clustering.

Cluster attention patterns across heads and layers to discover
common pattern archetypes, measure pattern diversity, and identify
functionally similar heads.
"""

import jax
import jax.numpy as jnp


def pattern_archetypes(model, tokens, n_clusters=4):
    """Discover common attention pattern archetypes via clustering.

    Uses k-means-style approach on flattened attention patterns.

    Args:
        model: HookedTransformer
        tokens: input token IDs
        n_clusters: number of archetypes to find

    Returns:
        dict with cluster assignments and archetype descriptions.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    # Collect all patterns
    all_patterns = []
    labels = []
    for l in range(n_layers):
        patterns = cache[f'blocks.{l}.attn.hook_pattern']  # [n_heads, seq, seq]
        for h in range(n_heads):
            # Summary features: row entropy, diag strength, prev-token strength
            P = patterns[h]  # [seq, seq]
            seq_len = P.shape[0]
            P_safe = jnp.maximum(P, 1e-10)
            row_entropies = -jnp.sum(P * jnp.log(P_safe), axis=-1)
            mean_entropy = float(jnp.mean(row_entropies))

            # Diagonal score
            diag_vals = jnp.diag(P)
            diag_score = float(jnp.mean(diag_vals))

            # Previous token score
            if seq_len > 1:
                prev_vals = jnp.diag(P, -1)
                prev_score = float(jnp.mean(prev_vals))
            else:
                prev_score = 0.0

            # Uniformity
            uniform_score = float(jnp.mean(jnp.max(P, axis=-1)))

            all_patterns.append([mean_entropy, diag_score, prev_score, uniform_score])
            labels.append((l, h))

    features = jnp.array(all_patterns)  # [n_total, 4]
    n_total = features.shape[0]

    # Simple k-means (few iterations)
    key = jax.random.PRNGKey(0)
    indices = jax.random.choice(key, n_total, shape=(min(n_clusters, n_total),), replace=False)
    centroids = features[indices]

    for _ in range(10):
        # Assign
        dists = jnp.sum((features[:, None, :] - centroids[None, :, :]) ** 2, axis=-1)
        assignments = jnp.argmin(dists, axis=1)
        # Update
        for k in range(min(n_clusters, n_total)):
            mask = assignments == k
            if jnp.sum(mask) > 0:
                centroids = centroids.at[k].set(jnp.mean(features[mask], axis=0))

    # Build results
    clusters = []
    for k in range(min(n_clusters, n_total)):
        mask = assignments == k
        members = [labels[i] for i in range(n_total) if mask[i]]
        if members:
            c = centroids[k]
            clusters.append({
                'cluster': k,
                'n_members': len(members),
                'members': [{'layer': l, 'head': h} for l, h in members],
                'mean_entropy': float(c[0]),
                'diag_score': float(c[1]),
                'prev_token_score': float(c[2]),
                'max_weight': float(c[3]),
            })

    return {
        'n_clusters': len(clusters),
        'clusters': clusters,
        'n_total_heads': n_total,
    }


def head_pattern_similarity(model, tokens, layer=None):
    """Compute pairwise similarity between all head attention patterns.

    Args:
        model: HookedTransformer
        tokens: input token IDs
        layer: if specified, only compare heads within this layer

    Returns:
        dict with similarity matrix.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    # Collect patterns
    flat_patterns = []
    head_labels = []
    layers_to_check = [layer] if layer is not None else range(n_layers)

    for l in layers_to_check:
        patterns = cache[f'blocks.{l}.attn.hook_pattern']
        for h in range(n_heads):
            flat_patterns.append(patterns[h].reshape(-1))
            head_labels.append((int(l), h))

    n = len(flat_patterns)
    sim_matrix = jnp.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            cos = float(jnp.sum(flat_patterns[i] * flat_patterns[j]) /
                        jnp.maximum(jnp.linalg.norm(flat_patterns[i]) *
                                    jnp.linalg.norm(flat_patterns[j]), 1e-10))
            sim_matrix = sim_matrix.at[i, j].set(cos)
            sim_matrix = sim_matrix.at[j, i].set(cos)

    # Find most/least similar pairs
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append({
                'head_a': {'layer': head_labels[i][0], 'head': head_labels[i][1]},
                'head_b': {'layer': head_labels[j][0], 'head': head_labels[j][1]},
                'cosine': float(sim_matrix[i, j]),
            })
    pairs.sort(key=lambda p: -p['cosine'])

    return {
        'head_labels': head_labels,
        'similarity_matrix': [[float(sim_matrix[i, j]) for j in range(n)] for i in range(n)],
        'most_similar': pairs[:5] if pairs else [],
        'least_similar': pairs[-5:] if pairs else [],
    }


def pattern_diversity(model, tokens):
    """Measure the diversity of attention patterns per layer.

    Returns:
        dict with per-layer diversity scores.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    results = []
    for l in range(n_layers):
        patterns = cache[f'blocks.{l}.attn.hook_pattern']  # [n_heads, seq, seq]

        # Flatten patterns and compute pairwise distances
        flat = patterns.reshape(n_heads, -1)
        # Mean pairwise cosine distance
        norms = jnp.linalg.norm(flat, axis=-1, keepdims=True)
        normalized = flat / jnp.maximum(norms, 1e-10)
        cos_matrix = normalized @ normalized.T
        # Average off-diagonal similarity
        mask = 1.0 - jnp.eye(n_heads)
        mean_sim = float(jnp.sum(cos_matrix * mask) / jnp.maximum(jnp.sum(mask), 1e-10))

        results.append({
            'layer': l,
            'mean_pairwise_similarity': mean_sim,
            'diversity': 1.0 - mean_sim,
        })

    return {
        'per_layer': results,
        'mean_diversity': float(jnp.mean(jnp.array([r['diversity'] for r in results]))),
    }


def pattern_stability_across_inputs(model, tokens1, tokens2):
    """Compare attention pattern stability between two different inputs.

    Args:
        model: HookedTransformer
        tokens1: first input
        tokens2: second input

    Returns:
        dict with per-head pattern stability.
    """
    _, cache1 = model.run_with_cache(tokens1)
    _, cache2 = model.run_with_cache(tokens2)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    results = []
    for l in range(n_layers):
        p1 = cache1[f'blocks.{l}.attn.hook_pattern']
        p2 = cache2[f'blocks.{l}.attn.hook_pattern']
        min_seq = min(p1.shape[1], p2.shape[1])

        for h in range(n_heads):
            pat1 = p1[h, :min_seq, :min_seq].reshape(-1)
            pat2 = p2[h, :min_seq, :min_seq].reshape(-1)
            cos = float(jnp.sum(pat1 * pat2) /
                        jnp.maximum(jnp.linalg.norm(pat1) * jnp.linalg.norm(pat2), 1e-10))
            results.append({
                'layer': l,
                'head': h,
                'cosine_similarity': cos,
                'is_stable': cos > 0.8,
            })

    stable_count = sum(1 for r in results if r['is_stable'])
    return {
        'per_head': results,
        'n_stable': stable_count,
        'n_total': len(results),
        'stability_fraction': stable_count / max(len(results), 1),
    }


def cross_layer_pattern_evolution(model, tokens):
    """Track how attention patterns evolve across layers.

    For each head index, measure how its pattern changes from layer to layer.

    Returns:
        dict with per-head-index pattern evolution.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    per_head_evolution = []
    for h in range(n_heads):
        layer_changes = []
        for l in range(n_layers - 1):
            p_curr = cache[f'blocks.{l}.attn.hook_pattern'][h].reshape(-1)
            p_next = cache[f'blocks.{l+1}.attn.hook_pattern'][h].reshape(-1)
            cos = float(jnp.sum(p_curr * p_next) /
                        jnp.maximum(jnp.linalg.norm(p_curr) * jnp.linalg.norm(p_next), 1e-10))
            layer_changes.append({
                'from_layer': l,
                'to_layer': l + 1,
                'cosine': cos,
            })

        per_head_evolution.append({
            'head': h,
            'layer_transitions': layer_changes,
            'mean_change': float(jnp.mean(jnp.array([1.0 - c['cosine'] for c in layer_changes])))
            if layer_changes else 0.0,
        })

    return {
        'per_head': per_head_evolution,
        'most_evolving_head': max(per_head_evolution, key=lambda h: h['mean_change'])['head']
        if per_head_evolution else 0,
    }
