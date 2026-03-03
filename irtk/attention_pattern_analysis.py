"""Deep attention pattern analysis.

Analyzes attention pattern structure beyond basic visualization: entropy
distribution, positional biases, attention distance, sparsity patterns,
and cross-head consistency.

References:
    Clark et al. (2019) "What Does BERT Look At?"
    Kovaleva et al. (2019) "Revealing the Dark Secrets of BERT"
"""

import jax
import jax.numpy as jnp
import numpy as np


def attention_entropy_profile(model, tokens):
    """Compute entropy of attention distributions for all heads.

    High entropy = uniform attention (diffuse). Low entropy = focused attention (sharp).

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].

    Returns:
        dict with:
            entropy_matrix: array [n_layers, n_heads] of mean attention entropy
            sharpest_head: tuple (layer, head) with lowest entropy
            most_diffuse_head: tuple (layer, head) with highest entropy
            entropy_by_layer: array [n_layers] of mean entropy per layer
            max_possible_entropy: float, entropy of uniform distribution
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    seq_len = len(tokens)

    hook_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=hook_state)
    cache = hook_state.cache

    entropy_matrix = np.zeros((n_layers, n_heads))

    for layer in range(n_layers):
        pattern = cache.get(f"blocks.{layer}.attn.hook_pattern")
        if pattern is not None:
            # pattern: [n_heads, seq_len, seq_len]
            for h in range(n_heads):
                attn = np.array(pattern[h])  # [seq, seq]
                # Compute per-query entropy, average over positions
                entropies = []
                for q in range(seq_len):
                    probs = attn[q, :q + 1]  # causal: only attend to positions 0..q
                    probs = probs + 1e-10
                    probs = probs / probs.sum()
                    ent = -np.sum(probs * np.log(probs))
                    entropies.append(ent)
                entropy_matrix[layer, h] = np.mean(entropies)

    sharpest = np.unravel_index(np.argmin(entropy_matrix), entropy_matrix.shape)
    most_diffuse = np.unravel_index(np.argmax(entropy_matrix), entropy_matrix.shape)
    layer_means = np.mean(entropy_matrix, axis=1)
    max_ent = float(np.log(seq_len))

    return {
        "entropy_matrix": entropy_matrix,
        "sharpest_head": (int(sharpest[0]), int(sharpest[1])),
        "most_diffuse_head": (int(most_diffuse[0]), int(most_diffuse[1])),
        "entropy_by_layer": layer_means,
        "max_possible_entropy": max_ent,
    }


def positional_attention_bias(model, tokens):
    """Measure how much attention is biased toward specific positions.

    Detects common patterns: BOS-attention, recency bias, positional periodicity.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].

    Returns:
        dict with:
            bos_attention: array [n_layers, n_heads] of mean attention to position 0
            recency_bias: array [n_layers, n_heads] of mean attention to most recent position
            diagonal_strength: array [n_layers, n_heads] of self-attention strength
            mean_attention_distance: array [n_layers, n_heads] of avg distance attended to
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    seq_len = len(tokens)

    hook_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=hook_state)
    cache = hook_state.cache

    bos = np.zeros((n_layers, n_heads))
    recency = np.zeros((n_layers, n_heads))
    diagonal = np.zeros((n_layers, n_heads))
    distance = np.zeros((n_layers, n_heads))

    for layer in range(n_layers):
        pattern = cache.get(f"blocks.{layer}.attn.hook_pattern")
        if pattern is not None:
            for h in range(n_heads):
                attn = np.array(pattern[h])  # [seq, seq]
                # BOS attention: mean attention to position 0
                bos[layer, h] = np.mean(attn[:, 0])
                # Recency: for each query, attention to the query position itself (diagonal)
                diag_vals = np.diag(attn)
                diagonal[layer, h] = np.mean(diag_vals)
                # Recency: for position > 0, attention to position q (self) or q-1
                rec_vals = []
                for q in range(1, seq_len):
                    rec_vals.append(attn[q, q])  # attention to self (most recent allowed)
                recency[layer, h] = np.mean(rec_vals) if rec_vals else 0.0
                # Mean attention distance
                dist_vals = []
                for q in range(seq_len):
                    positions = np.arange(q + 1)
                    distances = q - positions
                    mean_dist = np.sum(attn[q, :q + 1] * distances)
                    dist_vals.append(mean_dist)
                distance[layer, h] = np.mean(dist_vals)

    return {
        "bos_attention": bos,
        "recency_bias": recency,
        "diagonal_strength": diagonal,
        "mean_attention_distance": distance,
    }


def attention_sparsity_analysis(model, tokens, threshold=0.1):
    """Analyze sparsity of attention patterns.

    Measures how concentrated attention is: sparse patterns attend to
    few tokens, dense patterns spread attention broadly.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        threshold: Attention weight threshold for counting "attended" tokens.

    Returns:
        dict with:
            sparsity_matrix: array [n_layers, n_heads] of mean sparsity (fraction below threshold)
            mean_tokens_attended: array [n_layers, n_heads] of avg tokens with weight > threshold
            sparsest_head: tuple (layer, head) with highest sparsity
            densest_head: tuple (layer, head) with lowest sparsity
            gini_coefficients: array [n_layers, n_heads] of Gini coefficient (inequality measure)
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    seq_len = len(tokens)

    hook_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=hook_state)
    cache = hook_state.cache

    sparsity = np.zeros((n_layers, n_heads))
    mean_attended = np.zeros((n_layers, n_heads))
    gini = np.zeros((n_layers, n_heads))

    for layer in range(n_layers):
        pattern = cache.get(f"blocks.{layer}.attn.hook_pattern")
        if pattern is not None:
            for h in range(n_heads):
                attn = np.array(pattern[h])  # [seq, seq]
                sparsity_vals = []
                attended_vals = []
                gini_vals = []
                for q in range(seq_len):
                    probs = attn[q, :q + 1]
                    below = np.mean(probs < threshold)
                    sparsity_vals.append(below)
                    attended_vals.append(np.sum(probs >= threshold))
                    # Gini coefficient
                    sorted_probs = np.sort(probs)
                    n = len(sorted_probs)
                    if n > 0 and np.sum(sorted_probs) > 1e-10:
                        index = np.arange(1, n + 1)
                        g = (2 * np.sum(index * sorted_probs) - (n + 1) * np.sum(sorted_probs)) / (n * np.sum(sorted_probs))
                        gini_vals.append(g)

                sparsity[layer, h] = np.mean(sparsity_vals)
                mean_attended[layer, h] = np.mean(attended_vals)
                gini[layer, h] = np.mean(gini_vals) if gini_vals else 0.0

    sparsest = np.unravel_index(np.argmax(sparsity), sparsity.shape)
    densest = np.unravel_index(np.argmin(sparsity), sparsity.shape)

    return {
        "sparsity_matrix": sparsity,
        "mean_tokens_attended": mean_attended,
        "sparsest_head": (int(sparsest[0]), int(sparsest[1])),
        "densest_head": (int(densest[0]), int(densest[1])),
        "gini_coefficients": gini,
    }


def cross_head_attention_similarity(model, tokens):
    """Measure similarity between attention patterns of different heads.

    Heads with similar patterns may be redundant; diverse patterns suggest
    each head captures different relationships.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].

    Returns:
        dict with:
            within_layer_similarity: array [n_layers] of mean pairwise cosine within each layer
            across_layer_similarity: float, mean pairwise similarity across all heads
            most_similar_pair: tuple ((l1,h1), (l2,h2)) most similar head pair
            most_dissimilar_pair: tuple, most dissimilar head pair
            redundancy_score: float, fraction of pairs with similarity > 0.8
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    hook_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=hook_state)
    cache = hook_state.cache

    # Flatten all attention patterns
    all_patterns = []
    all_labels = []
    for layer in range(n_layers):
        pattern = cache.get(f"blocks.{layer}.attn.hook_pattern")
        if pattern is not None:
            for h in range(n_heads):
                flat = np.array(pattern[h]).flatten()
                norm = np.linalg.norm(flat)
                if norm > 1e-10:
                    flat = flat / norm
                all_patterns.append(flat)
                all_labels.append((layer, h))

    n = len(all_patterns)
    if n < 2:
        return {
            "within_layer_similarity": np.zeros(n_layers),
            "across_layer_similarity": 0.0,
            "most_similar_pair": ((0, 0), (0, 1)),
            "most_dissimilar_pair": ((0, 0), (0, 1)),
            "redundancy_score": 0.0,
        }

    patterns_matrix = np.stack(all_patterns)  # [n, flat_dim]
    sim_matrix = patterns_matrix @ patterns_matrix.T

    # Within-layer similarity
    within = np.zeros(n_layers)
    for layer in range(n_layers):
        layer_indices = [i for i, (l, h) in enumerate(all_labels) if l == layer]
        if len(layer_indices) >= 2:
            sims = []
            for i in range(len(layer_indices)):
                for j in range(i + 1, len(layer_indices)):
                    sims.append(sim_matrix[layer_indices[i], layer_indices[j]])
            within[layer] = np.mean(sims)

    # All pairs
    all_sims = []
    best_sim = -2.0
    worst_sim = 2.0
    best_pair = ((0, 0), (0, 1))
    worst_pair = ((0, 0), (0, 1))
    for i in range(n):
        for j in range(i + 1, n):
            s = sim_matrix[i, j]
            all_sims.append(s)
            if s > best_sim:
                best_sim = s
                best_pair = (all_labels[i], all_labels[j])
            if s < worst_sim:
                worst_sim = s
                worst_pair = (all_labels[i], all_labels[j])

    redundancy = float(np.mean(np.array(all_sims) > 0.8)) if all_sims else 0.0

    return {
        "within_layer_similarity": within,
        "across_layer_similarity": float(np.mean(all_sims)) if all_sims else 0.0,
        "most_similar_pair": best_pair,
        "most_dissimilar_pair": worst_pair,
        "redundancy_score": redundancy,
    }


def attention_head_classification(model, tokens):
    """Classify attention heads by their dominant pattern type.

    Categories: 'positional' (strong BOS/diagonal), 'content' (input-dependent),
    'previous_token' (strong off-diagonal), 'uniform' (high entropy).

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].

    Returns:
        dict with:
            classifications: array [n_layers, n_heads] of string labels
            class_counts: dict mapping class -> count
            confidence_scores: array [n_layers, n_heads] of classification confidence
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    seq_len = len(tokens)

    hook_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=hook_state)
    cache = hook_state.cache

    classifications = np.empty((n_layers, n_heads), dtype=object)
    confidences = np.zeros((n_layers, n_heads))

    for layer in range(n_layers):
        pattern = cache.get(f"blocks.{layer}.attn.hook_pattern")
        if pattern is not None:
            for h in range(n_heads):
                attn = np.array(pattern[h])
                # Features
                bos = np.mean(attn[:, 0])
                diag = np.mean(np.diag(attn))
                # Previous token: attention to position q-1
                prev_vals = [attn[q, q - 1] for q in range(1, seq_len)]
                prev = np.mean(prev_vals) if prev_vals else 0.0
                # Entropy
                entropies = []
                for q in range(seq_len):
                    probs = attn[q, :q + 1] + 1e-10
                    probs = probs / probs.sum()
                    entropies.append(-np.sum(probs * np.log(probs)))
                mean_ent = np.mean(entropies)
                max_ent = np.log(seq_len)
                uniformity = mean_ent / max_ent if max_ent > 0 else 0

                # Classify
                scores = {
                    "positional": max(bos, diag),
                    "previous_token": prev,
                    "uniform": uniformity,
                    "content": 1.0 - uniformity,  # low entropy = content-based
                }
                best = max(scores, key=scores.get)
                classifications[layer, h] = best
                confidences[layer, h] = scores[best]

    # Count
    class_counts = {}
    for c in classifications.flatten():
        class_counts[c] = class_counts.get(c, 0) + 1

    return {
        "classifications": classifications,
        "class_counts": class_counts,
        "confidence_scores": confidences,
    }
