"""Attention head probing.

Probes what information attention heads encode: positional information,
token identity, relative position, and per-head specialization.

References:
    Voita et al. (2019) "Analyzing Multi-Head Self-Attention"
    Clark et al. (2019) "What Does BERT Look At?"
"""

import jax
import jax.numpy as jnp
import numpy as np


def head_positional_probe(model, tokens):
    """Probe how much positional information each head encodes.

    Measures correlation between attention patterns and positional
    structure (distance, relative position).

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].

    Returns:
        dict with:
            distance_correlation: array [n_layers, n_heads] correlation with distance
            recency_score: array [n_layers, n_heads] bias toward recent tokens
            locality_score: array [n_layers, n_heads] attention concentration near diagonal
            most_positional_head: tuple (layer, head)
            least_positional_head: tuple (layer, head)
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    seq_len = len(tokens)

    hook_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=hook_state)
    cache = hook_state.cache

    # Distance matrix (causal)
    dist_matrix = np.zeros((seq_len, seq_len))
    for i in range(seq_len):
        for j in range(i + 1):
            dist_matrix[i, j] = i - j

    distance_corr = np.zeros((n_layers, n_heads))
    recency = np.zeros((n_layers, n_heads))
    locality = np.zeros((n_layers, n_heads))

    for layer in range(n_layers):
        pattern = cache.get(f"blocks.{layer}.attn.hook_pattern")
        if pattern is None:
            continue
        pattern = np.array(pattern)  # [n_heads, seq, seq]

        for h in range(n_heads):
            p = pattern[h]  # [seq, seq]

            # Distance correlation: mean weighted distance
            weighted_dist = np.sum(p * dist_matrix) / (seq_len + 1e-10)
            max_dist = np.sum(dist_matrix) / (seq_len * (seq_len + 1) / 2 + 1e-10)
            recency[layer, h] = 1.0 - weighted_dist / (max_dist + 1e-10) if max_dist > 0 else 0.0

            # Locality: fraction of attention within window of 3
            local_mask = np.zeros_like(p)
            for i in range(seq_len):
                for j in range(max(0, i - 2), i + 1):
                    local_mask[i, j] = 1.0
            locality[layer, h] = float(np.sum(p * local_mask) / (np.sum(p) + 1e-10))

            # Correlation with distance
            flat_p = p[np.tril(np.ones((seq_len, seq_len), dtype=bool))]
            flat_d = dist_matrix[np.tril(np.ones((seq_len, seq_len), dtype=bool))]
            if len(flat_p) > 1 and np.std(flat_p) > 1e-10 and np.std(flat_d) > 1e-10:
                distance_corr[layer, h] = float(np.corrcoef(flat_p, flat_d)[0, 1])

    # Combined positional score
    combined = np.abs(distance_corr) + recency + locality
    most_idx = np.unravel_index(np.argmax(combined), combined.shape)
    least_idx = np.unravel_index(np.argmin(combined), combined.shape)

    return {
        "distance_correlation": distance_corr,
        "recency_score": recency,
        "locality_score": locality,
        "most_positional_head": (int(most_idx[0]), int(most_idx[1])),
        "least_positional_head": (int(least_idx[0]), int(least_idx[1])),
    }


def head_token_identity_probe(model, tokens, pos=-1):
    """Probe whether heads encode token identity information.

    Uses the OV circuit to measure how well each head preserves or
    transforms token identity through its value-output pathway.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        pos: Position to analyze.

    Returns:
        dict with:
            identity_scores: array [n_layers, n_heads] of token identity preservation
            transformation_scores: array [n_layers, n_heads] of how much identity changes
            most_identity_preserving: tuple (layer, head)
            most_transforming: tuple (layer, head)
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    hook_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=hook_state)
    cache = hook_state.cache

    W_E = np.array(model.embed.W_E)  # [d_vocab, d_model]

    identity_scores = np.zeros((n_layers, n_heads))
    transform_scores = np.zeros((n_layers, n_heads))

    for layer in range(n_layers):
        block = model.blocks[layer]
        W_V = np.array(block.attn.W_V)  # [n_heads, d_model, d_head]
        W_O = np.array(block.attn.W_O)  # [n_heads, d_head, d_model]

        # Get the input embedding for the attended token
        # Approximate: use the residual at this layer
        if layer == 0:
            resid_key = "blocks.0.hook_resid_pre"
        else:
            resid_key = f"blocks.{layer - 1}.hook_resid_post"
        resid = cache.get(resid_key)
        if resid is None:
            continue
        input_vec = np.array(resid[pos])

        for h in range(n_heads):
            # OV circuit output: input -> V -> O -> output
            z = W_V[h].T @ input_vec  # [d_head]
            ov_out = z @ W_O[h]  # [d_model]

            # Identity: cosine similarity between input and output
            in_norm = np.linalg.norm(input_vec) + 1e-10
            out_norm = np.linalg.norm(ov_out) + 1e-10
            identity_scores[layer, h] = float(np.dot(input_vec, ov_out) / (in_norm * out_norm))

            # Transformation: how different the output is
            diff = ov_out - input_vec * (np.dot(input_vec, ov_out) / (in_norm ** 2))
            transform_scores[layer, h] = float(np.linalg.norm(diff) / (out_norm + 1e-10))

    most_id = np.unravel_index(np.argmax(identity_scores), identity_scores.shape)
    most_tr = np.unravel_index(np.argmax(transform_scores), transform_scores.shape)

    return {
        "identity_scores": identity_scores,
        "transformation_scores": transform_scores,
        "most_identity_preserving": (int(most_id[0]), int(most_id[1])),
        "most_transforming": (int(most_tr[0]), int(most_tr[1])),
    }


def head_specialization_profile(model, tokens):
    """Profile the specialization of each attention head.

    Combines positional, pattern, and output metrics to classify
    what each head specializes in.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].

    Returns:
        dict with:
            entropy_scores: array [n_layers, n_heads] of attention entropy
            bos_scores: array [n_layers, n_heads] of BOS attention fraction
            diagonal_scores: array [n_layers, n_heads] of self-attention fraction
            prev_token_scores: array [n_layers, n_heads] of prev-token attention
            specialization_labels: array [n_layers, n_heads] of str labels
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    seq_len = len(tokens)

    hook_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=hook_state)
    cache = hook_state.cache

    entropy_scores = np.zeros((n_layers, n_heads))
    bos_scores = np.zeros((n_layers, n_heads))
    diag_scores = np.zeros((n_layers, n_heads))
    prev_scores = np.zeros((n_layers, n_heads))
    labels = np.empty((n_layers, n_heads), dtype=object)

    for layer in range(n_layers):
        pattern = cache.get(f"blocks.{layer}.attn.hook_pattern")
        if pattern is None:
            for h in range(n_heads):
                labels[layer, h] = "unknown"
            continue
        pattern = np.array(pattern)  # [n_heads, seq, seq]

        for h in range(n_heads):
            p = pattern[h]  # [seq, seq]

            # Entropy per query, then average
            ents = []
            for i in range(seq_len):
                row = p[i, :i + 1]
                row = row + 1e-10
                ent = -np.sum(row * np.log(row))
                ents.append(ent)
            entropy_scores[layer, h] = float(np.mean(ents))

            # BOS attention (fraction attending to position 0)
            bos_attn = np.mean(p[1:, 0]) if seq_len > 1 else 0.0
            bos_scores[layer, h] = float(bos_attn)

            # Diagonal (self-attention)
            diag_sum = np.sum([p[i, i] for i in range(seq_len)])
            diag_scores[layer, h] = float(diag_sum / seq_len)

            # Previous token
            prev_sum = np.sum([p[i, i - 1] for i in range(1, seq_len)])
            prev_scores[layer, h] = float(prev_sum / max(1, seq_len - 1))

            # Classify
            if bos_scores[layer, h] > 0.5:
                labels[layer, h] = "bos"
            elif prev_scores[layer, h] > 0.4:
                labels[layer, h] = "prev_token"
            elif diag_scores[layer, h] > 0.4:
                labels[layer, h] = "identity"
            elif entropy_scores[layer, h] > np.log(seq_len) * 0.8:
                labels[layer, h] = "uniform"
            else:
                labels[layer, h] = "mixed"

    return {
        "entropy_scores": entropy_scores,
        "bos_scores": bos_scores,
        "diagonal_scores": diag_scores,
        "prev_token_scores": prev_scores,
        "specialization_labels": labels,
    }


def head_output_norm_analysis(model, tokens, pos=-1):
    """Analyze the norm of each head's output contribution.

    Measures how much each head contributes to the residual stream
    in terms of magnitude.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        pos: Position.

    Returns:
        dict with:
            output_norms: array [n_layers, n_heads]
            relative_norms: array [n_layers, n_heads] normalized per layer
            largest_head: tuple (layer, head)
            smallest_head: tuple (layer, head)
            layer_total_norms: array [n_layers]
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    hook_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=hook_state)
    cache = hook_state.cache

    output_norms = np.zeros((n_layers, n_heads))

    for layer in range(n_layers):
        z = cache.get(f"blocks.{layer}.attn.hook_z")
        if z is None:
            continue
        z_arr = np.array(z)  # [seq, n_heads, d_head]
        W_O = np.array(model.blocks[layer].attn.W_O)  # [n_heads, d_head, d_model]

        for h in range(n_heads):
            head_out = z_arr[pos, h, :] @ W_O[h]  # [d_model]
            output_norms[layer, h] = float(np.linalg.norm(head_out))

    # Relative norms per layer
    layer_totals = np.sum(output_norms, axis=1)
    relative_norms = output_norms / (layer_totals[:, None] + 1e-10)

    largest = np.unravel_index(np.argmax(output_norms), output_norms.shape)
    smallest = np.unravel_index(np.argmin(output_norms), output_norms.shape)

    return {
        "output_norms": output_norms,
        "relative_norms": relative_norms,
        "largest_head": (int(largest[0]), int(largest[1])),
        "smallest_head": (int(smallest[0]), int(smallest[1])),
        "layer_total_norms": layer_totals,
    }


def head_value_rank_analysis(model, tokens, pos=-1, top_k=5):
    """Analyze the effective rank of each head's value space.

    Low-rank value spaces indicate heads that project onto a small
    subspace, suggesting specialized function.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        pos: Position.
        top_k: Number of singular values to return.

    Returns:
        dict with:
            effective_ranks: array [n_layers, n_heads]
            top_singular_values: array [n_layers, n_heads, top_k]
            most_low_rank: tuple (layer, head)
            most_full_rank: tuple (layer, head)
            mean_rank: float
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    effective_ranks = np.zeros((n_layers, n_heads))
    d_head = model.cfg.d_head
    actual_k = min(top_k, d_head)
    top_svs = np.zeros((n_layers, n_heads, actual_k))

    for layer in range(n_layers):
        block = model.blocks[layer]
        W_V = np.array(block.attn.W_V)  # [n_heads, d_model, d_head]
        W_O = np.array(block.attn.W_O)  # [n_heads, d_head, d_model]

        for h in range(n_heads):
            # OV matrix: W_O[h] @ W_V[h].T -> [d_model, d_model] but go through d_head
            # Effective matrix is W_V[h] (d_model x d_head) -- analyze this
            S = np.linalg.svd(W_V[h], compute_uv=False)
            S2 = S ** 2
            total = np.sum(S2) + 1e-10
            probs = S2 / total
            probs = probs[probs > 1e-12]
            effective_ranks[layer, h] = float(np.exp(-np.sum(probs * np.log(probs + 1e-12))))
            top_svs[layer, h, :len(S[:actual_k])] = S[:actual_k]

    low_rank = np.unravel_index(np.argmin(effective_ranks), effective_ranks.shape)
    full_rank = np.unravel_index(np.argmax(effective_ranks), effective_ranks.shape)

    return {
        "effective_ranks": effective_ranks,
        "top_singular_values": top_svs,
        "most_low_rank": (int(low_rank[0]), int(low_rank[1])),
        "most_full_rank": (int(full_rank[0]), int(full_rank[1])),
        "mean_rank": float(np.mean(effective_ranks)),
    }
