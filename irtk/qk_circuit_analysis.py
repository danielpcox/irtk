"""Deep QK circuit analysis.

Analyze the query-key (QK) circuits of attention heads: pattern prediction
from weights alone, positional vs content-based QK contributions, QK
composition between layers, eigenvalue structure, and effective receptive field.

The QK circuit W_Q^T @ W_K determines what each head attends to.

References:
    Elhage et al. (2021) "A Mathematical Framework for Transformer Circuits"
"""

import jax
import jax.numpy as jnp
import numpy as np


def qk_eigenvalue_structure(model, layer, head, top_k=5):
    """Analyze the eigenvalue structure of the QK circuit.

    The QK matrix W_Q @ W_K^T has eigenvalues that reveal whether the
    head attends based on similarity, dissimilarity, or orthogonal features.

    Args:
        model: HookedTransformer model.
        layer: Layer index.
        head: Head index.
        top_k: Number of top eigenvalues.

    Returns:
        dict with:
            eigenvalues: [top_k] top eigenvalues (real parts)
            eigenvectors: [top_k, d_model] corresponding eigenvectors
            qk_trace: float
            qk_rank: float (effective rank)
            singular_values: list of top singular values
            positive_negative_ratio: float (ratio of positive to negative eigenvalues)
    """
    W_Q = np.array(model.blocks[layer].attn.W_Q)  # [n_heads, d_model, d_head]
    W_K = np.array(model.blocks[layer].attn.W_K)

    # QK matrix: W_Q @ W_K^T = [d_model, d_head] @ [d_head, d_model] = [d_model, d_model]
    qk = W_Q[head] @ W_K[head].T

    eigvals, eigvecs = np.linalg.eig(qk)
    order = np.argsort(-np.abs(eigvals.real))
    top_eigvals = eigvals[order[:top_k]].real
    top_eigvecs = eigvecs[:, order[:top_k]].real

    U, S, Vt = np.linalg.svd(qk, full_matrices=False)
    s = S[S > 1e-10]
    if len(s) > 0:
        p = s / np.sum(s)
        eff_rank = float(np.exp(-np.sum(p * np.log(p + 1e-10))))
    else:
        eff_rank = 0.0

    pos_eig = np.sum(eigvals.real > 0)
    neg_eig = np.sum(eigvals.real < 0)
    ratio = float(pos_eig / (neg_eig + 1e-10))

    return {
        "eigenvalues": top_eigvals,
        "eigenvectors": top_eigvecs.T,
        "qk_trace": float(np.trace(qk)),
        "qk_rank": eff_rank,
        "singular_values": S[:top_k].tolist(),
        "positive_negative_ratio": ratio,
    }


def positional_vs_content_qk(model, tokens, layer, head):
    """Separate the QK circuit into positional and content contributions.

    Decomposes attention scores into the part coming from positional
    embeddings vs the part from token content.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        layer: Layer index.
        head: Head index.

    Returns:
        dict with:
            content_scores: [seq_len, seq_len] attention scores from content
            positional_scores: [seq_len, seq_len] attention scores from position
            content_fraction: float (fraction of attention from content)
            positional_fraction: float (fraction from position)
            actual_pattern: [seq_len, seq_len] actual attention pattern
    """
    from irtk.hook_points import HookState

    seq_len = len(tokens)
    W_Q = np.array(model.blocks[layer].attn.W_Q)
    W_K = np.array(model.blocks[layer].attn.W_K)

    # Get embeddings
    W_E = np.array(model.embed.W_E)
    W_pos = np.array(model.pos_embed.W_pos)

    # Token and positional embeddings
    token_embeds = W_E[np.array(tokens)]  # [seq_len, d_model]
    pos_embeds = W_pos[:seq_len]  # [seq_len, d_model]

    # Content Q and K
    content_Q = token_embeds @ W_Q[head]  # [seq_len, d_head]
    content_K = token_embeds @ W_K[head]

    # Positional Q and K
    pos_Q = pos_embeds @ W_Q[head]
    pos_K = pos_embeds @ W_K[head]

    d_head = W_Q.shape[2]
    scale = np.sqrt(d_head)

    content_scores = (content_Q @ content_K.T) / scale
    positional_scores = (pos_Q @ pos_K.T) / scale

    # Actual pattern
    cache_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=cache_state)
    cache = cache_state.cache

    actual = np.zeros((seq_len, seq_len))
    pattern = cache.get(f"blocks.{layer}.attn.hook_pattern")
    if pattern is not None:
        actual = np.array(pattern)[head]

    # Fractions (based on score magnitudes)
    content_mag = float(np.mean(np.abs(content_scores)))
    pos_mag = float(np.mean(np.abs(positional_scores)))
    total_mag = content_mag + pos_mag + 1e-10

    return {
        "content_scores": content_scores,
        "positional_scores": positional_scores,
        "content_fraction": content_mag / total_mag,
        "positional_fraction": pos_mag / total_mag,
        "actual_pattern": actual,
    }


def qk_pattern_prediction(model, tokens, layer, head):
    """Predict attention pattern from QK circuit weights.

    Compare the predicted pattern (from W_Q, W_K applied to embeddings)
    with the actual pattern (which includes layernorm and residual stream effects).

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        layer: Layer index.
        head: Head index.

    Returns:
        dict with:
            predicted_pattern: [seq_len, seq_len] softmax of QK scores
            actual_pattern: [seq_len, seq_len] from actual forward pass
            correlation: float (correlation between predicted and actual)
            mse: float (mean squared error)
            max_error_position: (query_pos, key_pos) position with largest error
    """
    from irtk.hook_points import HookState

    seq_len = len(tokens)
    W_Q = np.array(model.blocks[layer].attn.W_Q)
    W_K = np.array(model.blocks[layer].attn.W_K)
    W_E = np.array(model.embed.W_E)
    W_pos = np.array(model.pos_embed.W_pos)

    # Full embedding
    embeds = W_E[np.array(tokens)] + W_pos[:seq_len]  # [seq_len, d_model]

    Q = embeds @ W_Q[head]  # [seq_len, d_head]
    K = embeds @ W_K[head]

    d_head = W_Q.shape[2]
    scores = (Q @ K.T) / np.sqrt(d_head)

    # Apply causal mask
    mask = np.triu(np.ones((seq_len, seq_len)), k=1) * (-1e9)
    scores = scores + mask

    # Softmax
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    predicted = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    # Actual pattern
    cache_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=cache_state)
    cache = cache_state.cache

    actual = np.zeros((seq_len, seq_len))
    pattern = cache.get(f"blocks.{layer}.attn.hook_pattern")
    if pattern is not None:
        actual = np.array(pattern)[head]

    # Correlation
    pred_flat = predicted.flatten()
    act_flat = actual.flatten()
    if np.std(pred_flat) > 1e-10 and np.std(act_flat) > 1e-10:
        corr = float(np.corrcoef(pred_flat, act_flat)[0, 1])
    else:
        corr = 0.0

    mse = float(np.mean((predicted - actual) ** 2))

    # Max error position
    error = np.abs(predicted - actual)
    max_pos = np.unravel_index(np.argmax(error), error.shape)

    return {
        "predicted_pattern": predicted,
        "actual_pattern": actual,
        "correlation": corr,
        "mse": mse,
        "max_error_position": (int(max_pos[0]), int(max_pos[1])),
    }


def qk_composition_analysis(model, src_layer, src_head, dst_layer, dst_head):
    """Analyze QK composition between two heads across layers.

    When a head in a later layer uses key vectors written by an earlier head
    (through the residual stream), this creates a composed QK circuit.

    Args:
        model: HookedTransformer model.
        src_layer: Source layer.
        src_head: Source head.
        dst_layer: Destination layer (should be > src_layer).
        dst_head: Destination head.

    Returns:
        dict with:
            q_composition_score: float (how much dst Q reads from src output)
            k_composition_score: float (how much dst K reads from src output)
            composed_qk_rank: float (effective rank of composed circuit)
            composition_strength: float (overall composition strength)
            singular_values: list of top singular values of composed circuit
    """
    W_V_src = np.array(model.blocks[src_layer].attn.W_V)
    W_O_src = np.array(model.blocks[src_layer].attn.W_O)
    W_Q_dst = np.array(model.blocks[dst_layer].attn.W_Q)
    W_K_dst = np.array(model.blocks[dst_layer].attn.W_K)

    # Source OV output
    ov_src = W_V_src[src_head] @ W_O_src[src_head]  # [d_model, d_model]

    # Q-composition: OV_src -> Q_dst
    q_comp = ov_src @ W_Q_dst[dst_head]  # [d_model, d_head]
    q_score = float(np.linalg.norm(q_comp, 'fro'))

    # K-composition: OV_src -> K_dst
    k_comp = ov_src @ W_K_dst[dst_head]  # [d_model, d_head]
    k_score = float(np.linalg.norm(k_comp, 'fro'))

    # Composed QK circuit
    composed = q_comp @ k_comp.T  # [d_model, d_model]
    U, S, Vt = np.linalg.svd(composed, full_matrices=False)

    s = S[S > 1e-10]
    if len(s) > 0:
        p = s / np.sum(s)
        eff_rank = float(np.exp(-np.sum(p * np.log(p + 1e-10))))
    else:
        eff_rank = 0.0

    strength = float(np.linalg.norm(composed, 'fro'))

    return {
        "q_composition_score": q_score,
        "k_composition_score": k_score,
        "composed_qk_rank": eff_rank,
        "composition_strength": strength,
        "singular_values": S[:5].tolist(),
    }


def effective_receptive_field(model, tokens, layer, head):
    """Compute the effective receptive field for a head.

    Shows the average attention pattern and identifies which positions
    each query position effectively attends to.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        layer: Layer index.
        head: Head index.

    Returns:
        dict with:
            attention_pattern: [seq_len, seq_len] the actual attention weights
            mean_attention_distance: float (average distance attended to)
            receptive_field_width: float (effective width based on entropy)
            peak_positions: [seq_len] most-attended position per query
            attention_entropy: [seq_len] entropy of attention distribution per position
    """
    from irtk.hook_points import HookState

    seq_len = len(tokens)

    cache_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=cache_state)
    cache = cache_state.cache

    pattern = cache.get(f"blocks.{layer}.attn.hook_pattern")
    if pattern is None:
        return {
            "attention_pattern": np.zeros((seq_len, seq_len)),
            "mean_attention_distance": 0.0,
            "receptive_field_width": 0.0,
            "peak_positions": np.zeros(seq_len, dtype=int),
            "attention_entropy": np.zeros(seq_len),
        }

    pat = np.array(pattern)[head]  # [seq_len, seq_len]

    # Mean distance
    total_dist = 0.0
    total_weight = 0.0
    for q in range(seq_len):
        for k in range(q + 1):
            total_dist += abs(q - k) * pat[q, k]
            total_weight += pat[q, k]
    mean_dist = total_dist / (total_weight + 1e-10)

    # Entropy per position
    entropies = np.zeros(seq_len)
    for q in range(seq_len):
        p = pat[q, :q + 1]
        p = p[p > 1e-10]
        entropies[q] = -float(np.sum(p * np.log(p + 1e-10)))

    # Receptive field width (mean entropy)
    rf_width = float(np.mean(np.exp(entropies)))

    # Peak positions
    peaks = np.zeros(seq_len, dtype=int)
    for q in range(seq_len):
        peaks[q] = np.argmax(pat[q, :q + 1])

    return {
        "attention_pattern": pat,
        "mean_attention_distance": float(mean_dist),
        "receptive_field_width": rf_width,
        "peak_positions": peaks,
        "attention_entropy": entropies,
    }
