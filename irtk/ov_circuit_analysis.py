"""Deep OV circuit analysis.

Analyze the output value (OV) circuits of attention heads: eigenvalue
decomposition, token copying strength, semantic role mapping, inter-layer
OV composition, and unembedding projection analysis.

The OV circuit W_V @ W_O determines what information each head writes
to the residual stream given the input values.

References:
    Elhage et al. (2021) "A Mathematical Framework for Transformer Circuits"
"""

import jax
import jax.numpy as jnp
import numpy as np


def ov_eigenvalue_decomposition(model, layer, head, top_k=5):
    """Decompose the OV circuit into eigenvalues and eigenvectors.

    The OV matrix W_V @ W_O maps from d_model to d_model. Its eigenvalues
    reveal the principal directions of information flow.

    Args:
        model: HookedTransformer model.
        layer: Layer index.
        head: Head index.
        top_k: Number of top eigenvalues to return.

    Returns:
        dict with:
            eigenvalues: [min(d_model, top_k)] top eigenvalues (real parts)
            eigenvectors: [top_k, d_model] corresponding eigenvectors
            ov_trace: float (trace of OV matrix)
            ov_rank: float (effective rank)
            singular_values: [top_k] top singular values
    """
    W_V = np.array(model.blocks[layer].attn.W_V)  # [n_heads, d_model, d_head]
    W_O = np.array(model.blocks[layer].attn.W_O)  # [n_heads, d_head, d_model]

    # OV matrix: [d_model, d_head] @ [d_head, d_model] = [d_model, d_model]
    ov = W_V[head] @ W_O[head]

    # Eigenvalue decomposition
    eigvals, eigvecs = np.linalg.eig(ov)
    # Sort by magnitude of real part
    order = np.argsort(-np.abs(eigvals.real))
    top_eigvals = eigvals[order[:top_k]].real
    top_eigvecs = eigvecs[:, order[:top_k]].real

    # SVD for singular values
    U, S, Vt = np.linalg.svd(ov, full_matrices=False)

    # Effective rank
    s = S[S > 1e-10]
    if len(s) > 0:
        p = s / np.sum(s)
        eff_rank = float(np.exp(-np.sum(p * np.log(p + 1e-10))))
    else:
        eff_rank = 0.0

    return {
        "eigenvalues": top_eigvals,
        "eigenvectors": top_eigvecs.T,  # [top_k, d_model]
        "ov_trace": float(np.trace(ov)),
        "ov_rank": eff_rank,
        "singular_values": S[:top_k].tolist(),
    }


def token_copying_strength(model, layer, head, top_k=10):
    """Measure how strongly the OV circuit copies token identity.

    Projects the OV circuit through the embedding and unembedding to
    measure the direct token-to-token copying strength.

    Args:
        model: HookedTransformer model.
        layer: Layer index.
        head: Head index.
        top_k: Number of top token pairs.

    Returns:
        dict with:
            copy_diagonal: [d_vocab] self-copying strength per token
            mean_copy_strength: float
            top_copied_tokens: list of (token, strength) best self-copiers
            top_token_pairs: list of (src_token, dst_token, strength) cross-copies
            copy_vs_suppress_ratio: float
    """
    W_V = np.array(model.blocks[layer].attn.W_V)
    W_O = np.array(model.blocks[layer].attn.W_O)
    W_E = np.array(model.embed.W_E)    # [d_vocab, d_model]
    W_U = np.array(model.unembed.W_U)  # [d_model, d_vocab]

    d_vocab = W_E.shape[0]

    # Full OV circuit through embeddings: W_E @ W_V @ W_O @ W_U
    # = [d_vocab, d_model] @ [d_model, d_head] @ [d_head, d_model] @ [d_model, d_vocab]
    ov = W_V[head] @ W_O[head]  # [d_model, d_model]

    # For efficiency, compute for a subset of tokens
    n_sample = min(d_vocab, 100)
    sample_idx = np.linspace(0, d_vocab - 1, n_sample).astype(int)

    copy_diagonal = np.zeros(d_vocab)
    for i in sample_idx:
        embed = W_E[i]  # [d_model]
        output = embed @ ov @ W_U  # [d_vocab]
        copy_diagonal[i] = output[i]  # self-copy strength

    mean_copy = float(np.mean(copy_diagonal[sample_idx]))

    # Top self-copiers
    sorted_idx = np.argsort(-copy_diagonal[sample_idx])
    top_copied = [(int(sample_idx[i]), float(copy_diagonal[sample_idx[i]]))
                  for i in sorted_idx[:top_k]]

    # Top cross-token pairs (sample a few source tokens)
    top_pairs = []
    for src in sample_idx[:10]:
        embed = W_E[src]
        output = embed @ ov @ W_U  # [d_vocab]
        top_dst = np.argsort(-output)[:3]
        for dst in top_dst:
            top_pairs.append((int(src), int(dst), float(output[dst])))
    top_pairs.sort(key=lambda x: -x[2])
    top_pairs = top_pairs[:top_k]

    # Copy vs suppress ratio
    positive = np.sum(copy_diagonal[sample_idx][copy_diagonal[sample_idx] > 0])
    negative = np.sum(np.abs(copy_diagonal[sample_idx][copy_diagonal[sample_idx] < 0]))
    ratio = float(positive / (negative + 1e-10))

    return {
        "copy_diagonal": copy_diagonal,
        "mean_copy_strength": mean_copy,
        "top_copied_tokens": top_copied,
        "top_token_pairs": top_pairs,
        "copy_vs_suppress_ratio": ratio,
    }


def ov_semantic_role(model, tokens, layer, head, pos=-1, top_k=5):
    """Analyze the semantic role of the OV circuit on actual inputs.

    For the given input, shows what the OV circuit writes to the residual
    stream at the target position.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        layer: Layer index.
        head: Head index.
        pos: Target position.
        top_k: Top tokens to show.

    Returns:
        dict with:
            ov_output: [d_model] the OV circuit's output at target position
            output_norm: float
            top_promoted_tokens: list of (token, logit_contribution)
            top_demoted_tokens: list of (token, logit_contribution)
            source_position_contributions: [seq_len] how much each source contributes
    """
    from irtk.hook_points import HookState

    cache_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=cache_state)
    cache = cache_state.cache

    z = cache.get(f"blocks.{layer}.attn.hook_z")
    pattern = cache.get(f"blocks.{layer}.attn.hook_pattern")
    v = cache.get(f"blocks.{layer}.attn.hook_v")

    seq_len = len(tokens)
    W_O = np.array(model.blocks[layer].attn.W_O)
    W_U = np.array(model.unembed.W_U)

    if z is None:
        d_model = model.cfg.d_model
        return {
            "ov_output": np.zeros(d_model),
            "output_norm": 0.0,
            "top_promoted_tokens": [],
            "top_demoted_tokens": [],
            "source_position_contributions": np.zeros(seq_len),
        }

    z_arr = np.array(z)
    ov_output = z_arr[pos, head] @ W_O[head]  # [d_model]
    output_norm = float(np.linalg.norm(ov_output))

    # Logit contribution
    logits = ov_output @ W_U  # [d_vocab]
    top_prom = np.argsort(-logits)[:top_k]
    top_dem = np.argsort(logits)[:top_k]

    promoted = [(int(t), float(logits[t])) for t in top_prom]
    demoted = [(int(t), float(logits[t])) for t in top_dem]

    # Source position contributions
    src_contribs = np.zeros(seq_len)
    if pattern is not None and v is not None:
        pat = np.array(pattern)
        v_arr = np.array(v)
        for src in range(seq_len):
            weighted_v = pat[head, pos, src] * v_arr[src, head]
            out = weighted_v @ W_O[head]
            src_contribs[src] = float(np.linalg.norm(out))

    return {
        "ov_output": ov_output,
        "output_norm": output_norm,
        "top_promoted_tokens": promoted,
        "top_demoted_tokens": demoted,
        "source_position_contributions": src_contribs,
    }


def ov_composition_between_layers(model, src_layer, src_head, dst_layer, dst_head):
    """Analyze how OV circuits compose between two layers.

    Measures whether the output of one head's OV circuit aligns with
    the input space of another head's QK or OV circuit.

    Args:
        model: HookedTransformer model.
        src_layer: Source layer.
        src_head: Source head.
        dst_layer: Destination layer (must be > src_layer).
        dst_head: Destination head.

    Returns:
        dict with:
            v_composition_score: float (V-composition strength)
            k_composition_score: float (K-composition strength)
            q_composition_score: float (Q-composition strength)
            composed_ov_rank: float (effective rank of composed circuit)
            composition_singular_values: list of top singular values
    """
    W_V_src = np.array(model.blocks[src_layer].attn.W_V)
    W_O_src = np.array(model.blocks[src_layer].attn.W_O)
    W_V_dst = np.array(model.blocks[dst_layer].attn.W_V)
    W_K_dst = np.array(model.blocks[dst_layer].attn.W_K)
    W_Q_dst = np.array(model.blocks[dst_layer].attn.W_Q)

    # OV of source: [d_model, d_model]
    ov_src = W_V_src[src_head] @ W_O_src[src_head]

    # V-composition: how well does src output feed into dst V
    # W_O_src @ W_V_dst = [d_head_src, d_model] @ [d_model, d_head_dst]
    v_comp = W_O_src[src_head] @ W_V_dst[dst_head]  # [d_head, d_head]
    v_score = float(np.linalg.norm(v_comp, 'fro'))

    # K-composition
    k_comp = W_O_src[src_head] @ W_K_dst[dst_head]  # [d_head, d_head]
    k_score = float(np.linalg.norm(k_comp, 'fro'))

    # Q-composition
    q_comp = W_O_src[src_head] @ W_Q_dst[dst_head]  # [d_head, d_head]
    q_score = float(np.linalg.norm(q_comp, 'fro'))

    # Composed OV circuit
    W_O_dst = np.array(model.blocks[dst_layer].attn.W_O)
    composed = ov_src @ W_V_dst[dst_head] @ W_O_dst[dst_head]  # [d_model, d_model]
    U, S, Vt = np.linalg.svd(composed, full_matrices=False)

    s = S[S > 1e-10]
    if len(s) > 0:
        p = s / np.sum(s)
        eff_rank = float(np.exp(-np.sum(p * np.log(p + 1e-10))))
    else:
        eff_rank = 0.0

    return {
        "v_composition_score": v_score,
        "k_composition_score": k_score,
        "q_composition_score": q_score,
        "composed_ov_rank": eff_rank,
        "composition_singular_values": S[:5].tolist(),
    }


def ov_unembedding_projection(model, layer, head, top_k=10):
    """Project the OV circuit through the unembedding to see token effects.

    Analyzes the full W_V @ W_O @ W_U circuit to understand which tokens
    each head promotes or suppresses in the output.

    Args:
        model: HookedTransformer model.
        layer: Layer index.
        head: Head index.
        top_k: Top tokens per analysis.

    Returns:
        dict with:
            ov_logit_matrix_rank: float (effective rank of OV @ W_U)
            top_positive_directions: list of (token, magnitude) tokens most promoted
            top_negative_directions: list of (token, magnitude) tokens most suppressed
            ov_wu_singular_values: list of top singular values
            projection_norm: float
    """
    W_V = np.array(model.blocks[layer].attn.W_V)
    W_O = np.array(model.blocks[layer].attn.W_O)
    W_U = np.array(model.unembed.W_U)  # [d_model, d_vocab]

    # OV @ W_U: [d_model, d_model] @ [d_model, d_vocab] = [d_model, d_vocab]
    ov = W_V[head] @ W_O[head]
    ov_wu = ov @ W_U  # [d_model, d_vocab]

    # SVD
    U, S, Vt = np.linalg.svd(ov_wu, full_matrices=False)
    s = S[S > 1e-10]
    if len(s) > 0:
        p = s / np.sum(s)
        eff_rank = float(np.exp(-np.sum(p * np.log(p + 1e-10))))
    else:
        eff_rank = 0.0

    # Which tokens have the strongest projection?
    # Sum over d_model dimension to get per-token sensitivity
    token_sensitivity = np.linalg.norm(ov_wu, axis=0)  # [d_vocab]

    # Top positive (most promoted by principal direction)
    principal = U[:, 0]  # [d_model]
    token_proj = principal @ ov_wu  # [d_vocab]

    top_pos = np.argsort(-token_proj)[:top_k]
    top_neg = np.argsort(token_proj)[:top_k]

    positive_dirs = [(int(t), float(token_proj[t])) for t in top_pos]
    negative_dirs = [(int(t), float(token_proj[t])) for t in top_neg]

    return {
        "ov_logit_matrix_rank": eff_rank,
        "top_positive_directions": positive_dirs,
        "top_negative_directions": negative_dirs,
        "ov_wu_singular_values": S[:min(top_k, len(S))].tolist(),
        "projection_norm": float(np.linalg.norm(ov_wu, 'fro')),
    }
