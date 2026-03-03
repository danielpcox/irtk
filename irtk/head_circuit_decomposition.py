"""Head circuit decomposition.

Decomposes attention head circuits into interpretable subcircuits:
QK circuit analysis, OV circuit decomposition, virtual attention heads,
and composition patterns.

References:
    Elhage et al. (2021) "A Mathematical Framework for Transformer Circuits"
    Olsson et al. (2022) "In-context Learning and Induction Heads"
"""

import jax
import jax.numpy as jnp
import numpy as np


def qk_circuit_analysis(model, layer, head):
    """Analyze the QK circuit of a specific head.

    The QK circuit (W_E^T W_Q^T W_K W_E) determines which tokens attend to which.

    Args:
        model: HookedTransformer model.
        layer: Layer index.
        head: Head index.

    Returns:
        dict with:
            qk_matrix: [d_model, d_model] effective QK matrix
            eigenvalues: top eigenvalues of QK matrix
            effective_rank: approximate rank
            top_query_dirs: top query-side directions [5, d_model]
            top_key_dirs: top key-side directions [5, d_model]
    """
    W_Q = np.array(model.blocks[layer].attn.W_Q[head])  # [d_model, d_head]
    W_K = np.array(model.blocks[layer].attn.W_K[head])  # [d_model, d_head]

    # QK matrix: W_Q @ W_K^T gives [d_model, d_model]
    qk = W_Q @ W_K.T

    # SVD for analysis
    U, s, Vt = np.linalg.svd(qk, full_matrices=False)

    # Effective rank
    s_norm = s / (s[0] + 1e-10)
    eff_rank = float(np.sum(s_norm > 0.01))

    return {
        "qk_matrix": qk,
        "eigenvalues": s[:10],
        "effective_rank": eff_rank,
        "top_query_dirs": U[:, :5].T,
        "top_key_dirs": Vt[:5],
    }


def ov_circuit_analysis(model, layer, head):
    """Analyze the OV circuit of a specific head.

    The OV circuit (W_V W_O) determines what information is moved when attending.

    Args:
        model: HookedTransformer model.
        layer: Layer index.
        head: Head index.

    Returns:
        dict with:
            ov_matrix: [d_model, d_model] effective OV matrix
            singular_values: top singular values
            effective_rank: approximate rank
            top_input_dirs: top input directions [5, d_model]
            top_output_dirs: top output directions [5, d_model]
            trace: trace of OV matrix (related to copying behavior)
    """
    W_V = np.array(model.blocks[layer].attn.W_V[head])  # [d_model, d_head]
    W_O = np.array(model.blocks[layer].attn.W_O[head])  # [d_head, d_model]

    # OV matrix: W_V^T gives [d_head, d_model], then @ W_O gives [d_head, d_model]
    # Full OV circuit: input [d_model] -> W_V [d_model, d_head] -> W_O [d_head, d_model]
    # So it's W_O^T @ W_V^T = (W_V @ W_O)^T... actually:
    # For input x [d_model]: output = (x @ W_V) @ W_O = x @ (W_V @ W_O)
    ov = W_V @ W_O  # [d_model, d_model]

    U, s, Vt = np.linalg.svd(ov, full_matrices=False)

    s_norm = s / (s[0] + 1e-10)
    eff_rank = float(np.sum(s_norm > 0.01))

    trace_val = float(np.trace(ov))

    return {
        "ov_matrix": ov,
        "singular_values": s[:10],
        "effective_rank": eff_rank,
        "top_input_dirs": Vt[:5],
        "top_output_dirs": U[:, :5].T,
        "trace": trace_val,
    }


def virtual_attention_head(model, layer_a, head_a, layer_b, head_b):
    """Compute the virtual attention head formed by composing two heads.

    The virtual head arises from head B attending based on output of head A.

    Args:
        model: HookedTransformer model.
        layer_a: First layer (must be < layer_b).
        head_a: Head in first layer.
        layer_b: Second layer.
        head_b: Head in second layer.

    Returns:
        dict with:
            qk_composition: float, how much A's output aligns with B's QK circuit
            ov_composition: float, how much A's output aligns with B's OV circuit
            virtual_ov: [d_model, d_model] composed OV matrix
            composition_strength: overall composition strength
    """
    # Head A's OV matrix
    W_V_a = np.array(model.blocks[layer_a].attn.W_V[head_a])
    W_O_a = np.array(model.blocks[layer_a].attn.W_O[head_a])
    ov_a = W_V_a @ W_O_a

    # Head B's matrices
    W_Q_b = np.array(model.blocks[layer_b].attn.W_Q[head_b])
    W_K_b = np.array(model.blocks[layer_b].attn.W_K[head_b])
    W_V_b = np.array(model.blocks[layer_b].attn.W_V[head_b])
    W_O_b = np.array(model.blocks[layer_b].attn.W_O[head_b])

    # QK composition: how much does A's output affect B's attention?
    # A's output @ B's W_Q or W_K
    qk_via_q = ov_a @ W_Q_b  # A output used as query
    qk_via_k = ov_a @ W_K_b  # A output used as key
    qk_comp = float(np.linalg.norm(qk_via_q) + np.linalg.norm(qk_via_k))

    # OV composition: A's output through B's OV
    ov_b = W_V_b @ W_O_b
    virtual_ov = ov_a @ ov_b
    ov_comp = float(np.linalg.norm(virtual_ov))

    # Normalize by individual norms
    norm_a = np.linalg.norm(ov_a) + 1e-10
    norm_b_qk = np.linalg.norm(W_Q_b @ W_K_b.T) + 1e-10
    norm_b_ov = np.linalg.norm(ov_b) + 1e-10

    qk_score = qk_comp / (norm_a * np.sqrt(np.linalg.norm(W_Q_b)**2 + np.linalg.norm(W_K_b)**2) + 1e-10)
    ov_score = ov_comp / (norm_a * norm_b_ov + 1e-10)

    return {
        "qk_composition": float(qk_score),
        "ov_composition": float(ov_score),
        "virtual_ov": virtual_ov,
        "composition_strength": float(qk_score + ov_score),
    }


def head_composition_pattern(model):
    """Map composition patterns between all pairs of heads.

    Args:
        model: HookedTransformer model.

    Returns:
        dict with:
            qk_composition_scores: [n_layers, n_heads, n_layers, n_heads]
            ov_composition_scores: [n_layers, n_heads, n_layers, n_heads]
            strongest_qk_pair: tuple (layer_a, head_a, layer_b, head_b)
            strongest_ov_pair: tuple
            mean_composition: float
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    qk_scores = np.zeros((n_layers, n_heads, n_layers, n_heads))
    ov_scores = np.zeros((n_layers, n_heads, n_layers, n_heads))

    for la in range(n_layers):
        for ha in range(n_heads):
            W_V_a = np.array(model.blocks[la].attn.W_V[ha])
            W_O_a = np.array(model.blocks[la].attn.W_O[ha])
            ov_a = W_V_a @ W_O_a
            norm_a = np.linalg.norm(ov_a) + 1e-10

            for lb in range(la + 1, n_layers):
                for hb in range(n_heads):
                    W_Q_b = np.array(model.blocks[lb].attn.W_Q[hb])
                    W_K_b = np.array(model.blocks[lb].attn.W_K[hb])
                    W_V_b = np.array(model.blocks[lb].attn.W_V[hb])
                    W_O_b = np.array(model.blocks[lb].attn.W_O[hb])

                    # QK composition
                    qk_via_q = np.linalg.norm(ov_a @ W_Q_b)
                    qk_via_k = np.linalg.norm(ov_a @ W_K_b)
                    qk_norm = np.sqrt(np.linalg.norm(W_Q_b)**2 + np.linalg.norm(W_K_b)**2) + 1e-10
                    qk_scores[la, ha, lb, hb] = (qk_via_q + qk_via_k) / (norm_a * qk_norm)

                    # OV composition
                    ov_b = W_V_b @ W_O_b
                    ov_scores[la, ha, lb, hb] = np.linalg.norm(ov_a @ ov_b) / (norm_a * np.linalg.norm(ov_b) + 1e-10)

    # Find strongest pairs
    best_qk = np.unravel_index(np.argmax(qk_scores), qk_scores.shape)
    best_ov = np.unravel_index(np.argmax(ov_scores), ov_scores.shape)

    total = qk_scores + ov_scores
    mean_comp = float(np.sum(total)) / max(1, np.sum(total > 0))

    return {
        "qk_composition_scores": qk_scores,
        "ov_composition_scores": ov_scores,
        "strongest_qk_pair": tuple(int(x) for x in best_qk),
        "strongest_ov_pair": tuple(int(x) for x in best_ov),
        "mean_composition": mean_comp,
    }


def head_logit_contribution(model, tokens, layer, head, pos=-1, top_k=5):
    """Decompose a head's contribution to the final logits.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        layer: Layer index.
        head: Head index.
        pos: Position to analyze.
        top_k: Number of top tokens.

    Returns:
        dict with:
            head_output: [d_model] the head's output vector at pos
            logit_contribution: [d_vocab] contribution to each logit
            top_promoted: list of (token_idx, logit_value) for most promoted
            top_demoted: list of (token_idx, logit_value) for most demoted
            output_norm: norm of head output
    """
    from irtk.hook_points import HookState

    cache_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=cache_state)

    z = cache_state.cache.get(f"blocks.{layer}.attn.hook_z")
    if z is None:
        return {
            "head_output": np.zeros(model.cfg.d_model),
            "logit_contribution": np.zeros(model.cfg.d_vocab),
            "top_promoted": [],
            "top_demoted": [],
            "output_norm": 0.0,
        }

    z_arr = np.array(z)
    W_O = np.array(model.blocks[layer].attn.W_O[head])
    head_out = z_arr[pos, head] @ W_O  # [d_model]

    # Project through unembed
    W_U = np.array(model.unembed.W_U)  # [d_model, d_vocab]
    logit_contrib = head_out @ W_U  # [d_vocab]

    # Top promoted/demoted
    top_idx = np.argsort(logit_contrib)[::-1][:top_k]
    bot_idx = np.argsort(logit_contrib)[:top_k]

    promoted = [(int(i), float(logit_contrib[i])) for i in top_idx]
    demoted = [(int(i), float(logit_contrib[i])) for i in bot_idx]

    return {
        "head_output": head_out,
        "logit_contribution": logit_contrib,
        "top_promoted": promoted,
        "top_demoted": demoted,
        "output_norm": float(np.linalg.norm(head_out)),
    }
