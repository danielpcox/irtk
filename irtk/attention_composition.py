"""Attention head composition analysis.

Analyze how attention heads compose across layers: Q-composition, K-composition,
V-composition, virtual attention heads, and composition path tracing.

References:
    Elhage et al. (2021) "A Mathematical Framework for Transformer Circuits"
"""

import jax
import jax.numpy as jnp
import numpy as np


def qk_composition_scores(model):
    """Compute Q-composition and K-composition scores between head pairs.

    Measures how much earlier heads' outputs contribute to later heads' Q and K
    inputs, via the composition of OV and QK circuits.

    Args:
        model: HookedTransformer model.

    Returns:
        dict with:
            q_composition: [n_layers, n_heads, n_layers, n_heads] Q-composition scores
            k_composition: [n_layers, n_heads, n_layers, n_heads] K-composition scores
            top_q_pairs: list of (src_layer, src_head, dst_layer, dst_head, score)
            top_k_pairs: list of (src_layer, src_head, dst_layer, dst_head, score)
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    q_comp = np.zeros((n_layers, n_heads, n_layers, n_heads))
    k_comp = np.zeros((n_layers, n_heads, n_layers, n_heads))

    for src_l in range(n_layers):
        W_O = np.array(model.blocks[src_l].attn.W_O)  # [n_heads, d_head, d_model]
        W_V = np.array(model.blocks[src_l].attn.W_V)  # [n_heads, d_model, d_head]

        for dst_l in range(src_l + 1, n_layers):
            W_Q = np.array(model.blocks[dst_l].attn.W_Q)  # [n_heads, d_model, d_head]
            W_K = np.array(model.blocks[dst_l].attn.W_K)

            for sh in range(n_heads):
                # OV output: W_V[sh] @ W_O[sh] -> [d_model, d_model]
                ov = W_V[sh] @ W_O[sh]  # [d_model, d_model]
                ov_norm = np.linalg.norm(ov, ord='fro')
                if ov_norm < 1e-10:
                    continue

                for dh in range(n_heads):
                    # Q-composition: how much does OV output project onto Q input
                    q_proj = ov @ W_Q[dh]  # [d_model, d_head]
                    q_comp[src_l, sh, dst_l, dh] = float(np.linalg.norm(q_proj, ord='fro') / ov_norm)

                    # K-composition: how much does OV output project onto K input
                    k_proj = ov @ W_K[dh]  # [d_model, d_head]
                    k_comp[src_l, sh, dst_l, dh] = float(np.linalg.norm(k_proj, ord='fro') / ov_norm)

    # Top pairs
    q_flat = q_comp.reshape(-1)
    q_indices = np.argsort(-q_flat)[:10]
    top_q = []
    for idx in q_indices:
        sl, sh, dl, dh = np.unravel_index(idx, q_comp.shape)
        if q_flat[idx] > 0:
            top_q.append((int(sl), int(sh), int(dl), int(dh), float(q_flat[idx])))

    k_flat = k_comp.reshape(-1)
    k_indices = np.argsort(-k_flat)[:10]
    top_k = []
    for idx in k_indices:
        sl, sh, dl, dh = np.unravel_index(idx, k_comp.shape)
        if k_flat[idx] > 0:
            top_k.append((int(sl), int(sh), int(dl), int(dh), float(k_flat[idx])))

    return {
        "q_composition": q_comp,
        "k_composition": k_comp,
        "top_q_pairs": top_q,
        "top_k_pairs": top_k,
    }


def v_composition_scores(model):
    """Compute V-composition scores between head pairs.

    Measures how much earlier heads' outputs affect the values that later heads attend to.

    Args:
        model: HookedTransformer model.

    Returns:
        dict with:
            v_composition: [n_layers, n_heads, n_layers, n_heads] V-composition scores
            top_v_pairs: list of (src_layer, src_head, dst_layer, dst_head, score)
            mean_v_composition: float
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    v_comp = np.zeros((n_layers, n_heads, n_layers, n_heads))

    for src_l in range(n_layers):
        W_O = np.array(model.blocks[src_l].attn.W_O)
        W_V_src = np.array(model.blocks[src_l].attn.W_V)

        for dst_l in range(src_l + 1, n_layers):
            W_V_dst = np.array(model.blocks[dst_l].attn.W_V)
            W_O_dst = np.array(model.blocks[dst_l].attn.W_O)

            for sh in range(n_heads):
                ov = W_V_src[sh] @ W_O[sh]  # [d_model, d_model]
                ov_norm = np.linalg.norm(ov, ord='fro')
                if ov_norm < 1e-10:
                    continue

                for dh in range(n_heads):
                    # V-composition: OV of src composed with V of dst
                    v_proj = ov @ W_V_dst[dh]  # [d_model, d_head]
                    v_comp[src_l, sh, dst_l, dh] = float(np.linalg.norm(v_proj, ord='fro') / ov_norm)

    v_flat = v_comp.reshape(-1)
    v_indices = np.argsort(-v_flat)[:10]
    top_v = []
    for idx in v_indices:
        sl, sh, dl, dh = np.unravel_index(idx, v_comp.shape)
        if v_flat[idx] > 0:
            top_v.append((int(sl), int(sh), int(dl), int(dh), float(v_flat[idx])))

    return {
        "v_composition": v_comp,
        "top_v_pairs": top_v,
        "mean_v_composition": float(np.mean(v_comp[v_comp > 0])) if np.any(v_comp > 0) else 0.0,
    }


def composition_path_tracing(model, tokens, metric_fn, max_depth=3):
    """Trace composition paths that contribute to a metric.

    Identifies chains of heads that compose to produce a specific behavior
    by measuring the effect of ablating sequential head pairs.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        metric_fn: Function mapping logits to scalar metric.
        max_depth: Maximum composition chain length.

    Returns:
        dict with:
            path_scores: dict of path_tuple -> metric effect
            top_paths: list of (path, score) sorted by importance
            n_significant_paths: int
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    # Baseline
    base_logits = np.array(model(tokens))
    base_metric = metric_fn(base_logits)

    path_scores = {}

    # Depth 1: individual heads
    for layer in range(n_layers):
        for head in range(n_heads):
            hook_key = f"blocks.{layer}.hook_z"
            h = head
            def make_hook(h_idx):
                def hook_fn(x, name):
                    return x.at[:, h_idx, :].set(0.0)
                return hook_fn
            state = HookState(hook_fns={hook_key: make_hook(h)}, cache={})
            abl_logits = np.array(model(tokens, hook_state=state))
            effect = abs(metric_fn(abl_logits) - base_metric)
            path_scores[(layer, head)] = float(effect)

    # Depth 2: head pairs
    if max_depth >= 2:
        for l1 in range(n_layers):
            for h1 in range(n_heads):
                for l2 in range(l1 + 1, n_layers):
                    for h2 in range(n_heads):
                        hook1 = f"blocks.{l1}.hook_z"
                        hook2 = f"blocks.{l2}.hook_z"
                        def make_hook2(h_idx):
                            def hook_fn(x, name):
                                return x.at[:, h_idx, :].set(0.0)
                            return hook_fn
                        hooks = {hook1: make_hook2(h1), hook2: make_hook2(h2)}
                        state = HookState(hook_fns=hooks, cache={})
                        abl_logits = np.array(model(tokens, hook_state=state))
                        joint = abs(metric_fn(abl_logits) - base_metric)
                        ind1 = path_scores.get((l1, h1), 0)
                        ind2 = path_scores.get((l2, h2), 0)
                        # Interaction = joint - sum of individual
                        interaction = joint - (ind1 + ind2)
                        path_scores[(l1, h1, l2, h2)] = float(abs(interaction))

    # Sort paths
    top_paths = sorted(path_scores.items(), key=lambda x: -x[1])[:20]
    n_sig = sum(1 for _, v in path_scores.items() if v > 0.01)

    return {
        "path_scores": path_scores,
        "top_paths": top_paths,
        "n_significant_paths": n_sig,
    }


def virtual_attention_patterns(model, tokens, src_layer=0, src_head=0, dst_layer=1, dst_head=0):
    """Compute virtual attention patterns from composed head pairs.

    The virtual attention pattern shows what a later head effectively attends to
    when composition with an earlier head is considered.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        src_layer: Source (earlier) layer.
        src_head: Source head index.
        dst_layer: Destination (later) layer.
        dst_head: Destination head index.

    Returns:
        dict with:
            src_pattern: [seq_len, seq_len] source attention pattern
            dst_pattern: [seq_len, seq_len] destination attention pattern
            virtual_pattern: [seq_len, seq_len] composed virtual pattern
            composition_strength: float
    """
    from irtk.hook_points import HookState

    cache_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=cache_state)
    cache = cache_state.cache

    src_pattern_key = f"blocks.{src_layer}.attn.hook_pattern"
    dst_pattern_key = f"blocks.{dst_layer}.attn.hook_pattern"

    src_pat = cache.get(src_pattern_key)
    dst_pat = cache.get(dst_pattern_key)

    if src_pat is None or dst_pat is None:
        seq_len = len(tokens)
        return {
            "src_pattern": np.zeros((seq_len, seq_len)),
            "dst_pattern": np.zeros((seq_len, seq_len)),
            "virtual_pattern": np.zeros((seq_len, seq_len)),
            "composition_strength": 0.0,
        }

    src_p = np.array(src_pat[src_head])  # [seq_len, seq_len]
    dst_p = np.array(dst_pat[dst_head])  # [seq_len, seq_len]

    # Virtual pattern: dst_pattern @ src_pattern
    # This shows what dst effectively attends to via src
    virtual = dst_p @ src_p  # [seq_len, seq_len]

    # Normalize rows
    row_sums = virtual.sum(axis=-1, keepdims=True)
    row_sums = np.where(row_sums < 1e-10, 1.0, row_sums)
    virtual = virtual / row_sums

    # Composition strength: how different is virtual from dst alone
    diff = np.mean(np.abs(virtual - dst_p))
    strength = float(diff)

    return {
        "src_pattern": src_p,
        "dst_pattern": dst_p,
        "virtual_pattern": virtual,
        "composition_strength": strength,
    }


def full_composition_matrix(model):
    """Compute full composition score matrix across all head pairs and types.

    Combines Q, K, and V composition into a single summary matrix.

    Args:
        model: HookedTransformer model.

    Returns:
        dict with:
            total_composition: [n_layers, n_heads, n_layers, n_heads] combined scores
            composition_type: [n_layers, n_heads, n_layers, n_heads] dominant type (0=Q, 1=K, 2=V)
            layer_composition_summary: [n_layers, n_layers] layer-level composition
            most_composing_pair: tuple (src_l, src_h, dst_l, dst_h)
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    qk_result = qk_composition_scores(model)
    v_result = v_composition_scores(model)

    q_comp = qk_result["q_composition"]
    k_comp = qk_result["k_composition"]
    v_comp = v_result["v_composition"]

    # Stack and find dominant
    stacked = np.stack([q_comp, k_comp, v_comp], axis=0)  # [3, nl, nh, nl, nh]
    total = np.max(stacked, axis=0)  # max of Q/K/V
    comp_type = np.argmax(stacked, axis=0)  # which type dominates

    # Layer-level summary
    layer_comp = np.zeros((n_layers, n_layers))
    for sl in range(n_layers):
        for dl in range(n_layers):
            layer_comp[sl, dl] = float(np.mean(total[sl, :, dl, :]))

    # Most composing pair
    flat_idx = np.argmax(total.reshape(-1))
    sl, sh, dl, dh = np.unravel_index(flat_idx, total.shape)

    return {
        "total_composition": total,
        "composition_type": comp_type,
        "layer_composition_summary": layer_comp,
        "most_composing_pair": (int(sl), int(sh), int(dl), int(dh)),
    }
