"""Systematic path-level circuit analysis.

Enumerates and evaluates computational paths through the residual stream.
Goes beyond component-level analysis to understand how information flows
through specific multi-hop paths (e.g., embed -> head 0.1 -> head 1.3 -> unembed).

References:
    Elhage et al. (2021) "A Mathematical Framework for Transformer Circuits"
    Conmy et al. (2023) "Towards Automated Circuit Discovery"
"""

import jax
import jax.numpy as jnp
import numpy as np
from itertools import product


def enumerate_paths(model, tokens, max_depth=2, top_k=5):
    """Enumerate top computational paths by contribution magnitude.

    A "path" is a sequence of components that information passes through:
    embed -> component_1 -> component_2 -> ... -> unembed.
    Each component is either an attention head (layer, head) or MLP (layer).

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        max_depth: Maximum path depth (number of intermediate components).
        top_k: Number of top paths to return.

    Returns:
        dict with:
            paths: list of path tuples, each element is ('attn', layer, head) or ('mlp', layer)
            path_contributions: array of contribution magnitudes for each path
            n_paths_enumerated: total paths considered
            top_paths: top_k paths sorted by contribution
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    # Get all component outputs
    hook_state = HookState(hook_fns={}, cache={})
    logits = model(tokens, hook_state=hook_state)
    cache = hook_state.cache

    # Get unembed direction (last position, top predicted token)
    top_token = int(jnp.argmax(logits[-1]))
    unembed_dir = model.unembed.W_U[:, top_token]  # [d_model]
    unembed_dir = unembed_dir / (jnp.linalg.norm(unembed_dir) + 1e-10)

    # Collect per-head outputs and MLP outputs
    components = []
    component_labels = []
    for layer in range(n_layers):
        # Attention heads via hook_result (already projected)
        attn_result = cache.get(f"blocks.{layer}.attn.hook_result")
        if attn_result is not None:
            # hook_result is [seq, d_model], but we need per-head
            hook_z = cache.get(f"blocks.{layer}.attn.hook_z")
            if hook_z is not None:
                # hook_z: [seq, n_heads, d_head], project each head
                W_O = model.blocks[layer].attn.W_O  # [n_heads, d_head, d_model]
                b_O = getattr(model.blocks[layer].attn, 'b_O', None)
                for h in range(n_heads):
                    head_out = jnp.einsum("sh,hm->sm", hook_z[:, h, :], W_O[h])
                    components.append(head_out[-1])  # last position [d_model]
                    component_labels.append(("attn", layer, h))
        # MLP output
        mlp_key = f"blocks.{layer}.hook_mlp_out"
        if mlp_key in cache:
            mlp_out = cache[mlp_key]
            components.append(mlp_out[-1])  # [d_model]
            component_labels.append(("mlp", layer))

    if not components:
        return {
            "paths": [],
            "path_contributions": np.array([]),
            "n_paths_enumerated": 0,
            "top_paths": [],
        }

    # Direct (depth-1) contributions
    direct_contribs = []
    for comp in components:
        contrib = float(jnp.dot(comp, unembed_dir))
        direct_contribs.append(contrib)

    # For depth-1, paths are just individual components
    all_paths = []
    all_contributions = []

    for i, label in enumerate(component_labels):
        all_paths.append((label,))
        all_contributions.append(abs(direct_contribs[i]))

    # For depth-2, consider pairs (component_i -> component_j)
    # Approximation: contribution = alignment of comp_i output with comp_j input direction
    if max_depth >= 2:
        for i, label_i in enumerate(component_labels):
            for j, label_j in enumerate(component_labels):
                # Only consider forward paths (i's layer < j's layer)
                layer_i = label_i[1]
                layer_j = label_j[1]
                if layer_i >= layer_j:
                    continue
                # Path contribution: (comp_i output) dot (comp_j direction) * (comp_j -> unembed)
                comp_i_out = components[i]
                comp_j_out = components[j]
                comp_j_dir = comp_j_out / (jnp.linalg.norm(comp_j_out) + 1e-10)
                path_strength = float(jnp.dot(comp_i_out, comp_j_dir) * jnp.dot(comp_j_out, unembed_dir))
                all_paths.append((label_i, label_j))
                all_contributions.append(abs(path_strength))

    contributions = np.array(all_contributions)
    sorted_indices = np.argsort(contributions)[::-1][:top_k]

    return {
        "paths": all_paths,
        "path_contributions": contributions,
        "n_paths_enumerated": len(all_paths),
        "top_paths": [(all_paths[i], float(contributions[i])) for i in sorted_indices],
    }


def path_contribution_matrix(model, tokens, pos=-1):
    """Compute contribution of each component to the final prediction.

    Returns a matrix showing how each attention head and MLP layer
    contributes to the logit of the top predicted token.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        pos: Position to analyze.

    Returns:
        dict with:
            attn_contributions: array [n_layers, n_heads] of attention head contributions
            mlp_contributions: array [n_layers] of MLP contributions
            embed_contribution: float, embedding layer contribution
            total_contribution: float, sum of all contributions
            dominant_component: tuple identifying the largest contributor
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    hook_state = HookState(hook_fns={}, cache={})
    logits = model(tokens, hook_state=hook_state)
    cache = hook_state.cache

    top_token = int(jnp.argmax(logits[pos]))
    unembed_dir = model.unembed.W_U[:, top_token]

    attn_contribs = np.zeros((n_layers, n_heads))
    mlp_contribs = np.zeros(n_layers)

    for layer in range(n_layers):
        hook_z = cache.get(f"blocks.{layer}.attn.hook_z")
        if hook_z is not None:
            W_O = model.blocks[layer].attn.W_O
            for h in range(n_heads):
                head_out = jnp.einsum("h,hm->m", hook_z[pos, h, :], W_O[h])
                attn_contribs[layer, h] = float(jnp.dot(head_out, unembed_dir))

        mlp_key = f"blocks.{layer}.hook_mlp_out"
        if mlp_key in cache:
            mlp_contribs[layer] = float(jnp.dot(cache[mlp_key][pos], unembed_dir))

    # Embedding contribution
    embed_key = "hook_embed"
    pos_key = "hook_pos_embed"
    embed_contrib = 0.0
    if embed_key in cache:
        embed_contrib += float(jnp.dot(cache[embed_key][pos], unembed_dir))
    if pos_key in cache:
        embed_contrib += float(jnp.dot(cache[pos_key][pos], unembed_dir))

    total = embed_contrib + float(np.sum(attn_contribs)) + float(np.sum(mlp_contribs))

    # Find dominant
    max_attn = float(np.max(np.abs(attn_contribs)))
    max_mlp = float(np.max(np.abs(mlp_contribs)))
    if abs(embed_contrib) >= max_attn and abs(embed_contrib) >= max_mlp:
        dominant = ("embed",)
    elif max_attn >= max_mlp:
        idx = np.unravel_index(np.argmax(np.abs(attn_contribs)), attn_contribs.shape)
        dominant = ("attn", int(idx[0]), int(idx[1]))
    else:
        dominant = ("mlp", int(np.argmax(np.abs(mlp_contribs))))

    return {
        "attn_contributions": attn_contribs,
        "mlp_contributions": mlp_contribs,
        "embed_contribution": embed_contrib,
        "total_contribution": total,
        "dominant_component": dominant,
    }


def virtual_weight_path(model, layer_a, head_a, layer_b, head_b):
    """Compute the virtual weight matrix for a two-hop attention path.

    The virtual weight OV_a @ QK_b captures how head A's output
    influences head B's attention pattern (Q-composition) or
    OV_a @ OV_b for output composition.

    Args:
        model: HookedTransformer model.
        layer_a: Layer index of first head.
        head_a: Head index of first head.
        layer_b: Layer index of second head.
        head_b: Head index of second head.

    Returns:
        dict with:
            ov_ov_composition: float, Frobenius norm of OV_a @ OV_b path
            ov_qk_composition: float, Frobenius norm of OV_a @ W_Q_b contribution
            composition_score: float, normalized composition strength
            ov_a_matrix: the OV matrix for head A [d_model, d_model]
    """
    attn_a = model.blocks[layer_a].attn
    attn_b = model.blocks[layer_b].attn

    # OV matrix for head A: W_V_a @ W_O_a -> [d_model, d_model]
    W_V_a = attn_a.W_V[head_a]  # [d_model, d_head]
    W_O_a = attn_a.W_O[head_a]  # [d_head, d_model]
    OV_a = W_V_a @ W_O_a  # [d_model, d_model]

    # OV matrix for head B
    W_V_b = attn_b.W_V[head_b]
    W_O_b = attn_b.W_O[head_b]
    OV_b = W_V_b @ W_O_b

    # OV-OV composition: how A's output affects B's output
    ov_ov = OV_a @ OV_b  # [d_model, d_model]
    ov_ov_norm = float(jnp.linalg.norm(ov_ov))

    # Q-composition: how A's output affects B's queries
    W_Q_b = attn_b.W_Q[head_b]  # [d_model, d_head]
    ov_q = OV_a @ W_Q_b  # [d_model, d_head]
    ov_qk_norm = float(jnp.linalg.norm(ov_q))

    # Normalize by individual norms
    ov_a_norm = float(jnp.linalg.norm(OV_a))
    ov_b_norm = float(jnp.linalg.norm(OV_b))
    composition_score = ov_ov_norm / (ov_a_norm * ov_b_norm + 1e-10)

    return {
        "ov_ov_composition": ov_ov_norm,
        "ov_qk_composition": ov_qk_norm,
        "composition_score": composition_score,
        "ov_a_matrix": np.array(OV_a),
    }


def residual_stream_decomposition(model, tokens, pos=-1):
    """Decompose the residual stream at every layer into component contributions.

    Shows how the residual stream builds up through embedding, attention,
    and MLP contributions at each layer.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        pos: Position to analyze.

    Returns:
        dict with:
            cumulative_norms: array [n_layers+1] of residual stream norms
            attn_added_norms: array [n_layers] of attention output norms
            mlp_added_norms: array [n_layers] of MLP output norms
            growth_rate: array [n_layers] of relative norm growth per layer
            embed_norm: float, initial embedding norm
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers

    hook_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=hook_state)
    cache = hook_state.cache

    cumulative_norms = np.zeros(n_layers + 1)
    attn_norms = np.zeros(n_layers)
    mlp_norms = np.zeros(n_layers)

    # Embedding norm
    resid = cache.get(f"blocks.0.hook_resid_pre")
    if resid is not None:
        cumulative_norms[0] = float(jnp.linalg.norm(resid[pos]))

    for layer in range(n_layers):
        # Attention output norm
        attn_key = f"blocks.{layer}.hook_attn_out"
        if attn_key in cache:
            attn_norms[layer] = float(jnp.linalg.norm(cache[attn_key][pos]))

        # MLP output norm
        mlp_key = f"blocks.{layer}.hook_mlp_out"
        if mlp_key in cache:
            mlp_norms[layer] = float(jnp.linalg.norm(cache[mlp_key][pos]))

        # Cumulative residual after this layer
        resid_key = f"blocks.{layer}.hook_resid_post"
        if resid_key in cache:
            cumulative_norms[layer + 1] = float(jnp.linalg.norm(cache[resid_key][pos]))

    # Growth rate
    growth = np.zeros(n_layers)
    for l in range(n_layers):
        if cumulative_norms[l] > 1e-10:
            growth[l] = (cumulative_norms[l + 1] - cumulative_norms[l]) / cumulative_norms[l]

    return {
        "cumulative_norms": cumulative_norms,
        "attn_added_norms": attn_norms,
        "mlp_added_norms": mlp_norms,
        "growth_rate": growth,
        "embed_norm": float(cumulative_norms[0]),
    }


def path_patching_matrix(model, tokens, corrupted_tokens, metric_fn, pos=-1):
    """Compute path patching effects for all single-component paths.

    For each component (attention head or MLP), patches its output from
    the corrupted run into the clean run and measures the effect on the metric.

    Args:
        model: HookedTransformer model.
        tokens: Clean input token IDs [seq_len].
        corrupted_tokens: Corrupted input token IDs [seq_len].
        metric_fn: Function from logits -> scalar metric.
        pos: Position to patch at.

    Returns:
        dict with:
            attn_effects: array [n_layers, n_heads] of patching effects
            mlp_effects: array [n_layers] of patching effects
            baseline_metric: float, metric on clean input
            corrupted_metric: float, metric on corrupted input
            most_important_component: tuple identifying largest effect
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    # Clean run
    clean_state = HookState(hook_fns={}, cache={})
    clean_logits = model(tokens, hook_state=clean_state)
    baseline = metric_fn(clean_logits)
    clean_cache = clean_state.cache

    # Corrupted run
    corrupt_state = HookState(hook_fns={}, cache={})
    corrupt_logits = model(corrupted_tokens, hook_state=corrupt_state)
    corrupted = metric_fn(corrupt_logits)
    corrupt_cache = corrupt_state.cache

    attn_effects = np.zeros((n_layers, n_heads))
    mlp_effects = np.zeros(n_layers)

    # Patch each attention head
    for layer in range(n_layers):
        hook_z_key = f"blocks.{layer}.attn.hook_z"
        clean_z = clean_cache.get(hook_z_key)
        corrupt_z = corrupt_cache.get(hook_z_key)
        if clean_z is None or corrupt_z is None:
            continue

        for h in range(n_heads):
            # Replace head h's output with corrupted version
            def make_patch_fn(l, head_idx, corrupt_val):
                def patch_fn(x, name):
                    patched = x.at[:, head_idx, :].set(corrupt_val[:, head_idx, :])
                    return patched
                return patch_fn

            patch_state = HookState(
                hook_fns={hook_z_key: make_patch_fn(layer, h, corrupt_z)},
                cache={},
            )
            patched_logits = model(tokens, hook_state=patch_state)
            patched_metric = metric_fn(patched_logits)
            attn_effects[layer, h] = baseline - patched_metric

        # Patch MLP
        mlp_key = f"blocks.{layer}.hook_mlp_out"
        clean_mlp = clean_cache.get(mlp_key)
        corrupt_mlp = corrupt_cache.get(mlp_key)
        if clean_mlp is not None and corrupt_mlp is not None:
            def make_mlp_patch(corrupt_val):
                def patch_fn(x, name):
                    return corrupt_val
                return patch_fn

            patch_state = HookState(
                hook_fns={mlp_key: make_mlp_patch(corrupt_mlp)},
                cache={},
            )
            patched_logits = model(tokens, hook_state=patch_state)
            mlp_effects[layer] = baseline - metric_fn(patched_logits)

    # Most important
    max_attn = float(np.max(np.abs(attn_effects)))
    max_mlp = float(np.max(np.abs(mlp_effects)))
    if max_attn >= max_mlp:
        idx = np.unravel_index(np.argmax(np.abs(attn_effects)), attn_effects.shape)
        most_important = ("attn", int(idx[0]), int(idx[1]))
    else:
        most_important = ("mlp", int(np.argmax(np.abs(mlp_effects))))

    return {
        "attn_effects": attn_effects,
        "mlp_effects": mlp_effects,
        "baseline_metric": float(baseline),
        "corrupted_metric": float(corrupted),
        "most_important_component": most_important,
    }
