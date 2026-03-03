"""Feature-level circuit tracing.

Traces how specific features propagate through the network, measures
feature composition scores, and attributes feature effects to paths.

References:
    Elhage et al. (2022) "Toy Models of Superposition"
    Bricken et al. (2023) "Towards Monosemanticity"
"""

import jax
import jax.numpy as jnp
import numpy as np


def feature_propagation_trace(model, tokens, direction, pos=-1):
    """Trace how a feature direction propagates through layers.

    Projects the residual stream onto a given direction at each layer
    to track how a feature builds up or decays.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        direction: Feature direction [d_model], unit vector.
        pos: Position to analyze.

    Returns:
        dict with:
            projections: array [n_layers+1] of projection onto direction
            attn_contributions: array [n_layers] of attention contribution
            mlp_contributions: array [n_layers] of MLP contribution
            peak_layer: int, layer with maximum projection
            emergence_layer: int, first layer where projection exceeds half max
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    direction = np.array(direction, dtype=np.float32)
    d_norm = np.linalg.norm(direction) + 1e-10
    direction = direction / d_norm

    hook_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=hook_state)
    cache = hook_state.cache

    projections = np.zeros(n_layers + 1)
    attn_contribs = np.zeros(n_layers)
    mlp_contribs = np.zeros(n_layers)

    for layer in range(n_layers + 1):
        if layer == 0:
            key = "blocks.0.hook_resid_pre"
        else:
            key = f"blocks.{layer - 1}.hook_resid_post"
        resid = cache.get(key)
        if resid is not None:
            projections[layer] = float(np.dot(np.array(resid[pos]), direction))

    for layer in range(n_layers):
        attn_out = cache.get(f"blocks.{layer}.hook_attn_out")
        mlp_out = cache.get(f"blocks.{layer}.hook_mlp_out")
        if attn_out is not None:
            attn_contribs[layer] = float(np.dot(np.array(attn_out[pos]), direction))
        if mlp_out is not None:
            mlp_contribs[layer] = float(np.dot(np.array(mlp_out[pos]), direction))

    max_proj = np.max(np.abs(projections))
    peak = int(np.argmax(np.abs(projections)))

    emergence = n_layers
    if max_proj > 1e-10:
        half = max_proj * 0.5
        for l in range(n_layers + 1):
            if abs(projections[l]) >= half:
                emergence = l
                break

    return {
        "projections": projections,
        "attn_contributions": attn_contribs,
        "mlp_contributions": mlp_contribs,
        "peak_layer": peak,
        "emergence_layer": emergence,
    }


def feature_composition_scores(model, direction_a, direction_b):
    """Measure how two features compose through OV and QK circuits.

    Computes the virtual weight that direction_a writes into direction_b
    via OV circuits, and how direction_a affects attention to direction_b
    via QK circuits.

    Args:
        model: HookedTransformer model.
        direction_a: First feature direction [d_model].
        direction_b: Second feature direction [d_model].

    Returns:
        dict with:
            ov_scores: array [n_layers, n_heads] of OV composition scores
            qk_scores: array [n_layers, n_heads] of QK composition scores
            max_ov_head: tuple (layer, head) with strongest OV composition
            max_qk_head: tuple (layer, head) with strongest QK composition
            total_ov_score: float
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    a = np.array(direction_a, dtype=np.float32)
    b = np.array(direction_b, dtype=np.float32)
    a = a / (np.linalg.norm(a) + 1e-10)
    b = b / (np.linalg.norm(b) + 1e-10)

    ov_scores = np.zeros((n_layers, n_heads))
    qk_scores = np.zeros((n_layers, n_heads))

    for layer in range(n_layers):
        block = model.blocks[layer]
        W_V = np.array(block.attn.W_V)  # [n_heads, d_model, d_head]
        W_O = np.array(block.attn.W_O)  # [n_heads, d_head, d_model]
        W_Q = np.array(block.attn.W_Q)  # [n_heads, d_model, d_head]
        W_K = np.array(block.attn.W_K)  # [n_heads, d_model, d_head]

        for h in range(n_heads):
            # OV circuit: a -> W_V[h].T @ a -> [d_head], then [d_head] @ W_O[h] -> [d_model]
            z = W_V[h].T @ a  # [d_head]
            ov_out = z @ W_O[h]  # [d_model]
            ov_scores[layer, h] = float(np.dot(b, ov_out))

            # QK: how much a in key affects b in query
            ka = W_K[h].T @ a  # [d_head]
            qb = W_Q[h].T @ b  # [d_head]
            qk_scores[layer, h] = float(np.dot(qb, ka))

    max_ov_idx = np.unravel_index(np.argmax(np.abs(ov_scores)), ov_scores.shape)
    max_qk_idx = np.unravel_index(np.argmax(np.abs(qk_scores)), qk_scores.shape)

    return {
        "ov_scores": ov_scores,
        "qk_scores": qk_scores,
        "max_ov_head": (int(max_ov_idx[0]), int(max_ov_idx[1])),
        "max_qk_head": (int(max_qk_idx[0]), int(max_qk_idx[1])),
        "total_ov_score": float(np.sum(np.abs(ov_scores))),
    }


def feature_path_attribution(model, tokens, direction, metric_fn, pos=-1):
    """Attribute feature effect to attention and MLP paths.

    For each component, measures how much of the feature's contribution
    to the metric flows through that component.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        direction: Feature direction [d_model].
        metric_fn: Function from logits -> scalar.
        pos: Position.

    Returns:
        dict with:
            baseline_metric: float
            attn_attributions: array [n_layers, n_heads]
            mlp_attributions: array [n_layers]
            dominant_path: str description of dominant path
            total_attribution: float
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    direction = np.array(direction, dtype=np.float32)
    d_norm = np.linalg.norm(direction) + 1e-10
    direction = direction / d_norm

    baseline_logits = model(tokens)
    baseline = metric_fn(baseline_logits)

    attn_attr = np.zeros((n_layers, n_heads))
    mlp_attr = np.zeros(n_layers)

    for layer in range(n_layers):
        # Ablate the feature direction from each head's output
        z_key = f"blocks.{layer}.attn.hook_z"
        for h in range(n_heads):
            def make_proj_ablate(head_idx, d=direction):
                def fn(x, name):
                    # Get OV output for this head, project out direction
                    W_O = np.array(model.blocks[layer].attn.W_O[head_idx])  # [d_head, d_model]
                    head_out = x[:, head_idx, :] @ jnp.array(W_O)  # [seq, d_model]
                    proj = jnp.sum(head_out[pos] * jnp.array(d)) * jnp.array(d)
                    # Zero out this head entirely to measure its contribution
                    return x.at[:, head_idx, :].set(0.0)
                return fn

            state = HookState(hook_fns={z_key: make_proj_ablate(h)}, cache={})
            logits = model(tokens, hook_state=state)
            effect = abs(baseline - metric_fn(logits))
            attn_attr[layer, h] = float(effect)

        # Ablate MLP
        mlp_key = f"blocks.{layer}.hook_mlp_out"
        def zero_fn(x, name):
            return jnp.zeros_like(x)
        state = HookState(hook_fns={mlp_key: zero_fn}, cache={})
        logits = model(tokens, hook_state=state)
        mlp_attr[layer] = abs(baseline - metric_fn(logits))

    # Find dominant
    max_attn = np.max(attn_attr)
    max_mlp = np.max(mlp_attr)
    if max_attn > max_mlp:
        idx = np.unravel_index(np.argmax(attn_attr), attn_attr.shape)
        dominant = f"attn L{idx[0]}H{idx[1]}"
    else:
        dominant = f"mlp L{np.argmax(mlp_attr)}"

    return {
        "baseline_metric": float(baseline),
        "attn_attributions": attn_attr,
        "mlp_attributions": mlp_attr,
        "dominant_path": dominant,
        "total_attribution": float(np.sum(attn_attr) + np.sum(mlp_attr)),
    }


def feature_interaction_matrix(model, directions, tokens, pos=-1):
    """Measure pairwise interactions between feature directions.

    For each pair of directions, computes how much they co-occur in the
    residual stream and how they interact through the OV circuit.

    Args:
        model: HookedTransformer model.
        directions: List of feature directions [n_features, d_model].
        tokens: Input token IDs [seq_len].
        pos: Position.

    Returns:
        dict with:
            coactivation_matrix: array [n_features, n_features] of cosine angles
            ov_interaction_matrix: array [n_features, n_features] of OV interaction
            most_interacting_pair: tuple of (i, j)
            mean_interaction: float
    """
    from irtk.hook_points import HookState

    directions = [np.array(d, dtype=np.float32) for d in directions]
    directions = [d / (np.linalg.norm(d) + 1e-10) for d in directions]
    n_features = len(directions)

    # Get final layer activations
    n_layers = model.cfg.n_layers
    hook_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=hook_state)
    cache = hook_state.cache

    resid = cache.get(f"blocks.{n_layers - 1}.hook_resid_post")
    act = np.array(resid[pos]) if resid is not None else np.zeros(model.cfg.d_model)

    # Coactivation: product of projections
    projections = np.array([np.dot(act, d) for d in directions])
    coact = np.outer(projections, projections)
    # Normalize to [-1, 1]
    norms = np.abs(projections) + 1e-10
    coact_norm = coact / np.outer(norms, norms)

    # OV interaction: sum over heads of d_i^T @ OV @ d_j
    ov_interaction = np.zeros((n_features, n_features))
    for layer in range(n_layers):
        block = model.blocks[layer]
        W_V = np.array(block.attn.W_V)
        W_O = np.array(block.attn.W_O)
        n_heads = W_V.shape[0]

        for h in range(n_heads):
            for i in range(n_features):
                z = W_V[h].T @ directions[i]  # [d_head]
                ov_out = z @ W_O[h]  # [d_model]
                for j in range(n_features):
                    ov_interaction[i, j] += np.dot(directions[j], ov_out)

    # Most interacting pair (off-diagonal)
    mask = np.ones((n_features, n_features)) - np.eye(n_features)
    masked = np.abs(ov_interaction) * mask
    if n_features > 1:
        idx = np.unravel_index(np.argmax(masked), masked.shape)
        most_pair = (int(idx[0]), int(idx[1]))
    else:
        most_pair = (0, 0)

    off_diag = ov_interaction[mask.astype(bool)]
    mean_int = float(np.mean(np.abs(off_diag))) if len(off_diag) > 0 else 0.0

    return {
        "coactivation_matrix": coact_norm,
        "ov_interaction_matrix": ov_interaction,
        "most_interacting_pair": most_pair,
        "mean_interaction": mean_int,
    }


def feature_logit_effect(model, tokens, direction, pos=-1, top_k=5):
    """Measure the effect of a feature direction on the output logits.

    Projects the unembedding matrix onto the feature direction to find
    which tokens are most promoted/demoted by this feature.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        direction: Feature direction [d_model].
        pos: Position.
        top_k: Number of top/bottom tokens.

    Returns:
        dict with:
            logit_effects: array [d_vocab] of effect per token
            top_promoted: array [top_k] of most promoted token indices
            top_demoted: array [top_k] of most demoted token indices
            promotion_scores: array [top_k] of promotion magnitudes
            demotion_scores: array [top_k] of demotion magnitudes
    """
    direction = np.array(direction, dtype=np.float32)
    direction = direction / (np.linalg.norm(direction) + 1e-10)

    W_U = np.array(model.unembed.W_U)  # [d_model, d_vocab]
    logit_effects = W_U.T @ direction  # [d_vocab]

    top_k = min(top_k, len(logit_effects))
    promoted_idx = np.argsort(logit_effects)[-top_k:][::-1]
    demoted_idx = np.argsort(logit_effects)[:top_k]

    return {
        "logit_effects": logit_effects,
        "top_promoted": promoted_idx,
        "top_demoted": demoted_idx,
        "promotion_scores": logit_effects[promoted_idx],
        "demotion_scores": logit_effects[demoted_idx],
    }
