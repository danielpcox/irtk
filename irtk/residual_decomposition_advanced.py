"""Advanced residual stream decomposition for mechanistic interpretability.

Orthogonal decomposition, per-token residual buildup, component interference,
residual subspace tracking, and contribution isolation.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Optional


def orthogonal_decomposition(
    model,
    tokens,
    layer: int = -1,
    pos: int = -1,
    n_components: int = 5,
) -> dict:
    """Decompose residual stream into orthogonal components.

    Projects the residual onto component contributions and decomposes
    into orthogonal directions.

    Args:
        model: HookedTransformer model.
        tokens: Input token array.
        layer: Layer to decompose.
        pos: Position.
        n_components: Number of orthogonal components.

    Returns:
        Dict with components, explained_variance, directions.
    """
    from irtk.hook_points import HookState

    if layer < 0:
        layer = model.cfg.n_layers + layer
    cache = {}
    hs = HookState(hook_fns={}, cache=cache)
    model(tokens, hook_state=hs)

    resid = np.array(cache[f"blocks.{layer}.hook_resid_post"][pos])

    # Collect component outputs at this position
    contributions = []
    labels = []
    for l in range(layer + 1):
        attn_key = f"blocks.{l}.hook_attn_out"
        if attn_key in cache:
            contributions.append(np.array(cache[attn_key][pos]))
            labels.append(f"attn_{l}")
        mlp_key = f"blocks.{l}.hook_mlp_out"
        if mlp_key in cache:
            contributions.append(np.array(cache[mlp_key][pos]))
            labels.append(f"mlp_{l}")

    if not contributions:
        return {"components": [], "explained_variance": [], "directions": jnp.array([])}

    C = np.stack(contributions)  # [n_contrib, d_model]

    # SVD for orthogonal decomposition
    U, S, Vt = np.linalg.svd(C, full_matrices=False)
    n_comp = min(n_components, len(S))

    # Project residual onto each direction
    explained = []
    for i in range(n_comp):
        proj = float(np.dot(resid, Vt[i]) ** 2)
        explained.append(proj)

    total_var = float(np.dot(resid, resid))
    explained_frac = [e / max(total_var, 1e-10) for e in explained]

    return {
        "components": labels,
        "singular_values": jnp.array(S[:n_comp]),
        "explained_variance": explained_frac,
        "directions": jnp.array(Vt[:n_comp]),
        "residual_norm": float(np.linalg.norm(resid)),
    }


def per_token_residual_buildup(
    model,
    tokens,
    target_pos: int = -1,
) -> dict:
    """Track how the residual stream builds up across layers for a position.

    Args:
        model: HookedTransformer model.
        tokens: Input token array.
        target_pos: Position to track.

    Returns:
        Dict with residual_norms, attn_contributions, mlp_contributions,
        cumulative_buildup.
    """
    from irtk.hook_points import HookState

    cache = {}
    hs = HookState(hook_fns={}, cache=cache)
    model(tokens, hook_state=hs)

    resid_norms = []
    attn_contribs = []
    mlp_contribs = []

    for l in range(model.cfg.n_layers):
        resid_key = f"blocks.{l}.hook_resid_post"
        if resid_key in cache:
            resid_norms.append(float(np.linalg.norm(np.array(cache[resid_key][target_pos]))))

        attn_key = f"blocks.{l}.hook_attn_out"
        if attn_key in cache:
            attn_contribs.append(float(np.linalg.norm(np.array(cache[attn_key][target_pos]))))

        mlp_key = f"blocks.{l}.hook_mlp_out"
        if mlp_key in cache:
            mlp_contribs.append(float(np.linalg.norm(np.array(cache[mlp_key][target_pos]))))

    # Cumulative buildup direction
    buildup = np.zeros(model.cfg.d_model)
    cumulative_norms = []
    for l in range(model.cfg.n_layers):
        attn_key = f"blocks.{l}.hook_attn_out"
        mlp_key = f"blocks.{l}.hook_mlp_out"
        if attn_key in cache:
            buildup += np.array(cache[attn_key][target_pos])
        if mlp_key in cache:
            buildup += np.array(cache[mlp_key][target_pos])
        cumulative_norms.append(float(np.linalg.norm(buildup)))

    return {
        "residual_norms": jnp.array(resid_norms),
        "attn_contributions": jnp.array(attn_contribs),
        "mlp_contributions": jnp.array(mlp_contribs),
        "cumulative_buildup": jnp.array(cumulative_norms),
    }


def component_interference(
    model,
    tokens,
    layer: int = -1,
    pos: int = -1,
) -> dict:
    """Measure interference between component contributions.

    Tests whether components add constructively or destructively.

    Args:
        model: HookedTransformer model.
        tokens: Input token array.
        layer: Layer.
        pos: Position.

    Returns:
        Dict with interference_matrix, constructive_pairs, destructive_pairs,
        net_interference.
    """
    from irtk.hook_points import HookState

    if layer < 0:
        layer = model.cfg.n_layers + layer
    cache = {}
    hs = HookState(hook_fns={}, cache=cache)
    model(tokens, hook_state=hs)

    contributions = []
    labels = []
    for l in range(layer + 1):
        attn_key = f"blocks.{l}.hook_attn_out"
        if attn_key in cache:
            contributions.append(np.array(cache[attn_key][pos]))
            labels.append(f"attn_{l}")
        mlp_key = f"blocks.{l}.hook_mlp_out"
        if mlp_key in cache:
            contributions.append(np.array(cache[mlp_key][pos]))
            labels.append(f"mlp_{l}")

    n = len(contributions)
    interference = np.zeros((n, n))
    constructive = []
    destructive = []

    for i in range(n):
        for j in range(n):
            cos = np.dot(contributions[i], contributions[j]) / (
                np.linalg.norm(contributions[i]) * np.linalg.norm(contributions[j]) + 1e-10
            )
            interference[i, j] = float(cos)

            if i < j:
                if cos > 0.1:
                    constructive.append((labels[i], labels[j], float(cos)))
                elif cos < -0.1:
                    destructive.append((labels[i], labels[j], float(cos)))

    # Net interference: how much norm is lost/gained from cancellation
    sum_vec = np.sum(np.stack(contributions), axis=0) if contributions else np.zeros(1)
    sum_norms = sum(np.linalg.norm(c) for c in contributions)
    actual_norm = float(np.linalg.norm(sum_vec))
    net = float(actual_norm / max(sum_norms, 1e-10) - 1.0)

    return {
        "interference_matrix": jnp.array(interference),
        "labels": labels,
        "constructive_pairs": sorted(constructive, key=lambda x: -x[2]),
        "destructive_pairs": sorted(destructive, key=lambda x: x[2]),
        "net_interference": net,
    }


def residual_subspace_tracking(
    model,
    tokens,
    pos: int = -1,
    n_dims: int = 3,
) -> dict:
    """Track how the residual occupies different subspaces across layers.

    Args:
        model: HookedTransformer model.
        tokens: Input token array.
        pos: Position.
        n_dims: Subspace dimensionality.

    Returns:
        Dict with subspace_overlap, principal_angles, effective_rank.
    """
    from irtk.hook_points import HookState

    cache = {}
    hs = HookState(hook_fns={}, cache=cache)
    model(tokens, hook_state=hs)

    residuals = []
    for l in range(model.cfg.n_layers):
        key = f"blocks.{l}.hook_resid_post"
        if key in cache:
            residuals.append(np.array(cache[key]))  # [seq_len, d_model]

    n_layers = len(residuals)
    overlaps = np.zeros((n_layers, n_layers))
    ranks = []

    for l in range(n_layers):
        R = residuals[l]
        svs = np.linalg.svd(R, compute_uv=False)
        svs_norm = svs / (np.sum(svs) + 1e-10)
        eff_rank = float(np.exp(-np.sum(svs_norm * np.log(svs_norm + 1e-10))))
        ranks.append(eff_rank)

    # Subspace overlap between layers
    for i in range(n_layers):
        for j in range(n_layers):
            Ui, _, _ = np.linalg.svd(residuals[i], full_matrices=False)
            Uj, _, _ = np.linalg.svd(residuals[j], full_matrices=False)
            k = min(n_dims, Ui.shape[1], Uj.shape[1])
            # Principal angles
            M = Ui[:, :k].T @ Uj[:, :k]
            svs = np.linalg.svd(M, compute_uv=False)
            overlaps[i, j] = float(np.mean(svs[:min(k, len(svs))]))

    return {
        "subspace_overlap": jnp.array(overlaps),
        "effective_rank": jnp.array(ranks),
        "n_layers": n_layers,
    }


def contribution_isolation(
    model,
    tokens,
    target_component: str,
    pos: int = -1,
    top_k: int = 5,
) -> dict:
    """Isolate a specific component's contribution to the output.

    Args:
        model: HookedTransformer model.
        tokens: Input token array.
        target_component: Hook name like "blocks.0.hook_attn_out".
        pos: Position.
        top_k: Number of top vocab tokens.

    Returns:
        Dict with contribution_vector, promoted_tokens, demoted_tokens,
        contribution_norm.
    """
    from irtk.hook_points import HookState

    cache = {}
    hs = HookState(hook_fns={}, cache=cache)
    model(tokens, hook_state=hs)

    if target_component not in cache:
        return {"contribution_vector": jnp.array([]), "promoted_tokens": [], "demoted_tokens": []}

    contribution = np.array(cache[target_component][pos])  # [d_model]

    # Project through unembedding
    W_U = np.array(model.unembed.W_U)  # [d_model, d_vocab]
    logit_effect = contribution @ W_U  # [d_vocab]

    promoted = np.argsort(logit_effect)[::-1][:top_k]
    demoted = np.argsort(logit_effect)[:top_k]

    return {
        "contribution_vector": jnp.array(contribution),
        "contribution_norm": float(np.linalg.norm(contribution)),
        "logit_effect": jnp.array(logit_effect),
        "promoted_tokens": [(int(t), float(logit_effect[t])) for t in promoted],
        "demoted_tokens": [(int(t), float(logit_effect[t])) for t in demoted],
    }
