"""Advanced logit decomposition for mechanistic interpretability.

Fine-grained logit decomposition: per-component contributions,
interaction terms, direct vs indirect paths, logit lens decomposition,
and token-specific attribution.

References:
- Elhage et al. (2021) "A Mathematical Framework for Transformer Circuits"
- nostalgebraist (2020) "interpreting GPT: the logit lens"
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Callable, Optional


def per_component_logit_contribution(
    model,
    tokens,
    target_token: Optional[int] = None,
    pos: int = -1,
) -> dict:
    """Decompose final logits into per-component contributions.

    Attributes the logit of each vocabulary token to each component
    (attention output and MLP output per layer).

    Args:
        model: HookedTransformer model.
        tokens: Input token array.
        target_token: Token to decompose logits for (default: top prediction).
        pos: Position.

    Returns:
        Dict with attn_contributions, mlp_contributions, embed_contribution,
        total_logit.
    """
    from irtk.hook_points import HookState

    cache = {}
    hs = HookState(hook_fns={}, cache=cache)
    logits = model(tokens, hook_state=hs)

    if target_token is None:
        target_token = int(jnp.argmax(logits[pos]))

    # Get unembedding direction
    W_U = np.array(model.unembed.W_U)  # [d_model, d_vocab]
    b_U = np.array(model.unembed.b_U)  # [d_vocab]
    unembed_dir = W_U[:, target_token]  # [d_model]

    attn_contribs = []
    mlp_contribs = []

    for l in range(model.cfg.n_layers):
        attn_key = f"blocks.{l}.hook_attn_out"
        if attn_key in cache:
            attn_out = np.array(cache[attn_key][pos])  # [d_model]
            attn_contribs.append(float(np.dot(attn_out, unembed_dir)))
        else:
            attn_contribs.append(0.0)

        mlp_key = f"blocks.{l}.hook_mlp_out"
        if mlp_key in cache:
            mlp_out = np.array(cache[mlp_key][pos])
            mlp_contribs.append(float(np.dot(mlp_out, unembed_dir)))
        else:
            mlp_contribs.append(0.0)

    # Embedding contribution
    embed_key = "hook_embed"
    pos_key = "hook_pos_embed"
    embed_contrib = 0.0
    if embed_key in cache:
        embed_contrib += float(np.dot(np.array(cache[embed_key][pos]), unembed_dir))
    if pos_key in cache:
        embed_contrib += float(np.dot(np.array(cache[pos_key][pos]), unembed_dir))

    total = embed_contrib + sum(attn_contribs) + sum(mlp_contribs) + float(b_U[target_token])

    return {
        "attn_contributions": jnp.array(attn_contribs),
        "mlp_contributions": jnp.array(mlp_contribs),
        "embed_contribution": embed_contrib,
        "bias_contribution": float(b_U[target_token]),
        "total_logit": total,
        "target_token": target_token,
    }


def logit_interaction_terms(
    model,
    tokens,
    target_token: Optional[int] = None,
    pos: int = -1,
) -> dict:
    """Compute interaction terms between components in logit space.

    Tests whether pairs of components have synergistic or canceling
    effects on the target logit.

    Args:
        model: HookedTransformer model.
        tokens: Input token array.
        target_token: Token to analyze.
        pos: Position.

    Returns:
        Dict with interaction_matrix, synergistic_pairs, canceling_pairs.
    """
    from irtk.hook_points import HookState

    logits = model(tokens)
    if target_token is None:
        target_token = int(jnp.argmax(logits[pos]))

    base_logit = float(logits[pos, target_token])

    n_layers = model.cfg.n_layers
    components = []
    for l in range(n_layers):
        components.append(("attn", l))
        components.append(("mlp", l))
    n = len(components)

    # Single ablation effects
    single_effects = np.zeros(n)
    for i, (ctype, l) in enumerate(components):
        hook_name = f"blocks.{l}.hook_{ctype}_out"

        def zero_hook(x, name):
            return jnp.zeros_like(x)

        hook_state = HookState(hook_fns={hook_name: zero_hook}, cache=None)
        abl_logits = model(tokens, hook_state=hook_state)
        single_effects[i] = base_logit - float(abl_logits[pos, target_token])

    # Pairwise interactions (sample a few pairs to keep it fast)
    interaction = np.zeros((n, n))
    synergistic = []
    canceling = []

    for i in range(n):
        interaction[i, i] = single_effects[i]
        for j in range(i + 1, min(i + 3, n)):  # Only nearby pairs
            ci_type, ci_l = components[i]
            cj_type, cj_l = components[j]
            hook_i = f"blocks.{ci_l}.hook_{ci_type}_out"
            hook_j = f"blocks.{cj_l}.hook_{cj_type}_out"

            def zero_hook(x, name):
                return jnp.zeros_like(x)

            hooks = {hook_i: zero_hook}
            if hook_j != hook_i:
                hooks[hook_j] = zero_hook

            hook_state = HookState(hook_fns=hooks, cache=None)
            abl_logits = model(tokens, hook_state=hook_state)
            joint = base_logit - float(abl_logits[pos, target_token])

            inter = joint - (single_effects[i] + single_effects[j])
            interaction[i, j] = inter
            interaction[j, i] = inter

            if inter > 1e-4:
                synergistic.append((components[i], components[j], float(inter)))
            elif inter < -1e-4:
                canceling.append((components[i], components[j], float(inter)))

    return {
        "interaction_matrix": jnp.array(interaction),
        "single_effects": jnp.array(single_effects),
        "synergistic_pairs": synergistic,
        "canceling_pairs": canceling,
        "components": components,
        "target_token": target_token,
    }


def direct_vs_indirect_logit(
    model,
    tokens,
    target_token: Optional[int] = None,
    pos: int = -1,
) -> dict:
    """Separate direct and indirect contributions to the logit.

    Direct: component output projected onto unembedding direction.
    Indirect: effect mediated through later layers.

    Args:
        model: HookedTransformer model.
        tokens: Input token array.
        target_token: Token to analyze.
        pos: Position.

    Returns:
        Dict with direct_contributions, indirect_contributions,
        direct_fraction, indirect_fraction.
    """
    from irtk.hook_points import HookState

    cache = {}
    hs = HookState(hook_fns={}, cache=cache)
    logits = model(tokens, hook_state=hs)

    if target_token is None:
        target_token = int(jnp.argmax(logits[pos]))

    W_U = np.array(model.unembed.W_U)
    unembed_dir = W_U[:, target_token]

    n_layers = model.cfg.n_layers
    base_logit = float(logits[pos, target_token])

    # Direct contributions (projection onto unembed)
    direct = np.zeros(n_layers * 2)  # attn + mlp per layer
    for l in range(n_layers):
        attn_key = f"blocks.{l}.hook_attn_out"
        if attn_key in cache:
            direct[2 * l] = float(np.dot(np.array(cache[attn_key][pos]), unembed_dir))

        mlp_key = f"blocks.{l}.hook_mlp_out"
        if mlp_key in cache:
            direct[2 * l + 1] = float(np.dot(np.array(cache[mlp_key][pos]), unembed_dir))

    # Indirect = ablation effect minus direct contribution
    indirect = np.zeros(n_layers * 2)
    for l in range(n_layers):
        for ci, ctype in enumerate(["attn", "mlp"]):
            hook_name = f"blocks.{l}.hook_{ctype}_out"

            def zero_hook(x, name):
                return jnp.zeros_like(x)

            hook_state = HookState(hook_fns={hook_name: zero_hook}, cache=None)
            abl_logits = model(tokens, hook_state=hook_state)
            total_effect = base_logit - float(abl_logits[pos, target_token])
            indirect[2 * l + ci] = total_effect - direct[2 * l + ci]

    total_direct = float(np.sum(np.abs(direct)))
    total_indirect = float(np.sum(np.abs(indirect)))
    total = total_direct + total_indirect + 1e-10

    return {
        "direct_contributions": jnp.array(direct),
        "indirect_contributions": jnp.array(indirect),
        "direct_fraction": float(total_direct / total),
        "indirect_fraction": float(total_indirect / total),
        "target_token": target_token,
    }


def logit_lens_decomposition(
    model,
    tokens,
    target_token: Optional[int] = None,
    pos: int = -1,
) -> dict:
    """Decompose the logit lens trajectory into component contributions.

    Shows how the predicted logit builds up across layers via the
    logit lens (projecting intermediate residuals to vocabulary).

    Args:
        model: HookedTransformer model.
        tokens: Input token array.
        target_token: Token to track.
        pos: Position.

    Returns:
        Dict with logit_trajectory, delta_per_layer, attn_delta, mlp_delta.
    """
    from irtk.hook_points import HookState

    cache = {}
    hs = HookState(hook_fns={}, cache=cache)
    logits = model(tokens, hook_state=hs)

    if target_token is None:
        target_token = int(jnp.argmax(logits[pos]))

    W_U = np.array(model.unembed.W_U)
    b_U = np.array(model.unembed.b_U)

    trajectory = []
    attn_delta = []
    mlp_delta = []

    for l in range(model.cfg.n_layers):
        resid_key = f"blocks.{l}.hook_resid_post"
        if resid_key in cache:
            resid = np.array(cache[resid_key][pos])
            logit = float(resid @ W_U[:, target_token] + b_U[target_token])
            trajectory.append(logit)

        attn_key = f"blocks.{l}.hook_attn_out"
        if attn_key in cache:
            attn_out = np.array(cache[attn_key][pos])
            attn_delta.append(float(attn_out @ W_U[:, target_token]))

        mlp_key = f"blocks.{l}.hook_mlp_out"
        if mlp_key in cache:
            mlp_out = np.array(cache[mlp_key][pos])
            mlp_delta.append(float(mlp_out @ W_U[:, target_token]))

    # Delta per layer
    delta = [trajectory[0]]
    for i in range(1, len(trajectory)):
        delta.append(trajectory[i] - trajectory[i - 1])

    return {
        "logit_trajectory": jnp.array(trajectory),
        "delta_per_layer": jnp.array(delta),
        "attn_delta": jnp.array(attn_delta),
        "mlp_delta": jnp.array(mlp_delta),
        "target_token": target_token,
        "final_logit": trajectory[-1] if trajectory else 0.0,
    }


def token_specific_attribution(
    model,
    tokens,
    target_tokens: Optional[list] = None,
    pos: int = -1,
    top_k: int = 5,
) -> dict:
    """Attribute logit changes to specific source tokens.

    For each target vocabulary token, finds which input positions
    contribute most to its logit.

    Args:
        model: HookedTransformer model.
        tokens: Input token array.
        target_tokens: Tokens to attribute (default: top predictions).
        pos: Output position.
        top_k: Number of top source positions per target.

    Returns:
        Dict with per_target attributions, most_important_positions.
    """
    from irtk.hook_points import HookState

    logits = model(tokens)
    if target_tokens is None:
        top_idx = np.argsort(np.array(logits[pos]))[::-1][:3]
        target_tokens = top_idx.tolist()

    seq_len = len(tokens)
    base_logits = np.array(logits)

    results = {}
    for target in target_tokens:
        base = float(base_logits[pos, target])

        # Ablate each source position (zero its embedding)
        position_effects = []
        for src in range(seq_len):
            hook_name = "hook_embed"

            def zero_pos(x, name, _src=src):
                return x.at[_src].set(jnp.zeros(x.shape[-1]))

            hook_state = HookState(hook_fns={hook_name: zero_pos}, cache=None)
            abl_logits = model(tokens, hook_state=hook_state)
            effect = base - float(abl_logits[pos, target])
            position_effects.append(effect)

        position_effects = np.array(position_effects)
        top_positions = np.argsort(np.abs(position_effects))[::-1][:top_k]

        results[target] = {
            "position_effects": jnp.array(position_effects),
            "top_positions": [(int(p), float(position_effects[p])) for p in top_positions],
            "base_logit": base,
        }

    return {
        "per_target": results,
        "target_tokens": target_tokens,
    }
