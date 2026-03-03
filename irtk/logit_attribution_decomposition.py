"""Fine-grained logit attribution decomposition.

Decomposes the final logit vector into per-component, per-position, and
per-token contributions. Provides the most detailed view of how the model
arrives at its prediction.

References:
    Elhage et al. (2021) "A Mathematical Framework for Transformer Circuits"
    Nostalgebraist (2020) "Interpreting GPT: the Logit Lens"
"""

import jax
import jax.numpy as jnp
import numpy as np


def full_logit_decomposition(model, tokens, target_token=None, pos=-1):
    """Decompose the logit for a target token into every component's contribution.

    Returns the contribution of embedding, each attention head, each MLP,
    and the bias to the logit of the target token.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        target_token: Token to decompose logit for (None = top predicted).
        pos: Position to analyze.

    Returns:
        dict with:
            embed_contribution: float
            attn_contributions: array [n_layers, n_heads]
            mlp_contributions: array [n_layers]
            bias_contribution: float
            total_logit: float
            reconstruction_error: float
            target_token: int
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    hook_state = HookState(hook_fns={}, cache={})
    logits = model(tokens, hook_state=hook_state)
    cache = hook_state.cache

    if target_token is None:
        target_token = int(jnp.argmax(logits[pos]))

    unembed_dir = model.unembed.W_U[:, target_token]  # [d_model]
    total_logit = float(logits[pos, target_token])

    # Embedding contribution
    embed_contrib = 0.0
    embed = cache.get("hook_embed")
    if embed is not None:
        embed_contrib += float(jnp.dot(embed[pos], unembed_dir))
    pos_embed = cache.get("hook_pos_embed")
    if pos_embed is not None:
        embed_contrib += float(jnp.dot(pos_embed[pos], unembed_dir))

    # Attention head contributions
    attn_contribs = np.zeros((n_layers, n_heads))
    for layer in range(n_layers):
        hook_z = cache.get(f"blocks.{layer}.attn.hook_z")
        if hook_z is not None:
            W_O = model.blocks[layer].attn.W_O
            for h in range(n_heads):
                head_out = jnp.einsum("h,hm->m", hook_z[pos, h, :], W_O[h])
                attn_contribs[layer, h] = float(jnp.dot(head_out, unembed_dir))

    # MLP contributions
    mlp_contribs = np.zeros(n_layers)
    for layer in range(n_layers):
        mlp_out = cache.get(f"blocks.{layer}.hook_mlp_out")
        if mlp_out is not None:
            mlp_contribs[layer] = float(jnp.dot(mlp_out[pos], unembed_dir))

    # Bias contribution
    b_U = getattr(model.unembed, 'b_U', None)
    bias_contrib = float(b_U[target_token]) if b_U is not None else 0.0

    reconstruction = embed_contrib + float(np.sum(attn_contribs)) + float(np.sum(mlp_contribs)) + bias_contrib
    error = abs(total_logit - reconstruction)

    return {
        "embed_contribution": embed_contrib,
        "attn_contributions": attn_contribs,
        "mlp_contributions": mlp_contribs,
        "bias_contribution": bias_contrib,
        "total_logit": total_logit,
        "reconstruction_error": error,
        "target_token": target_token,
    }


def per_position_logit_attribution(model, tokens, target_token=None, prediction_pos=-1):
    """Attribute the final logit to source positions via attention.

    For each attention head, measures how much each source position
    contributes to the prediction through that head's attention pattern.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        target_token: Token to analyze.
        prediction_pos: Position making the prediction.

    Returns:
        dict with:
            position_contributions: array [seq_len] total contribution from each source position
            head_position_contributions: array [n_layers, n_heads, seq_len]
            most_important_position: int
            position_fraction: array [seq_len] of fractional contributions
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    seq_len = len(tokens)

    hook_state = HookState(hook_fns={}, cache={})
    logits = model(tokens, hook_state=hook_state)
    cache = hook_state.cache

    if target_token is None:
        target_token = int(jnp.argmax(logits[prediction_pos]))

    unembed_dir = model.unembed.W_U[:, target_token]

    head_pos_contribs = np.zeros((n_layers, n_heads, seq_len))

    for layer in range(n_layers):
        pattern = cache.get(f"blocks.{layer}.attn.hook_pattern")
        hook_v = cache.get(f"blocks.{layer}.attn.hook_v")
        if pattern is None or hook_v is None:
            continue
        W_O = model.blocks[layer].attn.W_O

        for h in range(n_heads):
            # For each source position, compute its contribution through this head
            for src in range(seq_len):
                attn_weight = pattern[h, prediction_pos, src]
                v_src = hook_v[src, h, :]  # [d_head]
                head_out = jnp.einsum("h,hm->m", v_src, W_O[h])
                contrib = float(attn_weight * jnp.dot(head_out, unembed_dir))
                head_pos_contribs[layer, h, src] = contrib

    # Sum over heads and layers
    total_pos = np.sum(head_pos_contribs, axis=(0, 1))
    most_important = int(np.argmax(np.abs(total_pos)))
    total_abs = np.sum(np.abs(total_pos))
    fractions = np.abs(total_pos) / (total_abs + 1e-10)

    return {
        "position_contributions": total_pos,
        "head_position_contributions": head_pos_contribs,
        "most_important_position": most_important,
        "position_fraction": fractions,
    }


def top_promoted_demoted_tokens(model, tokens, pos=-1, top_k=5):
    """Find which tokens are most promoted and demoted by each component.

    For each attention head and MLP, identifies the tokens whose logits
    are most increased (promoted) or decreased (demoted).

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        pos: Position to analyze.
        top_k: Number of top tokens to return.

    Returns:
        dict with:
            embed_promoted: array [top_k] of tokens promoted by embedding
            embed_demoted: array [top_k] of tokens demoted by embedding
            attn_promoted: dict (layer, head) -> array [top_k]
            attn_demoted: dict (layer, head) -> array [top_k]
            mlp_promoted: dict layer -> array [top_k]
            mlp_demoted: dict layer -> array [top_k]
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    hook_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=hook_state)
    cache = hook_state.cache

    W_U = np.array(model.unembed.W_U)  # [d_model, d_vocab]

    # Embedding
    embed = cache.get("hook_embed")
    pos_embed = cache.get("hook_pos_embed")
    embed_vec = np.zeros(model.cfg.d_model)
    if embed is not None:
        embed_vec += np.array(embed[pos])
    if pos_embed is not None:
        embed_vec += np.array(pos_embed[pos])
    embed_logits = embed_vec @ W_U  # [d_vocab]
    embed_sorted = np.argsort(embed_logits)
    embed_promoted = embed_sorted[-top_k:][::-1]
    embed_demoted = embed_sorted[:top_k]

    attn_promoted = {}
    attn_demoted = {}
    mlp_promoted = {}
    mlp_demoted = {}

    for layer in range(n_layers):
        hook_z = cache.get(f"blocks.{layer}.attn.hook_z")
        if hook_z is not None:
            W_O = np.array(model.blocks[layer].attn.W_O)
            for h in range(n_heads):
                head_out = np.einsum("h,hm->m", np.array(hook_z[pos, h, :]), W_O[h])
                head_logits = head_out @ W_U
                sorted_idx = np.argsort(head_logits)
                attn_promoted[(layer, h)] = sorted_idx[-top_k:][::-1]
                attn_demoted[(layer, h)] = sorted_idx[:top_k]

        mlp_out = cache.get(f"blocks.{layer}.hook_mlp_out")
        if mlp_out is not None:
            mlp_vec = np.array(mlp_out[pos])
            mlp_logits = mlp_vec @ W_U
            sorted_idx = np.argsort(mlp_logits)
            mlp_promoted[layer] = sorted_idx[-top_k:][::-1]
            mlp_demoted[layer] = sorted_idx[:top_k]

    return {
        "embed_promoted": embed_promoted,
        "embed_demoted": embed_demoted,
        "attn_promoted": attn_promoted,
        "attn_demoted": attn_demoted,
        "mlp_promoted": mlp_promoted,
        "mlp_demoted": mlp_demoted,
    }


def logit_difference_decomposition(model, tokens, token_a, token_b, pos=-1):
    """Decompose the logit difference between two tokens into component contributions.

    Essential for analyzing binary classification tasks (e.g., IOI).

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        token_a: First token index.
        token_b: Second token index.
        pos: Position to analyze.

    Returns:
        dict with:
            logit_diff: float, logit(token_a) - logit(token_b)
            embed_diff: float
            attn_diffs: array [n_layers, n_heads]
            mlp_diffs: array [n_layers]
            bias_diff: float
            largest_contributor: tuple identifying component with largest |diff|
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    hook_state = HookState(hook_fns={}, cache={})
    logits = model(tokens, hook_state=hook_state)
    cache = hook_state.cache

    diff_dir = model.unembed.W_U[:, token_a] - model.unembed.W_U[:, token_b]
    logit_diff = float(logits[pos, token_a] - logits[pos, token_b])

    # Embedding
    embed_diff = 0.0
    embed = cache.get("hook_embed")
    if embed is not None:
        embed_diff += float(jnp.dot(embed[pos], diff_dir))
    pos_embed = cache.get("hook_pos_embed")
    if pos_embed is not None:
        embed_diff += float(jnp.dot(pos_embed[pos], diff_dir))

    # Attention
    attn_diffs = np.zeros((n_layers, n_heads))
    for layer in range(n_layers):
        hook_z = cache.get(f"blocks.{layer}.attn.hook_z")
        if hook_z is not None:
            W_O = model.blocks[layer].attn.W_O
            for h in range(n_heads):
                head_out = jnp.einsum("h,hm->m", hook_z[pos, h, :], W_O[h])
                attn_diffs[layer, h] = float(jnp.dot(head_out, diff_dir))

    # MLP
    mlp_diffs = np.zeros(n_layers)
    for layer in range(n_layers):
        mlp_out = cache.get(f"blocks.{layer}.hook_mlp_out")
        if mlp_out is not None:
            mlp_diffs[layer] = float(jnp.dot(mlp_out[pos], diff_dir))

    # Bias
    b_U = getattr(model.unembed, 'b_U', None)
    bias_diff = float(b_U[token_a] - b_U[token_b]) if b_U is not None else 0.0

    # Largest contributor
    all_contribs = {"embed": abs(embed_diff)}
    for l in range(n_layers):
        for h in range(n_heads):
            all_contribs[("attn", l, h)] = abs(attn_diffs[l, h])
        all_contribs[("mlp", l)] = abs(mlp_diffs[l])
    largest = max(all_contribs, key=all_contribs.get)

    return {
        "logit_diff": logit_diff,
        "embed_diff": embed_diff,
        "attn_diffs": attn_diffs,
        "mlp_diffs": mlp_diffs,
        "bias_diff": bias_diff,
        "largest_contributor": largest,
    }


def cumulative_logit_build(model, tokens, target_token=None, pos=-1):
    """Show how the logit for the target token builds up through the model.

    Returns the cumulative logit after each component is added.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        target_token: Token to track.
        pos: Position.

    Returns:
        dict with:
            component_labels: list of string labels for each component
            cumulative_logits: array of running total after each component
            component_deltas: array of each component's individual contribution
            final_logit: float
            biggest_jump_component: str, component causing largest absolute change
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    hook_state = HookState(hook_fns={}, cache={})
    logits = model(tokens, hook_state=hook_state)
    cache = hook_state.cache

    if target_token is None:
        target_token = int(jnp.argmax(logits[pos]))

    unembed_dir = model.unembed.W_U[:, target_token]

    labels = []
    deltas = []

    # Embedding
    embed_contrib = 0.0
    embed = cache.get("hook_embed")
    if embed is not None:
        embed_contrib += float(jnp.dot(embed[pos], unembed_dir))
    pos_embed = cache.get("hook_pos_embed")
    if pos_embed is not None:
        embed_contrib += float(jnp.dot(pos_embed[pos], unembed_dir))
    labels.append("embed")
    deltas.append(embed_contrib)

    for layer in range(n_layers):
        # Attention heads
        hook_z = cache.get(f"blocks.{layer}.attn.hook_z")
        if hook_z is not None:
            W_O = model.blocks[layer].attn.W_O
            for h in range(n_heads):
                head_out = jnp.einsum("h,hm->m", hook_z[pos, h, :], W_O[h])
                contrib = float(jnp.dot(head_out, unembed_dir))
                labels.append(f"L{layer}H{h}")
                deltas.append(contrib)

        # MLP
        mlp_out = cache.get(f"blocks.{layer}.hook_mlp_out")
        if mlp_out is not None:
            contrib = float(jnp.dot(mlp_out[pos], unembed_dir))
            labels.append(f"L{layer}MLP")
            deltas.append(contrib)

    deltas = np.array(deltas)
    cumulative = np.cumsum(deltas)
    biggest = labels[int(np.argmax(np.abs(deltas)))]

    return {
        "component_labels": labels,
        "cumulative_logits": cumulative,
        "component_deltas": deltas,
        "final_logit": float(cumulative[-1]) if len(cumulative) > 0 else 0.0,
        "biggest_jump_component": biggest,
    }
