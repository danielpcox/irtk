"""Logit lens variants.

Extended logit lens techniques: contrastive lens, causal lens (with intervention),
residual contribution lens, and token-level lens trajectories.

References:
    nostalgebraist (2020) "interpreting GPT: the logit lens"
    Belrose et al. (2023) "Eliciting Latent Predictions from Transformers with the Tuned Lens"
"""

import jax
import jax.numpy as jnp
import numpy as np


def contrastive_logit_lens(model, tokens_a, tokens_b, pos=-1, top_k=5):
    """Contrastive logit lens: compare logit lens between two inputs.

    Shows how intermediate predictions differ between two inputs at each layer.

    Args:
        model: HookedTransformer model.
        tokens_a: First input token IDs [seq_len].
        tokens_b: Second input token IDs [seq_len].
        pos: Position to analyze.
        top_k: Number of top tokens per layer.

    Returns:
        dict with:
            layer_logit_diffs: [n_layers, d_vocab] logit difference per layer
            top_divergent_tokens: list of (layer, token, diff) tuples
            divergence_per_layer: [n_layers] total divergence magnitude
            convergence_layer: int, layer where inputs converge most
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers

    cache_a = HookState(hook_fns={}, cache={})
    model(tokens_a, hook_state=cache_a)

    cache_b = HookState(hook_fns={}, cache={})
    model(tokens_b, hook_state=cache_b)

    W_U = np.array(model.unembed.W_U)
    b_U = np.array(model.unembed.b_U) if hasattr(model.unembed, 'b_U') and model.unembed.b_U is not None else np.zeros(W_U.shape[1])

    d_vocab = W_U.shape[1]
    layer_diffs = np.zeros((n_layers, d_vocab))
    divergence = np.zeros(n_layers)
    top_divergent = []

    for layer in range(n_layers):
        key = f"blocks.{layer}.hook_resid_post"
        ra = cache_a.cache.get(key)
        rb = cache_b.cache.get(key)
        if ra is not None and rb is not None:
            logits_a = np.array(ra[pos]) @ W_U + b_U
            logits_b = np.array(rb[pos]) @ W_U + b_U
            diff = logits_a - logits_b
            layer_diffs[layer] = diff
            divergence[layer] = float(np.linalg.norm(diff))

            top_idx = np.argsort(-np.abs(diff))[:top_k]
            for idx in top_idx:
                top_divergent.append((layer, int(idx), float(diff[idx])))

    convergence = int(np.argmin(divergence))

    return {
        "layer_logit_diffs": layer_diffs,
        "top_divergent_tokens": top_divergent,
        "divergence_per_layer": divergence,
        "convergence_layer": convergence,
    }


def causal_logit_lens(model, tokens, intervention_layer, intervention_fn, pos=-1, top_k=5):
    """Causal logit lens: apply intervention and track downstream predictions.

    Run logit lens at every layer after applying an intervention at a specific layer.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        intervention_layer: Layer at which to intervene.
        intervention_fn: Function(resid) -> new_resid to apply at intervention_layer.
        pos: Position to analyze.
        top_k: Top tokens to return per layer.

    Returns:
        dict with:
            clean_predictions: [n_layers, top_k] top tokens per layer (clean)
            intervened_predictions: [n_layers, top_k] top tokens per layer (intervened)
            prediction_shifts: [n_layers] logit change magnitude post-intervention
            first_affected_layer: int
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    W_U = np.array(model.unembed.W_U)
    b_U = np.array(model.unembed.b_U) if hasattr(model.unembed, 'b_U') and model.unembed.b_U is not None else np.zeros(W_U.shape[1])

    # Clean run
    clean_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=clean_state)

    # Intervened run
    hook_key = f"blocks.{intervention_layer}.hook_resid_pre"
    int_state = HookState(hook_fns={hook_key: lambda x, n: intervention_fn(x)}, cache={})
    model(tokens, hook_state=int_state)

    clean_preds = np.zeros((n_layers, top_k), dtype=int)
    int_preds = np.zeros((n_layers, top_k), dtype=int)
    shifts = np.zeros(n_layers)

    for layer in range(n_layers):
        key = f"blocks.{layer}.hook_resid_post"
        rc = clean_state.cache.get(key)
        ri = int_state.cache.get(key)
        if rc is not None:
            logits_c = np.array(rc[pos]) @ W_U + b_U
            clean_preds[layer] = np.argsort(-logits_c)[:top_k]
        if ri is not None:
            logits_i = np.array(ri[pos]) @ W_U + b_U
            int_preds[layer] = np.argsort(-logits_i)[:top_k]
        if rc is not None and ri is not None:
            shifts[layer] = float(np.linalg.norm(logits_c - logits_i))

    first_affected = int(np.argmax(shifts > 0.01)) if np.any(shifts > 0.01) else n_layers - 1

    return {
        "clean_predictions": clean_preds,
        "intervened_predictions": int_preds,
        "prediction_shifts": shifts,
        "first_affected_layer": first_affected,
    }


def residual_contribution_lens(model, tokens, pos=-1, top_k=5):
    """Residual contribution lens: decompose logit lens by component.

    For each layer, show the contribution of attention and MLP to the logit lens prediction.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        pos: Position to analyze.
        top_k: Top tokens per component.

    Returns:
        dict with:
            attn_logit_contributions: [n_layers, d_vocab] attention contribution to logits
            mlp_logit_contributions: [n_layers, d_vocab] MLP contribution to logits
            cumulative_logits: [n_layers, d_vocab] cumulative logits through layers
            top_attn_tokens: list of (layer, token, logit) tuples
            top_mlp_tokens: list of (layer, token, logit) tuples
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers

    cache_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=cache_state)
    cache = cache_state.cache

    W_U = np.array(model.unembed.W_U)
    b_U = np.array(model.unembed.b_U) if hasattr(model.unembed, 'b_U') and model.unembed.b_U is not None else np.zeros(W_U.shape[1])

    d_vocab = W_U.shape[1]
    attn_logits = np.zeros((n_layers, d_vocab))
    mlp_logits = np.zeros((n_layers, d_vocab))
    cum_logits = np.zeros((n_layers, d_vocab))

    top_attn = []
    top_mlp = []

    for layer in range(n_layers):
        attn = cache.get(f"blocks.{layer}.hook_attn_out")
        mlp = cache.get(f"blocks.{layer}.hook_mlp_out")
        resid = cache.get(f"blocks.{layer}.hook_resid_post")

        if attn is not None:
            a_logits = np.array(attn[pos]) @ W_U
            attn_logits[layer] = a_logits
            top_idx = np.argsort(-a_logits)[:top_k]
            for idx in top_idx:
                top_attn.append((layer, int(idx), float(a_logits[idx])))

        if mlp is not None:
            m_logits = np.array(mlp[pos]) @ W_U
            mlp_logits[layer] = m_logits
            top_idx = np.argsort(-m_logits)[:top_k]
            for idx in top_idx:
                top_mlp.append((layer, int(idx), float(m_logits[idx])))

        if resid is not None:
            cum_logits[layer] = np.array(resid[pos]) @ W_U + b_U

    return {
        "attn_logit_contributions": attn_logits,
        "mlp_logit_contributions": mlp_logits,
        "cumulative_logits": cum_logits,
        "top_attn_tokens": top_attn,
        "top_mlp_tokens": top_mlp,
    }


def token_lens_trajectory(model, tokens, target_tokens=None, pos=-1):
    """Track specific token probabilities through the logit lens.

    For specified target tokens, track their probability at each layer
    to see when/where they emerge or get suppressed.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        target_tokens: List of token IDs to track. If None, tracks top-5 final predictions.
        pos: Position to analyze.

    Returns:
        dict with:
            token_probs: dict of token_id -> [n_layers] probability per layer
            token_ranks: dict of token_id -> [n_layers] rank per layer
            emergence_layers: dict of token_id -> first layer where prob > 0.1
            final_prediction: int (top token at final layer)
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers

    cache_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=cache_state)
    cache = cache_state.cache

    W_U = np.array(model.unembed.W_U)
    b_U = np.array(model.unembed.b_U) if hasattr(model.unembed, 'b_U') and model.unembed.b_U is not None else np.zeros(W_U.shape[1])

    # Determine target tokens
    if target_tokens is None:
        last_key = f"blocks.{n_layers - 1}.hook_resid_post"
        r = cache.get(last_key)
        if r is not None:
            logits = np.array(r[pos]) @ W_U + b_U
            target_tokens = list(np.argsort(-logits)[:5])
        else:
            target_tokens = [0, 1, 2, 3, 4]

    token_probs = {t: np.zeros(n_layers) for t in target_tokens}
    token_ranks = {t: np.zeros(n_layers, dtype=int) for t in target_tokens}
    emergence = {t: n_layers - 1 for t in target_tokens}

    for layer in range(n_layers):
        key = f"blocks.{layer}.hook_resid_post"
        r = cache.get(key)
        if r is not None:
            logits = np.array(r[pos]) @ W_U + b_U
            probs = np.exp(logits - logits.max())
            probs = probs / probs.sum()
            ranks = np.argsort(-logits)

            for t in target_tokens:
                token_probs[t][layer] = float(probs[t])
                rank_pos = np.where(ranks == t)[0]
                token_ranks[t][layer] = int(rank_pos[0]) if len(rank_pos) > 0 else -1
                if token_probs[t][layer] > 0.1 and emergence[t] == n_layers - 1:
                    emergence[t] = layer

    # Final prediction
    last_key = f"blocks.{n_layers - 1}.hook_resid_post"
    r = cache.get(last_key)
    final_pred = 0
    if r is not None:
        logits = np.array(r[pos]) @ W_U + b_U
        final_pred = int(np.argmax(logits))

    return {
        "token_probs": token_probs,
        "token_ranks": token_ranks,
        "emergence_layers": emergence,
        "final_prediction": final_pred,
    }


def logit_lens_difference(model, tokens, pos_a=0, pos_b=-1, top_k=5):
    """Compare logit lens predictions between two positions.

    Shows how intermediate predictions differ between positions at each layer.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        pos_a: First position.
        pos_b: Second position.
        top_k: Number of top tokens to track.

    Returns:
        dict with:
            position_a_predictions: [n_layers, top_k] top tokens at pos_a per layer
            position_b_predictions: [n_layers, top_k] top tokens at pos_b per layer
            logit_diff_per_layer: [n_layers] total logit difference between positions
            shared_top_tokens: [n_layers] number of shared top-k tokens per layer
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers

    cache_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=cache_state)
    cache = cache_state.cache

    W_U = np.array(model.unembed.W_U)
    b_U = np.array(model.unembed.b_U) if hasattr(model.unembed, 'b_U') and model.unembed.b_U is not None else np.zeros(W_U.shape[1])

    preds_a = np.zeros((n_layers, top_k), dtype=int)
    preds_b = np.zeros((n_layers, top_k), dtype=int)
    logit_diff = np.zeros(n_layers)
    shared = np.zeros(n_layers, dtype=int)

    for layer in range(n_layers):
        key = f"blocks.{layer}.hook_resid_post"
        r = cache.get(key)
        if r is not None:
            logits_a = np.array(r[pos_a]) @ W_U + b_U
            logits_b = np.array(r[pos_b]) @ W_U + b_U

            top_a = np.argsort(-logits_a)[:top_k]
            top_b = np.argsort(-logits_b)[:top_k]

            preds_a[layer] = top_a
            preds_b[layer] = top_b

            logit_diff[layer] = float(np.linalg.norm(logits_a - logits_b))
            shared[layer] = len(set(top_a.tolist()) & set(top_b.tolist()))

    return {
        "position_a_predictions": preds_a,
        "position_b_predictions": preds_b,
        "logit_diff_per_layer": logit_diff,
        "shared_top_tokens": shared,
    }
