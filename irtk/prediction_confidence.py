"""Prediction confidence analysis.

Analyze prediction confidence mechanics: calibration, overconfidence
detection, confidence sources by component, and entropy decomposition.

References:
    Kadavath et al. (2022) "Language Models (Mostly) Know What They Know"
    Guo et al. (2017) "On Calibration of Modern Neural Networks"
"""

import jax
import jax.numpy as jnp
import numpy as np


def confidence_profile(model, tokens):
    """Profile prediction confidence at each position.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].

    Returns:
        dict with:
            top1_probs: [seq_len] probability of top prediction
            top1_tokens: [seq_len] top predicted token
            entropy: [seq_len] prediction entropy
            confidence_mean: float
            overconfident_positions: list of positions where top1 > 0.9
    """
    logits = np.array(model(tokens))
    seq_len = logits.shape[0]

    # Softmax
    probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs = probs / probs.sum(axis=-1, keepdims=True)

    top1_probs = np.max(probs, axis=-1)
    top1_tokens = np.argmax(probs, axis=-1)

    entropy = -np.sum(probs * np.log(probs + 1e-10), axis=-1)

    overconfident = [int(i) for i in range(seq_len) if top1_probs[i] > 0.9]

    return {
        "top1_probs": top1_probs,
        "top1_tokens": top1_tokens,
        "entropy": entropy,
        "confidence_mean": float(np.mean(top1_probs)),
        "overconfident_positions": overconfident,
    }


def layerwise_confidence_evolution(model, tokens, pos=-1):
    """Track how confidence evolves through layers.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        pos: Position.

    Returns:
        dict with:
            layer_entropy: [n_layers] entropy at each layer's logit lens
            layer_top1_prob: [n_layers] top-1 probability per layer
            confidence_emergence_layer: int, layer where top1 > 0.5
            entropy_reduction: [n_layers-1] entropy drop per layer
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers

    cache_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=cache_state)
    cache = cache_state.cache

    W_U = np.array(model.unembed.W_U)
    b_U = np.array(model.unembed.b_U) if hasattr(model.unembed, 'b_U') and model.unembed.b_U is not None else np.zeros(W_U.shape[1])

    layer_entropy = np.zeros(n_layers)
    layer_top1 = np.zeros(n_layers)

    for layer in range(n_layers):
        key = f"blocks.{layer}.hook_resid_post"
        r = cache.get(key)
        if r is not None:
            resid = np.array(r[pos])
            logits = resid @ W_U + b_U
            probs = np.exp(logits - logits.max())
            probs = probs / probs.sum()
            layer_entropy[layer] = -float(np.sum(probs * np.log(probs + 1e-10)))
            layer_top1[layer] = float(np.max(probs))

    # Emergence layer
    emergence = n_layers - 1
    for l in range(n_layers):
        if layer_top1[l] > 0.5:
            emergence = l
            break

    # Entropy reduction
    ent_red = np.zeros(max(0, n_layers - 1))
    for l in range(n_layers - 1):
        ent_red[l] = layer_entropy[l] - layer_entropy[l + 1]

    return {
        "layer_entropy": layer_entropy,
        "layer_top1_prob": layer_top1,
        "confidence_emergence_layer": emergence,
        "entropy_reduction": ent_red,
    }


def confidence_source_attribution(model, tokens, pos=-1, top_k=5):
    """Attribute confidence to individual components.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        pos: Position.
        top_k: Number of top tokens to track.

    Returns:
        dict with:
            component_confidence_effects: dict of component -> entropy change when ablated
            confidence_boosters: list of components that increase confidence
            confidence_reducers: list of components that decrease confidence
            top_token_attributions: dict of component -> change in top-1 logit
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers

    # Baseline
    logits = np.array(model(tokens))
    base_probs = np.exp(logits[pos] - logits[pos].max())
    base_probs = base_probs / base_probs.sum()
    base_entropy = -float(np.sum(base_probs * np.log(base_probs + 1e-10)))
    top_token = int(np.argmax(base_probs))

    effects = {}
    boosters = []
    reducers = []
    top_token_attr = {}

    for layer in range(n_layers):
        for comp_type, hook_key in [("attn", f"blocks.{layer}.hook_attn_out"),
                                     ("mlp", f"blocks.{layer}.hook_mlp_out")]:
            name = f"{comp_type}_L{layer}"

            state = HookState(hook_fns={hook_key: lambda x, n: jnp.zeros_like(x)}, cache={})
            abl_logits = np.array(model(tokens, hook_state=state))
            abl_probs = np.exp(abl_logits[pos] - abl_logits[pos].max())
            abl_probs = abl_probs / abl_probs.sum()
            abl_entropy = -float(np.sum(abl_probs * np.log(abl_probs + 1e-10)))

            # Entropy change: positive = removing component increases entropy (it was helping confidence)
            ent_change = abl_entropy - base_entropy
            effects[name] = float(ent_change)

            if ent_change > 0.01:
                boosters.append(name)
            elif ent_change < -0.01:
                reducers.append(name)

            # Top token logit change
            top_token_attr[name] = float(logits[pos, top_token] - abl_logits[pos, top_token])

    return {
        "component_confidence_effects": effects,
        "confidence_boosters": boosters,
        "confidence_reducers": reducers,
        "top_token_attributions": top_token_attr,
    }


def entropy_decomposition(model, tokens, pos=-1):
    """Decompose final entropy into per-component contributions.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        pos: Position.

    Returns:
        dict with:
            total_entropy: float
            component_entropy_contributions: dict of component -> float
            entropy_from_embedding: float
            entropy_from_attention: float (total across layers)
            entropy_from_mlp: float (total across layers)
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers

    # Get final logits
    logits = np.array(model(tokens))
    probs = np.exp(logits[pos] - logits[pos].max())
    probs = probs / probs.sum()
    total_ent = -float(np.sum(probs * np.log(probs + 1e-10)))

    # Get component logit contributions via unembed
    cache_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=cache_state)
    cache = cache_state.cache
    W_U = np.array(model.unembed.W_U)

    comp_ent = {}
    embed_ent = 0.0
    attn_ent = 0.0
    mlp_ent = 0.0

    # Embedding contribution
    r = cache.get("blocks.0.hook_resid_pre")
    if r is not None:
        embed_logits = np.array(r[pos]) @ W_U
        # Contribution to entropy via logit magnitude
        embed_ent = float(np.std(embed_logits))
        comp_ent["embed"] = embed_ent

    for layer in range(n_layers):
        attn = cache.get(f"blocks.{layer}.hook_attn_out")
        if attn is not None:
            attn_logits = np.array(attn[pos]) @ W_U
            val = float(np.std(attn_logits))
            comp_ent[f"attn_L{layer}"] = val
            attn_ent += val

        mlp = cache.get(f"blocks.{layer}.hook_mlp_out")
        if mlp is not None:
            mlp_logits = np.array(mlp[pos]) @ W_U
            val = float(np.std(mlp_logits))
            comp_ent[f"mlp_L{layer}"] = val
            mlp_ent += val

    return {
        "total_entropy": total_ent,
        "component_entropy_contributions": comp_ent,
        "entropy_from_embedding": embed_ent,
        "entropy_from_attention": attn_ent,
        "entropy_from_mlp": mlp_ent,
    }


def confidence_calibration(model, tokens_list, correct_tokens_list):
    """Measure calibration: how well confidence predicts correctness.

    Args:
        model: HookedTransformer model.
        tokens_list: List of input token arrays.
        correct_tokens_list: List of correct next tokens (one per input).

    Returns:
        dict with:
            confidences: list of confidence values
            correct: list of bool
            mean_confidence: float
            accuracy: float
            calibration_error: float (|confidence - accuracy| averaged)
    """
    confidences = []
    correct = []

    for tokens, correct_tok in zip(tokens_list, correct_tokens_list):
        logits = np.array(model(tokens))
        probs = np.exp(logits[-1] - logits[-1].max())
        probs = probs / probs.sum()

        top_tok = int(np.argmax(probs))
        conf = float(probs[top_tok])
        is_correct = top_tok == int(correct_tok)

        confidences.append(conf)
        correct.append(is_correct)

    mean_conf = float(np.mean(confidences))
    acc = float(np.mean(correct))
    cal_error = abs(mean_conf - acc)

    return {
        "confidences": confidences,
        "correct": correct,
        "mean_confidence": mean_conf,
        "accuracy": acc,
        "calibration_error": cal_error,
    }
