"""SAE feature-level steering and intervention.

Bridges sparse autoencoders and activation steering: steer model behavior
by manipulating individual SAE features rather than raw direction vectors.
This enables precise, interpretable interventions at the feature level.

Functions:
- feature_steer: Run model with a single SAE feature amplified/suppressed
- multi_feature_steer: Intervene on multiple features simultaneously
- find_steering_features: Find features that best steer toward a target behavior
- feature_ablation_effect: Measure the effect of zeroing out each feature
- clamped_feature_generation: Generate text with a feature clamped to a fixed value

References:
    - Templeton et al. (2024) "Scaling Monosemanticity" (Anthropic)
    - Bricken et al. (2023) "Towards Monosemanticity" (Anthropic)
"""

from typing import Optional, Callable

import jax
import jax.numpy as jnp
import numpy as np

from irtk.hooked_transformer import HookedTransformer
from irtk.sae import SparseAutoencoder


def feature_steer(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    sae: SparseAutoencoder,
    hook_name: str,
    feature_idx: int,
    alpha: float = 1.0,
    pos: Optional[int] = None,
) -> dict:
    """Run the model with a single SAE feature amplified or suppressed.

    Intercepts activations at hook_name, decomposes via the SAE, modifies
    the target feature's activation, and reconstructs.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        sae: Trained SparseAutoencoder for this hook point.
        hook_name: Hook point name to intervene at.
        feature_idx: Which SAE feature to modify.
        alpha: Multiplier for the feature (>1 amplifies, 0 ablates, <0 inverts).
        pos: If given, only modify at this token position.

    Returns:
        Dict with:
            "steered_logits": [seq_len, d_vocab] logits after intervention
            "clean_logits": [seq_len, d_vocab] logits without intervention
            "feature_activation": original activation of the target feature
            "logit_diff": max absolute logit change at last position
    """
    # Get clean logits and feature activation
    _, cache = model.run_with_cache(tokens)
    clean_act = cache.cache_dict.get(hook_name)
    clean_logits = model(tokens)

    if clean_act is None:
        return {
            "steered_logits": np.array(clean_logits),
            "clean_logits": np.array(clean_logits),
            "feature_activation": 0.0,
            "logit_diff": 0.0,
        }

    # Get feature activations
    feat_acts = sae.encode(clean_act)
    if pos is not None:
        orig_feat_val = float(feat_acts[pos, feature_idx])
    else:
        orig_feat_val = float(jnp.mean(feat_acts[:, feature_idx]))

    def steer_hook(x, name):
        f_acts = sae.encode(x)
        feature_dir = sae.W_dec[feature_idx]  # [d_model]
        if pos is not None:
            # Modify only at specified position
            current_val = f_acts[pos, feature_idx]
            delta = current_val * (alpha - 1.0)
            return x.at[pos].add(delta * feature_dir)
        else:
            # Modify at all positions
            current_vals = f_acts[:, feature_idx]  # [seq_len]
            deltas = current_vals * (alpha - 1.0)  # [seq_len]
            return x + deltas[:, None] * feature_dir[None, :]

    steered_logits = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, steer_hook)])

    logit_diff = float(jnp.max(jnp.abs(steered_logits[-1] - clean_logits[-1])))

    return {
        "steered_logits": np.array(steered_logits),
        "clean_logits": np.array(clean_logits),
        "feature_activation": orig_feat_val,
        "logit_diff": logit_diff,
    }


def multi_feature_steer(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    sae: SparseAutoencoder,
    hook_name: str,
    feature_interventions: dict,
    pos: Optional[int] = None,
) -> dict:
    """Run the model with multiple SAE features modified simultaneously.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        sae: Trained SparseAutoencoder for this hook point.
        hook_name: Hook point name to intervene at.
        feature_interventions: {feature_idx: alpha} mapping feature indices
            to their multipliers. E.g., {5: 2.0, 12: 0.0} amplifies feature 5
            and ablates feature 12.
        pos: If given, only modify at this token position.

    Returns:
        Dict with:
            "steered_logits": [seq_len, d_vocab] logits after intervention
            "clean_logits": [seq_len, d_vocab] logits without intervention
            "n_features_modified": number of features intervened on
            "logit_diff": max absolute logit change at last position
    """
    clean_logits = model(tokens)

    if not feature_interventions:
        return {
            "steered_logits": np.array(clean_logits),
            "clean_logits": np.array(clean_logits),
            "n_features_modified": 0,
            "logit_diff": 0.0,
        }

    def steer_hook(x, name):
        f_acts = sae.encode(x)
        total_delta = jnp.zeros_like(x)
        for fidx, alpha in feature_interventions.items():
            feature_dir = sae.W_dec[fidx]  # [d_model]
            if pos is not None:
                current_val = f_acts[pos, fidx]
                d = current_val * (alpha - 1.0)
                total_delta = total_delta.at[pos].add(d * feature_dir)
            else:
                current_vals = f_acts[:, fidx]  # [seq_len]
                deltas = current_vals * (alpha - 1.0)
                total_delta = total_delta + deltas[:, None] * feature_dir[None, :]
        return x + total_delta

    steered_logits = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, steer_hook)])
    logit_diff = float(jnp.max(jnp.abs(steered_logits[-1] - clean_logits[-1])))

    return {
        "steered_logits": np.array(steered_logits),
        "clean_logits": np.array(clean_logits),
        "n_features_modified": len(feature_interventions),
        "logit_diff": logit_diff,
    }


def find_steering_features(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    sae: SparseAutoencoder,
    hook_name: str,
    metric_fn: Callable,
    top_k: int = 10,
) -> dict:
    """Find which SAE features most affect a target metric when amplified.

    Tests each active feature by amplifying it (2x) and measuring the
    metric change.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        sae: Trained SparseAutoencoder for this hook point.
        hook_name: Hook point name.
        metric_fn: Function(logits) -> float. The target metric.
        top_k: Number of top features to return.

    Returns:
        Dict with:
            "feature_effects": {feature_idx: metric_change} for top features
            "top_positive": list of (feature_idx, effect) increasing the metric
            "top_negative": list of (feature_idx, effect) decreasing the metric
            "baseline_metric": metric value without intervention
    """
    # Get baseline metric
    clean_logits = model(tokens)
    baseline = metric_fn(clean_logits)

    # Get active features
    _, cache = model.run_with_cache(tokens)
    act = cache.cache_dict.get(hook_name)
    if act is None:
        return {
            "feature_effects": {},
            "top_positive": [],
            "top_negative": [],
            "baseline_metric": float(baseline),
        }

    feat_acts = sae.encode(act)
    # Mean activation across positions for each feature
    mean_acts = np.array(jnp.mean(feat_acts, axis=0))
    # Only test features that are actually active
    active_features = np.where(mean_acts > 1e-6)[0]

    if len(active_features) == 0:
        return {
            "feature_effects": {},
            "top_positive": [],
            "top_negative": [],
            "baseline_metric": float(baseline),
        }

    effects = {}
    for fidx in active_features[:min(len(active_features), 100)]:
        fidx = int(fidx)

        def steer_hook(x, name, _fidx=fidx):
            f_acts = sae.encode(x)
            feature_dir = sae.W_dec[_fidx]
            current_vals = f_acts[:, _fidx]
            return x + current_vals[:, None] * feature_dir[None, :]  # 2x

        steered_logits = model.run_with_hooks(
            tokens, fwd_hooks=[(hook_name, steer_hook)]
        )
        effect = float(metric_fn(steered_logits)) - float(baseline)
        effects[fidx] = effect

    # Sort by absolute effect
    sorted_effects = sorted(effects.items(), key=lambda x: abs(x[1]), reverse=True)
    top_effects = dict(sorted_effects[:top_k])

    positive = [(f, e) for f, e in sorted_effects if e > 0][:top_k]
    negative = [(f, e) for f, e in sorted_effects if e < 0][:top_k]

    return {
        "feature_effects": top_effects,
        "top_positive": positive,
        "top_negative": negative,
        "baseline_metric": float(baseline),
    }


def feature_ablation_effect(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    sae: SparseAutoencoder,
    hook_name: str,
    metric_fn: Callable,
    features: Optional[list] = None,
) -> dict:
    """Measure the effect of zeroing out each SAE feature.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        sae: Trained SparseAutoencoder for this hook point.
        hook_name: Hook point name.
        metric_fn: Function(logits) -> float.
        features: List of feature indices to test. If None, tests all
            active features.

    Returns:
        Dict with:
            "ablation_effects": {feature_idx: metric_change}
            "most_critical": feature_idx with largest absolute effect
            "total_effect": sum of absolute effects
            "baseline_metric": metric without ablation
    """
    clean_logits = model(tokens)
    baseline = metric_fn(clean_logits)

    # Get active features
    _, cache = model.run_with_cache(tokens)
    act = cache.cache_dict.get(hook_name)
    if act is None:
        return {
            "ablation_effects": {},
            "most_critical": -1,
            "total_effect": 0.0,
            "baseline_metric": float(baseline),
        }

    feat_acts = sae.encode(act)
    mean_acts = np.array(jnp.mean(feat_acts, axis=0))

    if features is None:
        features = list(np.where(mean_acts > 1e-6)[0].astype(int))

    if not features:
        return {
            "ablation_effects": {},
            "most_critical": -1,
            "total_effect": 0.0,
            "baseline_metric": float(baseline),
        }

    effects = {}
    for fidx in features:
        fidx = int(fidx)

        def ablate_hook(x, name, _fidx=fidx):
            f_acts = sae.encode(x)
            feature_dir = sae.W_dec[_fidx]
            current_vals = f_acts[:, _fidx]
            # Remove this feature's contribution
            return x - current_vals[:, None] * feature_dir[None, :]

        ablated_logits = model.run_with_hooks(
            tokens, fwd_hooks=[(hook_name, ablate_hook)]
        )
        effect = float(metric_fn(ablated_logits)) - float(baseline)
        effects[fidx] = effect

    most_critical = max(effects, key=lambda k: abs(effects[k]))
    total_effect = sum(abs(v) for v in effects.values())

    return {
        "ablation_effects": effects,
        "most_critical": most_critical,
        "total_effect": total_effect,
        "baseline_metric": float(baseline),
    }


def clamped_feature_generation(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    sae: SparseAutoencoder,
    hook_name: str,
    feature_idx: int,
    clamp_value: float = 5.0,
    max_new_tokens: int = 20,
    temperature: float = 1.0,
) -> dict:
    """Generate tokens with an SAE feature clamped to a fixed activation value.

    At each generation step, the feature is forced to the specified value,
    allowing exploration of what the feature "wants" the model to produce.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] prompt tokens.
        sae: Trained SparseAutoencoder for this hook point.
        hook_name: Hook point to intervene at.
        feature_idx: Which feature to clamp.
        clamp_value: Fixed activation value for the feature.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (0 = greedy).

    Returns:
        Dict with:
            "generated_tokens": [seq_len + new] full token sequence
            "n_generated": number of new tokens generated
            "clamped_feature": which feature was clamped
            "clamp_value": the clamped value
    """
    current_tokens = jnp.array(tokens)
    key = jax.random.PRNGKey(0)

    def clamp_hook(x, name):
        f_acts = sae.encode(x)
        feature_dir = sae.W_dec[feature_idx]
        # Set feature to clamp_value at last position
        current_val = f_acts[-1, feature_idx]
        delta = clamp_value - current_val
        return x.at[-1].add(delta * feature_dir)

    for _ in range(max_new_tokens):
        if len(current_tokens) > model.cfg.n_ctx:
            current_tokens = current_tokens[-model.cfg.n_ctx:]

        logits = model.run_with_hooks(
            current_tokens, fwd_hooks=[(hook_name, clamp_hook)]
        )
        next_logits = logits[-1]

        if temperature <= 0:
            next_token = jnp.argmax(next_logits)
        else:
            key, subkey = jax.random.split(key)
            next_token = jax.random.categorical(subkey, next_logits / temperature)

        current_tokens = jnp.concatenate([current_tokens, next_token[None]])

    return {
        "generated_tokens": np.array(current_tokens),
        "n_generated": max_new_tokens,
        "clamped_feature": feature_idx,
        "clamp_value": clamp_value,
    }
