"""Token erasure analysis.

Measures which input tokens matter most for predictions via erasure,
token necessity/sufficiency, and erasure curves.

References:
    Li et al. (2016) "Understanding Neural Networks through Representation Erasure"
    De Cao et al. (2020) "How Do Decisions Emerge across Layers in Neural Models?"
"""

import jax
import jax.numpy as jnp
import numpy as np


def token_erasure_effects(model, tokens, metric_fn, prediction_pos=-1):
    """Measure the effect of erasing each input token on the prediction.

    Replaces each token with a zero embedding and measures metric change.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        metric_fn: Function from logits -> scalar.
        prediction_pos: Position to evaluate.

    Returns:
        dict with:
            erasure_effects: array [seq_len] of metric change per erased token
            most_important_token: int, position with largest effect
            least_important_token: int
            importance_ranking: array of position indices sorted by importance
            mean_effect: float
    """
    from irtk.hook_points import HookState

    seq_len = len(tokens)
    baseline = metric_fn(model(tokens))

    effects = np.zeros(seq_len)
    for pos in range(seq_len):
        # Zero out the embedding at this position
        def make_erase_fn(erase_pos):
            def fn(x, name):
                return x.at[erase_pos].set(jnp.zeros(x.shape[-1]))
            return fn

        hook_name = "blocks.0.hook_resid_pre"
        state = HookState(hook_fns={hook_name: make_erase_fn(pos)}, cache={})
        logits = model(tokens, hook_state=state)
        effects[pos] = abs(baseline - metric_fn(logits))

    ranking = np.argsort(effects)[::-1]

    return {
        "erasure_effects": effects,
        "most_important_token": int(np.argmax(effects)),
        "least_important_token": int(np.argmin(effects)),
        "importance_ranking": ranking,
        "mean_effect": float(np.mean(effects)),
    }


def token_necessity_sufficiency(model, tokens, metric_fn, threshold=0.5):
    """Test necessity and sufficiency of each token.

    Necessity: metric drops significantly when token is removed.
    Sufficiency: token alone (with zeros elsewhere) recovers the metric.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        metric_fn: Function from logits -> scalar.
        threshold: Fraction of baseline to count as necessary/sufficient.

    Returns:
        dict with:
            necessity_scores: array [seq_len]
            sufficiency_scores: array [seq_len]
            necessary_tokens: list of positions
            sufficient_tokens: list of positions
            both_necessary_and_sufficient: list of positions
    """
    from irtk.hook_points import HookState

    seq_len = len(tokens)
    baseline = metric_fn(model(tokens))

    necessity = np.zeros(seq_len)
    sufficiency = np.zeros(seq_len)

    for pos in range(seq_len):
        # Necessity: erase this token
        def make_erase(erase_pos):
            def fn(x, name):
                return x.at[erase_pos].set(jnp.zeros(x.shape[-1]))
            return fn

        state = HookState(hook_fns={"blocks.0.hook_resid_pre": make_erase(pos)}, cache={})
        logits = model(tokens, hook_state=state)
        erased_metric = metric_fn(logits)
        if abs(baseline) > 1e-10:
            necessity[pos] = abs(baseline - erased_metric) / abs(baseline)

        # Sufficiency: keep only this token
        def make_keep_only(keep_pos):
            def fn(x, name):
                mask = jnp.zeros_like(x)
                mask = mask.at[keep_pos].set(x[keep_pos])
                return mask
            return fn

        state = HookState(hook_fns={"blocks.0.hook_resid_pre": make_keep_only(pos)}, cache={})
        logits = model(tokens, hook_state=state)
        solo_metric = metric_fn(logits)
        if abs(baseline) > 1e-10:
            sufficiency[pos] = abs(solo_metric) / abs(baseline)

    necessary = [int(i) for i in range(seq_len) if necessity[i] >= threshold]
    sufficient = [int(i) for i in range(seq_len) if sufficiency[i] >= threshold]
    both = [i for i in necessary if i in sufficient]

    return {
        "necessity_scores": necessity,
        "sufficiency_scores": sufficiency,
        "necessary_tokens": necessary,
        "sufficient_tokens": sufficient,
        "both_necessary_and_sufficient": both,
    }


def erasure_curve(model, tokens, metric_fn):
    """Build an erasure curve: metric vs number of tokens erased.

    Erases tokens in order of importance (most important first).

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        metric_fn: Function from logits -> scalar.

    Returns:
        dict with:
            n_erased: array [seq_len+1] of number erased
            metrics: array [seq_len+1] of metric at each step
            area_under_curve: float, normalized AUC
            tokens_for_50pct_drop: int
    """
    from irtk.hook_points import HookState

    seq_len = len(tokens)

    # Get importance order
    effects = token_erasure_effects(model, tokens, metric_fn)
    order = effects["importance_ranking"]

    baseline = metric_fn(model(tokens))
    n_erased = list(range(seq_len + 1))
    metrics = [float(baseline)]

    erased_set = set()
    for pos in order:
        erased_set.add(int(pos))

        def make_erase_set(to_erase):
            def fn(x, name):
                result = x
                for p in to_erase:
                    result = result.at[p].set(jnp.zeros(x.shape[-1]))
                return result
            return fn

        state = HookState(hook_fns={"blocks.0.hook_resid_pre": make_erase_set(erased_set)}, cache={})
        logits = model(tokens, hook_state=state)
        metrics.append(float(metric_fn(logits)))

    metrics = np.array(metrics)

    # AUC normalized
    auc = float(np.sum(np.abs(metrics))) / (seq_len + 1)
    auc_norm = auc / (abs(baseline) + 1e-10)

    # Tokens for 50% drop
    half = abs(baseline) * 0.5
    n_for_50 = seq_len
    for i in range(1, len(metrics)):
        if abs(baseline - metrics[i]) >= half:
            n_for_50 = i
            break

    return {
        "n_erased": np.array(n_erased),
        "metrics": metrics,
        "area_under_curve": float(auc_norm),
        "tokens_for_50pct_drop": n_for_50,
    }


def pairwise_token_interaction(model, tokens, metric_fn, pos_a=0, pos_b=1):
    """Measure interaction between two tokens via double erasure.

    Interaction = effect(erase both) - effect(erase a) - effect(erase b).

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        metric_fn: Function from logits -> scalar.
        pos_a: First position.
        pos_b: Second position.

    Returns:
        dict with:
            effect_a: float, effect of erasing a alone
            effect_b: float, effect of erasing b alone
            effect_both: float, effect of erasing both
            interaction: float, synergy/redundancy
            interaction_type: str, "synergistic" or "redundant" or "independent"
    """
    from irtk.hook_points import HookState

    baseline = metric_fn(model(tokens))

    def make_erase(positions):
        def fn(x, name):
            result = x
            for p in positions:
                result = result.at[p].set(jnp.zeros(x.shape[-1]))
            return result
        return fn

    # Erase a
    state = HookState(hook_fns={"blocks.0.hook_resid_pre": make_erase([pos_a])}, cache={})
    effect_a = abs(baseline - metric_fn(model(tokens, hook_state=state)))

    # Erase b
    state = HookState(hook_fns={"blocks.0.hook_resid_pre": make_erase([pos_b])}, cache={})
    effect_b = abs(baseline - metric_fn(model(tokens, hook_state=state)))

    # Erase both
    state = HookState(hook_fns={"blocks.0.hook_resid_pre": make_erase([pos_a, pos_b])}, cache={})
    effect_both = abs(baseline - metric_fn(model(tokens, hook_state=state)))

    interaction = effect_both - effect_a - effect_b

    if interaction > 0.01:
        itype = "synergistic"
    elif interaction < -0.01:
        itype = "redundant"
    else:
        itype = "independent"

    return {
        "effect_a": float(effect_a),
        "effect_b": float(effect_b),
        "effect_both": float(effect_both),
        "interaction": float(interaction),
        "interaction_type": itype,
    }


def layerwise_token_importance(model, tokens, metric_fn):
    """Measure token importance at each layer.

    At each layer, erases each token's contribution and measures
    the effect, showing where each token becomes important.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        metric_fn: Function from logits -> scalar.

    Returns:
        dict with:
            importance_matrix: array [n_layers, seq_len] of importance per layer
            emergence_layer: array [seq_len] of layer where token becomes important
            peak_layer: array [seq_len] of layer where token is most important
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    seq_len = len(tokens)
    baseline = metric_fn(model(tokens))

    importance = np.zeros((n_layers, seq_len))

    for layer in range(n_layers):
        hook_name = "blocks.0.hook_resid_pre" if layer == 0 else f"blocks.{layer - 1}.hook_resid_post"

        for pos in range(seq_len):
            def make_erase(erase_pos):
                def fn(x, name):
                    return x.at[erase_pos].set(jnp.zeros(x.shape[-1]))
                return fn

            state = HookState(hook_fns={hook_name: make_erase(pos)}, cache={})
            logits = model(tokens, hook_state=state)
            importance[layer, pos] = abs(baseline - metric_fn(logits))

    # Emergence: first layer where importance exceeds 50% of max
    emergence = np.zeros(seq_len, dtype=int)
    peak = np.zeros(seq_len, dtype=int)
    for pos in range(seq_len):
        max_imp = np.max(importance[:, pos])
        peak[pos] = int(np.argmax(importance[:, pos]))
        if max_imp > 1e-10:
            for l in range(n_layers):
                if importance[l, pos] >= max_imp * 0.5:
                    emergence[pos] = l
                    break

    return {
        "importance_matrix": importance,
        "emergence_layer": emergence,
        "peak_layer": peak,
    }
