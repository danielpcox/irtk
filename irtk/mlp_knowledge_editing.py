"""MLP knowledge editing.

Locate and modify factual knowledge stored in MLP layers. Implements
rank-one editing, fact localization, and side-effect analysis.

References:
    Meng et al. (2022) "Locating and Editing Factual Associations in GPT"
    Meng et al. (2023) "Mass-Editing Memory in a Transformer"
"""

import jax
import jax.numpy as jnp
import numpy as np


def locate_fact_in_mlps(model, tokens, metric_fn, pos=-1):
    """Locate which MLP layers store a factual association.

    Ablates each MLP layer and measures the effect on a metric to identify
    where a fact is stored.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        metric_fn: Function mapping logits to scalar metric.
        pos: Position to analyze.

    Returns:
        dict with:
            mlp_effects: [n_layers] metric change when each MLP is ablated
            decisive_layer: int, layer with largest effect
            top_layers: list of (layer, effect) sorted by importance
            fact_distributed: bool, True if fact spans multiple layers
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers

    # Baseline
    base_logits = np.array(model(tokens))
    base_metric = metric_fn(base_logits)

    effects = np.zeros(n_layers)
    for layer in range(n_layers):
        hook_key = f"blocks.{layer}.hook_mlp_out"
        state = HookState(hook_fns={hook_key: lambda x, n: jnp.zeros_like(x)}, cache={})
        abl_logits = np.array(model(tokens, hook_state=state))
        effects[layer] = abs(metric_fn(abl_logits) - base_metric)

    decisive = int(np.argmax(effects))
    top_layers = sorted(enumerate(effects), key=lambda x: -x[1])
    top_layers = [(int(l), float(e)) for l, e in top_layers if e > 0.001]

    # Distributed if top-2 layers have similar effects
    sorted_effects = np.sort(effects)[::-1]
    distributed = len(sorted_effects) >= 2 and sorted_effects[1] > 0.5 * sorted_effects[0]

    return {
        "mlp_effects": effects,
        "decisive_layer": decisive,
        "top_layers": top_layers,
        "fact_distributed": bool(distributed),
    }


def rank_one_mlp_edit(model, layer, key_vector, value_vector, scale=1.0):
    """Apply a rank-one edit to an MLP layer's weights.

    Implements the core of ROME: modifies W_out by adding a rank-one update
    to associate a new key-value pair.

    Args:
        model: HookedTransformer model.
        layer: Layer to edit.
        key_vector: [d_mlp] key direction in MLP hidden space.
        value_vector: [d_model] desired output direction.
        scale: Scale factor for the edit.

    Returns:
        dict with:
            edited_model: model with modified MLP weights
            edit_norm: float, norm of the weight update
            edit_direction: [d_model] normalized edit direction
            key_norm: float, norm of the key vector
    """
    import equinox as eqx

    W_out = np.array(model.blocks[layer].mlp.W_out)  # [d_mlp, d_model]
    key_vec = np.array(key_vector)
    val_vec = np.array(value_vector)

    # Normalize key
    key_norm = float(np.linalg.norm(key_vec))
    if key_norm < 1e-10:
        return {
            "edited_model": model,
            "edit_norm": 0.0,
            "edit_direction": val_vec,
            "key_norm": 0.0,
        }

    key_normalized = key_vec / key_norm

    # Rank-one update: W_out += scale * key @ value^T
    update = scale * np.outer(key_normalized, val_vec)
    new_W_out = jnp.array(W_out + update)

    edited = eqx.tree_at(lambda m: m.blocks[layer].mlp.W_out, model, new_W_out)

    edit_norm = float(np.linalg.norm(update, ord='fro'))
    val_norm = float(np.linalg.norm(val_vec))
    edit_dir = val_vec / val_norm if val_norm > 1e-10 else val_vec

    return {
        "edited_model": edited,
        "edit_norm": edit_norm,
        "edit_direction": edit_dir,
        "key_norm": key_norm,
    }


def verify_edit_effect(model, edited_model, tokens_list, metric_fn):
    """Verify that an edit achieves the desired effect.

    Compares metric values between original and edited models on multiple inputs.

    Args:
        model: Original model.
        edited_model: Edited model.
        tokens_list: List of input token arrays to test.
        metric_fn: Function mapping logits to scalar metric.

    Returns:
        dict with:
            original_metrics: list of metric values on original model
            edited_metrics: list of metric values on edited model
            metric_changes: list of differences
            mean_change: float
            success_rate: float (fraction with positive change)
    """
    orig_metrics = []
    edit_metrics = []
    changes = []

    for tokens in tokens_list:
        orig_logits = np.array(model(tokens))
        edit_logits = np.array(edited_model(tokens))

        orig_m = metric_fn(orig_logits)
        edit_m = metric_fn(edit_logits)

        orig_metrics.append(float(orig_m))
        edit_metrics.append(float(edit_m))
        changes.append(float(edit_m - orig_m))

    success = sum(1 for c in changes if c > 0) / max(len(changes), 1)

    return {
        "original_metrics": orig_metrics,
        "edited_metrics": edit_metrics,
        "metric_changes": changes,
        "mean_change": float(np.mean(changes)),
        "success_rate": success,
    }


def edit_side_effects(model, edited_model, test_tokens_list, metric_fns):
    """Measure unintended side effects of an edit.

    Tests the edited model on unrelated inputs to check for collateral damage.

    Args:
        model: Original model.
        edited_model: Edited model.
        test_tokens_list: List of unrelated input token arrays.
        metric_fns: Dict of metric_name -> metric_fn for various behaviors.

    Returns:
        dict with:
            metric_drifts: dict of metric_name -> mean absolute change
            max_drift: float
            max_drift_metric: str
            affected_metrics: list of metrics with drift > 0.1
    """
    drifts = {}

    for name, fn in metric_fns.items():
        changes = []
        for tokens in test_tokens_list:
            orig = fn(np.array(model(tokens)))
            edited = fn(np.array(edited_model(tokens)))
            changes.append(abs(float(edited) - float(orig)))
        drifts[name] = float(np.mean(changes))

    max_name = max(drifts, key=drifts.get) if drifts else ""
    max_val = drifts[max_name] if max_name else 0.0
    affected = [name for name, val in drifts.items() if val > 0.1]

    return {
        "metric_drifts": drifts,
        "max_drift": max_val,
        "max_drift_metric": max_name,
        "affected_metrics": affected,
    }


def mlp_fact_extraction(model, tokens, layer, pos=-1, top_k=10):
    """Extract what factual information an MLP layer contributes.

    Projects MLP output through the unembedding to see what tokens it promotes.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        layer: MLP layer to analyze.
        pos: Position.
        top_k: Number of top tokens.

    Returns:
        dict with:
            mlp_output_norm: float
            promoted_tokens: list of (token_id, logit_value)
            suppressed_tokens: list of (token_id, logit_value)
            output_entropy: float (entropy of MLP's logit distribution)
    """
    from irtk.hook_points import HookState

    cache_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=cache_state)
    cache = cache_state.cache

    W_U = np.array(model.unembed.W_U)

    mlp = cache.get(f"blocks.{layer}.hook_mlp_out")
    if mlp is None:
        return {
            "mlp_output_norm": 0.0,
            "promoted_tokens": [],
            "suppressed_tokens": [],
            "output_entropy": 0.0,
        }

    mlp_out = np.array(mlp[pos])
    out_norm = float(np.linalg.norm(mlp_out))

    logits = mlp_out @ W_U
    promoted_idx = np.argsort(-logits)[:top_k]
    suppressed_idx = np.argsort(logits)[:top_k]

    promoted = [(int(i), float(logits[i])) for i in promoted_idx]
    suppressed = [(int(i), float(logits[i])) for i in suppressed_idx]

    # Entropy of softmax of MLP logits
    probs = np.exp(logits - logits.max())
    probs = probs / probs.sum()
    entropy = -float(np.sum(probs * np.log(probs + 1e-10)))

    return {
        "mlp_output_norm": out_norm,
        "promoted_tokens": promoted,
        "suppressed_tokens": suppressed,
        "output_entropy": entropy,
    }
