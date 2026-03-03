"""Safety-relevant mechanistic interpretability tools.

Tools for safety-oriented analysis: refusal direction identification,
knowledge localization, deception detection signatures, alignment-relevant
circuit analysis, and safety feature monitoring.

These tools help understand model behavior from a safety perspective,
identifying internal representations related to refusal, compliance,
and potentially deceptive computation.

References:
    Arditi et al. (2024) "Refusal in Language Models Is Mediated by a Single Direction"
    Marks et al. (2024) "The Geometry of Truth"
    Li et al. (2024) "Inference-Time Intervention"
"""

import jax
import jax.numpy as jnp
import numpy as np


def refusal_direction_analysis(model, compliant_tokens_list, refused_tokens_list, layer=-1, pos=-1):
    """Identify the refusal direction in the residual stream.

    Compares activations on prompts that are complied with vs refused to
    find the direction that mediates refusal behavior.

    Args:
        model: HookedTransformer model.
        compliant_tokens_list: list of token arrays where model complies.
        refused_tokens_list: list of token arrays where model refuses.
        layer: Layer to analyze (-1 for last).
        pos: Position to analyze (-1 for last).

    Returns:
        dict with:
            refusal_direction: [d_model] the direction separating refusal from compliance
            direction_norm: float
            compliant_projections: list of float (projection onto refusal direction)
            refused_projections: list of float
            separation_score: float (how well the direction separates the two classes)
            layer_index: int
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    target_layer = layer if layer >= 0 else n_layers - 1

    # Collect activations for both classes
    compliant_acts = []
    for tokens in compliant_tokens_list:
        hook_key = f"blocks.{target_layer}.hook_resid_post"
        cache_state = HookState(hook_fns={}, cache={})
        model(tokens, hook_state=cache_state)
        act = cache_state.cache.get(hook_key)
        if act is not None:
            compliant_acts.append(np.array(act)[pos])

    refused_acts = []
    for tokens in refused_tokens_list:
        hook_key = f"blocks.{target_layer}.hook_resid_post"
        cache_state = HookState(hook_fns={}, cache={})
        model(tokens, hook_state=cache_state)
        act = cache_state.cache.get(hook_key)
        if act is not None:
            refused_acts.append(np.array(act)[pos])

    if not compliant_acts or not refused_acts:
        d_model = model.cfg.d_model
        return {
            "refusal_direction": np.zeros(d_model),
            "direction_norm": 0.0,
            "compliant_projections": [],
            "refused_projections": [],
            "separation_score": 0.0,
            "layer_index": target_layer,
        }

    compliant_mean = np.mean(compliant_acts, axis=0)
    refused_mean = np.mean(refused_acts, axis=0)

    # Refusal direction = difference of means
    direction = refused_mean - compliant_mean
    norm = float(np.linalg.norm(direction))
    if norm > 1e-10:
        direction_unit = direction / norm
    else:
        direction_unit = direction

    # Project all activations
    comp_proj = [float(np.dot(a, direction_unit)) for a in compliant_acts]
    ref_proj = [float(np.dot(a, direction_unit)) for a in refused_acts]

    # Separation score: difference of means / pooled std
    if comp_proj and ref_proj:
        mean_diff = abs(np.mean(ref_proj) - np.mean(comp_proj))
        pooled_std = np.sqrt((np.var(comp_proj) + np.var(ref_proj)) / 2 + 1e-10)
        separation = float(mean_diff / pooled_std)
    else:
        separation = 0.0

    return {
        "refusal_direction": direction_unit,
        "direction_norm": norm,
        "compliant_projections": comp_proj,
        "refused_projections": ref_proj,
        "separation_score": separation,
        "layer_index": target_layer,
    }


def knowledge_localization(model, tokens, target_positions, metric_fn):
    """Localize where specific knowledge is stored in the model.

    Uses activation patching to identify which layers and components
    are critical for producing a specific piece of knowledge.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        target_positions: list of int positions to analyze.
        metric_fn: Function mapping logits to scalar measuring knowledge expression.

    Returns:
        dict with:
            layer_importance: [n_layers] importance of each layer
            component_importance: dict of component -> importance
            critical_layers: list of int (most important layers)
            attn_vs_mlp: [n_layers] ratio of attention to MLP importance
            knowledge_concentrated: bool (whether knowledge is in few layers)
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers

    base_logits = np.array(model(tokens))
    base_metric = float(metric_fn(base_logits))

    layer_importance = np.zeros(n_layers)
    attn_importance = np.zeros(n_layers)
    mlp_importance = np.zeros(n_layers)
    component_importance = {}

    for layer in range(n_layers):
        # Ablate attention output
        hook_key = f"blocks.{layer}.hook_attn_out"
        def make_zero_hook():
            def hook_fn(x, name):
                return jnp.zeros_like(x)
            return hook_fn
        state = HookState(hook_fns={hook_key: make_zero_hook()}, cache={})
        abl_logits = np.array(model(tokens, hook_state=state))
        attn_effect = abs(base_metric - float(metric_fn(abl_logits)))
        attn_importance[layer] = attn_effect
        component_importance[f"attn_{layer}"] = attn_effect

        # Ablate MLP output
        hook_key = f"blocks.{layer}.hook_mlp_out"
        state = HookState(hook_fns={hook_key: make_zero_hook()}, cache={})
        abl_logits = np.array(model(tokens, hook_state=state))
        mlp_effect = abs(base_metric - float(metric_fn(abl_logits)))
        mlp_importance[layer] = mlp_effect
        component_importance[f"mlp_{layer}"] = mlp_effect

        layer_importance[layer] = attn_effect + mlp_effect

    # Critical layers (above mean + 1 std)
    threshold = np.mean(layer_importance) + np.std(layer_importance)
    critical = [int(l) for l in range(n_layers) if layer_importance[l] > threshold]

    # Attn vs MLP ratio
    attn_vs_mlp = np.zeros(n_layers)
    for l in range(n_layers):
        total = attn_importance[l] + mlp_importance[l]
        if total > 1e-10:
            attn_vs_mlp[l] = attn_importance[l] / total
        else:
            attn_vs_mlp[l] = 0.5

    # Is knowledge concentrated?
    if np.sum(layer_importance) > 1e-10:
        normalized = layer_importance / np.sum(layer_importance)
        entropy = -float(np.sum(normalized[normalized > 0] * np.log(normalized[normalized > 0] + 1e-10)))
        max_entropy = np.log(n_layers)
        concentrated = entropy < 0.5 * max_entropy
    else:
        concentrated = False

    return {
        "layer_importance": layer_importance,
        "component_importance": component_importance,
        "critical_layers": critical,
        "attn_vs_mlp": attn_vs_mlp,
        "knowledge_concentrated": concentrated,
    }


def deception_detection_signatures(model, tokens_honest, tokens_deceptive):
    """Detect signatures of deceptive vs honest computation.

    Compares internal activations between honest and potentially deceptive
    inputs to identify where the model's representations diverge.

    Args:
        model: HookedTransformer model.
        tokens_honest: Token array for honest input.
        tokens_deceptive: Token array for deceptive input.

    Returns:
        dict with:
            layer_divergence: [n_layers] divergence between honest/deceptive per layer
            cosine_similarity: [n_layers] cosine similarity per layer
            divergence_onset_layer: int (first layer where representations diverge significantly)
            max_divergence_layer: int
            attention_divergence: [n_layers, n_heads] per-head attention pattern divergence
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    # Run both inputs
    cache_honest = HookState(hook_fns={}, cache={})
    model(tokens_honest, hook_state=cache_honest)

    cache_deceptive = HookState(hook_fns={}, cache={})
    model(tokens_deceptive, hook_state=cache_deceptive)

    layer_div = np.zeros(n_layers)
    cosine_sim = np.zeros(n_layers)
    attn_div = np.zeros((n_layers, n_heads))

    for layer in range(n_layers):
        # Residual stream divergence
        key = f"blocks.{layer}.hook_resid_post"
        act_h = cache_honest.cache.get(key)
        act_d = cache_deceptive.cache.get(key)

        if act_h is not None and act_d is not None:
            h = np.array(act_h)[-1]  # last position
            d = np.array(act_d)[-1]
            layer_div[layer] = float(np.linalg.norm(h - d))
            norm_h = np.linalg.norm(h)
            norm_d = np.linalg.norm(d)
            if norm_h > 1e-10 and norm_d > 1e-10:
                cosine_sim[layer] = float(np.dot(h, d) / (norm_h * norm_d))

        # Attention pattern divergence
        pat_key = f"blocks.{layer}.attn.hook_pattern"
        pat_h = cache_honest.cache.get(pat_key)
        pat_d = cache_deceptive.cache.get(pat_key)

        if pat_h is not None and pat_d is not None:
            ph = np.array(pat_h)
            pd = np.array(pat_d)
            min_seq = min(ph.shape[1], pd.shape[1])
            for head in range(n_heads):
                diff = ph[head, :min_seq, :min_seq] - pd[head, :min_seq, :min_seq]
                attn_div[layer, head] = float(np.mean(np.abs(diff)))

    # Find onset of divergence
    mean_div = np.mean(layer_div)
    std_div = np.std(layer_div)
    onset = 0
    for l in range(n_layers):
        if layer_div[l] > mean_div + 0.5 * std_div:
            onset = l
            break

    max_div_layer = int(np.argmax(layer_div))

    return {
        "layer_divergence": layer_div,
        "cosine_similarity": cosine_sim,
        "divergence_onset_layer": onset,
        "max_divergence_layer": max_div_layer,
        "attention_divergence": attn_div,
    }


def alignment_circuit_analysis(model, tokens, behavior_metric_fn, safety_metric_fn):
    """Analyze circuits relevant to alignment properties.

    Identifies components that contribute to desired (aligned) behavior
    vs undesired behavior, and maps the overlap between safety-relevant
    and capability-relevant circuits.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        behavior_metric_fn: Function measuring desired behavior (e.g., helpfulness).
        safety_metric_fn: Function measuring safety property (e.g., harmlessness).

    Returns:
        dict with:
            behavior_importance: [n_layers, n_heads] importance for desired behavior
            safety_importance: [n_layers, n_heads] importance for safety
            overlap_score: float (correlation between behavior and safety importance)
            conflict_heads: list of (layer, head) where behavior and safety conflict
            synergy_heads: list of (layer, head) where both are high
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    base_logits = np.array(model(tokens))
    base_behavior = float(behavior_metric_fn(base_logits))
    base_safety = float(safety_metric_fn(base_logits))

    behavior_imp = np.zeros((n_layers, n_heads))
    safety_imp = np.zeros((n_layers, n_heads))

    for layer in range(n_layers):
        for head in range(n_heads):
            hook_key = f"blocks.{layer}.hook_z"
            h = head
            def make_hook(h_idx):
                def hook_fn(x, name):
                    return x.at[:, h_idx, :].set(0.0)
                return hook_fn
            state = HookState(hook_fns={hook_key: make_hook(h)}, cache={})
            abl_logits = np.array(model(tokens, hook_state=state))

            behavior_imp[layer, head] = base_behavior - float(behavior_metric_fn(abl_logits))
            safety_imp[layer, head] = base_safety - float(safety_metric_fn(abl_logits))

    # Overlap correlation
    b_flat = behavior_imp.flatten()
    s_flat = safety_imp.flatten()
    if np.std(b_flat) > 1e-10 and np.std(s_flat) > 1e-10:
        overlap = float(np.corrcoef(b_flat, s_flat)[0, 1])
    else:
        overlap = 0.0

    # Find conflict and synergy heads
    b_thresh = np.median(np.abs(b_flat))
    s_thresh = np.median(np.abs(s_flat))

    conflict_heads = []
    synergy_heads = []
    for l in range(n_layers):
        for h in range(n_heads):
            b_val = behavior_imp[l, h]
            s_val = safety_imp[l, h]
            # Conflict: helps behavior but hurts safety (or vice versa)
            if (b_val > b_thresh and s_val < -s_thresh) or (b_val < -b_thresh and s_val > s_thresh):
                conflict_heads.append((l, h))
            # Synergy: helps both
            if b_val > b_thresh and s_val > s_thresh:
                synergy_heads.append((l, h))

    return {
        "behavior_importance": behavior_imp,
        "safety_importance": safety_imp,
        "overlap_score": overlap,
        "conflict_heads": conflict_heads,
        "synergy_heads": synergy_heads,
    }


def safety_feature_monitoring(model, tokens, reference_directions=None, layer=-1, pos=-1):
    """Monitor safety-relevant features in the residual stream.

    Projects activations onto known safety-relevant directions (if provided)
    or identifies directions with high variance across positions.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        reference_directions: Optional dict of name -> [d_model] direction vectors.
        layer: Layer to monitor (-1 for last).
        pos: Position to analyze (-1 for last).

    Returns:
        dict with:
            activation_norm: float
            projections: dict of direction_name -> float (if reference_directions provided)
            top_active_dimensions: list of (dim_index, value) top active dimensions
            position_variance: [d_model] variance of activation across positions
            anomaly_score: float (how unusual this activation is)
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    target_layer = layer if layer >= 0 else n_layers - 1
    d_model = model.cfg.d_model

    hook_key = f"blocks.{target_layer}.hook_resid_post"
    cache_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=cache_state)
    act = cache_state.cache.get(hook_key)

    if act is None:
        return {
            "activation_norm": 0.0,
            "projections": {},
            "top_active_dimensions": [],
            "position_variance": np.zeros(d_model),
            "anomaly_score": 0.0,
        }

    act_arr = np.array(act)  # [seq_len, d_model]
    target_act = act_arr[pos]  # [d_model]

    activation_norm = float(np.linalg.norm(target_act))

    # Project onto reference directions
    projections = {}
    if reference_directions is not None:
        for name, direction in reference_directions.items():
            d = np.array(direction)
            d_norm = np.linalg.norm(d)
            if d_norm > 1e-10:
                projections[name] = float(np.dot(target_act, d / d_norm))
            else:
                projections[name] = 0.0

    # Top active dimensions
    abs_act = np.abs(target_act)
    top_dims = np.argsort(-abs_act)[:10]
    top_active = [(int(d), float(target_act[d])) for d in top_dims]

    # Position variance
    pos_var = np.var(act_arr, axis=0)

    # Anomaly score: how far is target position from mean activation
    mean_act = np.mean(act_arr, axis=0)
    std_act = np.std(act_arr, axis=0) + 1e-10
    z_scores = np.abs((target_act - mean_act) / std_act)
    anomaly_score = float(np.mean(z_scores))

    return {
        "activation_norm": activation_norm,
        "projections": projections,
        "top_active_dimensions": top_active,
        "position_variance": pos_var,
        "anomaly_score": anomaly_score,
    }
