"""Multi-model comparison and behavioral alignment.

Compare mechanistic solutions across models, measure behavioral
alignment at different depths, and test interpretability transfer.
"""

from typing import Optional, Callable

import jax
import jax.numpy as jnp
import numpy as np

from irtk.hooked_transformer import HookedTransformer


def mechanical_correspondence(
    model_a: HookedTransformer,
    model_b: HookedTransformer,
    token_sequences: list,
    layer_a: int = 0,
    layer_b: int = 0,
) -> dict:
    """Find which components solve similar subproblems across models.

    Compares activation patterns between two models to identify
    analogous heads and MLP layers.

    Args:
        model_a: First model.
        model_b: Second model.
        token_sequences: Shared test inputs.
        layer_a: Layer to analyze in model A.
        layer_b: Layer to analyze in model B.

    Returns:
        Dict with:
        - "head_correspondence": [n_heads_a, n_heads_b] similarity matrix
        - "best_matches": list of (head_a, head_b, score)
        - "residual_similarity": cosine similarity of residual streams
        - "mean_correspondence": average best-match score
    """
    n_heads_a = model_a.cfg.n_heads
    n_heads_b = model_b.cfg.n_heads

    # Collect attention patterns
    patterns_a = {h: [] for h in range(n_heads_a)}
    patterns_b = {h: [] for h in range(n_heads_b)}

    resid_sims = []

    for tokens in token_sequences:
        tokens = jnp.array(tokens)

        _, cache_a = model_a.run_with_cache(tokens)
        _, cache_b = model_b.run_with_cache(tokens)

        hook_a = f"blocks.{layer_a}.attn.hook_pattern"
        hook_b = f"blocks.{layer_b}.attn.hook_pattern"

        if hook_a in cache_a.cache_dict:
            pat_a = np.array(cache_a.cache_dict[hook_a])
            for h in range(min(n_heads_a, pat_a.shape[0] if pat_a.ndim == 3 else 1)):
                p = pat_a[h] if pat_a.ndim == 3 else pat_a
                patterns_a[h].append(p.flatten())

        if hook_b in cache_b.cache_dict:
            pat_b = np.array(cache_b.cache_dict[hook_b])
            for h in range(min(n_heads_b, pat_b.shape[0] if pat_b.ndim == 3 else 1)):
                p = pat_b[h] if pat_b.ndim == 3 else pat_b
                patterns_b[h].append(p.flatten())

        # Residual stream similarity
        resid_hook_a = f"blocks.{layer_a}.hook_resid_post"
        resid_hook_b = f"blocks.{layer_b}.hook_resid_post"
        if resid_hook_a in cache_a.cache_dict and resid_hook_b in cache_b.cache_dict:
            ra = np.array(cache_a.cache_dict[resid_hook_a]).flatten()
            rb = np.array(cache_b.cache_dict[resid_hook_b]).flatten()
            # Pad shorter to match
            min_len = min(len(ra), len(rb))
            ra, rb = ra[:min_len], rb[:min_len]
            na, nb = np.linalg.norm(ra), np.linalg.norm(rb)
            if na > 1e-10 and nb > 1e-10:
                resid_sims.append(float(np.dot(ra, rb) / (na * nb)))

    # Compute head correspondence matrix
    corr = np.zeros((n_heads_a, n_heads_b))
    for ha in range(n_heads_a):
        for hb in range(n_heads_b):
            if patterns_a[ha] and patterns_b[hb]:
                sims = []
                for pa, pb in zip(patterns_a[ha], patterns_b[hb]):
                    min_len = min(len(pa), len(pb))
                    pa_t, pb_t = pa[:min_len], pb[:min_len]
                    na, nb = np.linalg.norm(pa_t), np.linalg.norm(pb_t)
                    if na > 1e-10 and nb > 1e-10:
                        sims.append(float(np.dot(pa_t, pb_t) / (na * nb)))
                corr[ha, hb] = float(np.mean(sims)) if sims else 0.0

    # Best matches
    best_matches = []
    for ha in range(n_heads_a):
        best_hb = int(np.argmax(corr[ha]))
        best_matches.append((ha, best_hb, float(corr[ha, best_hb])))
    best_matches.sort(key=lambda x: x[2], reverse=True)

    mean_corr = float(np.mean([s for _, _, s in best_matches])) if best_matches else 0.0
    resid_sim = float(np.mean(resid_sims)) if resid_sims else 0.0

    return {
        "head_correspondence": corr,
        "best_matches": best_matches,
        "residual_similarity": resid_sim,
        "mean_correspondence": mean_corr,
    }


def solution_diversity(
    models: list,
    token_sequences: list,
    metric_fn: Callable,
) -> dict:
    """Measure how much mechanistic solutions vary across models.

    Compares predictions and internal representations across models
    to quantify solution diversity.

    Args:
        models: List of HookedTransformer models.
        token_sequences: Shared test inputs.
        metric_fn: Function from logits -> float.

    Returns:
        Dict with:
        - "metric_values": [n_models, n_inputs] metric per model per input
        - "metric_variance": variance of metrics across models
        - "logit_agreement": mean pairwise agreement on top-1 prediction
        - "diversity_score": 1 - mean agreement (0 = identical, 1 = max diverse)
    """
    n_models = len(models)
    n_inputs = len(token_sequences)

    metric_vals = np.zeros((n_models, n_inputs))
    top_preds = np.zeros((n_models, n_inputs), dtype=int)

    for mi, model in enumerate(models):
        for ii, tokens in enumerate(token_sequences):
            tokens = jnp.array(tokens)
            logits = np.array(model(tokens))
            metric_vals[mi, ii] = float(metric_fn(jnp.array(logits)))
            top_preds[mi, ii] = int(np.argmax(logits[-1]))

    # Agreement: fraction of inputs where models agree on top-1
    agreements = []
    for i in range(n_models):
        for j in range(i + 1, n_models):
            agree = float(np.mean(top_preds[i] == top_preds[j]))
            agreements.append(agree)

    mean_agreement = float(np.mean(agreements)) if agreements else 1.0
    metric_var = float(np.mean(np.var(metric_vals, axis=0)))

    return {
        "metric_values": metric_vals,
        "metric_variance": metric_var,
        "logit_agreement": mean_agreement,
        "diversity_score": 1.0 - mean_agreement,
    }


def behavioral_alignment_spectrum(
    model_a: HookedTransformer,
    model_b: HookedTransformer,
    token_sequences: list,
) -> dict:
    """Quantify behavioral similarity at different model depths.

    Measures how aligned two models are from input through each layer
    to the output.

    Args:
        model_a: First model.
        model_b: Second model.
        token_sequences: Shared test inputs.

    Returns:
        Dict with:
        - "layer_similarities": [min(n_layers)] cosine similarity per layer
        - "output_similarity": cosine similarity of final logits
        - "divergence_layer": first layer where similarity drops below 0.5
        - "alignment_auc": area under the similarity curve
    """
    n_layers = min(model_a.cfg.n_layers, model_b.cfg.n_layers)

    layer_sims = np.zeros(n_layers)
    output_sims = []

    for tokens in token_sequences:
        tokens = jnp.array(tokens)
        logits_a = np.array(model_a(tokens))
        logits_b = np.array(model_b(tokens))

        # Output similarity
        la, lb = logits_a[-1], logits_b[-1]
        min_d = min(len(la), len(lb))
        la, lb = la[:min_d], lb[:min_d]
        na, nb = np.linalg.norm(la), np.linalg.norm(lb)
        if na > 1e-10 and nb > 1e-10:
            output_sims.append(float(np.dot(la, lb) / (na * nb)))

        _, cache_a = model_a.run_with_cache(tokens)
        _, cache_b = model_b.run_with_cache(tokens)

        for layer in range(n_layers):
            hook = f"blocks.{layer}.hook_resid_post"
            if hook in cache_a.cache_dict and hook in cache_b.cache_dict:
                ra = np.array(cache_a.cache_dict[hook]).flatten()
                rb = np.array(cache_b.cache_dict[hook]).flatten()
                min_d = min(len(ra), len(rb))
                ra, rb = ra[:min_d], rb[:min_d]
                na, nb = np.linalg.norm(ra), np.linalg.norm(rb)
                if na > 1e-10 and nb > 1e-10:
                    layer_sims[layer] += float(np.dot(ra, rb) / (na * nb))

    n_inputs = max(len(token_sequences), 1)
    layer_sims /= n_inputs

    # Divergence layer
    divergence = n_layers
    for i in range(n_layers):
        if layer_sims[i] < 0.5:
            divergence = i
            break

    auc = float(np.sum(layer_sims))  # simple sum as AUC proxy
    output_sim = float(np.mean(output_sims)) if output_sims else 0.0

    return {
        "layer_similarities": layer_sims,
        "output_similarity": output_sim,
        "divergence_layer": divergence,
        "alignment_auc": auc,
    }


def interpretability_transfer(
    source_model: HookedTransformer,
    target_model: HookedTransformer,
    token_sequences: list,
    hook_name: str,
    metric_fn: Callable,
) -> dict:
    """Test whether interventions discovered in one model work in another.

    Applies the source model's activations as interventions in the
    target model to measure transferability.

    Args:
        source_model: Model where intervention was discovered.
        target_model: Model to test transfer on.
        token_sequences: Test inputs.
        hook_name: Hook point for the intervention.
        metric_fn: Function from logits -> float.

    Returns:
        Dict with:
        - "source_metrics": [n] metric on source model
        - "target_baseline": [n] target metric without intervention
        - "target_transferred": [n] target metric with source activations
        - "transfer_success": fraction of inputs where transfer improves metric
        - "mean_transfer_effect": average effect of transfer
    """
    source_metrics = []
    target_baselines = []
    target_transferred = []

    for tokens in token_sequences:
        tokens = jnp.array(tokens)

        # Source metric
        source_logits = source_model(tokens)
        source_metrics.append(float(metric_fn(source_logits)))

        # Target baseline
        target_logits = target_model(tokens)
        target_baselines.append(float(metric_fn(target_logits)))

        # Get source activations
        _, source_cache = source_model.run_with_cache(tokens)
        if hook_name not in source_cache.cache_dict:
            target_transferred.append(float(metric_fn(target_logits)))
            continue

        source_act = source_cache.cache_dict[hook_name]

        # Patch into target
        def make_patch(act):
            def patch(x, name):
                # Match shapes
                if x.shape == act.shape:
                    return act
                # Project if different d_model
                min_d = min(x.shape[-1], act.shape[-1])
                result = x.copy()
                if x.ndim > 1:
                    result = result.at[:, :min_d].set(act[:x.shape[0], :min_d])
                return result
            return patch

        transferred_logits = target_model.run_with_hooks(
            tokens, fwd_hooks=[(hook_name, make_patch(source_act))]
        )
        target_transferred.append(float(metric_fn(transferred_logits)))

    source_arr = np.array(source_metrics)
    baseline_arr = np.array(target_baselines)
    transferred_arr = np.array(target_transferred)

    # Transfer success: did transferring improve the metric?
    improvements = transferred_arr > baseline_arr
    success = float(np.mean(improvements))
    mean_effect = float(np.mean(transferred_arr - baseline_arr))

    return {
        "source_metrics": source_arr,
        "target_baseline": baseline_arr,
        "target_transferred": transferred_arr,
        "transfer_success": success,
        "mean_transfer_effect": mean_effect,
    }


def emergence_comparison(
    models: list,
    token_sequences: list,
    hook_name: str,
) -> dict:
    """Compare when features emerge across models of different sizes.

    Measures feature activation strength at a hook point across
    models to identify scaling-dependent emergence.

    Args:
        models: List of models (e.g., different sizes/checkpoints).
        token_sequences: Shared test inputs.
        hook_name: Hook point to compare.

    Returns:
        Dict with:
        - "activation_norms": [n_models, n_inputs] activation norm per model
        - "sparsity_levels": [n_models] mean sparsity per model
        - "feature_overlap": [n_models, n_models] overlap of active features
        - "emergence_order": which model shows strongest activation
    """
    n_models = len(models)
    n_inputs = len(token_sequences)

    norms = np.zeros((n_models, n_inputs))
    sparsities = np.zeros(n_models)
    active_sets = [set() for _ in range(n_models)]

    for mi, model in enumerate(models):
        all_sparsity = []
        for ii, tokens in enumerate(token_sequences):
            tokens = jnp.array(tokens)
            _, cache = model.run_with_cache(tokens)
            if hook_name in cache.cache_dict:
                acts = np.array(cache.cache_dict[hook_name])
                flat = acts.flatten()
                norms[mi, ii] = float(np.linalg.norm(flat))
                sp = float(np.mean(np.abs(flat) < 1e-6))
                all_sparsity.append(sp)

                # Track which features are active (above median)
                if acts.ndim > 1:
                    last = acts.reshape(-1, acts.shape[-1])[-1]
                    active = set(int(i) for i in np.where(np.abs(last) > np.median(np.abs(last)))[0])
                    active_sets[mi] |= active

        sparsities[mi] = float(np.mean(all_sparsity)) if all_sparsity else 0.0

    # Feature overlap
    overlap = np.zeros((n_models, n_models))
    for i in range(n_models):
        for j in range(n_models):
            if active_sets[i] and active_sets[j]:
                intersection = len(active_sets[i] & active_sets[j])
                union = len(active_sets[i] | active_sets[j])
                overlap[i, j] = intersection / max(union, 1)

    # Emergence order: which model has strongest mean activation
    mean_norms = np.mean(norms, axis=1)
    emergence_order = int(np.argmax(mean_norms))

    return {
        "activation_norms": norms,
        "sparsity_levels": sparsities,
        "feature_overlap": overlap,
        "emergence_order": emergence_order,
    }
