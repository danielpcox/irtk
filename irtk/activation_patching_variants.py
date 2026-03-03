"""Advanced activation patching methods.

Extends basic activation patching with denoising, noising, mean ablation,
resample ablation, and causal mediation analysis. Provides fine-grained
tools for causal analysis of model components.

References:
    Meng et al. (2022) "Locating and Editing Factual Associations in GPT"
    Vig et al. (2020) "Investigating Gender Bias in Language Models Using Causal Mediation Analysis"
    Chan et al. (2022) "Causal Scrubbing: A Method for Rigorously Testing Interpretability Hypotheses"
"""

import jax
import jax.numpy as jnp
import numpy as np


def denoising_patching(model, corrupted_tokens, clean_tokens, metric_fn):
    """Denoise patching: run on corrupted input, restore each component from clean.

    For each component, replaces its activation in the corrupted run with
    the clean version. Components that restore the metric are important
    for the computation.

    Args:
        model: HookedTransformer model.
        corrupted_tokens: Corrupted input token IDs [seq_len].
        clean_tokens: Clean input token IDs [seq_len].
        metric_fn: Function from logits -> scalar.

    Returns:
        dict with:
            attn_effects: array [n_layers, n_heads] of denoising effects
            mlp_effects: array [n_layers] of denoising effects
            baseline_metric: float, metric on corrupted input
            clean_metric: float, metric on clean input
            recovery_fractions: dict mapping component -> fraction of gap recovered
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    # Clean run to get clean activations
    clean_state = HookState(hook_fns={}, cache={})
    clean_logits = model(clean_tokens, hook_state=clean_state)
    clean_metric = metric_fn(clean_logits)
    clean_cache = clean_state.cache

    # Corrupted baseline
    corrupt_logits = model(corrupted_tokens)
    baseline = metric_fn(corrupt_logits)

    gap = clean_metric - baseline
    attn_effects = np.zeros((n_layers, n_heads))
    mlp_effects = np.zeros(n_layers)

    for layer in range(n_layers):
        # Denoise attention heads
        hook_z_key = f"blocks.{layer}.attn.hook_z"
        clean_z = clean_cache.get(hook_z_key)
        if clean_z is not None:
            for h in range(n_heads):
                def make_denoise_fn(head_idx, clean_val):
                    def fn(x, name):
                        return x.at[:, head_idx, :].set(clean_val[:, head_idx, :])
                    return fn

                state = HookState(
                    hook_fns={hook_z_key: make_denoise_fn(h, clean_z)},
                    cache={},
                )
                patched_logits = model(corrupted_tokens, hook_state=state)
                attn_effects[layer, h] = metric_fn(patched_logits) - baseline

        # Denoise MLP
        mlp_key = f"blocks.{layer}.hook_mlp_out"
        clean_mlp = clean_cache.get(mlp_key)
        if clean_mlp is not None:
            def make_mlp_denoise(clean_val):
                def fn(x, name):
                    return clean_val
                return fn

            state = HookState(
                hook_fns={mlp_key: make_mlp_denoise(clean_mlp)},
                cache={},
            )
            patched_logits = model(corrupted_tokens, hook_state=state)
            mlp_effects[layer] = metric_fn(patched_logits) - baseline

    # Recovery fractions
    recovery = {}
    if abs(gap) > 1e-10:
        for l in range(n_layers):
            for h in range(n_heads):
                recovery[("attn", l, h)] = float(attn_effects[l, h] / gap)
            recovery[("mlp", l)] = float(mlp_effects[l] / gap)

    return {
        "attn_effects": attn_effects,
        "mlp_effects": mlp_effects,
        "baseline_metric": float(baseline),
        "clean_metric": float(clean_metric),
        "recovery_fractions": recovery,
    }


def noising_patching(model, clean_tokens, corrupted_tokens, metric_fn):
    """Noising patching: run on clean input, corrupt each component.

    For each component, replaces its activation in the clean run with
    the corrupted version. Components that damage the metric are important.

    Args:
        model: HookedTransformer model.
        clean_tokens: Clean input token IDs [seq_len].
        corrupted_tokens: Corrupted input token IDs [seq_len].
        metric_fn: Function from logits -> scalar.

    Returns:
        dict with:
            attn_effects: array [n_layers, n_heads] of noising effects
            mlp_effects: array [n_layers] of noising effects
            baseline_metric: float, metric on clean input
            corrupted_metric: float, metric on corrupted input
            damage_fractions: dict mapping component -> fraction of gap caused
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    # Corrupted run to get corrupted activations
    corrupt_state = HookState(hook_fns={}, cache={})
    corrupt_logits = model(corrupted_tokens, hook_state=corrupt_state)
    corrupted_metric = metric_fn(corrupt_logits)
    corrupt_cache = corrupt_state.cache

    # Clean baseline
    clean_logits = model(clean_tokens)
    baseline = metric_fn(clean_logits)

    gap = baseline - corrupted_metric
    attn_effects = np.zeros((n_layers, n_heads))
    mlp_effects = np.zeros(n_layers)

    for layer in range(n_layers):
        hook_z_key = f"blocks.{layer}.attn.hook_z"
        corrupt_z = corrupt_cache.get(hook_z_key)
        if corrupt_z is not None:
            for h in range(n_heads):
                def make_noise_fn(head_idx, corrupt_val):
                    def fn(x, name):
                        return x.at[:, head_idx, :].set(corrupt_val[:, head_idx, :])
                    return fn

                state = HookState(
                    hook_fns={hook_z_key: make_noise_fn(h, corrupt_z)},
                    cache={},
                )
                patched_logits = model(clean_tokens, hook_state=state)
                attn_effects[layer, h] = baseline - metric_fn(patched_logits)

        mlp_key = f"blocks.{layer}.hook_mlp_out"
        corrupt_mlp = corrupt_cache.get(mlp_key)
        if corrupt_mlp is not None:
            def make_mlp_noise(corrupt_val):
                def fn(x, name):
                    return corrupt_val
                return fn

            state = HookState(
                hook_fns={mlp_key: make_mlp_noise(corrupt_mlp)},
                cache={},
            )
            patched_logits = model(clean_tokens, hook_state=state)
            mlp_effects[layer] = baseline - metric_fn(patched_logits)

    damage = {}
    if abs(gap) > 1e-10:
        for l in range(n_layers):
            for h in range(n_heads):
                damage[("attn", l, h)] = float(attn_effects[l, h] / gap)
            damage[("mlp", l)] = float(mlp_effects[l] / gap)

    return {
        "attn_effects": attn_effects,
        "mlp_effects": mlp_effects,
        "baseline_metric": float(baseline),
        "corrupted_metric": float(corrupted_metric),
        "damage_fractions": damage,
    }


def mean_ablation(model, tokens, metric_fn, n_samples=5, seed=42):
    """Mean ablation: replace each component's output with its mean activation.

    Computes the mean activation of each component across random inputs,
    then ablates by replacing with that mean.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        metric_fn: Function from logits -> scalar.
        n_samples: Number of random inputs for computing means.
        seed: Random seed.

    Returns:
        dict with:
            attn_effects: array [n_layers, n_heads] of ablation effects
            mlp_effects: array [n_layers] of ablation effects
            baseline_metric: float
            most_critical_component: tuple identifying largest effect
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    seq_len = len(tokens)

    # Baseline
    clean_logits = model(tokens)
    baseline = metric_fn(clean_logits)

    # Compute mean activations from random inputs
    rng = np.random.RandomState(seed)
    mean_z = {}
    mean_mlp = {}

    for _ in range(n_samples):
        rand_tokens = jnp.array(rng.randint(0, model.cfg.d_vocab, size=seq_len))
        state = HookState(hook_fns={}, cache={})
        model(rand_tokens, hook_state=state)

        for layer in range(n_layers):
            z_key = f"blocks.{layer}.attn.hook_z"
            if z_key in state.cache:
                if z_key not in mean_z:
                    mean_z[z_key] = state.cache[z_key] / n_samples
                else:
                    mean_z[z_key] = mean_z[z_key] + state.cache[z_key] / n_samples

            m_key = f"blocks.{layer}.hook_mlp_out"
            if m_key in state.cache:
                if m_key not in mean_mlp:
                    mean_mlp[m_key] = state.cache[m_key] / n_samples
                else:
                    mean_mlp[m_key] = mean_mlp[m_key] + state.cache[m_key] / n_samples

    attn_effects = np.zeros((n_layers, n_heads))
    mlp_effects = np.zeros(n_layers)

    for layer in range(n_layers):
        z_key = f"blocks.{layer}.attn.hook_z"
        if z_key in mean_z:
            for h in range(n_heads):
                def make_mean_fn(head_idx, mean_val):
                    def fn(x, name):
                        return x.at[:, head_idx, :].set(mean_val[:, head_idx, :])
                    return fn

                state = HookState(
                    hook_fns={z_key: make_mean_fn(h, mean_z[z_key])},
                    cache={},
                )
                patched_logits = model(tokens, hook_state=state)
                attn_effects[layer, h] = baseline - metric_fn(patched_logits)

        m_key = f"blocks.{layer}.hook_mlp_out"
        if m_key in mean_mlp:
            def make_mean_mlp(mean_val):
                def fn(x, name):
                    return mean_val
                return fn

            state = HookState(
                hook_fns={m_key: make_mean_mlp(mean_mlp[m_key])},
                cache={},
            )
            patched_logits = model(tokens, hook_state=state)
            mlp_effects[layer] = baseline - metric_fn(patched_logits)

    max_attn = float(np.max(np.abs(attn_effects)))
    max_mlp = float(np.max(np.abs(mlp_effects)))
    if max_attn >= max_mlp:
        idx = np.unravel_index(np.argmax(np.abs(attn_effects)), attn_effects.shape)
        most_critical = ("attn", int(idx[0]), int(idx[1]))
    else:
        most_critical = ("mlp", int(np.argmax(np.abs(mlp_effects))))

    return {
        "attn_effects": attn_effects,
        "mlp_effects": mlp_effects,
        "baseline_metric": float(baseline),
        "most_critical_component": most_critical,
    }


def resample_ablation(model, tokens, metric_fn, n_resamples=5, seed=42):
    """Resample ablation: replace each component's output with resampled activations.

    Instead of using a single corrupted input, averages the ablation effect
    over multiple random resamplings. More robust than single-corruption patching.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        metric_fn: Function from logits -> scalar.
        n_resamples: Number of random resamples.
        seed: Random seed.

    Returns:
        dict with:
            attn_effects: array [n_layers, n_heads] of average resample effects
            mlp_effects: array [n_layers] of average resample effects
            attn_std: array [n_layers, n_heads] of effect standard deviations
            mlp_std: array [n_layers] of effect standard deviations
            baseline_metric: float
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    seq_len = len(tokens)

    clean_logits = model(tokens)
    baseline = metric_fn(clean_logits)

    rng = np.random.RandomState(seed)

    all_attn = []
    all_mlp = []

    for _ in range(n_resamples):
        rand_tokens = jnp.array(rng.randint(0, model.cfg.d_vocab, size=seq_len))
        rand_state = HookState(hook_fns={}, cache={})
        model(rand_tokens, hook_state=rand_state)

        attn_eff = np.zeros((n_layers, n_heads))
        mlp_eff = np.zeros(n_layers)

        for layer in range(n_layers):
            z_key = f"blocks.{layer}.attn.hook_z"
            rand_z = rand_state.cache.get(z_key)
            if rand_z is not None:
                for h in range(n_heads):
                    def make_fn(head_idx, rand_val):
                        def fn(x, name):
                            return x.at[:, head_idx, :].set(rand_val[:, head_idx, :])
                        return fn

                    state = HookState(
                        hook_fns={z_key: make_fn(h, rand_z)},
                        cache={},
                    )
                    patched_logits = model(tokens, hook_state=state)
                    attn_eff[layer, h] = baseline - metric_fn(patched_logits)

            m_key = f"blocks.{layer}.hook_mlp_out"
            rand_mlp = rand_state.cache.get(m_key)
            if rand_mlp is not None:
                def make_mlp_fn(rand_val):
                    def fn(x, name):
                        return rand_val
                    return fn

                state = HookState(
                    hook_fns={m_key: make_mlp_fn(rand_mlp)},
                    cache={},
                )
                patched_logits = model(tokens, hook_state=state)
                mlp_eff[layer] = baseline - metric_fn(patched_logits)

        all_attn.append(attn_eff)
        all_mlp.append(mlp_eff)

    all_attn = np.stack(all_attn)
    all_mlp = np.stack(all_mlp)

    return {
        "attn_effects": np.mean(all_attn, axis=0),
        "mlp_effects": np.mean(all_mlp, axis=0),
        "attn_std": np.std(all_attn, axis=0),
        "mlp_std": np.std(all_mlp, axis=0),
        "baseline_metric": float(baseline),
    }


def causal_mediation_analysis(model, tokens, corrupted_tokens, metric_fn, mediator_layer=0):
    """Causal mediation analysis for a specific layer.

    Decomposes the total effect of corruption into:
    - Direct effect: bypassing the mediator layer
    - Indirect effect: through the mediator layer
    - Total effect: combined

    Args:
        model: HookedTransformer model.
        tokens: Clean input [seq_len].
        corrupted_tokens: Corrupted input [seq_len].
        metric_fn: Function from logits -> scalar.
        mediator_layer: Layer to analyze as mediator.

    Returns:
        dict with:
            total_effect: float, total effect of corruption
            indirect_effect: float, effect mediated through the layer
            direct_effect: float, effect not through the layer
            mediation_fraction: float, fraction of total mediated
            attn_indirect: float, attention's mediated effect
            mlp_indirect: float, MLP's mediated effect
    """
    from irtk.hook_points import HookState

    # Clean and corrupted baselines
    clean_logits = model(tokens)
    clean_metric = metric_fn(clean_logits)

    corrupt_logits = model(corrupted_tokens)
    corrupt_metric = metric_fn(corrupt_logits)

    total_effect = clean_metric - corrupt_metric

    # Get clean activations at mediator layer
    clean_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=clean_state)
    clean_cache = clean_state.cache

    # Indirect effect: run corrupted but restore mediator layer from clean
    resid_pre_key = f"blocks.{mediator_layer}.hook_resid_pre"
    resid_post_key = f"blocks.{mediator_layer}.hook_resid_post"

    # Restore both attn and mlp at mediator layer
    hooks = {}
    attn_key = f"blocks.{mediator_layer}.hook_attn_out"
    mlp_key = f"blocks.{mediator_layer}.hook_mlp_out"

    clean_attn = clean_cache.get(attn_key)
    clean_mlp = clean_cache.get(mlp_key)

    if clean_attn is not None and clean_mlp is not None:
        def make_restore(clean_val):
            def fn(x, name):
                return clean_val
            return fn

        # Indirect: restore both attn and mlp
        hooks_both = {
            attn_key: make_restore(clean_attn),
            mlp_key: make_restore(clean_mlp),
        }
        state = HookState(hook_fns=hooks_both, cache={})
        patched_logits = model(corrupted_tokens, hook_state=state)
        indirect_metric = metric_fn(patched_logits)
        indirect_effect = indirect_metric - corrupt_metric

        # Attn-only indirect
        state_attn = HookState(hook_fns={attn_key: make_restore(clean_attn)}, cache={})
        attn_logits = model(corrupted_tokens, hook_state=state_attn)
        attn_indirect = metric_fn(attn_logits) - corrupt_metric

        # MLP-only indirect
        state_mlp = HookState(hook_fns={mlp_key: make_restore(clean_mlp)}, cache={})
        mlp_logits = model(corrupted_tokens, hook_state=state_mlp)
        mlp_indirect = metric_fn(mlp_logits) - corrupt_metric
    else:
        indirect_effect = 0.0
        attn_indirect = 0.0
        mlp_indirect = 0.0

    direct_effect = total_effect - indirect_effect
    mediation_fraction = indirect_effect / (abs(total_effect) + 1e-10)

    return {
        "total_effect": float(total_effect),
        "indirect_effect": float(indirect_effect),
        "direct_effect": float(direct_effect),
        "mediation_fraction": float(mediation_fraction),
        "attn_indirect": float(attn_indirect),
        "mlp_indirect": float(mlp_indirect),
    }
