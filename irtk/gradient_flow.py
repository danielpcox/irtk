"""Gradient flow analysis through transformer layers.

Measures gradient norms, detects vanishing/exploding gradients, and
attributes gradient magnitude to specific components. Essential for
understanding training dynamics and feature importance.

References:
    Pascanu et al. (2013) "On the Difficulty of Training Recurrent Neural Networks"
    Zhang et al. (2019) "Improving Deep Transformer with Depth-Scaled Initialization"
"""

import jax
import jax.numpy as jnp
import numpy as np


def gradient_norm_by_layer(model, tokens, target_token=None, pos=-1):
    """Compute gradient norms at each residual stream position.

    Measures how strongly the loss signal propagates back through each layer.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        target_token: Token to compute gradient for (None = top predicted).
        pos: Position to analyze.

    Returns:
        dict with:
            layer_grad_norms: array [n_layers+1] of gradient norms at each residual position
            max_grad_layer: int, layer with largest gradient
            min_grad_layer: int, layer with smallest gradient
            gradient_ratio: float, ratio of first to last layer gradient
            vanishing: bool, True if gradient ratio > 100
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers

    # Forward pass to get target
    logits = model(tokens)
    if target_token is None:
        target_token = int(jnp.argmax(logits[pos]))

    # Compute gradients at each residual stream position
    grad_norms = np.zeros(n_layers + 1)

    for layer_idx in range(n_layers + 1):
        if layer_idx == 0:
            hook_name = "blocks.0.hook_resid_pre"
        else:
            hook_name = f"blocks.{layer_idx - 1}.hook_resid_post"

        def compute_grad(layer_i, hook_n):
            grad_val = [None]

            def capture_and_forward(x, name):
                grad_val[0] = x
                return x

            state = HookState(hook_fns={hook_n: capture_and_forward}, cache={})
            logits = model(tokens, hook_state=state)
            return logits[pos, target_token], grad_val[0]

        # Use finite differences as a simple gradient approximation
        state = HookState(hook_fns={}, cache={})
        model(tokens, hook_state=state)
        resid = state.cache.get(hook_name)
        if resid is not None:
            eps = 1e-4
            base_logit = float(logits[pos, target_token])
            # Estimate gradient norm via random direction
            rng = np.random.RandomState(42 + layer_idx)
            direction = rng.randn(*resid[pos].shape).astype(np.float32)
            direction = direction / (np.linalg.norm(direction) + 1e-10)
            direction = jnp.array(direction)

            def perturb_fn(delta, hook_n, p):
                def fn(x, name):
                    return x.at[p].add(delta)
                return fn

            perturbed_state = HookState(
                hook_fns={hook_name: perturb_fn(eps * direction, hook_name, pos)},
                cache={},
            )
            perturbed_logits = model(tokens, hook_state=perturbed_state)
            grad_approx = (float(perturbed_logits[pos, target_token]) - base_logit) / eps
            # Scale by sqrt(d_model) to get norm estimate
            grad_norms[layer_idx] = abs(grad_approx) * np.sqrt(resid.shape[-1])

    max_layer = int(np.argmax(grad_norms))
    min_layer = int(np.argmin(grad_norms[grad_norms > 0])) if np.any(grad_norms > 0) else 0
    first_nonzero = grad_norms[0] if grad_norms[0] > 1e-12 else 1e-12
    last_nonzero = grad_norms[-1] if grad_norms[-1] > 1e-12 else 1e-12

    return {
        "layer_grad_norms": grad_norms,
        "max_grad_layer": max_layer,
        "min_grad_layer": min_layer,
        "gradient_ratio": float(last_nonzero / first_nonzero),
        "vanishing": float(last_nonzero / first_nonzero) > 100,
    }


def component_gradient_attribution(model, tokens, target_token=None, pos=-1):
    """Attribute gradient magnitude to attention vs MLP at each layer.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        target_token: Token to compute gradient for.
        pos: Position to analyze.

    Returns:
        dict with:
            attn_grad_norms: array [n_layers] of attention gradient contribution norms
            mlp_grad_norms: array [n_layers] of MLP gradient contribution norms
            attn_fraction: array [n_layers] of attention fraction of total gradient
            dominant_component_per_layer: list of 'attn' or 'mlp' per layer
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers

    logits = model(tokens)
    if target_token is None:
        target_token = int(jnp.argmax(logits[pos]))

    base_logit = float(logits[pos, target_token])

    attn_norms = np.zeros(n_layers)
    mlp_norms = np.zeros(n_layers)

    for layer in range(n_layers):
        # Measure sensitivity to attention output
        attn_key = f"blocks.{layer}.hook_attn_out"
        mlp_key = f"blocks.{layer}.hook_mlp_out"

        for key, norms_arr in [(attn_key, attn_norms), (mlp_key, mlp_norms)]:
            state = HookState(hook_fns={}, cache={})
            model(tokens, hook_state=state)
            act = state.cache.get(key)
            if act is not None:
                eps = 1e-4
                rng = np.random.RandomState(42 + layer + hash(key) % 1000)
                direction = rng.randn(*act[pos].shape).astype(np.float32)
                direction = direction / (np.linalg.norm(direction) + 1e-10)
                direction_jnp = jnp.array(direction)

                def perturb(delta, hook_name, p):
                    def fn(x, name):
                        return x.at[p].add(delta)
                    return fn

                p_state = HookState(
                    hook_fns={key: perturb(eps * direction_jnp, key, pos)},
                    cache={},
                )
                p_logits = model(tokens, hook_state=p_state)
                grad_approx = abs(float(p_logits[pos, target_token]) - base_logit) / eps
                norms_arr[layer] = grad_approx * np.sqrt(act.shape[-1])

    total = attn_norms + mlp_norms + 1e-10
    attn_frac = attn_norms / total
    dominant = ["attn" if attn_norms[l] >= mlp_norms[l] else "mlp" for l in range(n_layers)]

    return {
        "attn_grad_norms": attn_norms,
        "mlp_grad_norms": mlp_norms,
        "attn_fraction": attn_frac,
        "dominant_component_per_layer": dominant,
    }


def gradient_saturation_analysis(model, tokens, pos=-1):
    """Detect gradient saturation in MLP activation functions.

    Measures how close MLP pre-activations are to saturation regions
    where gradients become very small.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        pos: Position to analyze.

    Returns:
        dict with:
            layer_saturation: array [n_layers] of mean saturation per layer
            max_saturation_layer: int, most saturated layer
            pre_activation_means: array [n_layers] of mean |pre-activation|
            fraction_saturated: array [n_layers] of fraction with |x| > 3
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers

    hook_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=hook_state)
    cache = hook_state.cache

    saturation = np.zeros(n_layers)
    pre_means = np.zeros(n_layers)
    frac_sat = np.zeros(n_layers)

    for layer in range(n_layers):
        pre_key = f"blocks.{layer}.mlp.hook_pre"
        pre_act = cache.get(pre_key)
        if pre_act is not None:
            vals = np.array(pre_act[pos])
            abs_vals = np.abs(vals)
            pre_means[layer] = float(np.mean(abs_vals))
            # Saturation: fraction of neurons with |x| > 3 (tanh/gelu saturation region)
            frac_sat[layer] = float(np.mean(abs_vals > 3.0))
            # Saturation score: mean of tanh(|x|) which approaches 1 for large values
            saturation[layer] = float(np.mean(np.tanh(abs_vals)))

    max_sat = int(np.argmax(saturation))

    return {
        "layer_saturation": saturation,
        "max_saturation_layer": max_sat,
        "pre_activation_means": pre_means,
        "fraction_saturated": frac_sat,
    }


def layernorm_gradient_effect(model, tokens, pos=-1):
    """Analyze how LayerNorm affects gradient flow at each layer.

    LayerNorm can both help (normalizing) and hurt (removing direction info)
    gradient flow. This measures the scale factors applied by LayerNorm.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        pos: Position to analyze.

    Returns:
        dict with:
            scale_factors: array [n_layers] of LayerNorm scale magnitudes
            input_norms: array [n_layers] of pre-LayerNorm residual norms
            output_norms: array [n_layers] of post-LayerNorm residual norms
            compression_ratio: array [n_layers] of output/input norm ratio
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers

    hook_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=hook_state)
    cache = hook_state.cache

    scale_factors = np.zeros(n_layers)
    input_norms = np.zeros(n_layers)
    output_norms = np.zeros(n_layers)

    for layer in range(n_layers):
        # Pre-LayerNorm residual
        resid_pre = cache.get(f"blocks.{layer}.hook_resid_pre")
        # Scale from LayerNorm (if captured)
        ln_scale = cache.get(f"blocks.{layer}.ln1.hook_scale")

        if resid_pre is not None:
            input_norms[layer] = float(jnp.linalg.norm(resid_pre[pos]))

        if ln_scale is not None:
            scale_factors[layer] = float(jnp.mean(jnp.abs(ln_scale[pos])))

        # Post-attention residual as a proxy for post-LN effect
        resid_mid = cache.get(f"blocks.{layer}.hook_resid_mid")
        if resid_mid is not None:
            output_norms[layer] = float(jnp.linalg.norm(resid_mid[pos]))
        elif resid_pre is not None:
            output_norms[layer] = input_norms[layer]  # fallback

    compression = np.where(input_norms > 1e-10, output_norms / input_norms, 1.0)

    return {
        "scale_factors": scale_factors,
        "input_norms": input_norms,
        "output_norms": output_norms,
        "compression_ratio": compression,
    }


def per_head_gradient_sensitivity(model, tokens, layer, target_token=None, pos=-1):
    """Measure gradient sensitivity of each attention head in a given layer.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        layer: Layer to analyze.
        target_token: Token to compute gradient for.
        pos: Position to analyze.

    Returns:
        dict with:
            head_sensitivities: array [n_heads] of gradient sensitivity per head
            most_sensitive_head: int
            least_sensitive_head: int
            sensitivity_ratio: float, max/min ratio
    """
    from irtk.hook_points import HookState

    n_heads = model.cfg.n_heads

    logits = model(tokens)
    if target_token is None:
        target_token = int(jnp.argmax(logits[pos]))
    base_logit = float(logits[pos, target_token])

    sensitivities = np.zeros(n_heads)
    hook_z_key = f"blocks.{layer}.attn.hook_z"

    state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=state)
    hook_z = state.cache.get(hook_z_key)

    if hook_z is not None:
        for h in range(n_heads):
            eps = 1e-4
            rng = np.random.RandomState(42 + h)
            d_head = hook_z.shape[-1]
            direction = rng.randn(d_head).astype(np.float32)
            direction = direction / (np.linalg.norm(direction) + 1e-10)
            direction_jnp = jnp.array(direction)

            def perturb_head(head_idx, delta):
                def fn(x, name):
                    return x.at[pos, head_idx, :].add(delta)
                return fn

            p_state = HookState(
                hook_fns={hook_z_key: perturb_head(h, eps * direction_jnp)},
                cache={},
            )
            p_logits = model(tokens, hook_state=p_state)
            grad_approx = abs(float(p_logits[pos, target_token]) - base_logit) / eps
            sensitivities[h] = grad_approx * np.sqrt(d_head)

    most = int(np.argmax(sensitivities))
    least = int(np.argmin(sensitivities))
    min_s = sensitivities[least] if sensitivities[least] > 1e-12 else 1e-12

    return {
        "head_sensitivities": sensitivities,
        "most_sensitive_head": most,
        "least_sensitive_head": least,
        "sensitivity_ratio": float(sensitivities[most] / min_s),
    }
