"""Causal graph inference.

Infer minimal causal structure from interventions: edge strengths between
components, bottleneck detection, path reconstruction, and graph robustness.
"""

import jax
import jax.numpy as jnp


def component_causal_edges(model, tokens):
    """Estimate causal edge strengths between adjacent components.

    Measures how much zeroing each component changes the next component's output.

    Args:
        model: HookedTransformer
        tokens: input token IDs

    Returns:
        dict with edges and strengths.
    """
    clean_logits = model(tokens)
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers

    edges = []
    for l in range(n_layers):
        # Attn -> residual: ablate attention
        def make_attn_hook(target_l):
            def hook_fn(x, name):
                return jnp.zeros_like(x)
            return hook_fn

        hook_name = f'blocks.{l}.hook_attn_out'
        mod_logits = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, make_attn_hook(l))])
        attn_effect = float(jnp.mean(jnp.abs(mod_logits - clean_logits)))
        edges.append({
            'source': f'L{l}_attn',
            'target': f'L{l}_resid_mid',
            'strength': attn_effect,
            'layer': l,
            'type': 'attn_to_resid',
        })

        # MLP -> residual: ablate MLP
        hook_name = f'blocks.{l}.hook_mlp_out'
        mod_logits = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, make_attn_hook(l))])
        mlp_effect = float(jnp.mean(jnp.abs(mod_logits - clean_logits)))
        edges.append({
            'source': f'L{l}_mlp',
            'target': f'L{l}_resid_post',
            'strength': mlp_effect,
            'layer': l,
            'type': 'mlp_to_resid',
        })

    edges.sort(key=lambda e: -e['strength'])
    return {
        'edges': edges,
        'strongest_edge': edges[0] if edges else None,
        'n_edges': len(edges),
    }


def information_bottleneck_detection(model, tokens):
    """Identify layers that act as information bottlenecks.

    A bottleneck is a layer where effective dimensionality drops.

    Args:
        model: HookedTransformer
        tokens: input token IDs

    Returns:
        dict with per-layer bottleneck scores.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers

    per_layer = []
    for l in range(n_layers):
        resid = cache[f'blocks.{l}.hook_resid_post']
        # Effective dimensionality via SVD
        U, S, Vt = jnp.linalg.svd(resid, full_matrices=False)
        S_norm = S / (jnp.sum(S) + 1e-10)
        entropy = -float(jnp.sum(S_norm * jnp.log(S_norm + 1e-10)))
        eff_dim = float(jnp.exp(entropy))

        # Residual norm
        norm = float(jnp.mean(jnp.linalg.norm(resid, axis=-1)))

        per_layer.append({
            'layer': l,
            'effective_dim': eff_dim,
            'residual_norm': norm,
        })

    # Bottleneck: layer with lowest effective dim
    if per_layer:
        dims = [p['effective_dim'] for p in per_layer]
        bottleneck_layer = int(jnp.argmin(jnp.array(dims)))
    else:
        bottleneck_layer = 0

    return {
        'per_layer': per_layer,
        'bottleneck_layer': bottleneck_layer,
        'min_effective_dim': per_layer[bottleneck_layer]['effective_dim'] if per_layer else 0.0,
    }


def causal_path_strength(model, tokens, source_layer, target_layer):
    """Measure causal path strength from source layer to target layer.

    Args:
        model: HookedTransformer
        tokens: input token IDs
        source_layer: starting layer
        target_layer: ending layer (must be > source_layer)

    Returns:
        dict with path strength analysis.
    """
    _, cache = model.run_with_cache(tokens)
    clean_logits = model(tokens)

    # Measure how much freezing the source layer's output affects the target
    hook_name = f'blocks.{source_layer}.hook_resid_post'
    frozen_resid = cache[hook_name]

    def freeze_hook(x, name):
        return frozen_resid

    # Replace with mean (disrupts info flow)
    def mean_hook(x, name):
        return jnp.broadcast_to(jnp.mean(x, axis=0, keepdims=True), x.shape)

    mod_logits = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, mean_hook)])
    disruption = float(jnp.mean(jnp.abs(mod_logits - clean_logits)))

    # Also check intermediate layers
    per_intermediate = []
    for l in range(source_layer, target_layer + 1):
        resid_source = cache[f'blocks.{source_layer}.hook_resid_post']
        resid_l = cache[f'blocks.{l}.hook_resid_post']
        min_len = min(resid_source.shape[0], resid_l.shape[0])
        cos = float(jnp.mean(jnp.sum(resid_source[:min_len] * resid_l[:min_len], axis=-1) /
            (jnp.linalg.norm(resid_source[:min_len], axis=-1) * jnp.linalg.norm(resid_l[:min_len], axis=-1) + 1e-10)))
        per_intermediate.append({
            'layer': l,
            'cosine_to_source': cos,
        })

    return {
        'source_layer': source_layer,
        'target_layer': target_layer,
        'path_disruption': disruption,
        'per_intermediate': per_intermediate,
        'information_preserved': per_intermediate[-1]['cosine_to_source'] if per_intermediate else 0.0,
    }


def critical_component_ordering(model, tokens):
    """Order components by their causal criticality for final predictions.

    Args:
        model: HookedTransformer
        tokens: input token IDs

    Returns:
        dict with ordered components by criticality.
    """
    clean_logits = model(tokens)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    components = []

    for l in range(n_layers):
        # Test each head
        for h in range(n_heads):
            def make_hook(target_h):
                def hook_fn(x, name):
                    return x.at[:, target_h, :].set(0.0)
                return hook_fn

            hook_name = f'blocks.{l}.attn.hook_z'
            mod_logits = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, make_hook(h))])
            effect = float(jnp.mean(jnp.abs(mod_logits - clean_logits)))
            components.append({
                'name': f'L{l}H{h}',
                'layer': l,
                'type': 'head',
                'criticality': effect,
            })

        # Test MLP
        def zero_hook(x, name):
            return jnp.zeros_like(x)

        hook_name = f'blocks.{l}.hook_mlp_out'
        mod_logits = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, zero_hook)])
        effect = float(jnp.mean(jnp.abs(mod_logits - clean_logits)))
        components.append({
            'name': f'MLP{l}',
            'layer': l,
            'type': 'mlp',
            'criticality': effect,
        })

    components.sort(key=lambda c: -c['criticality'])

    # Cumulative importance
    total = sum(c['criticality'] for c in components)
    cumulative = 0.0
    for c in components:
        cumulative += c['criticality']
        c['cumulative_fraction'] = cumulative / max(total, 1e-10)

    return {
        'components': components,
        'most_critical': components[0] if components else None,
        'top_80_pct_count': sum(1 for c in components if c['cumulative_fraction'] <= 0.8) + 1,
    }


def graph_robustness(model, tokens, n_ablations=3):
    """Test robustness of causal structure by ablating multiple components.

    Args:
        model: HookedTransformer
        tokens: input token IDs
        n_ablations: number of components to ablate simultaneously

    Returns:
        dict with robustness analysis.
    """
    clean_logits = model(tokens)
    n_layers = model.cfg.n_layers

    # Get individual effects first
    individual_effects = []
    for l in range(n_layers):
        for component in ['hook_attn_out', 'hook_mlp_out']:
            hook_name = f'blocks.{l}.{component}'
            def zero_hook(x, name):
                return jnp.zeros_like(x)
            mod_logits = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, zero_hook)])
            effect = float(jnp.mean(jnp.abs(mod_logits - clean_logits)))
            individual_effects.append({
                'hook': hook_name,
                'individual_effect': effect,
            })

    individual_effects.sort(key=lambda e: -e['individual_effect'])

    # Ablate top-n together
    top_hooks = [e['hook'] for e in individual_effects[:n_ablations]]

    def zero_hook(x, name):
        return jnp.zeros_like(x)

    fwd_hooks = [(h, zero_hook) for h in top_hooks]
    joint_logits = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)
    joint_effect = float(jnp.mean(jnp.abs(joint_logits - clean_logits)))
    sum_individual = sum(e['individual_effect'] for e in individual_effects[:n_ablations])

    # Check prediction survival
    clean_preds = jnp.argmax(clean_logits, axis=-1)
    joint_preds = jnp.argmax(joint_logits, axis=-1)
    pred_survival = float(jnp.mean(clean_preds == joint_preds))

    return {
        'ablated_hooks': top_hooks,
        'joint_effect': joint_effect,
        'sum_individual_effects': sum_individual,
        'interaction_ratio': joint_effect / max(sum_individual, 1e-10),
        'prediction_survival_rate': pred_survival,
        'is_robust': pred_survival > 0.5,
    }
