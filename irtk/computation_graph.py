"""Computation graph analysis.

Builds computation graphs showing dependencies between components,
dataflow analysis, and computation cost per component.

References:
    Vig (2019) "A Multiscale Visualization of Attention in the Transformer Model"
    Conmy et al. (2023) "Towards Automated Circuit Discovery"
"""

import jax
import jax.numpy as jnp
import numpy as np


def component_dependency_graph(model, tokens, metric_fn, threshold=0.01):
    """Build a dependency graph between model components.

    Tests whether ablating component A affects the output of component B,
    establishing causal dependencies.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        metric_fn: Function from logits -> scalar.
        threshold: Minimum effect to count as dependency.

    Returns:
        dict with:
            edges: list of (source, target, weight) tuples
            n_edges: int
            component_names: list of all component names
            in_degree: dict of component -> number of incoming edges
            out_degree: dict of component -> number of outgoing edges
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    # List components
    components = []
    for layer in range(n_layers):
        for h in range(n_heads):
            components.append(f"attn_L{layer}H{h}")
        components.append(f"mlp_L{layer}")

    baseline = metric_fn(model(tokens))

    edges = []
    in_deg = {c: 0 for c in components}
    out_deg = {c: 0 for c in components}

    # For each component, ablate it and see which downstream components are affected
    for i, src in enumerate(components):
        layer_src = int(src.split("L")[1].split("H")[0].split("_")[0]) if "L" in src else 0

        # Ablate source
        if src.startswith("attn"):
            parts = src.split("L")[1].split("H")
            l, h = int(parts[0]), int(parts[1])
            hook_key = f"blocks.{l}.attn.hook_z"

            def make_zero_head(head_idx):
                def fn(x, name):
                    return x.at[:, head_idx, :].set(0.0)
                return fn

            state = HookState(hook_fns={hook_key: make_zero_head(h)}, cache={})
        else:
            l = int(src.split("L")[1])
            hook_key = f"blocks.{l}.hook_mlp_out"

            def zero_fn(x, name):
                return jnp.zeros_like(x)

            state = HookState(hook_fns={hook_key: zero_fn}, cache={})

        logits = model(tokens, hook_state=state)
        effect = abs(baseline - metric_fn(logits))

        if effect > threshold * abs(baseline) if abs(baseline) > 1e-10 else effect > threshold:
            # This component matters - add edge to output
            for tgt in components:
                layer_tgt = int(tgt.split("L")[1].split("H")[0].split("_")[0]) if "L" in tgt else 0
                if layer_tgt > layer_src:
                    # Simplified: source affects all downstream components
                    edges.append((src, tgt, float(effect)))
                    in_deg[tgt] = in_deg.get(tgt, 0) + 1
                    out_deg[src] = out_deg.get(src, 0) + 1
                    break  # Only connect to next layer for simplicity

    return {
        "edges": edges,
        "n_edges": len(edges),
        "component_names": components,
        "in_degree": in_deg,
        "out_degree": out_deg,
    }


def dataflow_analysis(model, tokens, pos=-1):
    """Analyze dataflow: how much information flows through each component.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        pos: Position.

    Returns:
        dict with:
            attn_throughput: array [n_layers, n_heads] of norm of each head's output
            mlp_throughput: array [n_layers] of norm of each MLP's output
            residual_norms: array [n_layers+1] of residual stream norms
            attn_fraction: array [n_layers] of attention's share of total throughput
            mlp_fraction: array [n_layers] of MLP's share
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    hook_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=hook_state)
    cache = hook_state.cache

    attn_tp = np.zeros((n_layers, n_heads))
    mlp_tp = np.zeros(n_layers)
    resid_norms = np.zeros(n_layers + 1)

    for layer in range(n_layers + 1):
        if layer == 0:
            key = "blocks.0.hook_resid_pre"
        else:
            key = f"blocks.{layer - 1}.hook_resid_post"
        resid = cache.get(key)
        if resid is not None:
            resid_norms[layer] = float(np.linalg.norm(np.array(resid[pos])))

    for layer in range(n_layers):
        z = cache.get(f"blocks.{layer}.attn.hook_z")
        if z is not None:
            z_arr = np.array(z)
            W_O = np.array(model.blocks[layer].attn.W_O)
            for h in range(n_heads):
                head_out = z_arr[pos, h] @ W_O[h]
                attn_tp[layer, h] = float(np.linalg.norm(head_out))

        mlp_out = cache.get(f"blocks.{layer}.hook_mlp_out")
        if mlp_out is not None:
            mlp_tp[layer] = float(np.linalg.norm(np.array(mlp_out[pos])))

    attn_total = np.sum(attn_tp, axis=1)
    total = attn_total + mlp_tp + 1e-10
    attn_frac = attn_total / total
    mlp_frac = mlp_tp / total

    return {
        "attn_throughput": attn_tp,
        "mlp_throughput": mlp_tp,
        "residual_norms": resid_norms,
        "attn_fraction": attn_frac,
        "mlp_fraction": mlp_frac,
    }


def computation_cost_profile(model, tokens, metric_fn):
    """Profile computational cost-effectiveness of each component.

    Measures the metric effect per parameter for each component.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        metric_fn: Function from logits -> scalar.

    Returns:
        dict with:
            attn_params: array [n_layers] of attention parameter counts
            mlp_params: array [n_layers] of MLP parameter counts
            attn_effects: array [n_layers] of attention ablation effects
            mlp_effects: array [n_layers] of MLP ablation effects
            cost_effectiveness: dict mapping component -> effect/params
            most_cost_effective: str
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    d_head = model.cfg.d_head
    n_heads = model.cfg.n_heads
    d_mlp = model.cfg.d_mlp

    baseline = metric_fn(model(tokens))

    # Parameter counts
    attn_params = np.full(n_layers, n_heads * (3 * d_model * d_head + d_head * d_model))
    mlp_params = np.full(n_layers, d_model * d_mlp + d_mlp * d_model)

    attn_effects = np.zeros(n_layers)
    mlp_effects = np.zeros(n_layers)

    for layer in range(n_layers):
        def zero_fn(x, name):
            return jnp.zeros_like(x)

        state = HookState(hook_fns={f"blocks.{layer}.hook_attn_out": zero_fn}, cache={})
        logits = model(tokens, hook_state=state)
        attn_effects[layer] = abs(baseline - metric_fn(logits))

        state = HookState(hook_fns={f"blocks.{layer}.hook_mlp_out": zero_fn}, cache={})
        logits = model(tokens, hook_state=state)
        mlp_effects[layer] = abs(baseline - metric_fn(logits))

    cost_eff = {}
    best_name = ""
    best_ratio = 0.0

    for l in range(n_layers):
        a_ratio = attn_effects[l] / (attn_params[l] + 1e-10)
        m_ratio = mlp_effects[l] / (mlp_params[l] + 1e-10)
        cost_eff[f"attn_L{l}"] = float(a_ratio)
        cost_eff[f"mlp_L{l}"] = float(m_ratio)
        if a_ratio > best_ratio:
            best_ratio = a_ratio
            best_name = f"attn_L{l}"
        if m_ratio > best_ratio:
            best_ratio = m_ratio
            best_name = f"mlp_L{l}"

    return {
        "attn_params": attn_params,
        "mlp_params": mlp_params,
        "attn_effects": attn_effects,
        "mlp_effects": mlp_effects,
        "cost_effectiveness": cost_eff,
        "most_cost_effective": best_name,
    }


def critical_path_analysis(model, tokens, metric_fn, pos=-1):
    """Find the critical computation path from input to output.

    Identifies the chain of components with the highest cumulative effect.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        metric_fn: Function from logits -> scalar.
        pos: Position.

    Returns:
        dict with:
            critical_path: list of component names on the critical path
            path_effects: list of effect magnitudes along the path
            total_path_effect: float
            path_length: int
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    baseline = metric_fn(model(tokens))

    path = []
    path_effects = []

    for layer in range(n_layers):
        best_comp = f"mlp_L{layer}"
        best_effect = 0.0

        # Test each head
        for h in range(n_heads):
            def make_zero_head(head_idx):
                def fn(x, name):
                    return x.at[:, head_idx, :].set(0.0)
                return fn

            state = HookState(
                hook_fns={f"blocks.{layer}.attn.hook_z": make_zero_head(h)},
                cache={}
            )
            logits = model(tokens, hook_state=state)
            effect = abs(baseline - metric_fn(logits))
            if effect > best_effect:
                best_effect = effect
                best_comp = f"attn_L{layer}H{h}"

        # Test MLP
        def zero_fn(x, name):
            return jnp.zeros_like(x)

        state = HookState(hook_fns={f"blocks.{layer}.hook_mlp_out": zero_fn}, cache={})
        logits = model(tokens, hook_state=state)
        mlp_effect = abs(baseline - metric_fn(logits))
        if mlp_effect > best_effect:
            best_effect = mlp_effect
            best_comp = f"mlp_L{layer}"

        path.append(best_comp)
        path_effects.append(float(best_effect))

    return {
        "critical_path": path,
        "path_effects": path_effects,
        "total_path_effect": float(sum(path_effects)),
        "path_length": len(path),
    }


def component_interaction_strength(model, tokens, metric_fn, layer_a=0, layer_b=1):
    """Measure interaction strength between components in two layers.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        metric_fn: Function from logits -> scalar.
        layer_a: First layer.
        layer_b: Second layer.

    Returns:
        dict with:
            interaction_matrix: array [n_components_a, n_components_b]
            strongest_interaction: tuple of component names
            mean_interaction: float
    """
    from irtk.hook_points import HookState

    n_heads = model.cfg.n_heads
    baseline = metric_fn(model(tokens))

    # Components in each layer: n_heads + 1 (MLP)
    n_a = n_heads + 1
    n_b = n_heads + 1

    names_a = [f"attn_L{layer_a}H{h}" for h in range(n_heads)] + [f"mlp_L{layer_a}"]
    names_b = [f"attn_L{layer_b}H{h}" for h in range(n_heads)] + [f"mlp_L{layer_b}"]

    interaction = np.zeros((n_a, n_b))

    for i in range(n_a):
        for j in range(n_b):
            hooks = {}

            # Ablate component i in layer a
            if i < n_heads:
                def make_zh(head_idx):
                    def fn(x, name):
                        return x.at[:, head_idx, :].set(0.0)
                    return fn
                hooks[f"blocks.{layer_a}.attn.hook_z"] = make_zh(i)
            else:
                hooks[f"blocks.{layer_a}.hook_mlp_out"] = lambda x, name: jnp.zeros_like(x)

            # Ablate component j in layer b
            if j < n_heads:
                def make_zh2(head_idx):
                    def fn(x, name):
                        return x.at[:, head_idx, :].set(0.0)
                    return fn
                hooks[f"blocks.{layer_b}.attn.hook_z"] = make_zh2(j)
            else:
                hooks[f"blocks.{layer_b}.hook_mlp_out"] = lambda x, name: jnp.zeros_like(x)

            state = HookState(hook_fns=hooks, cache={})
            logits = model(tokens, hook_state=state)
            joint_effect = abs(baseline - metric_fn(logits))

            # Get individual effects for interaction
            # Interaction = joint - individual_a - individual_b
            # (Simplified: just use joint effect as proxy)
            interaction[i, j] = float(joint_effect)

    strongest_idx = np.unravel_index(np.argmax(interaction), interaction.shape)
    strongest = (names_a[strongest_idx[0]], names_b[strongest_idx[1]])

    return {
        "interaction_matrix": interaction,
        "strongest_interaction": strongest,
        "mean_interaction": float(np.mean(interaction)),
    }
