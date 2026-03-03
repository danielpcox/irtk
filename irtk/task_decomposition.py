"""Task decomposition analysis.

Analyze how models decompose tasks: subtask identification via ablation,
functional specialization mapping, and task-component alignment.

References:
    Wang et al. (2022) "Interpretability in the Wild"
    Conmy et al. (2023) "Towards Automated Circuit Discovery"
"""

import jax
import jax.numpy as jnp
import numpy as np


def subtask_identification(model, tokens, metric_fns):
    """Identify which components serve which subtasks.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        metric_fns: dict of subtask_name -> metric function.

    Returns:
        dict with:
            component_subtask_effects: dict of component -> {subtask: effect}
            primary_subtask: dict of component -> subtask with strongest effect
            subtask_components: dict of subtask -> list of important components
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    baselines = {name: fn(model(tokens)) for name, fn in metric_fns.items()}

    components = []
    for layer in range(n_layers):
        for h in range(n_heads):
            components.append((f"attn_L{layer}H{h}", layer, "attn", h))
        components.append((f"mlp_L{layer}", layer, "mlp", None))

    comp_effects = {}
    primary = {}
    subtask_comps = {name: [] for name in metric_fns}

    for name, layer, ctype, head_idx in components:
        if ctype == "attn":
            def make_zero_head(hi):
                def fn(x, hook_name):
                    return x.at[:, hi, :].set(0.0)
                return fn
            hooks = {f"blocks.{layer}.attn.hook_z": make_zero_head(head_idx)}
        else:
            hooks = {f"blocks.{layer}.hook_mlp_out": lambda x, hook_name: jnp.zeros_like(x)}

        state = HookState(hook_fns=hooks, cache={})
        logits = model(tokens, hook_state=state)

        effects = {}
        best_task = None
        best_effect = 0.0
        for task_name, metric_fn in metric_fns.items():
            effect = abs(baselines[task_name] - metric_fn(logits))
            effects[task_name] = float(effect)
            if effect > best_effect:
                best_effect = effect
                best_task = task_name

        comp_effects[name] = effects
        primary[name] = best_task
        if best_task and best_effect > 0.01:
            subtask_comps[best_task].append(name)

    return {
        "component_subtask_effects": comp_effects,
        "primary_subtask": primary,
        "subtask_components": subtask_comps,
    }


def functional_specialization(model, tokens, metric_fn):
    """Map functional specialization: how exclusively each component serves one role.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        metric_fn: Function from logits -> scalar.

    Returns:
        dict with:
            attn_effects: [n_layers, n_heads] ablation effect per head
            mlp_effects: [n_layers] ablation effect per MLP
            specialization_scores: dict of component -> float (0=generalist, 1=specialist)
            most_specialized: str
            most_general: str
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    baseline = metric_fn(model(tokens))

    attn_effects = np.zeros((n_layers, n_heads))
    mlp_effects = np.zeros(n_layers)

    for layer in range(n_layers):
        for h in range(n_heads):
            def make_zero_head(hi):
                def fn(x, name):
                    return x.at[:, hi, :].set(0.0)
                return fn
            state = HookState(hook_fns={f"blocks.{layer}.attn.hook_z": make_zero_head(h)}, cache={})
            logits = model(tokens, hook_state=state)
            attn_effects[layer, h] = abs(baseline - metric_fn(logits))

        state = HookState(hook_fns={f"blocks.{layer}.hook_mlp_out": lambda x, name: jnp.zeros_like(x)}, cache={})
        logits = model(tokens, hook_state=state)
        mlp_effects[layer] = abs(baseline - metric_fn(logits))

    # Specialization: how much of the total effect is in one component
    all_effects = list(attn_effects.flatten()) + list(mlp_effects)
    total = sum(all_effects) + 1e-10

    scores = {}
    best_name, best_score = "", 0.0
    worst_name, worst_score = "", 1.0

    for layer in range(n_layers):
        for h in range(n_heads):
            name = f"attn_L{layer}H{h}"
            s = attn_effects[layer, h] / total
            scores[name] = float(s)
            if s > best_score:
                best_score = s
                best_name = name
            if s < worst_score:
                worst_score = s
                worst_name = name

        name = f"mlp_L{layer}"
        s = mlp_effects[layer] / total
        scores[name] = float(s)
        if s > best_score:
            best_score = s
            best_name = name
        if s < worst_score:
            worst_score = s
            worst_name = name

    return {
        "attn_effects": attn_effects,
        "mlp_effects": mlp_effects,
        "specialization_scores": scores,
        "most_specialized": best_name,
        "most_general": worst_name,
    }


def task_component_alignment(model, tokens, metric_fns):
    """Measure alignment between components and multiple task metrics.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        metric_fns: dict of task_name -> metric function.

    Returns:
        dict with:
            alignment_matrix: dict of component -> dict of task -> effect
            task_overlap: dict of (task_a, task_b) -> overlap score
            component_selectivity: dict of component -> float (high = serves one task)
    """
    subtask = subtask_identification(model, tokens, metric_fns)
    effects = subtask["component_subtask_effects"]

    task_names = list(metric_fns.keys())
    comp_names = list(effects.keys())

    # Task overlap: how many components are shared
    overlap = {}
    for i, ta in enumerate(task_names):
        for j, tb in enumerate(task_names):
            if i < j:
                comps_a = set(subtask["subtask_components"].get(ta, []))
                comps_b = set(subtask["subtask_components"].get(tb, []))
                if len(comps_a | comps_b) > 0:
                    overlap[(ta, tb)] = len(comps_a & comps_b) / len(comps_a | comps_b)
                else:
                    overlap[(ta, tb)] = 0.0

    # Component selectivity
    selectivity = {}
    for comp in comp_names:
        task_effects = list(effects[comp].values())
        total = sum(task_effects) + 1e-10
        max_effect = max(task_effects) if task_effects else 0
        selectivity[comp] = float(max_effect / total)

    return {
        "alignment_matrix": effects,
        "task_overlap": overlap,
        "component_selectivity": selectivity,
    }


def component_cooperation_analysis(model, tokens, metric_fn):
    """Analyze how components cooperate: do they work together or independently?

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        metric_fn: Function from logits -> scalar.

    Returns:
        dict with:
            individual_effects: dict of component -> float
            pair_effects: dict of (comp_a, comp_b) -> float
            cooperation_scores: dict of (comp_a, comp_b) -> float (positive=cooperative)
            mean_cooperation: float
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    baseline = metric_fn(model(tokens))

    # Individual effects
    individual = {}
    for layer in range(n_layers):
        for comp_type, hook_key in [("attn", f"blocks.{layer}.hook_attn_out"),
                                     ("mlp", f"blocks.{layer}.hook_mlp_out")]:
            name = f"{comp_type}_L{layer}"
            state = HookState(hook_fns={hook_key: lambda x, n: jnp.zeros_like(x)}, cache={})
            logits = model(tokens, hook_state=state)
            individual[name] = abs(baseline - metric_fn(logits))

    # Pair effects (within and across layers)
    pair_effects = {}
    cooperation = {}
    names = list(individual.keys())

    for i in range(len(names)):
        for j in range(i + 1, min(i + 3, len(names))):  # limit pairs for speed
            na, nb = names[i], names[j]
            hooks = {}
            for n in [na, nb]:
                parts = n.split("_L")
                comp_type = parts[0]
                layer = int(parts[1])
                hook_key = f"blocks.{layer}.hook_{comp_type}_out"
                hooks[hook_key] = lambda x, name: jnp.zeros_like(x)

            state = HookState(hook_fns=hooks, cache={})
            logits = model(tokens, hook_state=state)
            joint = abs(baseline - metric_fn(logits))
            pair_effects[(na, nb)] = float(joint)
            # Cooperation = joint - individual_a - individual_b
            coop = joint - individual[na] - individual[nb]
            cooperation[(na, nb)] = float(coop)

    mean_coop = np.mean(list(cooperation.values())) if cooperation else 0.0

    return {
        "individual_effects": {k: float(v) for k, v in individual.items()},
        "pair_effects": pair_effects,
        "cooperation_scores": cooperation,
        "mean_cooperation": float(mean_coop),
    }


def task_difficulty_decomposition(model, tokens_easy, tokens_hard, metric_fn):
    """Compare component contributions between easy and hard inputs.

    Args:
        model: HookedTransformer model.
        tokens_easy: Easy input token IDs [seq_len].
        tokens_hard: Hard input token IDs [seq_len].
        metric_fn: Function from logits -> scalar.

    Returns:
        dict with:
            easy_effects: dict of component -> float
            hard_effects: dict of component -> float
            difficulty_sensitive: list of components more important for hard inputs
            difficulty_insensitive: list of components similar for both
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    baseline_easy = metric_fn(model(tokens_easy))
    baseline_hard = metric_fn(model(tokens_hard))

    easy_effects = {}
    hard_effects = {}

    for layer in range(n_layers):
        for comp_type, hook_key_tmpl in [("attn", "blocks.{}.hook_attn_out"),
                                          ("mlp", "blocks.{}.hook_mlp_out")]:
            name = f"{comp_type}_L{layer}"
            hook_key = hook_key_tmpl.format(layer)

            state = HookState(hook_fns={hook_key: lambda x, n: jnp.zeros_like(x)}, cache={})
            logits = model(tokens_easy, hook_state=state)
            easy_effects[name] = abs(baseline_easy - metric_fn(logits))

            state = HookState(hook_fns={hook_key: lambda x, n: jnp.zeros_like(x)}, cache={})
            logits = model(tokens_hard, hook_state=state)
            hard_effects[name] = abs(baseline_hard - metric_fn(logits))

    sensitive = []
    insensitive = []
    for name in easy_effects:
        ratio = hard_effects[name] / (easy_effects[name] + 1e-10)
        if ratio > 1.5:
            sensitive.append(name)
        elif 0.67 < ratio < 1.5:
            insensitive.append(name)

    return {
        "easy_effects": {k: float(v) for k, v in easy_effects.items()},
        "hard_effects": {k: float(v) for k, v in hard_effects.items()},
        "difficulty_sensitive": sensitive,
        "difficulty_insensitive": insensitive,
    }
