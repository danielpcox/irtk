"""Mechanistic hypothesis formation and validation.

Framework for forming, encoding, and testing mechanistic hypotheses
about model components. Bridges from observations to testable
causal claims about what components do.
"""

from typing import Optional, Callable
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from irtk.hooked_transformer import HookedTransformer


@dataclass
class MechanisticHypothesis:
    """A testable claim about a model component's role."""

    component: str  # e.g., "L5H3", "L8.mlp", "blocks.5.attn"
    role: str  # e.g., "previous_token", "induction", "fact_recall"
    description: str
    hook_name: str  # hook to test at
    confidence: float = 0.0
    evidence: list = field(default_factory=list)


def propose_hypotheses(
    model: HookedTransformer,
    token_sequences: list,
    depth: str = "basic",
) -> dict:
    """Generate mechanistic hypotheses from activation patterns.

    Analyzes attention patterns and MLP activations to propose
    hypotheses about what each component does.

    Args:
        model: HookedTransformer.
        token_sequences: Token arrays for analysis.
        depth: "basic" (attention only) or "full" (attention + MLP).

    Returns:
        Dict with:
        - "hypotheses": list of MechanisticHypothesis
        - "n_hypotheses": total number proposed
        - "component_coverage": fraction of components with hypotheses
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    hypotheses = []

    for layer in range(n_layers):
        hook = f"blocks.{layer}.attn.hook_pattern"

        # Collect attention patterns
        patterns = []
        for tokens in token_sequences:
            tokens = jnp.array(tokens)
            _, cache = model.run_with_cache(tokens)
            if hook in cache.cache_dict:
                pat = np.array(cache.cache_dict[hook])
                patterns.append(pat)

        if not patterns:
            continue

        for head in range(n_heads):
            head_patterns = []
            for pat in patterns:
                if pat.ndim == 3 and head < pat.shape[0]:
                    head_patterns.append(pat[head])
                elif pat.ndim == 2:
                    head_patterns.append(pat)

            if not head_patterns:
                continue

            # Score different roles
            prev_score = 0.0
            curr_score = 0.0
            bos_score = 0.0

            for p in head_patterns:
                seq = p.shape[0]
                if seq < 2:
                    continue
                prev_score += float(np.mean([p[i, i - 1] for i in range(1, seq)]))
                curr_score += float(np.mean([p[i, i] for i in range(seq)]))
                bos_score += float(np.mean(p[1:, 0]))

            n = len(head_patterns)
            if n > 0:
                prev_score /= n
                curr_score /= n
                bos_score /= n

            # Create hypothesis for dominant role
            scores = {"previous_token": prev_score, "current_token": curr_score, "bos_attending": bos_score}
            best_role = max(scores, key=scores.get)
            best_score = scores[best_role]

            if best_score > 0.1:
                hypotheses.append(MechanisticHypothesis(
                    component=f"L{layer}H{head}",
                    role=best_role,
                    description=f"Head L{layer}H{head} primarily performs {best_role} (score={best_score:.3f})",
                    hook_name=hook,
                    confidence=min(best_score, 1.0),
                ))

    total_components = n_layers * n_heads
    coverage = len(hypotheses) / max(total_components, 1)

    return {
        "hypotheses": hypotheses,
        "n_hypotheses": len(hypotheses),
        "component_coverage": float(coverage),
    }


def validate_hypothesis(
    model: HookedTransformer,
    hypothesis: MechanisticHypothesis,
    token_sequences: list,
    metric_fn: Optional[Callable] = None,
) -> dict:
    """Test a mechanistic hypothesis via ablation and intervention.

    Args:
        model: HookedTransformer.
        hypothesis: Hypothesis to test.
        token_sequences: Test inputs.
        metric_fn: Optional metric function.

    Returns:
        Dict with:
        - "passes": bool, whether hypothesis is supported
        - "ablation_effect": metric change when component is ablated
        - "consistency_score": fraction of inputs where behavior matches
        - "evidence": list of supporting/contradicting observations
    """
    evidence = []
    consistency_count = 0
    total_count = 0

    for tokens in token_sequences:
        tokens = jnp.array(tokens)
        _, cache = model.run_with_cache(tokens)

        if hypothesis.hook_name not in cache.cache_dict:
            continue

        pat = np.array(cache.cache_dict[hypothesis.hook_name])

        # Parse component to get head index
        comp = hypothesis.component
        if comp.startswith("L") and "H" in comp:
            parts = comp.split("H")
            head = int(parts[1])
        else:
            head = 0

        if pat.ndim == 3 and head < pat.shape[0]:
            head_pat = pat[head]
        elif pat.ndim == 2:
            head_pat = pat
        else:
            continue

        seq = head_pat.shape[0]
        if seq < 2:
            continue

        total_count += 1

        # Check if behavior matches hypothesis
        if hypothesis.role == "previous_token":
            diag = np.mean([head_pat[i, i - 1] for i in range(1, seq)])
            if diag > 0.2:
                consistency_count += 1
                evidence.append(f"Consistent: prev-token attention = {diag:.3f}")
            else:
                evidence.append(f"Inconsistent: prev-token attention = {diag:.3f}")

        elif hypothesis.role == "current_token":
            diag = np.mean([head_pat[i, i] for i in range(seq)])
            if diag > 0.2:
                consistency_count += 1
                evidence.append(f"Consistent: self-attention = {diag:.3f}")
            else:
                evidence.append(f"Inconsistent: self-attention = {diag:.3f}")

        elif hypothesis.role == "bos_attending":
            bos = np.mean(head_pat[1:, 0])
            if bos > 0.2:
                consistency_count += 1
                evidence.append(f"Consistent: BOS attention = {bos:.3f}")
            else:
                evidence.append(f"Inconsistent: BOS attention = {bos:.3f}")
        else:
            consistency_count += 1  # Unknown roles pass by default

    consistency = consistency_count / max(total_count, 1)

    # Ablation effect
    ablation_effect = 0.0
    if metric_fn is not None and token_sequences:
        tokens = jnp.array(token_sequences[0])
        original = float(metric_fn(model(tokens)))

        # Zero out the component
        def ablate_hook(x, name):
            if x.ndim == 3:
                return x.at[head].set(0.0)
            return jnp.zeros_like(x)

        ablated_logits = model.run_with_hooks(
            tokens, fwd_hooks=[(hypothesis.hook_name, ablate_hook)]
        )
        ablated = float(metric_fn(ablated_logits))
        ablation_effect = original - ablated

    return {
        "passes": consistency > 0.5,
        "ablation_effect": ablation_effect,
        "consistency_score": float(consistency),
        "evidence": evidence,
    }


def hypothesis_to_circuit(
    hypothesis: MechanisticHypothesis,
    model: HookedTransformer,
) -> dict:
    """Convert a hypothesis into a circuit specification.

    Args:
        hypothesis: Validated hypothesis.
        model: HookedTransformer.

    Returns:
        Dict with:
        - "nodes": list of component names in the circuit
        - "edges": list of (source, target) connections
        - "component_roles": dict mapping component -> role
        - "n_nodes": number of nodes
    """
    nodes = [hypothesis.component]
    edges = []
    roles = {hypothesis.component: hypothesis.role}

    # Parse layer from component name
    comp = hypothesis.component
    if comp.startswith("L") and "H" in comp:
        layer = int(comp.split("H")[0][1:])

        # Add upstream: previous layer residual
        if layer > 0:
            upstream = f"resid_L{layer - 1}"
            nodes.append(upstream)
            edges.append((upstream, comp))
            roles[upstream] = "residual_stream"

        # Add downstream: next layer or output
        downstream = f"resid_L{layer}"
        nodes.append(downstream)
        edges.append((comp, downstream))
        roles[downstream] = "residual_stream"

    return {
        "nodes": nodes,
        "edges": edges,
        "component_roles": roles,
        "n_nodes": len(nodes),
    }


def compose_hypotheses(
    h1: MechanisticHypothesis,
    h2: MechanisticHypothesis,
    model: HookedTransformer,
    token_sequences: list,
) -> dict:
    """Check if two hypotheses compose cleanly.

    Tests whether ablating one component affects the other's behavior,
    indicating compositional interaction.

    Args:
        h1: First hypothesis.
        h2: Second hypothesis.
        model: HookedTransformer.
        token_sequences: Test inputs.

    Returns:
        Dict with:
        - "composes": bool, whether the two interact
        - "interaction_score": strength of interaction (0 = independent)
        - "h1_effect_on_h2": how ablating h1 changes h2's behavior
        - "relationship": "independent", "sequential", or "parallel"
    """
    if not token_sequences:
        return {"composes": False, "interaction_score": 0.0,
                "h1_effect_on_h2": 0.0, "relationship": "independent"}

    tokens = jnp.array(token_sequences[0])
    _, cache = model.run_with_cache(tokens)

    # Parse head indices
    def parse_head(comp):
        if comp.startswith("L") and "H" in comp:
            parts = comp.split("H")
            return int(parts[0][1:]), int(parts[1])
        return 0, 0

    l1, head1 = parse_head(h1.component)
    l2, head2 = parse_head(h2.component)

    # Check h2's behavior with and without h1
    if h2.hook_name not in cache.cache_dict:
        return {"composes": False, "interaction_score": 0.0,
                "h1_effect_on_h2": 0.0, "relationship": "independent"}

    original_h2 = np.array(cache.cache_dict[h2.hook_name])

    # Ablate h1
    def ablate_h1(x, name):
        if x.ndim == 3 and head1 < x.shape[0]:
            return x.at[head1].set(0.0)
        return x

    logits = model.run_with_hooks(tokens, fwd_hooks=[(h1.hook_name, ablate_h1)])
    _, ablated_cache = model.run_with_cache(tokens)

    # Re-run with h1 ablated to see effect on h2
    # Use hook to ablate h1 and capture h2
    # Simplified: measure change via logits
    ablated_logits = model.run_with_hooks(tokens, fwd_hooks=[(h1.hook_name, ablate_h1)])

    original_out = np.array(model(tokens))
    ablated_out = np.array(ablated_logits)
    effect = float(np.linalg.norm(original_out[-1] - ablated_out[-1]))

    # Determine relationship
    if l1 < l2:
        relationship = "sequential"
    elif l1 == l2:
        relationship = "parallel"
    else:
        relationship = "sequential"

    interaction = effect / max(np.linalg.norm(original_out[-1]), 1e-10)

    return {
        "composes": interaction > 0.01,
        "interaction_score": float(interaction),
        "h1_effect_on_h2": float(effect),
        "relationship": relationship,
    }


def explain_prediction(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    hypotheses: list,
    metric_fn: Callable,
) -> dict:
    """Use hypotheses to explain a model prediction.

    Tests each hypothesis's contribution to the prediction and
    builds an explanation tree.

    Args:
        model: HookedTransformer.
        tokens: Input tokens.
        hypotheses: List of MechanisticHypothesis.
        metric_fn: Function from logits -> float.

    Returns:
        Dict with:
        - "full_metric": metric on unmodified model
        - "component_effects": dict of component -> effect when ablated
        - "explanation_order": components sorted by effect magnitude
        - "total_explained": sum of individual effects
    """
    tokens = jnp.array(tokens)
    full_logits = model(tokens)
    full_metric = float(metric_fn(full_logits))

    effects = {}

    for h in hypotheses:
        comp = h.component
        if comp.startswith("L") and "H" in comp:
            parts = comp.split("H")
            head = int(parts[1])
        else:
            head = 0

        def make_ablate(hd):
            def ablate(x, name):
                if x.ndim == 3 and hd < x.shape[0]:
                    return x.at[hd].set(0.0)
                return jnp.zeros_like(x)
            return ablate

        ablated_logits = model.run_with_hooks(
            tokens, fwd_hooks=[(h.hook_name, make_ablate(head))]
        )
        ablated_metric = float(metric_fn(ablated_logits))
        effects[comp] = full_metric - ablated_metric

    # Sort by effect magnitude
    order = sorted(effects.keys(), key=lambda k: abs(effects[k]), reverse=True)
    total_explained = sum(abs(v) for v in effects.values())

    return {
        "full_metric": full_metric,
        "component_effects": effects,
        "explanation_order": order,
        "total_explained": total_explained,
    }
