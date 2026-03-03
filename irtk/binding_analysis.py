"""Entity-attribute binding analysis.

Analyzes how transformer models bind attributes to entities -- for example,
how "red" gets bound to "ball" in "the red ball and the blue box." This is
a fundamental question in understanding how transformers represent structured
information.

Functions:
- entity_attribute_binding: Measure binding strength between positions
- binding_attention_pattern: Find attention heads mediating binding
- cross_position_binding_score: Pairwise binding scores across all positions
- binding_through_layers: Track binding strength across layers
- multi_entity_disambiguation: How model distinguishes multiple entities

References:
    - Feng & Steinhardt (2023) "How do Language Models Bind Entities in Context?"
    - Variengien & Winsor (2023) "Look Before You Leap" (binding in IOI)
"""

from typing import Optional, Callable

import jax
import jax.numpy as jnp
import numpy as np

from irtk.hooked_transformer import HookedTransformer


def entity_attribute_binding(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    entity_pos: int,
    attribute_pos: int,
    layer: Optional[int] = None,
) -> dict:
    """Measure binding strength between an entity and attribute position.

    Uses the cosine similarity of residual stream representations and
    attention flow from attribute to entity as proxies for binding strength.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        entity_pos: Position of the entity token.
        attribute_pos: Position of the attribute token.
        layer: If specified, measure at this layer. Otherwise, measure at all layers.

    Returns:
        Dict with:
            "binding_strength": float or array of cosine similarities
            "attention_flow": attention weight from entity_pos to attribute_pos
            "residual_similarity": cosine similarity of residual streams
            "layer_binding": per-layer binding scores (if layer is None)
    """
    _, cache = model.run_with_cache(tokens)

    n_layers = model.cfg.n_layers

    if layer is not None:
        layers_to_check = [layer]
    else:
        layers_to_check = list(range(n_layers))

    layer_binding = []
    attn_flow_total = 0.0

    for l in layers_to_check:
        # Residual stream similarity
        resid_key = f"blocks.{l}.hook_resid_post"
        if resid_key in cache.cache_dict:
            resid = cache.cache_dict[resid_key]  # [seq, d_model]
            entity_vec = resid[entity_pos]
            attr_vec = resid[attribute_pos]
            cos_sim = float(jnp.dot(entity_vec, attr_vec) / (
                jnp.linalg.norm(entity_vec) * jnp.linalg.norm(attr_vec) + 1e-10
            ))
        else:
            cos_sim = 0.0

        # Attention from entity to attribute
        pattern_key = f"blocks.{l}.attn.hook_pattern"
        if pattern_key in cache.cache_dict:
            pattern = cache.cache_dict[pattern_key]  # [seq, n_heads, seq]
            # Mean attention from entity to attribute across heads
            attn_score = float(jnp.mean(pattern[entity_pos, :, attribute_pos]))
            attn_flow_total += attn_score
        else:
            attn_score = 0.0

        layer_binding.append(cos_sim)

    layer_binding = np.array(layer_binding)
    mean_binding = float(np.mean(layer_binding)) if len(layer_binding) > 0 else 0.0

    # Overall residual similarity at final layer
    final_resid_key = f"blocks.{n_layers - 1}.hook_resid_post"
    if final_resid_key in cache.cache_dict:
        resid = cache.cache_dict[final_resid_key]
        entity_vec = resid[entity_pos]
        attr_vec = resid[attribute_pos]
        final_sim = float(jnp.dot(entity_vec, attr_vec) / (
            jnp.linalg.norm(entity_vec) * jnp.linalg.norm(attr_vec) + 1e-10
        ))
    else:
        final_sim = 0.0

    return {
        "binding_strength": mean_binding,
        "attention_flow": attn_flow_total / max(len(layers_to_check), 1),
        "residual_similarity": final_sim,
        "layer_binding": layer_binding,
    }


def binding_attention_pattern(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    entity_pos: int,
    attribute_pos: int,
) -> dict:
    """Find attention heads that mediate entity-attribute binding.

    Identifies which heads attend strongly from the entity position to
    the attribute position, suggesting they copy attribute information.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        entity_pos: Position of the entity token.
        attribute_pos: Position of the attribute token.

    Returns:
        Dict with:
            "head_scores": [n_layers, n_heads] attention from entity to attribute
            "top_binding_heads": list of (layer, head, score) top heads
            "max_score": highest attention score
    """
    _, cache = model.run_with_cache(tokens)

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    head_scores = np.zeros((n_layers, n_heads))

    for l in range(n_layers):
        pattern_key = f"blocks.{l}.attn.hook_pattern"
        if pattern_key in cache.cache_dict:
            pattern = cache.cache_dict[pattern_key]  # [seq, n_heads, seq]
            for h in range(n_heads):
                head_scores[l, h] = float(pattern[entity_pos, h, attribute_pos])

    # Top binding heads
    flat_scores = [(l, h, head_scores[l, h])
                   for l in range(n_layers) for h in range(n_heads)]
    flat_scores.sort(key=lambda x: x[2], reverse=True)
    top_heads = flat_scores[:min(5, len(flat_scores))]

    return {
        "head_scores": head_scores,
        "top_binding_heads": top_heads,
        "max_score": float(np.max(head_scores)),
    }


def cross_position_binding_score(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    layer: int = -1,
) -> dict:
    """Compute pairwise binding scores between all token positions.

    Uses cosine similarity of residual stream representations as a
    proxy for binding strength.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        layer: Which layer to analyze (-1 for final).

    Returns:
        Dict with:
            "binding_matrix": [seq_len, seq_len] pairwise cosine similarities
            "strongest_pair": (pos_i, pos_j) most strongly bound positions
            "mean_binding": mean binding score across all pairs
    """
    _, cache = model.run_with_cache(tokens)
    seq_len = len(tokens)

    actual_layer = layer if layer >= 0 else model.cfg.n_layers - 1
    resid_key = f"blocks.{actual_layer}.hook_resid_post"

    if resid_key not in cache.cache_dict:
        return {
            "binding_matrix": np.zeros((seq_len, seq_len)),
            "strongest_pair": (0, 0),
            "mean_binding": 0.0,
        }

    resid = cache.cache_dict[resid_key]  # [seq_len, d_model]
    # Normalize
    norms = jnp.linalg.norm(resid, axis=-1, keepdims=True)
    normalized = resid / (norms + 1e-10)
    # Pairwise cosine similarity
    binding_matrix = np.array(normalized @ normalized.T)

    # Find strongest off-diagonal pair
    mask = np.ones_like(binding_matrix) - np.eye(seq_len)
    masked = binding_matrix * mask
    flat_idx = np.argmax(np.abs(masked))
    i, j = np.unravel_index(flat_idx, binding_matrix.shape)

    # Mean off-diagonal
    mean_binding = float(np.sum(np.abs(masked))) / max(seq_len * (seq_len - 1), 1)

    return {
        "binding_matrix": binding_matrix,
        "strongest_pair": (int(i), int(j)),
        "mean_binding": mean_binding,
    }


def binding_through_layers(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    entity_pos: int,
    attribute_pos: int,
) -> dict:
    """Track how binding between entity and attribute evolves through layers.

    Measures cosine similarity at each layer's residual output to see
    when binding emerges or dissolves.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        entity_pos: Position of the entity token.
        attribute_pos: Position of the attribute token.

    Returns:
        Dict with:
            "layer_similarities": [n_layers] cosine similarity at each layer
            "peak_layer": layer with strongest binding
            "binding_emerges": first layer where similarity exceeds threshold
            "binding_trend": "increasing", "decreasing", or "non_monotonic"
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers

    sims = []
    for l in range(n_layers):
        resid_key = f"blocks.{l}.hook_resid_post"
        if resid_key in cache.cache_dict:
            resid = cache.cache_dict[resid_key]
            e_vec = resid[entity_pos]
            a_vec = resid[attribute_pos]
            sim = float(jnp.dot(e_vec, a_vec) / (
                jnp.linalg.norm(e_vec) * jnp.linalg.norm(a_vec) + 1e-10
            ))
            sims.append(sim)
        else:
            sims.append(0.0)

    sims = np.array(sims)
    peak_layer = int(np.argmax(np.abs(sims)))

    # When does binding emerge?
    threshold = 0.3
    binding_emerges = -1
    for l, s in enumerate(sims):
        if abs(s) >= threshold:
            binding_emerges = l
            break

    # Trend analysis
    if len(sims) >= 2:
        diffs = np.diff(sims)
        if np.all(diffs >= -0.01):
            trend = "increasing"
        elif np.all(diffs <= 0.01):
            trend = "decreasing"
        else:
            trend = "non_monotonic"
    else:
        trend = "flat"

    return {
        "layer_similarities": sims,
        "peak_layer": peak_layer,
        "binding_emerges": binding_emerges,
        "binding_trend": trend,
    }


def multi_entity_disambiguation(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    entity_positions: list,
    query_pos: int,
    metric_fn: Optional[Callable] = None,
) -> dict:
    """Analyze how the model distinguishes between multiple entities.

    Given positions of multiple entities, measures how the model at a
    query position attends to and represents each entity differently.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        entity_positions: List of positions for different entities.
        query_pos: Position where disambiguation is measured.
        metric_fn: Optional Function(logits) -> float for ablation comparison.

    Returns:
        Dict with:
            "entity_similarities": pairwise similarity between entity representations
            "query_to_entity_attention": mean attention from query to each entity
            "discrimination_score": how well entities are distinguished (0-1)
            "ablation_effects": effect of ablating each entity (if metric_fn given)
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    n_entities = len(entity_positions)

    if n_entities < 2:
        return {
            "entity_similarities": np.array([]),
            "query_to_entity_attention": np.array([]),
            "discrimination_score": 0.0,
            "ablation_effects": {},
        }

    # Entity representations at final layer
    final_key = f"blocks.{n_layers - 1}.hook_resid_post"
    if final_key in cache.cache_dict:
        resid = cache.cache_dict[final_key]
        entity_vecs = np.array(resid[jnp.array(entity_positions)])  # [n_entities, d_model]
        # Pairwise similarity
        norms = np.linalg.norm(entity_vecs, axis=-1, keepdims=True)
        normalized = entity_vecs / (norms + 1e-10)
        entity_sims = normalized @ normalized.T
    else:
        entity_sims = np.eye(n_entities)

    # Attention from query to each entity
    query_attn = np.zeros(n_entities)
    for l in range(n_layers):
        pattern_key = f"blocks.{l}.attn.hook_pattern"
        if pattern_key in cache.cache_dict:
            pattern = cache.cache_dict[pattern_key]  # [seq, n_heads, seq]
            for ei, epos in enumerate(entity_positions):
                query_attn[ei] += float(jnp.mean(pattern[query_pos, :, epos]))
    query_attn /= max(n_layers, 1)

    # Discrimination: how different are the entity representations?
    off_diag = entity_sims[np.triu_indices(n_entities, k=1)]
    if len(off_diag) > 0:
        # Lower similarity = better discrimination
        discrimination = 1.0 - float(np.mean(np.abs(off_diag)))
    else:
        discrimination = 0.0

    # Ablation effects
    ablation_effects = {}
    if metric_fn is not None:
        clean_logits = model(tokens)
        baseline = float(metric_fn(clean_logits))

        for ei, epos in enumerate(entity_positions):
            # Zero out the entity position in residual stream
            hook_name = f"blocks.{n_layers - 1}.hook_resid_pre"

            def ablate_hook(x, name, _pos=epos):
                return x.at[_pos].set(0.0)

            ablated_logits = model.run_with_hooks(
                tokens, fwd_hooks=[(hook_name, ablate_hook)]
            )
            effect = float(metric_fn(ablated_logits)) - baseline
            ablation_effects[epos] = effect

    return {
        "entity_similarities": entity_sims,
        "query_to_entity_attention": query_attn,
        "discrimination_score": discrimination,
        "ablation_effects": ablation_effects,
    }
