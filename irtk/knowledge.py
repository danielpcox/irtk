"""Knowledge and factual recall analysis.

Tools for studying how models store and recall factual knowledge:
- knowledge_neurons: Identify neurons that store a specific fact
- causal_knowledge_tracing: Trace where factual knowledge is computed
- fact_editing_vector: Compute a rank-1 update vector for editing facts
- attribute_to_mlp_vs_attn: Break down predictions into MLP vs attention contributions
"""

from typing import Optional, Callable

import jax
import jax.numpy as jnp
import numpy as np

from irtk.hooked_transformer import HookedTransformer
from irtk.activation_cache import ActivationCache


def knowledge_neurons(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    target_token: int,
    top_k: int = 20,
) -> list[dict]:
    """Identify neurons most responsible for predicting a target token.

    Uses the neuron's contribution to the target logit via W_out @ W_U[:, target].
    Weighted by the neuron's activation value for a multiplicative attribution score.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        target_token: Token ID whose prediction we're attributing.
        top_k: Number of top neurons to return.

    Returns:
        List of dicts with "layer", "neuron", "attribution", "activation"
        sorted by attribution magnitude descending.
    """
    _, cache = model.run_with_cache(tokens)
    W_U = np.array(model.unembed.W_U)  # [d_model, d_vocab]
    target_dir = W_U[:, target_token]  # [d_model]

    results = []
    for l in range(model.cfg.n_layers):
        # Get MLP post-activations at last position
        hook_name = f"blocks.{l}.mlp.hook_post"
        acts = np.array(cache[hook_name])[-1]  # [d_mlp]

        # W_out for this layer
        W_out = np.array(model.blocks[l].mlp.W_out)  # [d_mlp, d_model]

        # Each neuron's contribution to the target logit
        # neuron_i contributes: acts[i] * (W_out[i] @ target_dir)
        logit_contribs = W_out @ target_dir  # [d_mlp]
        attributions = acts * logit_contribs  # [d_mlp]

        for n_idx in range(len(acts)):
            results.append({
                "layer": l,
                "neuron": int(n_idx),
                "attribution": float(attributions[n_idx]),
                "activation": float(acts[n_idx]),
            })

    # Sort by absolute attribution
    results.sort(key=lambda x: abs(x["attribution"]), reverse=True)
    return results[:top_k]


def causal_knowledge_tracing(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    subject_pos: list[int],
    target_token: int,
    noise_std: float = 3.0,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Trace where factual knowledge is computed by corrupting and restoring.

    For a factual prompt like "The Eiffel Tower is in [Paris]", corrupts the
    subject tokens ("Eiffel Tower") and measures how much restoring the residual
    stream at each layer recovers the correct prediction.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        subject_pos: List of positions corresponding to the subject.
        target_token: Correct answer token ID.
        noise_std: Standard deviation of corruption noise.
        seed: Random seed for noise.

    Returns:
        Dict with:
        - "clean_logit": Logit for target token with clean input
        - "corrupted_logit": Logit with corrupted subject
        - "restored_resid": [n_layers] logit when restoring residual at each layer
        - "restored_mlp": [n_layers] logit when restoring MLP output at each layer
        - "restored_attn": [n_layers] logit when restoring attention output at each layer
    """
    # Clean run
    clean_logits = model(tokens)
    clean_logit = float(clean_logits[-1, target_token])

    # Generate corruption noise
    _, clean_cache = model.run_with_cache(tokens)
    rng = np.random.RandomState(seed)
    embed_act = np.array(clean_cache["hook_embed"])
    noise = rng.randn(*embed_act.shape).astype(np.float32) * noise_std

    # Corrupted embedding: add noise at subject positions
    corrupted_embed = np.array(embed_act).copy()
    for pos in subject_pos:
        corrupted_embed[pos] += noise[pos]

    # Corrupted run
    def corrupt_hook(x, name):
        return jnp.array(corrupted_embed)

    corrupted_logits = model.run_with_hooks(
        tokens, fwd_hooks=[("hook_embed", corrupt_hook)]
    )
    corrupted_logit = float(corrupted_logits[-1, target_token])

    n_layers = model.cfg.n_layers
    restored_resid = np.zeros(n_layers)
    restored_mlp = np.zeros(n_layers)
    restored_attn = np.zeros(n_layers)

    for l in range(n_layers):
        # Restore residual stream at layer l
        clean_resid = np.array(clean_cache[f"blocks.{l}.hook_resid_post"])

        def make_restore_hook(clean_act):
            def hook(x, name):
                return jnp.array(clean_act)
            return hook

        logits = model.run_with_hooks(
            tokens,
            fwd_hooks=[
                ("hook_embed", corrupt_hook),
                (f"blocks.{l}.hook_resid_post", make_restore_hook(clean_resid)),
            ]
        )
        restored_resid[l] = float(logits[-1, target_token])

        # Restore MLP output at layer l
        clean_mlp = np.array(clean_cache[f"blocks.{l}.hook_mlp_out"])
        logits = model.run_with_hooks(
            tokens,
            fwd_hooks=[
                ("hook_embed", corrupt_hook),
                (f"blocks.{l}.hook_mlp_out", make_restore_hook(clean_mlp)),
            ]
        )
        restored_mlp[l] = float(logits[-1, target_token])

        # Restore attention output at layer l
        clean_attn = np.array(clean_cache[f"blocks.{l}.hook_attn_out"])
        logits = model.run_with_hooks(
            tokens,
            fwd_hooks=[
                ("hook_embed", corrupt_hook),
                (f"blocks.{l}.hook_attn_out", make_restore_hook(clean_attn)),
            ]
        )
        restored_attn[l] = float(logits[-1, target_token])

    return {
        "clean_logit": clean_logit,
        "corrupted_logit": corrupted_logit,
        "restored_resid": restored_resid,
        "restored_mlp": restored_mlp,
        "restored_attn": restored_attn,
    }


def fact_editing_vector(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    old_token: int,
    new_token: int,
    layer: int,
) -> np.ndarray:
    """Compute a direction vector for editing a fact in the residual stream.

    The vector points from the old answer's unembedding direction to the new
    answer's unembedding direction, projected through the MLP at the given layer.
    This is a simplified ROME-inspired approach.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        old_token: Current answer token ID.
        new_token: Desired new answer token ID.
        layer: Layer to compute the editing vector at.

    Returns:
        [d_model] editing direction vector.
    """
    W_U = np.array(model.unembed.W_U)  # [d_model, d_vocab]
    old_dir = W_U[:, old_token]
    new_dir = W_U[:, new_token]

    # The editing direction is the difference in unembedding space
    edit_dir = new_dir - old_dir

    # Normalize
    norm = np.linalg.norm(edit_dir)
    if norm > 1e-10:
        edit_dir = edit_dir / norm

    return edit_dir


def attribute_to_mlp_vs_attn(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    target_token: int,
    pos: int = -1,
) -> dict[str, np.ndarray]:
    """Break down the prediction into MLP vs attention contributions per layer.

    Computes how much each layer's MLP and attention output contributes
    to the target token's logit at the specified position.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        target_token: Token ID to attribute.
        pos: Position to analyze (-1 for last).

    Returns:
        Dict with:
        - "mlp_contrib": [n_layers] MLP contribution to target logit per layer
        - "attn_contrib": [n_layers] attention contribution per layer
        - "embed_contrib": scalar embedding + positional contribution
        - "total_logit": scalar total logit for target token
    """
    _, cache = model.run_with_cache(tokens)
    W_U = np.array(model.unembed.W_U)  # [d_model, d_vocab]

    # Apply final layer norm if present
    if hasattr(model, 'ln_final') and model.ln_final is not None:
        # Get the logit direction after layer norm scaling
        target_dir = W_U[:, target_token]
    else:
        target_dir = W_U[:, target_token]

    n_layers = model.cfg.n_layers
    mlp_contrib = np.zeros(n_layers)
    attn_contrib = np.zeros(n_layers)

    for l in range(n_layers):
        # MLP output contribution
        mlp_out = np.array(cache[f"blocks.{l}.hook_mlp_out"])[pos]  # [d_model]
        mlp_contrib[l] = float(np.dot(mlp_out, target_dir))

        # Attention output contribution
        attn_out = np.array(cache[f"blocks.{l}.hook_attn_out"])[pos]  # [d_model]
        attn_contrib[l] = float(np.dot(attn_out, target_dir))

    # Embedding contribution
    embed = np.array(cache["hook_embed"])[pos]  # [d_model]
    if "hook_pos_embed" in cache.cache_dict:
        pos_embed = np.array(cache["hook_pos_embed"])[pos]
        embed_val = float(np.dot(embed + pos_embed, target_dir))
    else:
        embed_val = float(np.dot(embed, target_dir))

    # Total logit
    logits = model(tokens)
    total_logit = float(logits[pos, target_token])

    return {
        "mlp_contrib": mlp_contrib,
        "attn_contrib": attn_contrib,
        "embed_contrib": embed_val,
        "total_logit": total_logit,
    }
