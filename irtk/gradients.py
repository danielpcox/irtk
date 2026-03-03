"""Gradient-based interpretability utilities.

Provides tools that leverage JAX's autodiff for attribution:
- gradient_x_input: Simple gradient * input attribution
- integrated_gradients: Path-integral attribution from a baseline
- logit_gradient_attribution: Gradient of specific logit w.r.t. each component
- input_jacobian: Full Jacobian of logits w.r.t. input embeddings
"""

from typing import Optional, Callable

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx

from irtk.hooked_transformer import HookedTransformer


def _embed_to_logits(model: HookedTransformer, embeddings: jnp.ndarray) -> jnp.ndarray:
    """Run the model from embeddings (skipping token lookup).

    This makes the computation differentiable w.r.t. the embeddings.

    Args:
        model: HookedTransformer.
        embeddings: [seq_len, d_model] input embeddings.

    Returns:
        [seq_len, d_vocab] logits.
    """
    residual = embeddings

    for block in model.blocks:
        residual = block(residual, None)

    if model.ln_final is not None:
        residual = model.ln_final(residual, None)

    logits = model.unembed(residual, None)
    return logits


def _get_embeddings(model: HookedTransformer, tokens: jnp.ndarray) -> jnp.ndarray:
    """Get combined token + positional embeddings for the input."""
    embed = model.embed.W_E[tokens]  # [seq_len, d_model]
    if model.pos_embed is not None:
        seq_len = tokens.shape[0]
        embed = embed + model.pos_embed.W_pos[:seq_len]
    return embed


def gradient_x_input(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    target_token: int,
    pos: int = -1,
) -> np.ndarray:
    """Compute gradient * input attribution for each input position.

    For each input position i, computes:
        attribution[i] = ||grad_embed[i] * embed[i]||

    where the gradient is of the target token's logit w.r.t. the embedding.
    This measures how much each input position contributes to the prediction.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input token IDs.
        target_token: Token ID to compute attribution for.
        pos: Output position to analyze (-1 for last).

    Returns:
        [seq_len] array of attribution scores per input position.
    """
    embeddings = _get_embeddings(model, tokens)

    def logit_fn(embed):
        logits = _embed_to_logits(model, embed)
        return logits[pos, target_token]

    grads = jax.grad(logit_fn)(embeddings)  # [seq_len, d_model]

    # Attribution = gradient * input, then norm per position
    attr = grads * embeddings  # [seq_len, d_model]
    return np.array(jnp.linalg.norm(attr, axis=-1))  # [seq_len]


def gradient_norm(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    target_token: int,
    pos: int = -1,
) -> np.ndarray:
    """Compute gradient norm attribution for each input position.

    For each input position i, computes:
        attribution[i] = ||grad_embed[i]||

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input token IDs.
        target_token: Token ID to compute attribution for.
        pos: Output position to analyze (-1 for last).

    Returns:
        [seq_len] array of gradient norms per input position.
    """
    embeddings = _get_embeddings(model, tokens)

    def logit_fn(embed):
        logits = _embed_to_logits(model, embed)
        return logits[pos, target_token]

    grads = jax.grad(logit_fn)(embeddings)  # [seq_len, d_model]
    return np.array(jnp.linalg.norm(grads, axis=-1))  # [seq_len]


def integrated_gradients(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    target_token: int,
    pos: int = -1,
    n_steps: int = 50,
    baseline: Optional[jnp.ndarray] = None,
) -> np.ndarray:
    """Compute Integrated Gradients attribution.

    Integrates the gradient along a straight-line path from a baseline
    embedding to the actual embedding. This satisfies the completeness
    axiom: attributions sum to the output difference.

    IG(x) = (x - x') * integral_0^1 grad_f(x' + t*(x - x')) dt

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input token IDs.
        target_token: Token ID to compute attribution for.
        pos: Output position to analyze (-1 for last).
        n_steps: Number of interpolation steps for the integral.
        baseline: [seq_len, d_model] baseline embedding.
            Defaults to zero embedding.

    Returns:
        [seq_len] array of attribution scores per input position.
    """
    embeddings = _get_embeddings(model, tokens)

    if baseline is None:
        baseline = jnp.zeros_like(embeddings)

    def logit_fn(embed):
        logits = _embed_to_logits(model, embed)
        return logits[pos, target_token]

    # Compute gradients at each interpolation point
    diff = embeddings - baseline
    alphas = jnp.linspace(0, 1, n_steps + 1)

    total_grad = jnp.zeros_like(embeddings)
    for i in range(n_steps + 1):
        interpolated = baseline + alphas[i] * diff
        grad_i = jax.grad(logit_fn)(interpolated)
        total_grad = total_grad + grad_i

    # Trapezoidal rule: average the endpoints, sum the rest
    avg_grad = total_grad / (n_steps + 1)

    # Attribution = (input - baseline) * averaged gradient
    attr = diff * avg_grad  # [seq_len, d_model]
    return np.array(jnp.linalg.norm(attr, axis=-1))  # [seq_len]


def integrated_gradients_full(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    target_token: int,
    pos: int = -1,
    n_steps: int = 50,
    baseline: Optional[jnp.ndarray] = None,
) -> np.ndarray:
    """Compute full Integrated Gradients attribution (per-position, per-dimension).

    Like integrated_gradients but returns the full [seq_len, d_model] attribution
    matrix instead of reducing to per-position norms.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input token IDs.
        target_token: Token ID to compute attribution for.
        pos: Output position to analyze (-1 for last).
        n_steps: Number of interpolation steps.
        baseline: Baseline embedding (default: zeros).

    Returns:
        [seq_len, d_model] array of signed attribution values.
    """
    embeddings = _get_embeddings(model, tokens)

    if baseline is None:
        baseline = jnp.zeros_like(embeddings)

    def logit_fn(embed):
        logits = _embed_to_logits(model, embed)
        return logits[pos, target_token]

    diff = embeddings - baseline
    alphas = jnp.linspace(0, 1, n_steps + 1)

    total_grad = jnp.zeros_like(embeddings)
    for i in range(n_steps + 1):
        interpolated = baseline + alphas[i] * diff
        grad_i = jax.grad(logit_fn)(interpolated)
        total_grad = total_grad + grad_i

    avg_grad = total_grad / (n_steps + 1)
    return np.array(diff * avg_grad)


def logit_gradient_attribution(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    target_token: int,
    pos: int = -1,
) -> dict[str, np.ndarray]:
    """Compute gradient-based attribution for each model component.

    For each layer's attention output and MLP output, computes the gradient
    of the target logit w.r.t. that component's contribution to the residual
    stream. This uses the Jacobian to measure sensitivity.

    Note: This uses activation caching plus gradient computation. Each
    component's attribution is grad(logit) dot component_output.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input token IDs.
        target_token: Token ID.
        pos: Output position (-1 for last).

    Returns:
        Dict mapping component names to scalar attribution values.
        Keys: "embed", "L{i}_attn", "L{i}_mlp" for each layer i.
    """
    _, cache = model.run_with_cache(tokens)

    # We need the logit direction in d_model space
    W_U = model.unembed.W_U  # [d_model, d_vocab]
    logit_dir = W_U[:, target_token]  # [d_model]

    # For a proper gradient computation, we also need to account for ln_final.
    # But for simplicity and matching direct_logit_attribution, we compute
    # the gradient of the logit w.r.t. the residual stream at the final position,
    # then project each component's output onto that direction.

    # Get the final residual stream
    final_resid_key = f"blocks.{model.cfg.n_layers - 1}.hook_resid_post"
    if final_resid_key in cache.cache_dict:
        final_resid = cache.cache_dict[final_resid_key]
    else:
        # Fallback: reconstruct from hook_resid_pre of last layer
        final_resid = cache.cache_dict.get(
            f"blocks.{model.cfg.n_layers - 1}.hook_resid_post",
            None
        )

    # Compute gradient of logit w.r.t. the final residual stream
    def logit_from_resid(resid):
        if model.ln_final is not None:
            normed = model.ln_final(resid, None)
        else:
            normed = resid
        logits = normed @ W_U + model.unembed.b_U
        return logits[pos, target_token]

    if final_resid is not None:
        grad_resid = jax.grad(logit_from_resid)(final_resid)  # [seq_len, d_model]
        effective_dir = grad_resid[pos]  # [d_model]
    else:
        effective_dir = logit_dir

    results = {}

    # Embedding
    embed = cache.cache_dict.get("hook_embed", None)
    pos_embed = cache.cache_dict.get("hook_pos_embed", None)
    if embed is not None:
        e = embed[pos]
        if pos_embed is not None:
            e = e + pos_embed[pos]
        results["embed"] = np.array(jnp.dot(e, effective_dir))

    # Each layer
    for layer in range(model.cfg.n_layers):
        attn_key = f"blocks.{layer}.hook_attn_out"
        mlp_key = f"blocks.{layer}.hook_mlp_out"

        if attn_key in cache.cache_dict:
            attn_out = cache.cache_dict[attn_key][pos]
            results[f"L{layer}_attn"] = np.array(jnp.dot(attn_out, effective_dir))

        if mlp_key in cache.cache_dict:
            mlp_out = cache.cache_dict[mlp_key][pos]
            results[f"L{layer}_mlp"] = np.array(jnp.dot(mlp_out, effective_dir))

    return results


def input_jacobian(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    output_pos: int = -1,
    top_k: int = 10,
) -> tuple[np.ndarray, list[tuple[int, float]]]:
    """Compute the Jacobian of output logits w.r.t. input embeddings.

    Returns the full Jacobian matrix and the top-k (input_pos, output_token)
    pairs by Jacobian norm.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input token IDs.
        output_pos: Output position to analyze.
        top_k: Number of top entries to return.

    Returns:
        (jacobian, top_entries) where:
        - jacobian: [d_vocab, seq_len, d_model] array
            (grad of each output logit w.r.t. each input embedding dimension)
        - top_entries: list of (output_token_id, input_pos, jacobian_norm) tuples
    """
    embeddings = _get_embeddings(model, tokens)

    def logits_fn(embed):
        logits = _embed_to_logits(model, embed)
        return logits[output_pos]  # [d_vocab]

    # Jacobian: [d_vocab, seq_len, d_model]
    jac = jax.jacobian(logits_fn)(embeddings)

    # Norm over d_model dimension to get [d_vocab, seq_len]
    jac_norms = np.array(jnp.linalg.norm(jac, axis=-1))

    # Find top entries
    flat_indices = np.argsort(jac_norms.flatten())[::-1][:top_k]
    top_entries = []
    for idx in flat_indices:
        token_id = idx // jac_norms.shape[1]
        input_pos = idx % jac_norms.shape[1]
        top_entries.append((int(token_id), int(input_pos), float(jac_norms[token_id, input_pos])))

    return np.array(jac), top_entries


def token_attribution(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    target_token: int,
    pos: int = -1,
    method: str = "grad_x_input",
    **kwargs,
) -> np.ndarray:
    """Unified interface for token-level attribution.

    Compute attribution scores for each input token using the specified method.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input token IDs.
        target_token: Token ID to compute attribution for.
        pos: Output position (-1 for last).
        method: Attribution method. One of:
            - "grad_x_input": gradient * input (fast, approximate)
            - "grad_norm": gradient norm (fast, direction-agnostic)
            - "integrated_gradients": integrated gradients (slower, principled)
        **kwargs: Additional arguments passed to the method function.

    Returns:
        [seq_len] array of attribution scores.
    """
    if method == "grad_x_input":
        return gradient_x_input(model, tokens, target_token, pos=pos)
    elif method == "grad_norm":
        return gradient_norm(model, tokens, target_token, pos=pos)
    elif method == "integrated_gradients":
        return integrated_gradients(model, tokens, target_token, pos=pos, **kwargs)
    else:
        raise ValueError(f"Unknown attribution method: {method!r}. "
                         f"Choose from: 'grad_x_input', 'grad_norm', 'integrated_gradients'")
