"""GPT-2 equivalence test: verify our model matches HuggingFace output.

This is THE critical validation test. It loads GPT-2 Small, runs the same
input through both our model and HF's model, and checks logit equivalence.

Requires: transformers, torch (for HF model loading)
Run with: pytest irtk/tests/test_gpt2_equivalence.py -v -s
"""

import pytest
import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np


def _hf_available():
    try:
        import transformers
        import torch
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _hf_available(), reason="transformers/torch not installed")
class TestGPT2Equivalence:
    @pytest.fixture(scope="class")
    def models_and_tokens(self):
        """Load both models and prepare test tokens. Cached per test class."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Load HF model
        hf_model = AutoModelForCausalLM.from_pretrained(
            "openai-community/gpt2", torch_dtype=torch.float32
        )
        hf_model.eval()
        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

        # Load our model
        from irtk.hooked_transformer import HookedTransformer
        irtk_model = HookedTransformer.from_pretrained("gpt2")

        # Test input
        text = "The quick brown fox jumps over the lazy dog"
        tokens_pt = tokenizer(text, return_tensors="pt")["input_ids"]
        tokens_jax = jnp.array(tokens_pt.numpy()[0])

        # Get HF output
        with torch.no_grad():
            hf_output = hf_model(tokens_pt)
            hf_logits = hf_output.logits[0].numpy()

        return irtk_model, hf_logits, tokens_jax

    def test_logit_equivalence(self, models_and_tokens):
        """Logits should match HF within float32 tolerance."""
        irtk_model, hf_logits, tokens_jax = models_and_tokens
        irtk_logits = np.array(irtk_model(tokens_jax))

        # Check shapes match
        assert irtk_logits.shape == hf_logits.shape, (
            f"Shape mismatch: {irtk_logits.shape} vs {hf_logits.shape}"
        )

        # Check values are close
        max_diff = np.max(np.abs(irtk_logits - hf_logits))
        mean_diff = np.mean(np.abs(irtk_logits - hf_logits))
        print(f"\nMax logit diff: {max_diff:.6f}")
        print(f"Mean logit diff: {mean_diff:.6f}")

        assert max_diff < 1e-3, f"Max logit difference too large: {max_diff}"

    def test_top_predictions_match(self, models_and_tokens):
        """Top-1 predictions should match at every position."""
        irtk_model, hf_logits, tokens_jax = models_and_tokens
        irtk_logits = np.array(irtk_model(tokens_jax))

        irtk_top1 = np.argmax(irtk_logits, axis=-1)
        hf_top1 = np.argmax(hf_logits, axis=-1)

        assert np.array_equal(irtk_top1, hf_top1), (
            f"Top-1 predictions differ at positions: "
            f"{np.where(irtk_top1 != hf_top1)[0]}"
        )

    def test_cache_completeness(self, models_and_tokens):
        """run_with_cache should capture all expected hook points."""
        irtk_model, _, tokens_jax = models_and_tokens
        _, cache = irtk_model.run_with_cache(tokens_jax)
        cfg = irtk_model.cfg

        # Check embeddings
        assert "hook_embed" in cache
        assert "hook_pos_embed" in cache

        # Check all layers
        for l in range(cfg.n_layers):
            assert f"blocks.{l}.hook_resid_pre" in cache
            assert f"blocks.{l}.hook_resid_mid" in cache
            assert f"blocks.{l}.hook_resid_post" in cache
            assert f"blocks.{l}.hook_attn_out" in cache
            assert f"blocks.{l}.hook_mlp_out" in cache
            assert f"blocks.{l}.attn.hook_q" in cache
            assert f"blocks.{l}.attn.hook_k" in cache
            assert f"blocks.{l}.attn.hook_v" in cache
            assert f"blocks.{l}.attn.hook_z" in cache
            assert f"blocks.{l}.attn.hook_attn_scores" in cache
            assert f"blocks.{l}.attn.hook_pattern" in cache
            assert f"blocks.{l}.attn.hook_result" in cache
            assert f"blocks.{l}.mlp.hook_pre" in cache
            assert f"blocks.{l}.mlp.hook_post" in cache

    def test_hook_intervention(self, models_and_tokens):
        """run_with_hooks should modify activations correctly."""
        irtk_model, _, tokens_jax = models_and_tokens

        # Zero out attention output at layer 0
        def zero_attn(x, name):
            return jnp.zeros_like(x)

        logits_normal = irtk_model(tokens_jax)
        logits_zeroed = irtk_model.run_with_hooks(
            tokens_jax,
            fwd_hooks=[("blocks.0.hook_attn_out", zero_attn)],
        )

        # Logits should be different
        assert not jnp.allclose(logits_normal, logits_zeroed, atol=1e-5)

    def test_jit_without_hooks(self, models_and_tokens):
        """Model should work under eqx.filter_jit without hooks."""
        irtk_model, _, tokens_jax = models_and_tokens
        jitted = eqx.filter_jit(irtk_model)
        logits = jitted(tokens_jax)
        assert logits.shape == (tokens_jax.shape[0], irtk_model.cfg.d_vocab_out)
