"""Tests for config conversion and model name resolution.

Note: Weight loading tests require downloading models and are in test_gpt2_equivalence.py.
"""

import pytest

from irtk.loading.model_names import get_official_model_name, MODEL_ALIASES
from irtk.hooked_transformer_config import HookedTransformerConfig


class TestModelNames:
    def test_gpt2_aliases(self):
        assert get_official_model_name("gpt2") == "openai-community/gpt2"
        assert get_official_model_name("gpt2-small") == "openai-community/gpt2"
        assert get_official_model_name("gpt2-medium") == "openai-community/gpt2-medium"

    def test_pythia_aliases(self):
        assert get_official_model_name("pythia-70m") == "EleutherAI/pythia-70m"

    def test_case_insensitive(self):
        assert get_official_model_name("GPT2") == "openai-community/gpt2"

    def test_unknown_passthrough(self):
        assert get_official_model_name("some/custom-model") == "some/custom-model"


class TestHookedTransformerConfig:
    def test_defaults(self):
        cfg = HookedTransformerConfig(
            n_layers=12, d_model=768, n_ctx=1024, d_head=64, n_heads=12, d_vocab=50257,
        )
        assert cfg.d_mlp == 4 * 768
        assert cfg.d_vocab_out == 50257
        assert cfg.n_key_value_heads == 12
        assert cfg.attn_scale == 64**0.5

    def test_explicit_overrides(self):
        cfg = HookedTransformerConfig(
            n_layers=2, d_model=32, n_ctx=64, d_head=8, n_heads=4, d_vocab=100,
            d_mlp=256, d_vocab_out=200, n_key_value_heads=2,
        )
        assert cfg.d_mlp == 256
        assert cfg.d_vocab_out == 200
        assert cfg.n_key_value_heads == 2

    def test_rotary_default_dim(self):
        cfg = HookedTransformerConfig(
            n_layers=2, d_model=32, n_ctx=64, d_head=8, n_heads=4, d_vocab=100,
            positional_embedding_type="rotary",
        )
        assert cfg.rotary_dim == 8  # defaults to d_head
