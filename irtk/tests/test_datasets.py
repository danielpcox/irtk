"""Tests for dataset utilities."""

import jax.numpy as jnp
import numpy as np
import pytest

from irtk.datasets import (
    to_tokens,
    to_string,
    to_single_token,
    to_token_labels,
    make_ioi_dataset,
    make_repeated_tokens,
    make_greater_than_dataset,
    get_tokenizer,
)


@pytest.fixture(scope="module")
def tokenizer():
    return get_tokenizer("openai-community/gpt2")


class TestToTokens:
    def test_basic(self, tokenizer):
        tokens = to_tokens("Hello world", tokenizer=tokenizer)
        assert isinstance(tokens, jnp.ndarray)
        assert tokens.ndim == 1
        assert len(tokens) > 0

    def test_prepend_bos(self, tokenizer):
        tokens_no_bos = to_tokens("Hello", tokenizer=tokenizer)
        tokens_with_bos = to_tokens("Hello", tokenizer=tokenizer, prepend_bos=True)
        assert len(tokens_with_bos) == len(tokens_no_bos) + 1

    def test_list_of_strings(self, tokenizer):
        result = to_tokens(["Hello", "World"], tokenizer=tokenizer)
        assert isinstance(result, list)
        assert len(result) == 2


class TestToString:
    def test_roundtrip(self, tokenizer):
        text = "Hello world"
        tokens = to_tokens(text, tokenizer=tokenizer)
        decoded = to_string(tokens, tokenizer=tokenizer)
        assert decoded == text


class TestToSingleToken:
    def test_single(self, tokenizer):
        # " the" should be a single GPT-2 token
        token_id = to_single_token(" the", tokenizer=tokenizer)
        assert isinstance(token_id, int)

    def test_multi_raises(self, tokenizer):
        with pytest.raises(ValueError, match="not 1"):
            to_single_token("antidisestablishmentarianism", tokenizer=tokenizer)


class TestToTokenLabels:
    def test_basic(self, tokenizer):
        tokens = to_tokens("Hello world", tokenizer=tokenizer)
        labels = to_token_labels(tokens, tokenizer=tokenizer)
        assert isinstance(labels, list)
        assert len(labels) == len(tokens)
        assert all(isinstance(l, str) for l in labels)


class TestMakeIOIDataset:
    def test_basic(self, tokenizer):
        ds = make_ioi_dataset(n_prompts=10, tokenizer=tokenizer)
        assert len(ds.clean_prompts) > 0
        assert len(ds.clean_prompts) == len(ds.corrupted_prompts)
        assert len(ds.answer_tokens) == len(ds.clean_prompts)
        assert len(ds.wrong_tokens) == len(ds.clean_prompts)

    def test_answer_tokens_are_ints(self, tokenizer):
        ds = make_ioi_dataset(n_prompts=5, tokenizer=tokenizer)
        for t in ds.answer_tokens:
            assert isinstance(t, int)

    def test_deterministic(self, tokenizer):
        ds1 = make_ioi_dataset(n_prompts=10, tokenizer=tokenizer, seed=42)
        ds2 = make_ioi_dataset(n_prompts=10, tokenizer=tokenizer, seed=42)
        assert ds1.clean_prompts == ds2.clean_prompts


class TestMakeRepeatedTokens:
    def test_shape(self):
        tokens = make_repeated_tokens(seq_half=20)
        assert tokens.shape == (41,)  # 1 BOS + 20 + 20

    def test_repeated(self):
        tokens = make_repeated_tokens(seq_half=10)
        np.testing.assert_array_equal(
            np.array(tokens[1:11]), np.array(tokens[11:21])
        )

    def test_deterministic(self):
        t1 = make_repeated_tokens(seed=42)
        t2 = make_repeated_tokens(seed=42)
        np.testing.assert_array_equal(np.array(t1), np.array(t2))


class TestMakeGreaterThanDataset:
    def test_basic(self, tokenizer):
        ds = make_greater_than_dataset(n_prompts=10, tokenizer=tokenizer)
        assert len(ds["prompts"]) == 10
        assert len(ds["years"]) == 10
        assert len(ds["tokens"]) == 10
        for year in ds["years"]:
            assert 10 <= year < 90
