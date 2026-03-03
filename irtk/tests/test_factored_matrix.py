"""Tests for FactoredMatrix."""

import jax
import jax.numpy as jnp
import pytest

from irtk.factored_matrix import FactoredMatrix


class TestFactoredMatrix:
    def test_shape(self):
        A = jnp.ones((5, 3))
        B = jnp.ones((3, 7))
        fm = FactoredMatrix(A, B)
        assert fm.shape == (5, 7)

    def test_AB(self):
        A = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        B = jnp.array([[5.0, 6.0], [7.0, 8.0]])
        fm = FactoredMatrix(A, B)
        expected = A @ B
        assert jnp.allclose(fm.AB, expected)

    def test_BA(self):
        A = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        B = jnp.array([[5.0, 6.0], [7.0, 8.0]])
        fm = FactoredMatrix(A, B)
        assert jnp.allclose(fm.BA, B @ A)

    def test_transpose(self):
        A = jnp.ones((5, 3))
        B = jnp.ones((3, 7))
        fm = FactoredMatrix(A, B)
        fm_t = fm.T
        assert fm_t.shape == (7, 5)
        assert jnp.allclose(fm_t.AB, (A @ B).T)

    def test_svd_accuracy(self):
        """SVD should reconstruct the original matrix."""
        key = jax.random.PRNGKey(42)
        k1, k2 = jax.random.split(key)
        A = jax.random.normal(k1, (10, 5))
        B = jax.random.normal(k2, (5, 8))
        fm = FactoredMatrix(A, B)
        U, S, Vh = fm.svd()
        reconstructed = U * S[None, :] @ Vh
        assert jnp.allclose(reconstructed, fm.AB, atol=1e-4)

    def test_eigenvalues_square(self):
        """Eigenvalues should work for square matrices."""
        A = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        B = jnp.array([[2.0, 0.0], [0.0, 3.0]])
        fm = FactoredMatrix(A, B)
        eigs = fm.eigenvalues
        # AB = [[2,0],[0,3]], eigenvalues = {2, 3}
        eig_sorted = jnp.sort(jnp.abs(eigs))
        assert jnp.allclose(eig_sorted, jnp.array([2.0, 3.0]), atol=1e-5)

    def test_eigenvalues_non_square_raises(self):
        A = jnp.ones((5, 3))
        B = jnp.ones((3, 7))
        fm = FactoredMatrix(A, B)
        with pytest.raises(ValueError, match="square"):
            _ = fm.eigenvalues

    def test_norm(self):
        A = jnp.eye(3)
        B = jnp.eye(3) * 2.0
        fm = FactoredMatrix(A, B)
        # ||2I||_F = sqrt(3 * 4) = sqrt(12)
        expected = jnp.sqrt(12.0)
        assert jnp.allclose(fm.norm(), expected, atol=1e-4)

    def test_matmul_with_array(self):
        A = jnp.ones((4, 3))
        B = jnp.ones((3, 5))
        fm = FactoredMatrix(A, B)
        C = jnp.ones((5, 2))
        result = fm @ C
        assert isinstance(result, FactoredMatrix)
        assert result.shape == (4, 2)
        assert jnp.allclose(result.AB, (A @ B) @ C)

    def test_matmul_with_factored_matrix(self):
        A = jnp.ones((4, 3))
        B = jnp.ones((3, 5))
        C = jnp.ones((5, 2))
        D = jnp.ones((2, 6))
        fm1 = FactoredMatrix(A, B)
        fm2 = FactoredMatrix(C, D)
        result = fm1 @ fm2
        assert isinstance(result, FactoredMatrix)
        assert result.shape == (4, 6)
        assert jnp.allclose(result.AB, (A @ B) @ (C @ D), atol=1e-4)

    def test_rmatmul(self):
        A = jnp.ones((4, 3))
        B = jnp.ones((3, 5))
        fm = FactoredMatrix(A, B)
        C = jnp.ones((2, 4))
        result = C @ fm
        assert isinstance(result, FactoredMatrix)
        assert result.shape == (2, 5)

    def test_repr(self):
        A = jnp.ones((4, 3))
        B = jnp.ones((3, 5))
        fm = FactoredMatrix(A, B)
        r = repr(fm)
        assert "FactoredMatrix" in r
        assert "(4, 5)" in r

    def test_batched(self):
        """Should work with batch dimensions."""
        A = jnp.ones((2, 4, 3))
        B = jnp.ones((2, 3, 5))
        fm = FactoredMatrix(A, B)
        assert fm.shape == (2, 4, 5)
        assert fm.AB.shape == (2, 4, 5)
