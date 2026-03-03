"""FactoredMatrix: represents A @ B without materializing the full product.

Useful for analyzing QK and OV circuits where the composed matrix
would be very large but we can work with SVD/eigenvalues efficiently.
"""

from __future__ import annotations

import jax.numpy as jnp


class FactoredMatrix:
    """A matrix represented as the product A @ B.

    Supports efficient computation of SVD, eigenvalues, norms, etc.
    without materializing the full (potentially huge) product.
    """

    def __init__(self, A: jnp.ndarray, B: jnp.ndarray):
        """Initialize with two factor matrices.

        Args:
            A: [..., m, k] left factor
            B: [..., k, n] right factor
        """
        self.A = A
        self.B = B

    @property
    def AB(self) -> jnp.ndarray:
        """Compute the full materialized product A @ B."""
        return self.A @ self.B

    @property
    def BA(self) -> jnp.ndarray:
        """Compute B @ A (useful when k < min(m, n))."""
        return self.B @ self.A

    @property
    def shape(self) -> tuple:
        """Shape of the full product [... , m, n]."""
        return self.A.shape[:-1] + (self.B.shape[-1],)

    @property
    def T(self) -> "FactoredMatrix":
        """Transpose: (A @ B)^T = B^T @ A^T."""
        return FactoredMatrix(self.B.mT, self.A.mT)

    def svd(self) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Compute SVD of A @ B efficiently.

        Uses the smaller of BA or AB for efficiency.

        Returns:
            (U, S, Vh) such that A @ B ≈ U @ diag(S) @ Vh
        """
        m = self.A.shape[-2]
        n = self.B.shape[-1]
        k = self.A.shape[-1]

        if k <= min(m, n):
            # Compute SVD via the smaller matrix
            # QR decompose A and B
            Q_A, R_A = jnp.linalg.qr(self.A)
            Q_B, R_B = jnp.linalg.qr(self.B.mT)
            # SVD of the small k x k matrix R_A @ R_B^T
            U_small, S, Vh_small = jnp.linalg.svd(R_A @ R_B.mT, full_matrices=False)
            U = Q_A @ U_small
            Vh = Vh_small @ Q_B.mT
            return U, S, Vh
        else:
            # Fall back to full SVD
            return jnp.linalg.svd(self.AB, full_matrices=False)

    @property
    def eigenvalues(self) -> jnp.ndarray:
        """Eigenvalues (only meaningful for square matrices)."""
        if self.A.shape[-2] != self.B.shape[-1]:
            raise ValueError(
                f"Eigenvalues require square matrix, got {self.shape}"
            )
        # Use the smaller matrix BA for efficiency
        k = self.A.shape[-1]
        m = self.A.shape[-2]
        if k < m:
            return jnp.linalg.eigvals(self.BA)
        return jnp.linalg.eigvals(self.AB)

    def norm(self) -> jnp.ndarray:
        """Frobenius norm of A @ B, computed efficiently."""
        # ||AB||_F^2 = tr(B^T A^T A B) = ||A^T A B||_F via trace trick
        # More efficient: use singular values
        _, S, _ = self.svd()
        return jnp.sqrt(jnp.sum(S**2, axis=-1))

    def __matmul__(self, other) -> "FactoredMatrix | jnp.ndarray":
        """Matrix multiplication.

        If other is a FactoredMatrix: returns FactoredMatrix(A, B @ other.A @ other.B)
        if other is an array: returns FactoredMatrix(A, B @ other)
        """
        if isinstance(other, FactoredMatrix):
            # (A @ B) @ (C @ D) = A @ (B @ C @ D)
            # Represent as FactoredMatrix(A, B @ C @ D) -- collapse B @ C into new right factor
            return FactoredMatrix(self.A, self.B @ other.A @ other.B)
        return FactoredMatrix(self.A, self.B @ other)

    def __rmatmul__(self, other) -> "FactoredMatrix":
        """Right matrix multiplication: other @ (A @ B) = (other @ A) @ B."""
        return FactoredMatrix(other @ self.A, self.B)

    def __repr__(self) -> str:
        return (
            f"FactoredMatrix(shape={self.shape}, "
            f"A.shape={self.A.shape}, B.shape={self.B.shape})"
        )
