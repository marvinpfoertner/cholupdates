"""Utility functions."""

from typing import Optional, Tuple

import numpy as np
import scipy.stats


def random_spd_eigendecomposition(
    N: int, random_state: Optional[np.random.RandomState] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates a random eigendecomposition of a symmetric positive definite matrix.

    The spectrum of the matrix will be drawn from a shifted gamma distribution, while
    the eigenbasis is drawn uniformly from the Haar measure.

    Parameters
    ----------
    N :
        Dimension of the matrix.
    random_state :
        The random number generator to be used to sample the eigendecomposition.

    Returns
    -------
    spectrum :
        The spectrum of the matrix as a :class:`numpy.ndarray` of shape :code:`(N,)`.
    basis :
        The eigenbasis as the columns of a :class:`numpy.ndarray` of shape
        :code:`(N, N)`.
    """
    # Generate a random positive spectrum
    spectrum = scipy.stats.gamma.rvs(
        a=10.0,  # "Shape" parameter
        loc=1.0,
        scale=1.0,
        size=N,
        random_state=random_state,
    )

    spectrum.sort()

    # Generate a random orthonormal eigenbasis
    basis = scipy.stats.special_ortho_group.rvs(N, random_state=random_state)

    return spectrum, basis


def random_spd_matrix(
    N: int, fast: bool = False, random_state: Optional[np.random.RandomState] = None
) -> np.ndarray:
    """Generates a random symmetric positive-definite matrix.

    Parameters
    ----------
    N :
        Dimension of the matrix.
    fast:
        If this is set to :code:`True`, the method will use a fast but biased method to
        draw the matrix. Otherwise, a random eigendecomposition will be drawn.
    random_state :
        The random number generator to be used to sample the matrix.

    Returns
    -------
    A random symmetrix positive-definite matrix.
    """
    if fast:
        # Generate positive-semidefinite matrix from square-root
        A = scipy.stats.norm.rvs(size=(N, N), random_state=random_state)
        A = A @ A.T

        # Make positive definite
        A += np.eye(N)

        # Apply Jacobi preconditioner to improve condition number
        D = np.sqrt(np.diag(A))
        A = D[:, None] * A * D[None, :]

        return A

    # Sample a random Eigendecomposition
    spectrum, Q = random_spd_eigendecomposition(N, random_state=random_state)

    # Assemble matrix
    M = Q @ np.diag(spectrum) @ Q.T

    # Symmetrize
    M = 0.5 * (M + M.T)

    return M


def random_rank_1_downdate(
    L: np.ndarray, random_state: Optional[np.random.RandomState] = None
) -> np.ndarray:
    """Generates a random rank-1 downdate for a given Cholesky factor which, when
    applied, will result in a positive-definite matrix again.

    Parameters
    ----------
    L :
        The lower-triangular Cholesky factor of the matrix to be downdated.
    random_state :
        The random number generator to be used to sample the matrix.

    Returns
    -------
    The vector :math:`v` which defines the downdate as a :class:`numpy.ndarray` of shape
    :code:`(N,)`, where :code:`(N, N)` is the shape of :code:`L`.
    """
    N = L.shape[0]

    # Sample uniformly random direction
    v_dir = scipy.stats.norm.rvs(size=N, random_state=random_state)
    v_dir /= np.linalg.norm(v_dir, ord=2)

    # The downdated matrix is positive semi-definite if and only if p^T p < 1 for
    # L * p = v. Hence, a vector v = ||v||_2 * u, where `u` is a unit vector leads to a
    # valid downdate if ||v||_2^2 < (1 / p^T p).
    p_dir = scipy.linalg.solve_triangular(L, v_dir, lower=True)

    v_norm_sq = scipy.stats.uniform.rvs(
        loc=0.2, scale=0.9 - 0.2, size=N, random_state=random_state
    ) / np.dot(p_dir, p_dir)

    v_norm = np.sqrt(v_norm_sq)

    return v_norm * v_dir
