"""Common fixtures for all rank-1 modification tests."""

# pylint: disable=redefined-outer-name

from typing import Tuple

import numpy as np
import pytest

import cholupdates.utils


@pytest.fixture(params=[pytest.param(N, id=f"dim{N}") for N in [1, 2, 3, 5, 10, 100]])
def N(request) -> int:
    """Dimension of the matrix to be modified."""
    return request.param


@pytest.fixture(
    params=[pytest.param(np.double, id="double"), pytest.param(np.single, id="single")]
)
def dtype(request) -> np.dtype:
    """Dtype of the matrix to be modified."""
    return request.param


@pytest.fixture
def A_eigh(
    N: int, dtype: np.dtype, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """Random eigendecomposition of a symmetric positive definite matrix of dimension
    :func:`N`, sampled from :func:`rng`"""
    return cholupdates.utils.random_spd_eigendecomposition(N, dtype=dtype, rng=rng)


@pytest.fixture
def A(A_eigh: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Symmetric positive definite matrix of dimension :func:`N` defined by the
    eigendecomposition `A_eigh`, sampled from :func:`rng`"""
    spectrum, Q = A_eigh

    return Q @ np.diag(spectrum) @ Q.T


@pytest.fixture(params=["C", "F"], ids=["orderC", "orderF"])
def L(request, A: np.ndarray):
    """Lower Cholesky factor of :func:`A`. This fixture is parameterized to emit both
    row- and column major Cholesky factors for every input matrix."""
    L = np.linalg.cholesky(A)

    return np.array(L, order=request.param)
