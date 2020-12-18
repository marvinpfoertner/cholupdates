""" Common fixtures for all rank-1 modification tests """

# pylint: disable=redefined-outer-name

from typing import Tuple

import numpy as np
import pytest
import scipy.stats


@pytest.fixture(params=[pytest.param(N, id=f"dim{N}") for N in [2, 3, 5, 10, 100]])
def N(request) -> int:
    """Dimension of the matrix to be updated. This is mostly used for test
    parameterization."""
    return request.param


@pytest.fixture(params=[pytest.param(seed, id=f"seed{seed}") for seed in range(5)])
def random_state(request):
    """Random states used to sample the test case input matrices. This is mostly used
    for test parameterization."""
    return np.random.RandomState(seed=request.param)


@pytest.fixture(params=["cho_factor", "seeger_python", "seeger_cython"])
def method(request) -> str:
    """The update algorithm to be tested."""
    return request.param


@pytest.fixture()
def A_eigh(
    N: int, random_state: np.random.RandomState
) -> Tuple[np.ndarray, np.ndarray]:
    """Random eigendecomposition of a symmetric positive definite matrix of dimension
    :func:`N`, sampled from :func:`random_state`"""
    # Generate a random orthonormal eigenbasis
    basis = scipy.stats.special_ortho_group.rvs(N, random_state=random_state)

    # Generate a random spectrum
    spectrum = scipy.stats.gamma.rvs(
        a=10.0,  # "Shape" parameter
        loc=1.0,
        scale=1.0,
        size=N,
        random_state=random_state,
    )

    spectrum.sort()

    return (spectrum, basis)


@pytest.fixture
def A(A_eigh: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Symmetric positive definite matrix of dimension :func:`N` defined by the
    eigendecomposition `A_eigh`, sampled from :func:`random_state`"""
    spectrum, Q = A_eigh

    return Q @ np.diag(spectrum) @ Q.T


@pytest.fixture(params=["C", "F"], ids=["orderC", "orderF"])
def L(request, A):
    """Lower Cholesky factor of :func:`A`. This fixture is parameterized to emit both
    row- and column major Cholesky factors for every input matrix."""
    L = np.linalg.cholesky(A)

    return np.array(L, order=request.param)
