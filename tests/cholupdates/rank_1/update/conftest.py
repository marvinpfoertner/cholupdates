"""Common fixtures for all rank-1 update tests"""

# pylint: disable=redefined-outer-name

import numpy as np
import pytest

import cholupdates


@pytest.fixture
def v(N: int, random_state: np.random.RandomState) -> np.ndarray:
    """Random vector of shape :func:`N` which defines a symmetric rank-1 update to
    :func:`A`"""
    return random_state.normal(scale=10, size=N)


@pytest.fixture
def A_prime(A: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Updated input matrix, i.e. :func:`A` after application of the symmetric rank-1
    update defined by :func:`v`"""
    return A + np.outer(v, v)


@pytest.fixture(params=["cho_factor", "seeger_python", "seeger_cython"])
def method(request) -> str:
    """The update algorithm to be tested."""
    return request.param


@pytest.fixture
def L_prime(L: np.ndarray, v: np.ndarray, method: str) -> np.ndarray:
    """Lower cholesky factor of :func:`A_prime` computed via
    :func:`cholupdates.rank_1.update`"""
    return cholupdates.rank_1.update(
        L=L.copy(order="K"),
        v=v.copy(),
        overwrite_L=True,
        overwrite_v=True,
        method=method,
    )
