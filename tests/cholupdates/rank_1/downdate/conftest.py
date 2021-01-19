"""Common fixtures for all rank-1 downdate tests"""

# pylint: disable=redefined-outer-name

from typing import Any, Dict

import numpy as np
import pytest
import scipy.linalg

import cholupdates


@pytest.fixture
def v(N: int, L: np.ndarray, random_state: np.random.RandomState) -> np.ndarray:
    """Random vector of shape :func:`N` which defines a symmetric rank-1 downdate to
    :func:`A`"""

    # Sample random direction
    v_dir = random_state.normal(size=N)
    v_dir /= np.linalg.norm(v_dir, ord=2)

    # The downdated matrix is positive definite if and only if p^T p < 1 for L * p = v.
    # Hence, a vector v = ||v||_2 * u, where `u` is a unit vector leads to a valid
    # downdate if ||v||_2^2 < (1 / p^T p).
    p_dir = scipy.linalg.solve_triangular(L, v_dir, lower=True)

    v_norm_sq = random_state.uniform(0.2, 0.9) / np.dot(p_dir, p_dir)

    v_norm = np.sqrt(v_norm_sq)

    return v_norm * v_dir


@pytest.fixture
def A_prime(A: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Downdated input matrix, i.e. :func:`A` after application of the symmetric rank-1
    downdate defined by :func:`v`"""
    return A - np.outer(v, v)


@pytest.fixture(
    params=[
        pytest.param({"method": "cho_factor"}, id="cho_factor"),
        pytest.param({"method": "seeger"}, id="seeger_default"),
        pytest.param({"method": "seeger", "impl": "python"}, id="seeger_python"),
        pytest.param({"method": "seeger", "impl": "cython"}, id="seeger_cython"),
    ]
)
def method_kwargs(request) -> Dict[str, Any]:
    """The downdate algorithm to be tested."""
    return request.param


@pytest.fixture
def L_prime(L: np.ndarray, v: np.ndarray, method_kwargs: Dict[str, Any]) -> np.ndarray:
    """Lower cholesky factor of :func:`A_prime` computed via
    :func:`cholupdates.rank_1.downdate`"""
    return cholupdates.rank_1.downdate(
        L=L.copy(order="K"),
        v=v.copy(),
        overwrite_L=True,
        overwrite_v=True,
        **method_kwargs,
    )
