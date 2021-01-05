"""Common fixtures for all rank-1 downdate tests"""

# pylint: disable=redefined-outer-name

from typing import Any, Dict

import numpy as np
import pytest

import cholupdates


@pytest.fixture
def v(N: int, A_eigh, random_state: np.random.RandomState) -> np.ndarray:
    """Random vector of shape :func:`N` which defines a symmetric rank-1 downdate to
    :func:`A`"""

    Lambda, Q = A_eigh

    # Sample random direction
    v_dir = random_state.normal(size=N)
    v_dir /= np.linalg.norm(v_dir, ord=2)

    # Project direction onto eigenvectors
    v_dir_eigen = Q.T @ v_dir

    # Compute maximum squared norm of downdate
    v_max_sq_norm = np.sqrt(v_dir @ (Lambda * v_dir))

    # Choose norm of downdate
    v_norm = np.sqrt(0.5 * v_max_sq_norm)

    return v_norm * v_dir_eigen


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
