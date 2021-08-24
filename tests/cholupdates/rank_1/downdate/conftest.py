"""Common fixtures for all rank-1 downdate tests"""

# pylint: disable=redefined-outer-name

from typing import Any, Dict

import numpy as np
import pytest

import cholupdates
import cholupdates.utils


@pytest.fixture
def v(L: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random vector of shape :func:`N` which defines a symmetric rank-1 downdate to
    :func:`A`"""

    return cholupdates.utils.random_rank_1_downdate(L, rng=rng)


@pytest.fixture
def A_prime(A: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Downdated input matrix, i.e. :func:`A` after application of the symmetric rank-1
    downdate defined by :func:`v`"""
    return A - np.outer(v, v)


@pytest.fixture(
    params=(
        [
            pytest.param({"method": method}, id=method)
            for method in cholupdates.rank_1.downdate.available_methods
        ]
        + [
            pytest.param({"method": "seeger", "impl": impl}, id=f"seeger_{impl}")
            for impl in cholupdates.rank_1.downdate_seeger.available_impls
        ]
    )
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
