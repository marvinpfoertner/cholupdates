"""Common fixtures for all rank-1 update tests"""

# pylint: disable=redefined-outer-name

from typing import Any, Dict

import numpy as np
import pytest

import cholupdates


@pytest.fixture
def v(N: int, dtype: np.dtype, rng: np.random.Generator) -> np.ndarray:
    """Random vector of shape :func:`N` which defines a symmetric rank-1 update to
    :func:`A`"""
    return rng.normal(scale=10, size=N).astype(dtype, copy=False)


@pytest.fixture
def A_ud(A: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Updated input matrix, i.e. :func:`A` after application of the symmetric rank-1
    update defined by :func:`v`"""
    return A + np.outer(v, v)


@pytest.fixture(
    params=(
        [
            pytest.param({"method": method}, id=method)
            for method in cholupdates.rank_1.update.available_methods
        ]
        + [
            pytest.param({"method": "seeger", "impl": impl}, id=f"seeger_{impl}")
            for impl in cholupdates.rank_1.update_seeger.available_impls
        ]
    )
)
def method_kwargs(request) -> Dict[str, Any]:
    """Configuration of the update algorithm to be tested."""
    return request.param


@pytest.fixture
def L_ud(L: np.ndarray, v: np.ndarray, method_kwargs: Dict[str, Any]) -> np.ndarray:
    """Lower cholesky factor of :func:`A_ud` computed via
    :func:`cholupdates.rank_1.update`"""
    return cholupdates.rank_1.update(
        L=L.copy(order="K"),
        v=v.copy(),
        overwrite_L=True,
        overwrite_v=True,
        **method_kwargs,
    )
