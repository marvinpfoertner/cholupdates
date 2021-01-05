"""Specific tests for the function :func:`cholupdates.rank_1.downdate_seeger`."""

import numpy as np
import pytest

import cholupdates


@pytest.mark.parametrize(
    "impl", [None] + cholupdates.rank_1.downdate_seeger.available_impls
)
def test_memory_order(L: np.ndarray, v: np.ndarray, impl: str):
    """Assert that the resulting array has the same memory order as the input array"""

    L_prime = cholupdates.rank_1.downdate_seeger(L, v, impl=impl)

    if L.flags.c_contiguous:
        assert L_prime.flags.c_contiguous
    else:
        assert L.flags.f_contiguous
        assert L_prime.flags.f_contiguous


@pytest.mark.parametrize(
    "L_dtype,v_dtype,impl",
    [
        (L_dtype, v_dtype, impl)
        for L_dtype in [np.float64, np.float32, np.float16, np.complex64, np.int64]
        for v_dtype in [np.float64, np.float32, np.float16, np.complex64, np.int64]
        for impl in [None] + cholupdates.rank_1.downdate_seeger.available_impls
        # There seems to be a bug in pylint, since it marks `L_dtype` and `v_dtype` as
        # undefined here
        # pylint: disable=undefined-variable
        if L_dtype is not np.float64 or v_dtype is not np.float64
    ],
)
def test_raise_on_wrong_dtype(L_dtype, v_dtype, impl):
    """Tests whether a :class:`TypeError` is raised if the Cholesky factor or the vector
    :code:`v` have an unsupported dtype."""

    with pytest.raises(TypeError):
        cholupdates.rank_1.downdate_seeger(
            L=np.eye(5, dtype=L_dtype), v=np.zeros(5, dtype=v_dtype), impl=impl
        )
