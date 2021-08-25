"""Specific tests for the function :func:`cholupdates.rank_1.downdate_seeger`."""

import contextlib

import numpy as np
import pytest

import cholupdates


@pytest.mark.parametrize(
    "impl", [None] + cholupdates.rank_1.downdate_seeger.available_impls
)
def test_memory_order(L: np.ndarray, v: np.ndarray, impl: str):
    """Assert that the resulting array has the same memory order as the input array"""

    L_dd = cholupdates.rank_1.downdate_seeger(L, v, impl=impl)

    if L.flags.c_contiguous:
        assert L_dd.flags.c_contiguous
    else:
        assert L.flags.f_contiguous
        assert L_dd.flags.f_contiguous


@pytest.mark.parametrize(
    "L_dtype,v_dtype,impl",
    [
        (L_dtype, v_dtype, impl)
        for L_dtype in [np.double, np.single, np.half, np.cdouble, np.intc]
        for v_dtype in [np.double, np.single, np.half, np.cdouble, np.intc]
        for impl in [None] + cholupdates.rank_1.downdate_seeger.available_impls
    ],
)
def test_raise_on_wrong_dtype(L_dtype: np.dtype, v_dtype: np.dtype, impl: str):
    """Tests whether a :class:`TypeError` is raised if the Cholesky factor or the vector
    :code:`v` have an unsupported dtype."""

    with (
        pytest.raises(TypeError)
        if not (L_dtype == v_dtype and L_dtype in (np.single, np.double))
        else contextlib.nullcontext()
    ):
        cholupdates.rank_1.downdate_seeger(
            L=np.eye(5, dtype=L_dtype), v=np.zeros(5, dtype=v_dtype), impl=impl
        )
