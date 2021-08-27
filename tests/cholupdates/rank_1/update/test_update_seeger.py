"""Specific tests for the function :func:`cholupdates.rank_1.update_seeger`."""

import numpy as np
import pytest

import cholupdates


@pytest.mark.parametrize("impl", cholupdates.rank_1.update_seeger.available_impls)
def test_memory_order(L: np.ndarray, v: np.ndarray, impl: str):
    """Assert that the resulting array has the same memory order as the input array"""

    L_ud = cholupdates.rank_1.update_seeger(L, v, impl=impl)

    if L.flags.c_contiguous:
        assert L_ud.flags.c_contiguous
    else:
        assert L.flags.f_contiguous
        assert L_ud.flags.f_contiguous


@pytest.mark.parametrize(
    "impl", [None] + cholupdates.rank_1.update_seeger.available_impls
)
def test_non_contiguous(N: int, L: np.ndarray, v: np.ndarray, impl: str):
    """Assert that a non-contiguous array leads to a `ValueError`"""

    if N > 1:
        L_noncontig = np.stack([np.eye(N, dtype=L.dtype) for _ in range(8)], axis=1)

        with pytest.raises(ValueError):
            cholupdates.rank_1.update_seeger(L_noncontig[:, 3, :], v, impl=impl)

        v_noncontig = np.zeros((N, 3), dtype=v.dtype, order="C")

        with pytest.raises(ValueError):
            cholupdates.rank_1.update_seeger(L, v_noncontig[:, 1], impl=impl)


@pytest.mark.parametrize(
    "L_dtype,v_dtype,impl",
    [
        (L_dtype, v_dtype, impl)
        for L_dtype in [np.double, np.single, np.half, np.cdouble, np.intc]
        for v_dtype in [np.double, np.single, np.half, np.cdouble, np.intc]
        for impl in [None] + cholupdates.rank_1.update_seeger.available_impls
    ],
)
def test_raise_on_wrong_dtype(L_dtype: np.dtype, v_dtype: np.dtype, impl: str):
    """Tests whether a :class:`TypeError` is raised if the Cholesky factor or the vector
    :code:`v` have an unsupported dtype."""

    if not (L_dtype == v_dtype and L_dtype in (np.single, np.double)):
        with pytest.raises(TypeError):
            cholupdates.rank_1.update_seeger(
                L=np.eye(5, dtype=L_dtype), v=np.zeros(5, dtype=v_dtype), impl=impl
            )


def test_unknown_impl(L: np.ndarray, v: np.ndarray):
    """Tests whether requesting an unknown implementation results in an exception."""
    with pytest.raises(NotImplementedError):
        cholupdates.rank_1.update_seeger(L, v, impl="doesnotexist")
