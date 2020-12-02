# pylint: disable=redefined-outer-name

from typing import Optional

import numpy as np
import pytest
import scipy.linalg
import scipy.stats

import cholupdates


def random_spd_matrix(
    n: int,
    spectrum: np.ndarray = None,
    spectrum_shape: float = 10.0,
    spectrum_scale: float = 1.0,
    spectrum_offset: float = 0.0,
    random_state: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    Q = scipy.stats.special_ortho_group.rvs(n, random_state=random_state)

    if spectrum is None:
        spectrum = scipy.stats.gamma.rvs(
            spectrum_shape,
            loc=spectrum_offset,
            scale=spectrum_scale,
            size=n,
            random_state=random_state,
        )

        # TODO: Sort the spectrum?

    return Q @ np.diag(spectrum) @ Q.T


@pytest.fixture(params=[pytest.param(N, id=f"dim{N}") for N in [2, 3, 5, 10, 100]])
def N(request) -> int:
    return request.param


@pytest.fixture(params=[pytest.param(seed, id=f"seed{seed}") for seed in range(5)])
def random_state(request):
    return np.random.RandomState(seed=request.param)


@pytest.fixture
def A(N, random_state) -> np.ndarray:
    return random_spd_matrix(
        N, spectrum_shape=10.0, spectrum_offset=1.0, random_state=random_state
    )


@pytest.fixture(params=["C", "F"], ids=["orderC", "orderF"])
def L(request, A):
    L = np.linalg.cholesky(A)

    return np.array(L, order=request.param)


@pytest.fixture
def v(N, random_state) -> np.ndarray:
    return random_state.normal(scale=10, size=N)


@pytest.fixture
def A_prime(A, v) -> np.ndarray:
    return A + np.outer(v, v)


@pytest.fixture
def L_prime(L, v) -> np.ndarray:
    return cholupdates.rank_1_update(
        L=L.copy(order="K"), v=v.copy(), overwrite_L=True, overwrite_v=True
    )


def test_valid_matrix_square_root(A_prime: np.ndarray, L_prime: np.ndarray):
    assert L_prime @ L_prime.T == pytest.approx(A_prime)


def test_positive_diagonal(L_prime: np.ndarray):
    np.testing.assert_array_less(0.0, np.diag(L_prime))


def test_memory_order(L: np.ndarray, L_prime: np.ndarray):
    if L.flags.c_contiguous:
        assert L_prime.flags.c_contiguous
    else:
        assert L.flags.f_contiguous
        assert L_prime.flags.f_contiguous


def test_upper_triangular_part_not_accessed(L, v, L_prime):
    N = L.shape[0]

    # Modify the lower triangular part of L_T to check that the same result
    # is produced and that the lower triangular part is not modified.
    L_mod = L.copy(order="K")
    L_mod[np.triu_indices(N, k=1)] = np.random.rand((N * (N - 1)) // 2)

    L_mod_upd = cholupdates.rank_1_update(L_mod, v)

    np.testing.assert_array_equal(
        np.triu(L_mod, k=1),
        np.triu(L_mod_upd, k=1),
        err_msg=(
            "The rank-1 update modified the upper triangular part of the Cholesky "
            "factor"
        ),
    )

    np.testing.assert_array_equal(
        np.tril(L_mod_upd),
        np.tril(L_prime),
        err_msg=(
            "The rank-1 update did not ignore the upper triangular part of the "
            "original Cholesky factor"
        ),
    )


@pytest.mark.parametrize(
    "overwrite_L,overwrite_v", [(False, False), (True, False), (False, True)]
)
def test_no_input_mutation(L, v, overwrite_L, overwrite_v):
    L_copy = L.copy(order="K")
    v_copy = v.copy()

    cholupdates.rank_1_update(
        L_copy, v_copy, overwrite_L=overwrite_L, overwrite_v=overwrite_v
    )

    if not overwrite_L:
        np.testing.assert_array_equal(L_copy, L)

    if not overwrite_v:
        np.testing.assert_array_equal(v_copy, v)


@pytest.mark.parametrize("shape", [(3, 2), (3,), (1, 3, 3)])
def test_raise_on_non_square_cholesky_factor(shape):
    with pytest.raises(ValueError):
        cholupdates.rank_1_update(L=np.ones(shape), v=np.ones(shape[-1]))


@pytest.mark.parametrize("shape", [(3, 2), (3, 1), (1, 3, 3)])
def test_raise_on_invalid_vector_dimension(shape):
    with pytest.raises(ValueError):
        cholupdates.rank_1_update(L=np.eye(shape[0]), v=np.ones(shape))


def test_raise_on_vector_dimension_mismatch(L):
    N = L.shape[0]

    # Generate arbitrary v with incompatible length
    v_len = N + np.random.randint(-N, N) + 1

    if v_len == N:
        v_len += 1

    v = np.random.rand(v_len)

    with pytest.raises(ValueError):
        cholupdates.rank_1_update(L=L, v=v)


@pytest.mark.parametrize(
    "L_dtype,v_dtype",
    [
        (L_dtype, v_dtype)
        for L_dtype in [np.float64, np.float32, np.float16, np.complex64, np.int64]
        for v_dtype in [np.float64, np.float32, np.float16, np.complex64, np.int64]
        # pylint: disable=undefined-variable
        if L_dtype is not np.float64 or v_dtype is not np.float64
    ],
)
def test_raise_on_wrong_dtype(L_dtype, v_dtype):
    with pytest.raises(TypeError):
        cholupdates.rank_1_update(
            L=np.eye(5, dtype=L_dtype), v=np.zeros(5, dtype=v_dtype)
        )


def test_raise_on_zero_diagonal(L, v):
    L = L.copy(order="K")

    k = np.random.randint(L.shape[0])

    L[k, k] = 0.0

    with pytest.raises(np.linalg.LinAlgError):
        cholupdates.rank_1_update(L, v)
