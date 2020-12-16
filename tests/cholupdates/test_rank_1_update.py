"""Tests for the implementation of symmetric rank-1 updates to a Cholesky factor"""

# pylint: disable=redefined-outer-name


from typing import Tuple

import numpy as np
import pytest
import scipy.stats

import cholupdates


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


@pytest.fixture
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


@pytest.fixture
def v(N, random_state) -> np.ndarray:
    """Random vector of shape :func:`N` which defines a symmetric rank-1 update to
    :func:`A`"""
    return random_state.normal(scale=10, size=N)


@pytest.fixture
def A_prime(A, v) -> np.ndarray:
    """Updated input matrix, i.e. :func:`A` after application of the symmetric rank-1
    update defined by :func:`v`"""
    return A + np.outer(v, v)


@pytest.fixture
def L_prime(L, v) -> np.ndarray:
    """Lower cholesky factor of :func:`A_prime` computed via
    :func:`cholupdates.rank_1_updates`"""
    return cholupdates.rank_1.update(
        L=L.copy(order="K"), v=v.copy(), overwrite_L=True, overwrite_v=True
    )


def test_valid_matrix_square_root(A_prime: np.ndarray, L_prime: np.ndarray):
    """Assert that the resulting Cholesky factor right-multiplied with its transpose
    is (up to numerical imprecisions) equal to the updated matrix, i.e. that the
    Cholesky factor is a valid matrix square root"""
    assert L_prime @ L_prime.T == pytest.approx(A_prime)


def test_positive_diagonal(L_prime: np.ndarray):
    """Assert that the resulting Cholesky factor has a positive diagonal"""
    np.testing.assert_array_less(0.0, np.diag(L_prime))


def test_memory_order(L: np.ndarray, L_prime: np.ndarray):
    """Assert that the resulting array has the same memory order as the input array"""
    if L.flags.c_contiguous:
        assert L_prime.flags.c_contiguous
    else:
        assert L.flags.f_contiguous
        assert L_prime.flags.f_contiguous


def test_upper_triangular_part_not_accessed(L, v, L_prime):
    """Assert that the upper triangular part of the Cholesky factor does neither read
    from nor write to the upper triangular part of the Cholesky factor"""
    N = L.shape[0]

    # Modify the lower triangular part of L_T to check that the same result
    # is produced and that the lower triangular part is not modified.
    L_mod = L.copy(order="K")
    L_mod[np.triu_indices(N, k=1)] = np.random.rand((N * (N - 1)) // 2)

    L_mod_upd = cholupdates.rank_1.update(L_mod, v)

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
    """Test whether the input arrays are left unmodified if the respective overwrite
    flag is set to :code:`False`"""
    L_copy = L.copy(order="K")
    v_copy = v.copy()

    cholupdates.rank_1.update(
        L_copy, v_copy, overwrite_L=overwrite_L, overwrite_v=overwrite_v
    )

    if not overwrite_L:
        np.testing.assert_array_equal(L_copy, L)

    if not overwrite_v:
        np.testing.assert_array_equal(v_copy, v)


@pytest.mark.parametrize("shape", [(3, 2), (3,), (1, 3, 3)])
def test_raise_on_invalid_cholesky_factor_shape(shape):
    """Tests whether a :class:`ValueError` is raised if the shape of the Cholesky factor
    is not :code:`(N, N)` for some N"""
    with pytest.raises(ValueError):
        cholupdates.rank_1.update(L=np.ones(shape), v=np.ones(shape[-1]))


@pytest.mark.parametrize("shape", [(3, 2), (3, 1), (1, 3, 3)])
def test_raise_on_invalid_vector_shape(shape):
    """Tests whether a :class:`ValueError` is raised if the vector has more than one
    dimension"""
    with pytest.raises(ValueError):
        cholupdates.rank_1.update(L=np.eye(shape[0]), v=np.ones(shape))


def test_raise_on_vector_dimension_mismatch(L):
    """Tests whether a :class:`ValueError` is raised if the shape of the vector is not
    compatible with the shape of the Cholesky factor"""
    N = L.shape[0]

    # Generate arbitrary v with incompatible length
    v_len = N + np.random.randint(-N, N) + 1

    if v_len == N:
        v_len += 1

    v = np.random.rand(v_len)

    with pytest.raises(ValueError):
        cholupdates.rank_1.update(L=L, v=v)


@pytest.mark.parametrize(
    "L_dtype,v_dtype",
    [
        (L_dtype, v_dtype)
        for L_dtype in [np.float64, np.float32, np.float16, np.complex64, np.int64]
        for v_dtype in [np.float64, np.float32, np.float16, np.complex64, np.int64]
        # There seems to be a bug in pylint, since it marks `L_dtype` and `v_dtype` as
        # undefined here
        # pylint: disable=undefined-variable
        if L_dtype is not np.float64 or v_dtype is not np.float64
    ],
)
def test_raise_on_wrong_dtype(L_dtype, v_dtype):
    """Tests whether a :class:`TypeError` is raised if the Cholesky factor or the vector
    :code:`v` have an unsupported dtype."""
    with pytest.raises(TypeError):
        cholupdates.rank_1.update(
            L=np.eye(5, dtype=L_dtype), v=np.zeros(5, dtype=v_dtype)
        )


def test_raise_on_zero_diagonal(L, v):
    """Tests whether a :class:`numpy.linalg.LinAlgError` is raised if the diagonal of
    the Cholesky factor contains zeros."""
    L = L.copy(order="K")

    k = np.random.randint(L.shape[0])

    L[k, k] = 0.0

    with pytest.raises(np.linalg.LinAlgError):
        cholupdates.rank_1.update(L, v)


def test_ill_conditioned_matrix(A, A_eigh, L):
    """Tests whether the algorithm still works if the update blows up the condition
    number of the updated matrix."""
    spectrum, Q = A_eigh

    # Generate adverse update vector
    v = Q[:, -1]  # Select eigenvector corresponding to largest eigenvalue
    v *= np.sqrt(spectrum[-1] * 100000)  # Update multiplies condition number by 100000

    # Compute update
    L_upd = cholupdates.rank_1.update(L, v)

    # Check quality
    assert L_upd @ L_upd.T == pytest.approx(A + np.outer(v, v))
