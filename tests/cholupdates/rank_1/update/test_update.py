"""Tests for the implementation of symmetric rank-1 updates to a Cholesky factor"""

import numpy as np
import pytest

import cholupdates


def test_valid_matrix_square_root(A_prime: np.ndarray, L_prime: np.ndarray):
    """Assert that the resulting Cholesky factor right-multiplied with its transpose
    is (up to numerical imprecisions) equal to the updated matrix, i.e. that the
    Cholesky factor is a valid matrix square root"""
    np.testing.assert_allclose(L_prime @ L_prime.T, A_prime)


def test_positive_diagonal(L_prime: np.ndarray):
    """Assert that the resulting Cholesky factor has a positive diagonal"""
    np.testing.assert_array_less(0.0, np.diag(L_prime))


def test_upper_triangular_part_not_accessed(L, v, L_prime, method):
    """Assert that the upper triangular part of the Cholesky factor does neither read
    from nor write to the upper triangular part of the Cholesky factor"""
    N = L.shape[0]

    # Modify the lower triangular part of L_T to check that the same result
    # is produced and that the lower triangular part is not modified.
    L_mod = L.copy(order="K")
    L_mod[np.triu_indices(N, k=1)] = np.random.rand((N * (N - 1)) // 2)

    L_mod_upd = cholupdates.rank_1.update(L_mod, v, method=method)

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
def test_no_input_mutation(L, v, overwrite_L, overwrite_v, method):
    """Test whether the input arrays are left unmodified if the respective overwrite
    flag is set to :code:`False`"""
    L_copy = L.copy(order="K")
    v_copy = v.copy()

    cholupdates.rank_1.update(
        L_copy,
        v_copy,
        overwrite_L=overwrite_L,
        overwrite_v=overwrite_v,
        method=method,
    )

    if not overwrite_L:
        np.testing.assert_array_equal(L_copy, L)

    if not overwrite_v:
        np.testing.assert_array_equal(v_copy, v)


@pytest.mark.parametrize("shape", [(3, 2), (3,), (1, 3, 3)])
def test_raise_on_invalid_cholesky_factor_shape(shape, method):
    """Tests whether a :class:`ValueError` is raised if the shape of the Cholesky factor
    is not :code:`(N, N)` for some N"""
    with pytest.raises(ValueError):
        cholupdates.rank_1.update(L=np.ones(shape), v=np.ones(shape[-1]), method=method)


@pytest.mark.parametrize("shape", [(3, 2), (3, 1), (1, 3, 3)])
def test_raise_on_invalid_vector_shape(shape, method):
    """Tests whether a :class:`ValueError` is raised if the vector has more than one
    dimension"""
    with pytest.raises(ValueError):
        cholupdates.rank_1.update(L=np.eye(shape[0]), v=np.ones(shape), method=method)


def test_raise_on_vector_dimension_mismatch(L, method):
    """Tests whether a :class:`ValueError` is raised if the shape of the vector is not
    compatible with the shape of the Cholesky factor"""
    N = L.shape[0]

    # Generate arbitrary v with incompatible length
    v_len = N + np.random.randint(-N, N) + 1

    if v_len == N:
        v_len += 1

    v = np.random.rand(v_len)

    with pytest.raises(ValueError):
        cholupdates.rank_1.update(L=L, v=v, method=method)


def test_raise_on_zero_diagonal(L, v, method):
    """Tests whether a :class:`numpy.linalg.LinAlgError` is raised if the diagonal of
    the Cholesky factor contains zeros."""
    L = L.copy(order="K")

    k = np.random.randint(L.shape[0])

    L[k, k] = 0.0

    with pytest.raises(np.linalg.LinAlgError):
        cholupdates.rank_1.update(L, v, method=method)


def test_ill_conditioned_matrix(A, A_eigh, L, method):
    """Tests whether the algorithm still works if the update blows up the condition
    number of the updated matrix."""
    spectrum, Q = A_eigh

    # Generate adverse update vector, which increases the condition number by 100000
    v = Q[:, -1]  # Select eigenvector corresponding to largest eigenvalue
    v *= np.sqrt(spectrum[-1] * (100000.0 - 1.0))

    # Compute update
    L_up = cholupdates.rank_1.update(L, v, method=method)

    # Check quality
    A_up = A + np.outer(v, v)

    np.testing.assert_allclose(L_up @ L_up.T, A_up)
