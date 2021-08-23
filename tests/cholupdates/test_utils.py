"""Test utility code."""

# pylint: disable=redefined-outer-name

import numpy as np
import pytest

import cholupdates.utils


@pytest.fixture(params=[pytest.param(N, id=f"dim{N}") for N in [2, 3, 5, 10, 100]])
def N(request) -> int:
    """Dimension of the matrix to be updated. This is mostly used for test
    parameterization."""
    return request.param


@pytest.fixture
def random_spd_matrix(N, random_state):
    """Random symmetric, positive-definite matrix whose properties will be tested."""
    return cholupdates.utils.random_spd_matrix(N, random_state=random_state)


def test_random_spd_matrix_symmetric(random_spd_matrix):
    """Test whether the random symmetric, positive definite matrix is indeed
    symmetric."""
    np.testing.assert_equal(random_spd_matrix, random_spd_matrix.T)


def test_random_spd_matrix_positive_definite(random_spd_matrix):
    """Test whether the random symmetric, positive-definite matrix is indeed positive
    definite."""
    np.testing.assert_array_equal(np.linalg.eigvalsh(random_spd_matrix) > 0.0, True)
