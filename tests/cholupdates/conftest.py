"""Common fixtures for all tests."""

import numpy as np
import pytest


@pytest.fixture(params=[pytest.param(seed, id=f"seed{seed}") for seed in range(5)])
def random_state(request):
    """Random number generators used for test randomization."""
    return np.random.RandomState(seed=request.param)
