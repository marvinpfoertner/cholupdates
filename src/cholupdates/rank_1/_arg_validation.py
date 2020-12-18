"""Argument validation functions for the concrete update functions"""

import numpy as np


def _validate_update_args(
    L: np.ndarray,
    v: np.ndarray,
    check_diag: bool,
):
    # Validate L
    if L.ndim != 2 or L.shape[0] != L.shape[1]:
        raise ValueError(
            f"The given Cholesky factor `L` is not a square matrix (given shape: "
            f"{L.shape})."
        )

    if check_diag:
        if np.any(np.diag(L) == 0):
            raise np.linalg.LinAlgError(
                "The given Cholesky factor `L` contains zeros on its diagonal"
            )

    # Validate v
    if v.ndim != 1 or v.shape[0] != L.shape[0]:
        raise ValueError(
            f"The shape of the given vector `v` is incompatible with the shape of the "
            f"given Cholesky factor `L`. Expected shape {(L.shape[0],)} but got "
            f"{v.shape}."
        )
