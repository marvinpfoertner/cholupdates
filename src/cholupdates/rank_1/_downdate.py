""" Implementations of symmetric rank-1 downdates to Cholesky factorized matrices. """

import numpy as np
import scipy.linalg

from ._arg_validation import _validate_update_args
from ._seeger import downdate_seeger


def downdate(
    L: np.ndarray,
    v: np.ndarray,
    check_diag: bool = True,
    overwrite_L: bool = False,
    overwrite_v: bool = False,
    method: str = "seeger",
) -> np.ndarray:
    r"""Compute the Cholesky factorization of a symmetric rank-1 udowndate to a
    symmetric positive definite matrix with given Cholesky factorization in a fast and
    stable manner.

    Specifically, given the upper triangular Cholesky factor :math:`L^T` of
    :math:`A = L L^T \in \R^{N \times N}`, this function computes the upper triangular
    Cholesky factor :math:`L'^T` of :math:`A' = L' L'^T`, where :math:`A' = A - v v^T`
    for some vector :math:`v \in \R^n`, if :math:`A'` is positive definite.

    We implement the method in [1, section 3]. This algorithm computes the Cholesky
    decomposition of :math:`A'` from :math:`L_T` in :math:`O(N^2)` time, which is faster
    than the :math:`O(N^3)` time complexity of naively applying a Cholesky algorithm to
    :math:`A'` directly.

    Args:
        L_T:
            Upper triangular Cholesky factor of :math:`A` of shape `(N, N)`. Dtypes
            other than :class:`np.float64` will be cast to :class:`np.float64`
            (essentially triggering a copy). The algorithm is more efficient if this
            array is stored in C-contiguous (i.e. row-major) memory order. The entries
            in the lower triangular part of `L_T` will be ignored by the algorithm.
        v:
            The vector :math:`v` defining the symmetric rank-1 downdate with shape
            `(N,)`. Dtypes other than :class:`np.float64` will be cast to
            :class:`np.float64` (essentially triggering a copy).
        reuse_L_T:
            If set to `True`, the function might reuse the array `L_T` to store the
            upper Cholesky factor of :math:`A'`. In this case, the result is computed
            essentially in-place. Note that passing `True` here does not guarantee that
            `L_T` is reused. However, in this case, additional memory is only allocated
            if absolutely necessary, e.g. if the array has the wrong dtype.
            Passing `False` here will ensure that the array `L_T` is not modified.
        reuse_v:
            If set to `True`, the function might reuse the array `v` as an internal
            computation buffer. In this case, the array `v` might be modified. Note that
            passing `True` here does not guarantee that `v` is reused. However, in this
            case, additional memory is only allocated if absolutely necessary, e.g. if
            the array has the wrong dtype.
            Passing `False` here will ensure that the array `v` is not modified and an
            additional array of shape `(N)` and dtype :class:`np.float64` will always
            be allocated.

    Returns:
        Upper triangular Cholesky factor of :math:`A - v v^T` with dtype
        :class:`np.float64` and shape `(N, N)`. The diagonal entries of this matrix are
        guaranteed to be positive. The entries in the lower triangular part of this
        matrix will be the same as those in the input array `L_T`.

    Raises:
        ValueError: If `L_T` does not have shape `(N, N)` for some `N`.
        ValueError: If `v` does not have shape `(N,)`, while `L_T` has shape `(N, N)`.
        scipy.linalg.LinAlgError: If :math:`A'` is not positive definite.
        ValueError: If `L_T` has zeros among its diagonal entries.

    References:
        [1] Seeger, Matthias, Low Rank Updates for the Cholesky Decomposition, 2008.
    """

    if method == "cho_factor":
        _validate_update_args(L, v, check_diag)

        L_tril = np.tril(L)

        L_upd, _ = scipy.linalg.cho_factor(
            L_tril @ L_tril.T - np.outer(v, v),
            lower=True,
            overwrite_a=True,
        )

        L_upd[np.triu_indices(L.shape[0], k=1)] = L[np.triu_indices(L.shape[0], k=1)]
    elif method == "seeger":
        L_upd = downdate_seeger(
            L,
            v,
            check_diag=check_diag,
            overwrite_L=overwrite_L,
            overwrite_v=overwrite_v,
        )
    elif method == "seeger_cython":
        L_upd = downdate_seeger(
            L,
            v,
            check_diag=check_diag,
            overwrite_L=overwrite_L,
            overwrite_v=overwrite_v,
            impl="cython",
        )
    elif method == "seeger_python":
        L_upd = downdate_seeger(
            L,
            v,
            check_diag=check_diag,
            overwrite_L=overwrite_L,
            overwrite_v=overwrite_v,
            impl="python",
        )
    else:
        raise ValueError(f"Unknown method: '{method}'")

    return L_upd
