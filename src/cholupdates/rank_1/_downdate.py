""" Implementations of symmetric rank-1 updates to Cholesky factorized matrices. """

import numpy as np
import scipy.linalg


def downdate(
    L_T: np.ndarray,
    v: np.ndarray,
    reuse_L_T: bool = False,
    reuse_v: bool = False,
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

    if L_T.ndim != 2 or L_T.shape[0] != L_T.shape[1]:
        raise ValueError("The given Cholesky factor `L_T` is not a square matrix.")

    N = L_T.shape[0]

    if v.ndim != 1 or v.shape[0] != L_T.shape[0]:
        raise ValueError(
            f"The shape of the given vector `v` is compatible with the shape of the "
            f"given Cholesky factor `L_T`. Expected shape {(L_T.shape[0],)} but got "
            f"{v.shape}."
        )

    # Copy
    if reuse_L_T or L_T.dtype is not np.float64:
        L_T = np.array(L_T, dtype=np.float64, order="C")

    if reuse_v or v.dtype is not np.float64:
        v = np.array(v, dtype=np.float64)

    # Compute p
    scipy.linalg.blas.dtrsv(
        a=L_T,
        x=v,
        trans=1,
        overwrite_x=True,
    )

    p = v

    # Compute rho
    rho_sq = 1 - scipy.linalg.blas.ddot(p, p)

    if rho_sq <= 0.0:
        # The updated matrix is positive definite if and only if rho ** 2 is positive
        raise scipy.linalg.LinAlgError(
            "The downdate would not result in a positive definite matrix."
        )

    rho = np.sqrt(rho_sq)

    # "Create" q
    q_0 = rho
    q_1_to_n = p

    # Create temporary vector accumulating Givens rotations of the appended zero vector
    # in the augmented matrix from the left hand side of [1, equation 2]
    temp = np.zeros(N, dtype=np.float64)

    for k in range(N - 1, -1, -1):
        # Generate Givens rotation
        c, s = scipy.linalg.blas.drotg(
            q_0,
            q_1_to_n[k],
        )

        # Apply Givens rotation to q
        q_0, q_1_to_n[k] = scipy.linalg.blas.drot(q_0, q_1_to_n[k], c, s)

        # Givens rotations generated by BLAS' `drotg` might rotate q_0 to a negative
        # value. However, for the algorithm to work, it is important that q_0 remains
        # positive. As a remedy, we add another 180 degree rotation to the Givens
        # rotation matrix. This flips the sign of q_0 while ensuring that the resulting
        # transformation is still a Givens rotation.
        if q_0 < 0.0:
            q_0 = -q_0
            c = -c
            s = -s

        # Apply (possibly modified) Givens rotation to the augmented matrix [0 L]^T
        if L_T[k, k] == 0.0:
            # This can only happen if L_T is not an upper triangular matrix with
            # non-zero diagonal
            raise ValueError(
                "The given Cholesky factor `L_T` does not have a non-zero diagonal."
            )

        scipy.linalg.blas.drot(
            temp[k:], L_T[k, k:], c, -s, overwrite_x=True, overwrite_y=True
        )

        # Applying the Givens rotation might lead to a negative diagonal element in L_T.
        # However, by convention, the diagonal entries of a Cholesky factor are
        # positive. As a remedy, we simply rescale the whole row. Note that this is
        # possible, since rescaling a row is equivalent to a mirroring along one
        # dimension which is in turn an orthogonal transformation.
        if L_T[k, k] < 0.0:
            L_T[k, k:] = -L_T[k, k:]

    return L_T
