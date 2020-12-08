""" Implementations of symmetric rank-1 downdates to Cholesky factorized matrices. """

import numpy as np
import scipy.linalg


def rank_1_update(
    L: np.ndarray,
    v: np.ndarray,
    check_diag: bool = True,
    overwrite_L: bool = False,
    overwrite_v: bool = False,
) -> np.ndarray:
    r"""Compute the Cholesky factorization of a symmetric rank-1 update to a symmetric
    positive definite matrix with given Cholesky factorization in a fast and stable
    manner.

    Specifically, given the lower triangular Cholesky factor :math:`L` of
    :math:`A = L L^T \in \mathbb{R}^{N \times N}`, this function computes the lower
    triangular Cholesky factor :math:`L'` of :math:`A' = L' L'^T`, where
    :math:`A' = A + v v^T` for some vector :math:`v \in \mathbb{R}^n`.

    This function computes the Cholesky decomposition of :math:`A'` from :math:`L` in
    :math:`O(N^2)` time, which is faster than the :math:`O(N^3)` time complexity of
    naively applying a Cholesky algorithm to :math:`A'` directly.

    Parameters
    ----------
    L :
        Lower triangular Cholesky factor of :math:`A` with shape :code:`(N, N)` and
        dtype :class:`numpy.float64`.
        Must have a non-zero diagonal.
        The algorithm is most efficient if this array is given in column-major layout,
        a.k.a. Fortran-contiguous or f-contiguous memory order. Hint: Lower triangular
        Cholesky factors can be obtained efficiently (i.e. without requiring an
        additional copy) from :func:`scipy.linalg.cho_factor`.
        The entries in the strict upper triangular part of :code:`L` can contain
        arbitrary values, since the algorithm neither reads from nor writes to this part
        of the matrix. This behavior is useful when using the Cholesky factors returned
        by :func:`scipy.linalg.cho_factor` which contain arbitrary values on the
        irrelevant triangular part of the matrix.
    v :
        The vector :math:`v` with shape :code:`(N, N)` and dtype :class:`numpy.float64`
        defining the symmetric rank-1 matrix :math:`v v^T`.
    check_diag :
        If set to :code:`True`, the function will check whether the diagonal of the
        given Cholesky factor :code:`L` is non-zeros and raise a :class:`ValueError` if
        this is not the case.
        Setting :code:`check_diag` to `False` can be used to speed up computations if
        it is clear that the Cholesky factor can not have zeros on its diagonal.
        Caution: If this argument is set to :code:`False` and the Cholesky factor does
        contain zeros on its diagonal, the output will be undefined.
    overwrite_L :
        If set to :code:`True`, the function will overwrite the array :code:`L` with the
        upper Cholesky factor :math:`L'` of :math:`A'`, i.e. the result is computed
        in-place.
        Passing `False` here ensures that the array :code:`L` is not modified.
    overwrite_v :
        If set to `True`, the function will reuse the array :code:`v` as an internal
        computation buffer, which will modify :code:`v`.
        Passing `False` here ensures that the array :code:`v` is not modified.
        In this case, an additional array of shape :code:`(N,)` and dtype
        :class:`numpy.float64` must be allocated.

    Returns
    -------
        Lower triangular Cholesky factor :math:`L'` of :math:`A + v v^T = L' L'^T` with
        shape :code:`(N, N)` and dtype :class:`numpy.float64`.
        The diagonal entries of this matrix are guaranteed to be positive.
        The strict upper triangular part of this matrix will contain arbitrary values.
        The matrix will inherit the memory order from :code:`L`.

    Raises
    ------
    ValueError
        If :code:`L` does not have shape :code:`(N, N)` for some :code:`N`.
    TypeError
        If :code:`L` does not have dtype :class:`numpy.float64`.
    numpy.linalg.LinAlgError
        If the diagonal of :code:`L` contains zeros and :code:`check_diag` is set to
        true.
    ValueError
        If :code:`v` does not have shape :code:`(N,)`, while :code:`L` has shape
        :code:`(N, N)`.
    TypeError
        If :code:`v` does not have dtype :class:`numpy.float64`.

    See Also
    --------
    cholupdates.rank_1_downdate : A similar function which performs a symmetric rank 1
        downdate instead of an update.

    Notes
    -----
    This function implements the algorithm from [1]_, section 2.
    In the following, we will briefly summarize the theory behind the algorithm.
    Let :math:`A \in \mathbb{R}^{n \times n}` be symmetric and positive definite, and
    :math:`v \in \mathbb{R}^n`.
    The vector :math:`v` defines a symmetric rank-1 update :math:`v v^T` to :math:`A`,
    i.e.

    .. math::
       A' := A + v v^T.

    Assume that the Cholesky factorization of :math:`A`, i.e. a lower-triangular matrix
    :math:`L \in \mathbb{R}^{n \times n}` with :math:`A = L L^T`, is given.
    We want to find a fast and stable method to update the Cholesky factorization
    :math:`L L^T` of :math:`A` to the Cholesky factorization of the updated matrix
    :math:`A'`.

    To this end, we rewrite the update equation as

    .. math::
        A'
        =
        \underbrace{L L^T}_{= A} + v v^T
        =
        \begin{pmatrix}
            L & v
        \end{pmatrix}
        \begin{pmatrix}
            L^T \\
            v^T
        \end{pmatrix}.

    We can now construct a sequence of orthogonal transformations (we use Givens
    rotations) :math:`Q := Q_1 \dotsb Q_n \in \mathbb{R}^{(n + 1) \times (n + 1)}` such
    that

    .. math::
        \begin{pmatrix}
            L & v
        \end{pmatrix}
        Q
        =
        \begin{pmatrix}
            L' & 0
        \end{pmatrix},

    where :math:`L' \in \mathbb{R}^{n \times n}` is lower triangular.
    In other words, :math:`Q` eliminates the last column, i.e. :math:`v`, from the
    augmented matrix :math:`(L \mid v)`, while preserving the lower-triangular structure
    of the left block.
    But now we have

    .. math::
        L' L'^T
        & =
        \begin{pmatrix}
            L' & 0
        \end{pmatrix}
        \begin{pmatrix}
            L' & 0
        \end{pmatrix}^T \\
        & =
        \begin{pmatrix}
            L & v
        \end{pmatrix}
        \underbrace{Q Q^T}_{= I}
        \begin{pmatrix}
            L^T \\
            v^T
        \end{pmatrix} \\
        & = \underbrace{L L^T}_{= A} + v v^T \\
        & = A',

    so :math:`L'` is already the lower-triangular Cholesky factor of :math:`A'`.

    As mentioned above, we need to multiply :math:`n` Givens rotation matrices to
    :math:`(L \mid v)` which takes :math:`O(n)` arithmetic operations for each rotation
    matrix.
    Hence, we end up with a total time complexity of :math:`O(n^2)`.

    References
    ----------
    .. [1] M. Seeger, "Low Rank Updates for the Cholesky Decomposition", 2008.

    Examples
    --------
    Consider the following matrix-vector pair

    >>> A = np.diag([1.0, 2.0, 3.0]) + 0.1
    >>> A
    array([[1.1, 0.1, 0.1],
           [0.1, 2.1, 0.1],
           [0.1, 0.1, 3.1]])
    >>> v = np.array([1.0, 25.0, 10.0])
    >>> v
    array([ 1., 25., 10.])

    We want to compute the lower triangular Cholesky factor :code:`L_prime` of

    >>> A_prime = A + np.outer(v, v)
    >>> A_prime
    array([[  2.1,  25.1,  10.1],
           [ 25.1, 627.1, 250.1],
           [ 10.1, 250.1, 103.1]])

    We assume that the lower triangular Cholesky factor of :code:`A` is given

    >>> L = scipy.linalg.cho_factor(A, lower=True)[0]
    >>> np.tril(L)
    array([[1.04880885, 0.        , 0.        ],
           [0.09534626, 1.44599761, 0.        ],
           [0.09534626, 0.06286946, 1.75697368]])

    The function :func:`cholesky_rank_1_update` can compute :code:`L_prime` from
    :code:`L` efficiently

    >>> import cholupdates
    >>> L_prime = cholupdates.rank_1_update(L, v)
    >>> np.tril(L_prime)
    array([[ 1.44913767,  0.        ,  0.        ],
           [17.32064554, 18.08577447,  0.        ],
           [ 6.96966215,  7.15374133,  1.82969791]])

    Did it work?

    >>> np.allclose(A_prime, np.tril(L_prime) @ np.tril(L_prime).T)
    True

    We could also compute :code:`L_prime` by directly computing the Cholesky
    factorization of :code:`A_prime` (which is however less efficient)

    >>> L_prime_cho = scipy.linalg.cho_factor(A_prime, lower=True)[0]
    >>> np.tril(L_prime_cho)
    array([[ 1.44913767,  0.        ,  0.        ],
           [17.32064554, 18.08577447,  0.        ],
           [ 6.96966215,  7.15374133,  1.82969791]])
    >>> np.allclose(np.tril(L_prime), np.tril(L_prime_cho))
    True
    """

    # Validate L
    if L.ndim != 2 or L.shape[0] != L.shape[1]:
        raise ValueError(
            f"The given Cholesky factor `L_T` is not a square matrix (given shape: "
            f"{L.shape})."
        )

    if L.dtype != np.float64:
        raise TypeError(
            f"The given Cholesky factor `L_T` does not have dtype `np.float64` (given "
            f"dtype: {L.dtype.name})"
        )

    if check_diag:
        if np.any(np.diag(L) == 0):
            raise np.linalg.LinAlgError(
                "The given Cholesky factor `L` contains zeros on its diagonal"
            )

    # Validate v
    if v.ndim != 1 or v.shape[0] != L.shape[0]:
        raise ValueError(
            f"The shape of the given vector `v` is compatible with the shape of the "
            f"given Cholesky factor `L_T`. Expected shape {(L.shape[0],)} but got "
            f"{v.shape}."
        )

    if v.dtype != np.float64:
        raise TypeError(
            f"The given vector `v` does not have dtype `np.float64` (given dtype: "
            f"{L.dtype.name})"
        )

    # Copy on demand
    if not overwrite_L:
        L = L.copy(order="K")

    if not overwrite_v:
        v = v.copy()

    # The algorithm updates the transpose of L instead of L itself
    L_T = L.T

    # L and L_T now refer to the same memory buffer, so updating L_T implicitly updates
    # L correspondingly
    assert not L_T.flags.owndata
    assert np.may_share_memory(L, L_T)

    if L_T.flags.c_contiguous:
        assert L.flags.f_contiguous

        _rank_1_update_row_major(L_T, v)
    else:
        assert L.flags.c_contiguous
        assert L_T.flags.f_contiguous

        _cholesky_rank_1_update_column_major(L_T, v)

    return L


def _rank_1_update_row_major(L_T: np.ndarray, v: np.ndarray) -> None:
    N = L_T.shape[0]

    for k in range(N):
        # Generate Givens rotation
        c, s = scipy.linalg.blas.drotg(L_T[k, k], v[k])

        # Apply Givens rotation to diagonal term and corresponding entry
        L_T[k, k], v[k] = scipy.linalg.blas.drot(L_T[k, k], v[k], c, s)

        # Givens rotations generated by BLAS' `drotg` might rotate the diagonal entry to
        # a negative value. However, by convention, the diagonal entries of a Cholesky
        # factor are positive. As a remedy, we add another 180 degree rotation to the
        # Givens rotation matrix. This flips the sign of the diagonal entry while
        # ensuring that the resulting transformation is still a Givens rotation.
        if L_T[k, k] < 0.0:
            L_T[k, k] = -L_T[k, k]
            c = -c
            s = -s

        # Apply (modified) Givens rotation to the remaining entries of L_T and v
        if k + 1 < N:
            scipy.linalg.blas.drot(
                L_T[k, (k + 1) :],
                v[(k + 1) :],
                c,
                s,
                overwrite_x=True,
                overwrite_y=True,
            )


def _cholesky_rank_1_update_column_major(L_T: np.ndarray, v: np.ndarray) -> None:
    N = L_T.shape[0]

    row_inc = 1
    column_inc = N

    k = 0

    drot_n = N - 1
    drot_L_T = L_T.ravel("F")
    drot_off_L_T = column_inc
    drot_inc_L_T = column_inc
    drot_v = v
    drot_offv = 1
    drot_incv = 1

    while k < N:
        # Generate Givens rotation
        c, s = scipy.linalg.blas.drotg(L_T[k, k], v[k])

        # Apply Givens rotation to diagonal term and corresponding entry
        L_T[k, k], v[k] = scipy.linalg.blas.drot(L_T[k, k], v[k], c, s)

        # Givens rotations generated by BLAS' `drotg` might rotate the diagonal entry to
        # a negative value. However, by convention, the diagonal entries of a Cholesky
        # factor are positive. As a remedy, we add another 180 degree rotation to the
        # Givens rotation matrix. This flips the sign of the diagonal entry while
        # ensuring that the resulting transformation is still a Givens rotation.
        if L_T[k, k] < 0.0:
            L_T[k, k] = -L_T[k, k]
            c = -c
            s = -s

        # Apply (modified) Givens rotation to the remaining entries of L_T and v
        if drot_n > 0:
            scipy.linalg.blas.drot(
                n=drot_n,
                x=drot_L_T,
                offx=drot_off_L_T,
                incx=drot_inc_L_T,
                y=drot_v,
                offy=drot_offv,
                incy=drot_incv,
                c=c,
                s=s,
                overwrite_x=True,
                overwrite_y=True,
            )

        k += 1

        drot_n -= 1
        drot_off_L_T += row_inc + column_inc
        drot_offv += 1
