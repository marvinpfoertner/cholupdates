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

    # Update L
    _seeger_rank_1_update_inplace(L, v)

    return L


def _seeger_rank_1_update_inplace(L: np.ndarray, v: np.ndarray) -> None:
    """Implementation of the rank-1 Cholesky update algorithm from section 2 in [1]_.

    Warning: The validity of the arguments will not be checked by this method, so
    passing invalid argument will result in undefined behavior.

    Parameters
    ----------
    L :
        The lower-triangular Cholesky factor of the matrix to be updated.
        Must have shape `(N, N)` and dtype `np.float64`.
        Must not contain zeros on the diagonal.
        The entries in the strict upper triangular part of :code:`L` can contain
        arbitrary values, since the algorithm neither reads from nor writes to this part
        of the matrix
        Will be overridden with the Cholesky factor of the matrix to be updated.
    v :
        The vector :math:`v` with shape :code:`(N, N)` and dtype :class:`numpy.float64`
        defining the symmetric rank-1 update :math:`v v^T`.

    References
    ----------
    .. [1] M. Seeger, "Low Rank Updates for the Cholesky Decomposition", 2008.
    """
    N = L.shape[0]

    # Generate a contiguous view of the underling memory buffer of L, emulating raw
    # pointer access
    L_buf = L.ravel(order="K")

    assert np.may_share_memory(L, L_buf)

    if L.flags.f_contiguous:
        # In column-major memory layout, moving to the next row means moving the pointer
        # by 1 entry, while moving to the next column means moving the pointer by N
        # entries, i.e. the number of entries per column
        row_inc = 1
        column_inc = N
    else:
        assert L.flags.c_contiguous

        # In row-major memory layout, moving to the next column means moving the pointer
        # by 1 entry, while moving to the next row means moving the pointer by N
        # entries, i.e. the number of entries per row
        row_inc = N
        column_inc = 1

    # Create a "pointer" into the contiguous view of L's memory buffer
    # Points to the k-th diagonal entry of L at the beginning of the loop body
    L_buf_off = 0

    for k in range(N):
        # At this point L/L_buf contains a lower triangular matrix and the first k
        # entries of v are zeros

        # Generate Givens rotation which eliminates the k-th entry of v by rotating onto
        # the k-th diagonal entry of L and apply it only to these entries of (L|v)
        # Note: The following two operations will be performed by a single call to
        # `drotg` in C/Fortran. However, Python can not modify `Float` arguments.
        c, s = scipy.linalg.blas.drotg(L_buf[L_buf_off], v[k])
        L_buf[L_buf_off], v[k] = scipy.linalg.blas.drot(L_buf[L_buf_off], v[k], c, s)

        # Givens rotations generated by BLAS' `drotg` might rotate the diagonal entry to
        # a negative value. However, by convention, the diagonal entries of a Cholesky
        # factor are positive. As a remedy, we add another 180 degree rotation to the
        # Givens rotation matrix. This flips the sign of the diagonal entry while
        # ensuring that the resulting transformation is still a Givens rotation.
        if L_buf[L_buf_off] < 0.0:
            L_buf[L_buf_off] = -L_buf[L_buf_off]
            c = -c
            s = -s

        # Apply (modified) Givens rotation to the remaining entries in the k-th column
        # of L and the remaining entries in v

        # The first k entries in the k-th column of L are zero, since L is lower
        # triangular, so we only need to consider indices larger than or equal to i
        i = k + 1

        if i < N:
            # Move the pointer to the entry of L at index (i, k)
            L_buf_off += row_inc

            scipy.linalg.blas.drot(
                # We only need to rotate the last N - i entries
                n=N - i,
                # This constructs the memory adresses of the last N - i entries of the
                # k-th column in L
                x=L_buf,
                offx=L_buf_off,
                incx=row_inc,
                # This constructs the memory adresses of the last N - i entries of v
                y=v,
                offy=i,
                incy=1,
                c=c,
                s=s,
                overwrite_x=True,
                overwrite_y=True,
            )

            # In the beginning of the next iteration, the buffer offset must point to
            # the (k + 1)-th diagonal entry of L
            L_buf_off += column_inc
