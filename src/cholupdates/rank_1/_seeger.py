"""Interface functions for the rank-1 up- and downdate algorithms specified in sections
2 and 3 of [1]_.

References
----------
.. [1] M. Seeger, "Low Rank Updates for the Cholesky Decomposition", 2008.
"""

from typing import Optional

import numpy as np

from ._arg_validation import _validate_update_args
from ._seeger_impl_python import update as _update_impl_python

_update_available_impls = ["python"]

try:
    from ._seeger_impl_cython import update as _update_impl_cython

    _cython_available = True
    _update_available_impls.append("cython")

    _update_impl_default = _update_impl_cython
except ImportError:
    _cython_available = False

    _update_impl_default = _update_impl_python


def update_seeger(
    L: np.ndarray,
    v: np.ndarray,
    check_diag: bool = True,
    overwrite_L: bool = False,
    overwrite_v: bool = False,
    impl: Optional[str] = None,
) -> np.ndarray:
    r"""Update a Cholesky factorization after addition of a positive semidefinite
    symmetric rank-1 matrix using the algorithm from section 2 of [1]_.

    In other words, given :math:`A = L L^T \in \mathbb{R}^{N \times N}` and
    :math:`v \in \mathbb{R}^N`, compute :math:`L'` such that

    .. math::
        A' := A + v v^T = L' L'^T.

    This function computes the Cholesky factorization of :math:`A'` from :math:`L` in
    :math:`O(N^2)` time, which is faster than the :math:`O(N^3)` time complexity of
    naively applying a Cholesky algorithm to :math:`A'` directly.

    Parameters
    ----------
    L : (N, N) numpy.ndarray, dtype=numpy.double
        Lower-triangular Cholesky factor of :math:`A`.
        Must have a non-zero diagonal.
        The algorithm is most efficient if this array is given in column-major layout,
        a.k.a. Fortran-contiguous or f-contiguous memory order. Hint: Lower-triangular
        Cholesky factors in column-major memory layout can be obtained efficiently (i.e.
        without requiring an additional copy) from :func:`scipy.linalg.cho_factor`.
        The entries in the strict upper-triangular part of :code:`L` can contain
        arbitrary values, since the algorithm neither reads from nor writes to this part
        of the matrix. This behavior is useful when using the Cholesky factors returned
        by :func:`scipy.linalg.cho_factor` which contain arbitrary values on the
        irrelevant triangular part of the matrix.
    v : (N,) numpy.ndarray, dtype=numpy.double
        The vector :math:`v` which defines the symmetric rank-1 update matrix
        :math:`v v^T`.
    check_diag :
        If set to :code:`True`, the function will check whether the diagonal of the
        given Cholesky factor :code:`L` is non-zero and raise a :class:`ValueError` if
        this is not the case.
        Setting :code:`check_diag` to :code:`False` can be used to speed up computations
        if it is clear that the Cholesky factor can not have zeros on its diagonal.
        Caution: If this argument is set to :code:`False` and the Cholesky factor does
        contain zeros on its diagonal, the behavior of the function will be undefined.
    overwrite_L :
        If set to :code:`True`, the function will overwrite the array :code:`L` with the
        upper Cholesky factor :math:`L'` of :math:`A'`, i.e. the result is computed
        in-place.
        Passing :code:`False` here ensures that the array :code:`L` is not modified.
    overwrite_v :
        If set to :code:`True`, the function will reuse the array :code:`v` as an
        internal computation buffer, which will modify :code:`v`.
        Passing :code:`False` here ensures that the array :code:`v` is not modified.
        In this case, an additional array of shape :code:`(N,)` and dtype
        :class:`numpy.double` must be allocated.
    impl :
        Defines which implementation of the algorithm to use. Must be one of

        - :class:`None`
            Choose the Cython implementation if it is available, otherwise use the
            Python implementation.
        - "cython"
            Use the Cython implementation. Throws a :class:`ValueError` if the Cython
            implementation is not available.
        - "python"
            Use the Python implementation.

        Defaults to None.

    Returns
    -------
    (N, N) numpy.ndarray, dtype=numpy.double
        Lower-triangular Cholesky factor :math:`L'` of :math:`A + v v^T` with shape
        :code:`(N, N)` and dtype :class:`numpy.double`.
        The diagonal entries of this matrix are guaranteed to be positive.
        The strict upper-triangular part of this matrix will contain the values from the
        upper-triangular part of :code:`L`.
        The matrix will inherit the memory order from :code:`L`.

    Raises
    ------
    ValueError
        If :code:`L` does not have shape :code:`(N, N)` for some :code:`N`.
    numpy.linalg.LinAlgError
        If the diagonal of :code:`L` contains zeros and :code:`check_diag` is set to
        true.
    ValueError
        If :code:`v` does not have shape :code:`(N,)`, while :code:`L` has shape
        :code:`(N, N)`.
    TypeError
        If :code:`L` does not have dtype :class:`numpy.double`.
    TypeError
        If :code:`v` does not have dtype :class:`numpy.double`.
    ValueError
        If :code:`impl` was set to :code:`"cython"`, but the Cython implementation is
        not available.

    See Also
    --------
    cholupdates.rank_1.update : Interface function for all rank-1 update methods. Can
        be used to call this function conveniently.

    Notes
    -----
    This function implements the algorithm from [1]_, section 2.
    In the following, we will briefly summarize the theory behind the algorithm.
    Let :math:`A \in \mathbb{R}^{n \times n}` be symmetric and positive-definite, and
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

    We can now construct a sequence of orthogonal transformations
    :math:`Q := Q_1 \dotsb Q_n \in \mathbb{R}^{(n + 1) \times (n + 1)}`, in this case
    Givens rotations, such that

    .. math::
        \begin{pmatrix}
            L & v
        \end{pmatrix}
        Q
        =
        \begin{pmatrix}
            L' & 0
        \end{pmatrix},

    where :math:`L' \in \mathbb{R}^{n \times n}` is lower-triangular.
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

    Note that this algorithm is a minor modification of the `LINPACK` [2]_ routine
    :code:`dchud`. The exact modifications are described in [1]_.

    References
    ----------
    .. [1] M. Seeger, "Low Rank Updates for the Cholesky Decomposition", 2008.
    .. [2] J. Dongarra, C. Moler, J. Bunch, and G. Steward. "LINPACK User's Guide".
        Society for Industrial and Applied Mathematics, 1979.
    """

    # Validate arguments
    _validate_update_args(L, v, check_diag)

    if L.dtype != np.double:
        raise TypeError(
            f"The given Cholesky factor `L` does not have dtype `np.double` (given "
            f"dtype: {L.dtype.name})"
        )

    if v.dtype != np.double:
        raise TypeError(
            f"The given vector `v` does not have dtype `np.double` (given dtype: "
            f"{L.dtype.name})"
        )

    # Copy on demand
    if not overwrite_L:
        L = L.copy(order="K")

    if not overwrite_v:
        v = v.copy()

    # Update L
    if impl is None:
        _update_impl_default(L, v)
    elif impl == "cython":
        try:
            _update_impl_cython(L, v)
        except NameError as ne:
            raise ValueError("The Cython implementation is not available.") from ne
    elif impl == "python":
        _update_impl_python(L, v)
    else:
        raise ValueError(
            f"Unknown implementation '{impl}'. Available implementations: "
            f"{', '.join(_update_available_impls)}"
        )

    return L


update_seeger.available_impls = _update_available_impls
