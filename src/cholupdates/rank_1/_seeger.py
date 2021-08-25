"""Interface functions for the rank-1 up- and downdate algorithms specified in sections
2 and 3 of [1]_.

References
----------
.. [1] M. Seeger, "Low Rank Updates for the Cholesky Decomposition", 2008.
"""

from typing import Optional

import numpy as np

from ._arg_validation import _validate_update_args
from ._seeger_impl_python import downdate as _downdate_impl_python
from ._seeger_impl_python import update as _update_impl_python

# Check if Cython implementations are available
_update_available_impls = ["python"]
_update_impl_default = _update_impl_python

_downdate_available_impls = ["python"]
_downdate_impl_default = _downdate_impl_python

try:
    from ._seeger_impl_cython import downdate as _downdate_impl_cython
    from ._seeger_impl_cython import update as _update_impl_cython

    _update_available_impls.append("cython")
    _update_impl_default = _update_impl_cython

    _downdate_available_impls.append("cython")
    _downdate_impl_default = _downdate_impl_cython
except ImportError:  # pragma: no cover
    pass


def update_seeger(
    L: np.ndarray,
    v: np.ndarray,
    check_diag: bool = True,
    overwrite_L: bool = False,
    overwrite_v: bool = False,
    impl: Optional[str] = None,
) -> np.ndarray:
    r"""Update a Cholesky factorization after addition of a positive semidefinite
    symmetric rank-1 matrix using the algorithm from section 2 of [Seeger, 2008].

    In other words, given :math:`A = L L^T \in \mathbb{R}^{N \times N}` and
    :math:`v \in \mathbb{R}^N`, compute :math:`L^+` such that

    .. math::
        A^+ := A + v v^T = L^+ (L^+)^T.

    This function computes the Cholesky factorization of :math:`A^+` from :math:`L` in
    :math:`O(N^2)` time, which is faster than the :math:`O(N^3)` time complexity of
    naively applying a Cholesky algorithm to :math:`A^+` directly.

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
        upper Cholesky factor :math:`L^+` of :math:`A^+`, i.e. the result is computed
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
        Lower-triangular Cholesky factor :math:`L^+` of :math:`A + v v^T` with shape
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
        A^+ := A + v v^T.

    Assume that the Cholesky factorization of :math:`A`, i.e. a lower-triangular matrix
    :math:`L \in \mathbb{R}^{n \times n}` with :math:`A = L L^T`, is given.
    We want to find a fast and stable method to update the Cholesky factorization
    :math:`L L^T` of :math:`A` to the Cholesky factorization of the updated matrix
    :math:`A^+`.

    To this end, we rewrite the update equation as

    .. math::
        A^+
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
            L^+ & 0
        \end{pmatrix},

    where :math:`L^+ \in \mathbb{R}^{n \times n}` is lower-triangular.
    In other words, :math:`Q` eliminates the last column, i.e. :math:`v`, from the
    augmented matrix :math:`(L \mid v)`, while preserving the lower-triangular structure
    of the left block.
    But now we have

    .. math::
        L^+ (L^+)^T
        & =
        \begin{pmatrix}
            L^+ & 0
        \end{pmatrix}
        \begin{pmatrix}
            L^+ & 0
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
        & = A^+,

    so :math:`L^+` is already the lower-triangular Cholesky factor of :math:`A^+`.

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

    if L.dtype not in (np.single, np.double):
        raise TypeError(
            f"The given Cholesky factor `L` does not have dtype `np.double` (given "
            f"dtype: {L.dtype.name})"
        )

    if v.dtype not in (np.single, np.double):
        raise TypeError(
            f"The given vector `v` does not have dtype `np.double` (given dtype: "
            f"{L.dtype.name})"
        )

    if L.dtype != v.dtype:
        raise TypeError  # TODO: Error message

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
        except NameError as ne:  # pragma: no cover
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


def downdate_seeger(
    L: np.ndarray,
    v: np.ndarray,
    check_diag: bool = True,
    overwrite_L: bool = False,
    overwrite_v: bool = False,
    impl: Optional[str] = None,
) -> np.ndarray:
    r"""Update a Cholesky factorization after subtraction of a positive semidefinite
    symmetric rank-1 matrix using the algorithm from section 3 of [Seeger, 2008].

    In other words, given :math:`A = L L^T \in \mathbb{R}^{N \times N}`, and
    :math:`v \in \mathbb{R}^N`, compute :math:`L^-` such that

    .. math::
        A^- := A - v v^T = L^- (L^-)^T.

    Note that the Cholesky factorization of the downdated matrix may not exist, since a
    downdate might result in a matrix which is not positive definite.

    This function computes the Cholesky factorization of :math:`A^-` from :math:`L` in
    :math:`O(N^2)` time, which is faster than the :math:`O(N^3)` time complexity of
    naively applying a Cholesky algorithm to :math:`A^-` directly.

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
        The vector :math:`v` which defines the symmetric rank-1 downdate matrix
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
        upper Cholesky factor :math:`L^-` of :math:`A^-`, i.e. the result is computed
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
        Lower-triangular Cholesky factor :math:`L^-` of :math:`A - v v^T` with shape
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
    numpy.linalg.LinAlgError
        If the downdate results in a matrix :math:`L^-`, which is not positive definite.

    See Also
    --------
    cholupdates.rank_1.downdate : Interface function for all rank-1 downdate methods.
        Can be used to call this function conveniently.

    Notes
    -----
    This function implements the algorithm from [1]_, section 3.
    In the following, we will elaborate on the theory behind the algorithm.
    Let :math:`A \in \mathbb{R}^{n \times n}` be symmetric and positive-definite, and
    :math:`v \in \mathbb{R}^n`.
    The vector :math:`v` defines a symmetric rank-1 downdate :math:`-v v^T` to
    :math:`A`, i.e.

    .. math::
        A^- := A - v v^T.

    Note that :math:`A^-` need not be positive definite.
    In the following, we will derive an efficient criterion to check whether :math:`A^-`
    is positive definite.
    For the derivation of the algorithm, we assume that :math:`A^-` is positive
    definite.

    Assume that the Cholesky factorization of :math:`A`, i.e. a lower-triangular matrix
    :math:`L \in \mathbb{R}^{n \times n}` with :math:`A = L L^T`, is given.
    We want to find a fast and stable method to update the Cholesky factorization
    :math:`L L^T` of :math:`A` to the Cholesky factorization :math:`A^- = L^- (L^-)^T`
    of the updated matrix :math:`A^-`, where, again,
    :math:`L^- \in \mathbb{R}^{n \times n}` is lower triangular.

    If we can construct an orthogonal transformation
    :math:`Q \in \mathbb{R}^{(n + 1) \times (n + 1)}` such that

    .. math::
        :label: Q-constraint-L

        \begin{pmatrix}
            L & 0
        \end{pmatrix}
        Q^T
        =
        \begin{pmatrix}
            L^- & v
        \end{pmatrix},

    for some lower-triangular matrix :math:`L^-`, then

    .. math::
        L^- (L^-)^T + v v^T
        =
        \begin{pmatrix}
            L^- & v
        \end{pmatrix}
        \begin{pmatrix}
            L^- & v
        \end{pmatrix}^T
        =
        \begin{pmatrix}
            L & 0
        \end{pmatrix}
        \underbrace{Q^T Q}_{= I}
        \begin{pmatrix}
            L^T \\
            0
        \end{pmatrix}
        = L L^T,

    which is equivalent to :math:`L^- (L^-)^T = L L^T - v v^T`.
    This means that we can retrieve the updated Cholesky factor as
    :math:`L^- = L \cdot Q_{1:n, 1:n}^T`.

    In the remainder of this section, we will derive another constraint on :math:`Q`
    which will directly give rise to an algorithm to compute :math:`Q`.

    If we multiply equation :eq:`Q-constraint-L` with the standard basis vector
    :math:`e_{n + 1} \in \mathbb{R}^{n + 1}` from the right-hand side, we arrive at

    .. math::
        \begin{pmatrix}
            L & 0
        \end{pmatrix}
        \underbrace{Q^T e_{n + 1}}_{=: q}
        =
        v.

    With :math:`p := q_{1:n} \in \mathbb{R}^n`, this implies :math:`L p = v`, i.e.
    :math:`p = L^{-1} v`, since :math:`L` is invertible.

    The vector :math:`p` can be used to cheaply check whether the downdate results in a
    positive definite matrix, since :math:`p^T p < 1` if and only if :math:`A^-` is
    positive definite.
    To see this, note that :math:`L` is invertible and that :math:`A^-` is hence
    positive definite if and only if

    .. math::
        L^{-1} A^- L^{-T}
        = L^{-1} (L L^T - v v^T) L^{-T}
        = I - (L^{-1} v) (v^T L^{-T})
        = I - p p^T

    is positive definite.
    One can show that the eigenvectors of :math:`I - p p^T` are given by any :math:`u`
    in the orthogonal complement of :math:`p` with eigenvalue :math:`1 > 0` and
    :math:`p` with eigenvalue :math:`1 - p^T p`.
    Hence, :math:`I - p p^T` is positive definite if and only if :math:`p^T p < 1`.

    In order to find :math:`q_{n + 1}`, we note that

    .. math::
        \lVert q \rVert_2^2
        = e_{n + 1}^T \underbrace{Q Q^T}_{= I} e_{n + 1}
        = e_{n + 1}^T e_{n + 1}
        = 1,

    which implies

    .. math::
        1
        =\lVert q \rVert_2^2
        = p^T p + q_{n + 1}^2
        \quad \Leftrightarrow \quad
        q_{n + 1}^2 = 1 - p^T p.

    Note that this is well-defined, since :math:`A^-` is assumed to be positive
    definite, which means that :math:`p^T p < 1`, i.e. :math:`1 - p^T p > 0`.
    All in all, our additional constraint on :math:`Q` can be formulated as

    .. math::
        :label: Q-constraint-q

        Q q = e_{n + 1}
        \quad
        \text{with}
        \quad
        q =
        \begin{pmatrix}
            p \\
            \rho
        \end{pmatrix},
        \quad
        L p = v,
        \quad
        \text{and}
        \quad
        \rho^2 := 1 - p^T p.

    From this constraint, we can now generate the desired orthogonal matrix :math:`Q` in
    a principled way.
    Obviously, there are multiple different orthogonal matrices which fulfill
    :eq:`Q-constraint-q`.
    However, our particular choice must also fulfill :eq:`Q-constraint-L`.
    One particular way to construct :math:`Q` is as a sequence of Givens rotations
    :math:`Q = Q_1 \dotsb Q_n`, where :math:`Q_i` eliminates the :math:`i`-th entry of

    .. math::
        q^{(i + 1)} = \left( \prod_{j = i + 1}^n Q_j \right) q

    with :math:`q^{(i + 1)}_{n + 1}`, i.e. :math:`Q_i` transforms :math:`q^{(i + 1)}`
    to :math:`q^{(i)}` with :math:`q^{(i)}_i = 0` and

    .. math::
        q^{(i)}_{n + 1}
        = \sqrt{(q^{(i + 1)}_i)^2 + (q^{(i + 1)}_{n + 1})^2},

    while preserving all other entries.

    Obviously, this choice of :math:`Q` fulfills :eq:`Q-constraint-q` and we can show by
    induction that it also fulfills :eq:`Q-constraint-L`.
    Denote

    .. math::
        \begin{pmatrix}
            L^{(i)} & v^{(i)}
        \end{pmatrix}
        & :=
        \begin{pmatrix}
            L & 0
        \end{pmatrix}
        \left( Q_i \dotsb Q_n \right)^T
        \\
        & =
        \begin{pmatrix}
            L & 0
        \end{pmatrix}
        Q_n^T \dotsb Q_i^T
        \\
        & =
        \begin{pmatrix}
            L^{(i + 1)} & v^{(i + 1)}
        \end{pmatrix}
        Q_i^T.

    For :math:`i = n + 1`, we have :math:`L^{(i)} = L`, which is lower-triangular, and
    :math:`v^{(i)} = 0`, i.e. the first :math:`i - 1` components of :math:`v^{(i)}`
    contain zeros.
    Now assume that :math:`L^{(i + 1)}` is lower triangular and that the first
    :math:`(i + 1) - 1` components of :math:`v^{(i + 1)}` are zeros.

    .. math::
        &
        \begin{pmatrix}
            L^{(i)} & v^{(i)}
        \end{pmatrix} \\
        = \ &
        \begin{pmatrix}
            L^{(i + 1)} & v^{(i + 1)}
        \end{pmatrix}
        Q_i^T \\
        = \ &
        \begin{pmatrix}
            L_{1:(i - 1), 1:(i - 1)}^{(i + 1)} & 0 & 0 & 0 \\
            L_{i, 1:(i - 1)}^{(i + 1)} & L_{ii}^{(i + 1)} & 0 & 0 \\
            L_{(i + 1):n, 1:i}^{(i + 1)} & L_{(i + 1):n, i}^{(i + 1)} &
            L_{(i + 1):n, (i + 1):n}^{(i + 1)} & v_{(i + 1):n}^{(i + 1)} \\
        \end{pmatrix}
        \cdot
        \begin{pmatrix}
            I &  0   & 0 & 0 \\
            0 &  c_i & 0 & s_i \\
            0 &  0   & I & 0 \\
            0 & -s_i & 0 & c_i \\
        \end{pmatrix} \\
        = \ &
        \begin{pmatrix}
            L_{1:(i - 1), 1:(i - 1)}^{(i + 1)} & 0 & 0 & 0 \\
            L_{i, 1:(i - 1)}^{(i + 1)} & L_{ii}^{(i)} & 0 & v_i^{(i)} \\
            L_{(i + 1):n, 1:i}^{(i + 1)} & L_{(i + 1):n, i}^{(i)}
            & L_{(i + 1):n, (i + 1):n}^{(i + 1)} & v_{(i + 1):n}^{(i)} \\
        \end{pmatrix},

    where the :math:`0` and :math:`I` blocks are chosen such that the matrix product
    makes sense.
    One can now read off that :math:`L^{(i)}` is lower-triangular and that the first
    :math:`i - 1` entries of :math:`v^{(i)}` are zeros.
    By induction, we can thus conclude that :math:`L^{(1)} = LQ^T_{1:n, 1:n}` is
    lower-triangular.
    Moreover, since :math:`Q` also fulfills :eq:`Q-constraint-q`, we know that
    :math:`v^{(1)} = v`.
    It follows that

    .. math::
        \begin{pmatrix}
            L & 0
        \end{pmatrix}
        Q^T
        =
        \begin{pmatrix}
            L^- & v
        \end{pmatrix},

    i.e. :math:`L^- = L Q_{1:n, 1:n}^T`.

    Note that this algorithm is a minor modification of the `LINPACK` [2]_ routine
    :code:`dchdd`. The exact modifications are described in [1]_.

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

    # Downdate L
    if impl is None:
        _downdate_impl_default(L, v)
    elif impl == "cython":
        try:
            _downdate_impl_cython(L, v)
        except NameError as ne:  # pragma: no cover
            raise ValueError("The Cython implementation is not available.") from ne
    elif impl == "python":
        _downdate_impl_python(L, v)
    else:
        raise ValueError(
            f"Unknown implementation '{impl}'. Available implementations: "
            f"{', '.join(_update_available_impls)}"
        )

    return L


downdate_seeger.available_impls = _downdate_available_impls
