"""Interface function for all symmetric rank-1 update methods"""

import numpy as np
import scipy.linalg

from ._arg_validation import _validate_update_args
from ._seeger import update_seeger


def update(
    L: np.ndarray,
    v: np.ndarray,
    check_diag: bool = True,
    overwrite_L: bool = False,
    overwrite_v: bool = False,
    method: str = "seeger",
) -> np.ndarray:
    r"""Update a Cholesky factorization after addition of a positive-semidefinite
    symmetric rank-1 matrix.

    In other words, given :math:`A = L L^T \in \mathbb{R}^{N \times N}` and
    :math:`v \in \mathbb{R}^N`, compute :math:`L'` such that

    .. math::
        A' := A + v v^T = L' L'^T.

    Parameters
    ----------
    L :
        Lower-triangular Cholesky factor of :math:`A` with shape :code:`(N, N)`.
        Must have a non-zero diagonal.
        The entries in the strict upper-triangular part of :code:`L` can contain
        arbitrary values, since the algorithm neither reads from nor writes to this part
        of the matrix. This behavior is useful when using the Cholesky factors returned
        by :func:`scipy.linalg.cho_factor` which contain arbitrary values on the
        irrelevant triangular part of the matrix.
    v :
        The vector :math:`v` with shape :code:`(N,)` defining the symmetric rank-1
        update matrix :math:`v v^T`.
    check_diag :
        If set to :code:`True`, the function will check whether the diagonal of the
        given Cholesky factor :code:`L` is non-zero and raise a :class:`ValueError` if
        this is not the case.
        Setting :code:`check_diag` to `False` can be used to speed up computations if
        it is clear that the Cholesky factor can not have zeros on its diagonal.
        Caution: If this argument is set to :code:`False` and the Cholesky factor does
        contain zeros on its diagonal, the behavior of the function will be undefined.
    overwrite_L :
        If set to :code:`True`, the function may overwrite the array :code:`L` with the
        upper Cholesky factor :math:`L'` of :math:`A'`, i.e. the result is computed
        in-place.
        Passing `False` here ensures that the array :code:`L` is not modified.
    overwrite_v :
        If set to `True`, the function may reuse the array :code:`v` as an internal
        computation buffer, which will modify :code:`v`.
        Passing `False` here ensures that the array :code:`v` is not modified.
    method :
        Algorithm to be used to compute the updated Cholesky factor. Must be one of

        - "cho_factor":
            Directly uses :func:`scipy.linalg.cho_factor` on :math:`L L^T + v v^T`.
            This is just here for convenience and should be slower than all other
            methods.
        - "seeger":
            Calls :func:`cholupdates.rank_1.update_seeger`.
        - "seeger_cython":
            Calls :func:`cholupdates.rank_1.update_seeger` with :code:`impl="cython"`.
        - "seeger_python":
            Calls :func:`cholupdates.rank_1.update_seeger` with :code:`impl="python"`.

        Defaults to "seeger".

    Returns
    -------
        Lower triangular Cholesky factor :math:`L'` of :math:`A + v v^T` with shape
        :code:`(N, N)` and the same dtype as :code:`L`.
        The diagonal entries of this matrix are guaranteed to be positive.
        The strict upper-triangular part of this matrix will contain the values from the
        upper-triangular part of :code:`L`.

    Raises
    ------
    ValueError
        If :code:`L` does not have shape :code:`(N, N)` for some :code:`N`.
    numpy.linalg.LinAlgError
        If the diagonal of :code:`L` contains zeros and :code:`check_diag` is set to
        :code:`True`.
    ValueError
        If :code:`v` does not have shape :code:`(N,)`, while :code:`L` has shape
        :code:`(N, N)`.
    Exception
        Any exception raised by the function specified by :code:`method`.


    See Also
    --------
    cholupdates.rank_1.downdate : A similar function which performs a symmetric rank 1
        downdate instead of an update.

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

    >>> import scipy.linalg
    >>> L = scipy.linalg.cho_factor(A, lower=True)[0]
    >>> np.tril(L)
    array([[1.04880885, 0.        , 0.        ],
           [0.09534626, 1.44599761, 0.        ],
           [0.09534626, 0.06286946, 1.75697368]])

    The function :func:`cholupdates.rank_1.update` can compute :code:`L_prime` from
    :code:`L` efficiently

    >>> import cholupdates
    >>> L_prime = cholupdates.rank_1.update(L, v, method="seeger")
    >>> np.tril(L_prime)
    array([[ 1.44913767,  0.        ,  0.        ],
           [17.32064554, 18.08577447,  0.        ],
           [ 6.96966215,  7.15374133,  1.82969791]])

    Did it work?

    >>> np.allclose(A_prime, np.tril(L_prime) @ np.tril(L_prime).T)
    True

    We could also compute :code:`L_prime` by directly computing the Cholesky
    factorization of :code:`A_prime` (which is however less efficient)

    >>> L_prime_cho = cholupdates.rank_1.update(L, v, method="cho_factor")
    >>> np.tril(L_prime_cho)
    array([[ 1.44913767,  0.        ,  0.        ],
           [17.32064554, 18.08577447,  0.        ],
           [ 6.96966215,  7.15374133,  1.82969791]])
    >>> np.allclose(np.tril(L_prime), np.tril(L_prime_cho))
    True
    """

    if method == "cho_factor":
        _validate_update_args(L, v, check_diag)

        L_tril = np.tril(L)

        L_upd, _ = scipy.linalg.cho_factor(
            L_tril @ L_tril.T + np.outer(v, v),
            lower=True,
            overwrite_a=True,
        )

        L_upd[np.triu_indices(L.shape[0], k=1)] = L[np.triu_indices(L.shape[0], k=1)]
    elif method == "seeger":
        L_upd = update_seeger(
            L,
            v,
            check_diag=check_diag,
            overwrite_L=overwrite_L,
            overwrite_v=overwrite_v,
        )
    elif method == "seeger_cython":
        L_upd = update_seeger(
            L,
            v,
            check_diag=check_diag,
            overwrite_L=overwrite_L,
            overwrite_v=overwrite_v,
            impl="cython",
        )
    elif method == "seeger_python":
        L_upd = update_seeger(
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
