"""Interface function for all symmetric rank-1 downdate algorithms"""

from typing import Any, Dict

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
    **method_kwargs: Dict[str, Any],
) -> np.ndarray:
    r"""Update a Cholesky factorization after subtraction of a symmetric positive
    semidefinite rank-1 matrix.

    In other words, given :math:`A = L L^T \in \mathbb{R}^{N \times N}` and
    :math:`v \in \mathbb{R}^N`, compute :math:`L'` such that

    .. math::
        A' := A - v v^T = L' L'^T.

    Parameters
    ----------
    L : (N, N) numpy.ndarray
        Lower-triangular Cholesky factor of :math:`A`.
        Must have a non-zero diagonal.
        The entries in the strict upper-triangular part of :code:`L` can contain
        arbitrary values, since the algorithm neither reads from nor writes to this part
        of the matrix. This behavior is useful when using the Cholesky factors returned
        by :func:`scipy.linalg.cho_factor` which contain arbitrary values on the
        irrelevant triangular part of the matrix.
    v : (N,) numpy.ndarray
        The vector :math:`v` with shape :code:`(N,)` defining the symmetric rank-1
        update matrix :math:`v v^T`.
    check_diag :
        If set to :code:`True`, the function will check whether the diagonal of the
        given Cholesky factor :code:`L` is non-zero and raise a :class:`ValueError` if
        this is not the case.
        Setting :code:`check_diag` to :code:`False` can be used to speed up computations
        if it is clear that the Cholesky factor can not have zeros on its diagonal.
        Caution: If this argument is set to :code:`False` and the Cholesky factor does
        contain zeros on its diagonal, the behavior of the function will be undefined.
    overwrite_L :
        If set to :code:`True`, the function may overwrite the array :code:`L` with the
        upper Cholesky factor :math:`L'` of :math:`A'`, i.e. the result is computed
        in-place.
        Passing :code:`False` here ensures that the array :code:`L` is not modified.
    overwrite_v :
        If set to :code:`True`, the function may reuse the array :code:`v` as an
        internal computation buffer, which will modify :code:`v`.
        Passing :code:`False` here ensures that the array :code:`v` is not modified.
    method :
        Algorithm to be used to compute the updated Cholesky factor. Must be one of

        - "cho_factor"
            Directly uses :func:`scipy.linalg.cho_factor` on :math:`L L^T + v v^T`.
            This is just here for convenience and should be slower than all other
            methods.
        - "seeger"
            Calls :func:`cholupdates.rank_1.update_seeger`.

        Defaults to "seeger".
    method_kwargs :
        Additional keyword arguments which will be passed to the function selected by
        :code:`method`.

    Returns
    -------
    (N, N) numpy.ndarray, dtype=L.dtype
        Lower triangular Cholesky factor :math:`L'` of :math:`A - v v^T`.
        The diagonal entries of this matrix are guaranteed to be positive.
        The strict upper-triangular part of this matrix will contain the values from the
        upper-triangular part of :code:`L`.

    Raises
    ------
    ValueError
        If :code:`L` does not have shape :code:`(N, N)` for some :code:`N`.
    ValueError
        If :code:`v` does not have shape :code:`(N,)`, while :code:`L` has shape
        :code:`(N, N)`.
    numpy.linalg.LinAlgError
        If the diagonal of :code:`L` contains zeros and :code:`check_diag` is set to
        :code:`True`.
    numpy.linalg.LinAlgError
        If the downdate results in a matrix :math:`L'`, which is not positive definite.
    Exception
        Any exception raised by the function specified by :code:`method`.

    See Also
    --------
    cholupdates.rank_1.update : A similar function which performs a symmetric rank 1
        update instead of a downdate.

    Examples
    --------
    Consider the following matrix-vector pair

    >>> A = np.array([[ 7.77338976,  1.27256923,  1.58075291],
    ...               [ 1.27256923,  8.29126934,  0.80466256],
    ...               [ 1.58075291,  0.80466256, 13.65749896]])
    >>> v = np.array([1.60994441, 0.21482681, 0.78780241])

    We want to compute the lower-triangular Cholesky factor :code:`L_dd` of

    >>> A_dd = A - np.outer(v, v)
    >>> A_dd
    array([[ 5.18146876,  0.92671001,  0.31243482],
           [ 0.92671001,  8.24511878,  0.63542148],
           [ 0.31243482,  0.63542148, 13.03686632]])

    We assume that the lower-triangular Cholesky factor :code:`L` of :code:`A` is given

    >>> import scipy.linalg
    >>> L = scipy.linalg.cholesky(A, lower=True)
    >>> L
    array([[2.78807994, 0.        , 0.        ],
           [0.45643212, 2.84305101, 0.        ],
           [0.56696829, 0.19200501, 3.64680408]])

    The function :func:`cholupdates.rank_1.update` computes :code:`L_dd` efficiently
    from :code:`L` and :code:`v`

    >>> import cholupdates
    >>> L_dd = cholupdates.rank_1.downdate(L, v, method="seeger")
    >>> L_dd
    array([[2.27628398, 0.        , 0.        ],
           [0.40711529, 2.8424243 , 0.        ],
           [0.13725652, 0.20389013, 3.6022848 ]])
    >>> np.allclose(L_dd @ L_dd.T, A_dd)
    True

    We could also compute :code:`L_dd` by applying a Cholesky factorization algorithm
    directly to :code:`A_dd` (which is however less efficient)

    >>> L_dd_cho = cholupdates.rank_1.downdate(L, v, method="cho_factor")
    >>> L_dd_cho
    array([[2.27628398, 0.        , 0.        ],
           [0.40711529, 2.8424243 , 0.        ],
           [0.13725652, 0.20389013, 3.6022848 ]])
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
            **method_kwargs,
        )
    else:
        raise NotImplementedError(f"Unknown method: '{method}'")

    return L_upd


downdate.available_methods = ["cho_factor", "seeger"]
