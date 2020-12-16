""" Implementation of the rank-1 update algorithm from section 2 in [1] in Python.

References
----------
.. [1] M. Seeger, "Low Rank Updates for the Cholesky Decomposition", 2008.
"""

import numpy as np
import scipy.linalg


def _update_inplace(L: np.ndarray, v: np.ndarray) -> None:
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
