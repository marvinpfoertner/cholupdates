# cython: language_level = 3

"""Polymorphic wrappers for the functions/subroutines included in the BLAS library.

The function names simply omit the type code from the name of the corresponding BLAS
functions/subroutines."""

cimport scipy.linalg.cython_blas

# Level 1

cdef void rotg(blas_dtype *a, blas_dtype *b, blas_dtype *c, blas_dtype *s):
    if blas_dtype is float:
        scipy.linalg.cython_blas.srotg(a, b, c, s)
    elif blas_dtype is double:
        scipy.linalg.cython_blas.drotg(a, b, c, s)


cdef void rot(
    int n,
    blas_dtype *x,
    int incx,
    blas_dtype *y,
    int incy,
    blas_dtype c,
    blas_dtype s,
):
    if blas_dtype is float:
        scipy.linalg.cython_blas.srot(&n, x, &incx, y, &incy, &c, &s)
    elif blas_dtype is double:
        scipy.linalg.cython_blas.drot(&n, x, &incx, y, &incy, &c, &s)


cdef void scal(int n, blas_dtype alpha, blas_dtype *x, int incx):
    if blas_dtype is float:
        scipy.linalg.cython_blas.sscal(&n, &alpha, x, &incx)
    elif blas_dtype is double:
        scipy.linalg.cython_blas.dscal(&n, &alpha, x, &incx)


cdef blas_dtype dot(int n, blas_dtype *x, int incx, blas_dtype *y, int incy):
    if blas_dtype is float:
        return scipy.linalg.cython_blas.sdot(&n, x, &incx, y, &incy)
    elif blas_dtype is double:
        return scipy.linalg.cython_blas.ddot(&n, x, &incx, y, &incy)

# Level 2

cdef void trsv(
    char *uplo,
    char *trans,
    char *diag,
    int n,
    blas_dtype *a,
    int lda,
    blas_dtype *x,
    int inxc,
):
    if blas_dtype is float:
        scipy.linalg.cython_blas.strsv(
            uplo,
            trans,
            diag,
            &n,
            a,
            &lda,
            x,
            &inxc,
        )
    elif blas_dtype is double:
        scipy.linalg.cython_blas.dtrsv(
            uplo,
            trans,
            diag,
            &n,
            a,
            &lda,
            x,
            &inxc,
        )
