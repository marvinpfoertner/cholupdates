# cython: language_level = 3

"""Polymorphic wrappers for the functions/subroutines included in the BLAS library.

The function names simply omit the type code from the name of the corresponding BLAS
functions/subroutines."""

ctypedef fused blas_dtype:
    float
    double
    # float complex
    # double complex

# Level 1

cdef void rotg(blas_dtype *a, blas_dtype *b, blas_dtype *c, blas_dtype *s)

cdef void rot(
    int n,
    blas_dtype *x,
    int incx,
    blas_dtype *y,
    int incy,
    blas_dtype c,
    blas_dtype s,
)

cdef void scal(int n, blas_dtype alpha, blas_dtype *x, int incx)

cdef blas_dtype dot(int n, blas_dtype *x, int incx, blas_dtype *y, int incy)

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
)
