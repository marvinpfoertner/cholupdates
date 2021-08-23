r"""This subpackage contains implementations of rank-1 modifications to Cholesky
factorizations.

.. rubric:: Rank-1 Up- and Downdates

Consider a symmetric positive-definite matrix :math:`A \in \mathbb{R}^{n \times n}` and
a vector :math:`v \in \mathbb{R}^n`. The vector :math:`v \in \mathbb{R}^n` defines a
symmetric positive semi-definite rank-1 matrix :math:`v v^T \in \mathbb{R}^{n \times n}`
which we can apply to :math:`A` to obtain

.. math::
    A^\pm := A \pm v v^T.

If :math:`v v^T` is added to :math:`A`, we refer to the operation and :math:`v v^T`
itself as a rank-1 update to :math:`A`, and to :math:`A^+` as the updated matrix.
Conversely, if :math:`v v^T` is subtracted from :math:`A`, we refer to the operation and
:math:`v v^T` itself as a rank-1 downdate to :math:`A`, and to :math:`A^-` as the
downdated matrix.

Rank-1 updates are needed in (online) frequentist (general) linear regression,
while both up- and downdates appear in (online) linear Gaussian inference, and hence
in (online) Bayesian (general) linear regression, as well as in Quasi-Newton methods for
unconstrained minimization.

In these applications, one usually wants to compute a Cholesky factorization of
:math:`A^\pm`, i.e. :math:`A^\pm = L^\pm (L^\pm)^T` with
:math:`L^\pm \in \mathbb{R}^{n \times n}` lower-triangular.
Note that :math:`A^+` is always symmetric and positive-definite, while, depending on the
choice of :math:`v`, :math:`A^-` need not be positive definite.
Hence, rank-1 downdates are not always well-defined, in that computing a Cholesky
actorization of the downdated matrix is not always possible.

If a Cholesky factor :math:`L` of :math:`A` is given, there are more efficient methods
to compute :math:`L^\pm` from :math:`L` than simply running a Cholesky factorization
algorithm on :math:`A^\pm`, which generally have complexity :math:`O(n^3)`.
This submodule contains implementations of such methods.
"""

from ._update import update  # isort: skip
from ._downdate import downdate  # isort: skip

# Concrete update functions
from ._seeger import update_seeger  # isort: skip
from ._seeger import downdate_seeger  # isort: skip

__all__ = [
    "update",
    "downdate",
]
