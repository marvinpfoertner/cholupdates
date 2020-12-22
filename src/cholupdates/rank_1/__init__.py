r"""This submodule contains implementations of symmetric rank-1 updates to Cholesky
factorizations.

Consider a symmetric positive-definite matrix :math:`A \in \mathbb{R}^{n \times n}` and
a vector :math:`v \in \mathbb{R}^n`. The vector :math:`v \in \mathbb{R}^n` defines a
symmetric rank-1 update :math:`v v^T \in \mathbb{R}^{n \times n}` which we can apply to
:math:`A` to obtain the updated matrix

.. math::
    A' = A + v v^T.

Note that :math:`A'` is also symmetric and positive-definite.
In several applications, one needs to compute a Cholesky factorization of the updated
matrix :math:`A'`, i.e. :math:`A' = L' L'^T` where the Cholesky factor
:math:`L' \in \mathbb{R}^{n \times n}` is lower-triangular.
Examples of these applications include online and offline versions of (general) linear
regression and linear Gaussian inference.
If a Cholesky factor :math:`L` of :math:`A` is given, there are more efficient methods
to compute :math:`L'` from :math:`L` than simply running a Cholesky factorization
algorithm on :math:`A'`, which generally have complexity :math:`O(n^3)`.
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
