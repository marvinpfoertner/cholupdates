""" This package provides algorithms which efficiently compute the Cholesky
factorization of low-rank up-/downdates to a matrix with a known Cholesky factorization.
"""

from . import rank_1
from ._version import version as _version_str

__version__ = _version_str
