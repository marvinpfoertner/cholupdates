""" Utility functions needed for running the tests. """

from typing import Optional

import numpy as np
import scipy.stats


def random_spd_matrix(
    n: int,
    spectrum: np.ndarray = None,
    spectrum_shape: float = 10.0,
    spectrum_scale: float = 1.0,
    spectrum_loc: float = 0.0,
    random_state: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """Sample a random symmetric positive definite matrix.

    Parameters
    ----------
    n :
        Dimension of the matrix.
    spectrum :
        Spectrum of the matrix. Must be an array of shape :code:`(n,)`. If the spectrum
        is not given, it will be sampled from :code:`n` i.i.d. from a gamma distribution
        with parameters specified by :code:`spectrum_shape`, :code:`spectrum_scale` and
        :code:`spectrum_loc`.
    spectrum_shape :
        Shape parameter of the gamma distribution from which the spectrum is sampled.
        See :meth:`scipy.stats.gamma.rvs`.
    spectrum_scale :
        Scale parameter of the gamma distribution from which the spectrum is sampled.
        See :meth:`scipy.stats.gamma.rvs`.
    spectrum_loc :
        Location parameter of the gamma distribution from which the spectrum is sampled.
        See :meth:`scipy.stats.gamma.rvs`.
    random_state :
        Random state which will be used to sample the spd matrix. Uses the global random
        state by default.

    Returns
    -------
        A random symmetric positive definite matrix.
    """
    Q = scipy.stats.special_ortho_group.rvs(n, random_state=random_state)

    if spectrum is None:
        spectrum = scipy.stats.gamma.rvs(
            spectrum_shape,
            loc=spectrum_loc,
            scale=spectrum_scale,
            size=n,
            random_state=random_state,
        )

        spectrum.sort()

    return Q @ np.diag(spectrum) @ Q.T
