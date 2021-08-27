from setuptools import setup

# Cython Extensions
ext_modules = []

try:
    # isort: off

    # This import must come after the setuptools import
    from Cython.Build import cythonize
    import scipy  # pylint: disable=unused-import

    import os

    # isort: on

    cython_available = True
except ImportError:
    cython_available = False

build_cython = cython_available and not (
    "CHOLUPDATES_DISABLE_CYTHON_BUILD" in os.environ
    and os.environ["CHOLUPDATES_DISABLE_CYTHON_BUILD"] == "1"
)

if build_cython:
    ext_modules.extend(
        cythonize(
            [
                "src/cholupdates/_blas_polymorphic.pyx",
                "src/cholupdates/rank_1/_seeger_impl_cython.pyx",
            ]
        ),
    )
else:
    print("Not building Cython extensions")

setup(
    ext_modules=ext_modules,
)
