from setuptools import setup

try:
    # isort: off

    # This import must come after the setuptools import
    from Cython.Build import cythonize
    import scipy  # pylint: disable=unused-import

    # isort: on

    build_cython = True
except ImportError:
    build_cython = False

    print("Not building Cython extensions")

# Extensions
ext_modules = []

if build_cython:
    ext_modules.extend(
        cythonize("src/cholupdates/rank_1/_seeger_impl_cython.pyx"),
    )

setup(
    ext_modules=ext_modules,
)
