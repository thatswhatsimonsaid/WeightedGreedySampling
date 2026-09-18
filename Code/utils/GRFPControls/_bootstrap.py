"""
Import shim for the GRFP control scripts.

`utils/Auxiliary/__init__.py` eagerly imports plotting / data-generation modules
that import `seaborn`, `kagglehub` and `imageio` at module level -- and at least
one of them calls `sys.exit(1)` on ImportError. Those are optional viz deps that
the simulation itself never needs, but because `utils` is a namespace package,
importing anything under `utils.Auxiliary` (e.g. DataFrameUtils) triggers the
whole chain and hard-blocks the learning loop.

Importing this module FIRST installs harmless stubs for the optional deps. It
changes no environment and no repository file.
"""

import os
import sys
import types

### Pin BLAS to one thread per process ###
# These runs are many small linear-algebra calls, parallelised across processes.
# Left alone, each worker's BLAS spawns its own thread pool and the workers
# fight for cores. Must be set before NumPy is first imported, which is why this
# module is imported first by every entry point.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

_OPTIONAL = ("seaborn", "kagglehub", "imageio", "imageio.v3")


def install_stubs():
    for name in _OPTIONAL:
        if name not in sys.modules:
            try:
                __import__(name)
            except Exception:
                sys.modules[name] = types.ModuleType(name)
    # imageio.v3 is accessed as an attribute of imageio #
    if isinstance(sys.modules.get("imageio"), types.ModuleType):
        if not hasattr(sys.modules["imageio"], "v3"):
            sys.modules["imageio"].v3 = sys.modules.get(
                "imageio.v3", types.ModuleType("imageio.v3")
            )


install_stubs()
