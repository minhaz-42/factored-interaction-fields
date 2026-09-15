"""FIF: factored interaction-field estimation on SHOW3D (research code)."""
__version__ = "0.1.0"

import warnings as _w
# numpy on Apple Accelerate emits spurious divide/overflow RuntimeWarnings in matmul on finite data.
_w.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)
