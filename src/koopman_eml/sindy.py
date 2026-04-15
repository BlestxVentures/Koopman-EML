"""
PySINDy integration -- EMLLibrary feature library.

Provides EML tree compositions as a PySINDy-compatible feature library,
enabling ``eml`` to serve as a drop-in function basis for SINDy.

Two modes:
    - **Enumerative:** generate all distinct EML compositions up to a given depth.
    - **Pre-trained:** load snapped EML trees and expose them as fixed observables.
"""

from __future__ import annotations

from itertools import product
from typing import Callable, Optional, Sequence

import numpy as np

try:
    from pysindy.feature_library.base import BaseFeatureLibrary
    from sklearn.utils.validation import check_is_fitted

    _HAS_PYSINDY = True
except ImportError:
    _HAS_PYSINDY = False
    BaseFeatureLibrary = object

from koopman_eml.eml_ops import eml_numpy


# ---------------------------------------------------------------------------
# Enumerate all EML expressions to a given depth
# ---------------------------------------------------------------------------


def eml_enumerate(
    depth: int, n_vars: int, *, include_identity: bool = True,
) -> list[tuple[str, Callable]]:
    """Return (name, callable) pairs for every distinct EML expression up to *depth*.

    Depth 0 terminals are ``1`` and ``x0, x1, ...``.
    Depth d expressions are ``eml(L, R)`` where L and R have depth < d.
    """
    terminals: list[tuple[str, Callable]] = [("1", lambda *_xs: np.ones_like(_xs[0]) if len(_xs) else np.array(1.0))]
    for v in range(n_vars):
        def _var(v=v):
            return lambda *xs: xs[v]
        terminals.append((f"x{v}", _var()))

    if include_identity:
        for v in range(n_vars):
            def _id(v=v):
                return lambda *xs: xs[v]
            terminals.append((f"x{v}", _id()))

    levels: list[list[tuple[str, Callable]]] = [terminals]

    for d in range(1, depth + 1):
        current: list[tuple[str, Callable]] = []
        all_prev = [expr for lvl in levels for expr in lvl]
        for (name_l, fn_l), (name_r, fn_r) in product(all_prev, repeat=2):
            name = f"eml({name_l}, {name_r})"

            def _make_fn(fl=fn_l, fr=fn_r):
                def fn(*xs):
                    return eml_numpy(fl(*xs), fr(*xs))
                return fn

            current.append((name, _make_fn()))
        levels.append(current)

    all_exprs = [expr for lvl in levels for expr in lvl]
    # Deduplicate by name
    seen: set[str] = set()
    unique: list[tuple[str, Callable]] = []
    for name, fn in all_exprs:
        if name not in seen:
            seen.add(name)
            unique.append((name, fn))
    return unique


# ---------------------------------------------------------------------------
# PySINDy-compatible library
# ---------------------------------------------------------------------------


if _HAS_PYSINDY:

    class EMLLibrary(BaseFeatureLibrary):
        """PySINDy feature library built from EML tree compositions.

        Parameters
        ----------
        depth : int
            Maximum EML tree depth for enumerative mode.
        n_vars : int, optional
            Number of input variables.  Inferred from data if not given.
        functions : list of (name, callable), optional
            Pre-built list of EML observables (e.g., from snapped trees).
            If given, *depth* is ignored.
        include_bias : bool
            Whether to prepend a constant-1 column.
        """

        def __init__(
            self,
            depth: int = 2,
            n_vars: Optional[int] = None,
            functions: Optional[Sequence[tuple[str, Callable]]] = None,
            include_bias: bool = True,
        ):
            super().__init__()
            self.depth = depth
            self._n_vars = n_vars
            self._preset_functions = functions
            self.include_bias = include_bias
            self._functions: list[tuple[str, Callable]] = []

        def fit(self, x_full, y=None):
            n_samples, n_features = x_full.shape
            n_vars = self._n_vars or n_features

            if self._preset_functions is not None:
                self._functions = list(self._preset_functions)
            else:
                self._functions = eml_enumerate(self.depth, n_vars, include_identity=False)

            self.n_features_in_ = n_features
            self.n_output_features_ = len(self._functions) + (1 if self.include_bias else 0)
            return self

        def transform(self, x_full):
            check_is_fitted(self, ["n_features_in_"])
            cols: list[np.ndarray] = []
            if self.include_bias:
                cols.append(np.ones((x_full.shape[0], 1)))
            for _name, fn in self._functions:
                col_args = [x_full[:, i] for i in range(x_full.shape[1])]
                vals = fn(*col_args)
                if vals.ndim == 0:
                    vals = np.full(x_full.shape[0], vals)
                cols.append(vals.reshape(-1, 1))
            return np.hstack(cols)

        def get_feature_names(self, input_features=None):
            names: list[str] = []
            if self.include_bias:
                names.append("1")
            for name, _fn in self._functions:
                names.append(name)
            return names

else:

    class EMLLibrary:  # type: ignore[no-redef]
        """Stub when PySINDy is not installed."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PySINDy is required for EMLLibrary.  Install with: pip install pysindy"
            )
