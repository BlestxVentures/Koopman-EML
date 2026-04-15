"""Tests for PySINDy EMLLibrary integration."""

import numpy as np
import pytest

try:
    import pysindy  # noqa: F401
    _HAS_PYSINDY = True
except ImportError:
    _HAS_PYSINDY = False

from koopman_eml.sindy import EMLLibrary, eml_enumerate


class TestEMLEnumerate:
    """Test the EML expression enumeration."""

    def test_depth_0(self):
        exprs = eml_enumerate(0, n_vars=2, include_identity=False)
        names = [n for n, _ in exprs]
        assert "1" in names
        assert "x0" in names
        assert "x1" in names

    def test_depth_1_count(self):
        exprs = eml_enumerate(1, n_vars=1, include_identity=False)
        depth1 = [n for n, _ in exprs if n.startswith("eml(")]
        assert len(depth1) > 0

    def test_callables_work(self):
        exprs = eml_enumerate(1, n_vars=2, include_identity=False)
        x0 = np.array([1.0, 2.0, 3.0])
        x1 = np.array([0.5, 1.0, 1.5])
        for name, fn in exprs:
            result = fn(x0, x1)
            assert result.shape == (3,) or np.isscalar(result), (
                f"Function {name} returned unexpected shape {getattr(result, 'shape', 'scalar')}"
            )
            assert np.all(np.isfinite(result)), f"Non-finite output from {name}"

    def test_deduplication(self):
        exprs = eml_enumerate(1, n_vars=1, include_identity=False)
        names = [n for n, _ in exprs]
        assert len(names) == len(set(names)), "Duplicate names found"


@pytest.mark.skipif(not _HAS_PYSINDY, reason="PySINDy not installed")
class TestEMLLibrary:
    """Test PySINDy compatibility."""

    def test_fit_transform(self):
        X = np.random.randn(50, 2)
        lib = EMLLibrary(depth=1, n_vars=2)
        lib.fit(X)
        Xt = lib.transform(X)
        assert Xt.shape[0] == 50
        assert Xt.shape[1] == lib.n_output_features_

    def test_feature_names(self):
        X = np.random.randn(20, 3)
        lib = EMLLibrary(depth=1, n_vars=3)
        lib.fit(X)
        names = lib.get_feature_names()
        assert len(names) == lib.n_output_features_
        assert names[0] == "1"  # bias column

    def test_no_nan(self):
        X = np.abs(np.random.randn(30, 2)) + 0.1
        lib = EMLLibrary(depth=1, n_vars=2)
        lib.fit(X)
        Xt = lib.transform(X)
        assert np.all(np.isfinite(Xt)), "NaN or Inf in EMLLibrary output"
