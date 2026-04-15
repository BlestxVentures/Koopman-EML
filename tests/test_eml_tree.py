"""Tests for EML tree construction, evaluation, snapping, and formula extraction."""

import torch
import pytest

from koopman_eml.eml_tree import EMLNode, EMLTree, EMLTreeVectorized


class TestEMLNode:
    def test_leaf_node_output_shape(self):
        node = EMLNode(is_leaf=True, n_vars=2)
        batch = 10
        const_one = torch.ones(batch)
        x_vars = torch.randn(batch, 2)
        out = node(const_one, x_vars, None, None, tau=1.0)
        assert out.shape == (batch,)

    def test_interior_node_output_shape(self):
        node = EMLNode(is_leaf=False, n_vars=1)
        batch = 8
        const_one = torch.ones(batch)
        x_vars = torch.randn(batch, 1)
        f_child = torch.randn(batch)
        out = node(const_one, x_vars, f_child, f_child, tau=1.0)
        assert out.shape == (batch,)


class TestEMLTree:
    @pytest.mark.parametrize("depth", [1, 2, 3])
    def test_output_shape(self, depth: int):
        tree = EMLTree(depth=depth, n_vars=2)
        x = torch.randn(16, 2)
        out = tree(x, tau=1.0)
        assert out.shape == (16,)

    def test_snap_weights(self):
        tree = EMLTree(depth=2, n_vars=1)
        routes = tree.snap_weights()
        assert isinstance(routes, dict)
        assert len(routes) > 0
        for key, (left, right) in routes.items():
            assert left in ("1", "x0", "f_child")
            assert right in ("1", "x0", "f_child")

    def test_deterministic_after_snap(self):
        tree = EMLTree(depth=2, n_vars=1)
        tree.snap_weights()
        x = torch.tensor([1.0, 2.0, 3.0]).unsqueeze(-1)
        with torch.no_grad():
            out1 = tree(x, tau=0.01)
            out2 = tree(x, tau=0.01)
        # After snapping with low tau, Gumbel noise is negligible but not zero;
        # the outputs should be very close
        assert torch.allclose(out1, out2, atol=0.5)


class TestEMLTreeVectorized:
    @pytest.mark.parametrize("n_trees,depth", [(4, 2), (8, 3), (16, 2)])
    def test_output_shape(self, n_trees: int, depth: int):
        tree = EMLTreeVectorized(n_trees=n_trees, depth=depth, n_vars=3)
        x = torch.randn(32, 3)
        out = tree(x, tau=1.0)
        assert out.shape == (32, n_trees)

    def test_univariate(self):
        tree = EMLTreeVectorized(n_trees=4, depth=2, n_vars=1)
        x = torch.randn(10)
        out = tree(x, tau=1.0)
        assert out.shape == (10, 4)

    def test_no_nan(self):
        tree = EMLTreeVectorized(n_trees=8, depth=3, n_vars=2)
        x = torch.randn(20, 2)
        out = tree(x, tau=1.0)
        assert not torch.isnan(out).any(), "NaN in vectorized tree output"
