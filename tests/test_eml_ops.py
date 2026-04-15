"""Tests for Taylor-series primitives and the EML operator."""

import numpy as np
import pytest
import torch

from koopman_eml.eml_ops import (
    eml,
    eml_numpy,
    taylor_exp,
    taylor_exp_complex,
    taylor_ln,
    taylor_ln_complex,
)


class TestTaylorExp:
    """Verify taylor_exp matches torch.exp across a range of inputs."""

    @pytest.mark.parametrize("order", [8, 12, 16])
    def test_accuracy_positive(self, order: int):
        x = torch.linspace(0.0, 10.0, 200)
        approx = taylor_exp(x, order=order)
        exact = torch.exp(x)
        assert torch.allclose(approx, exact, atol=1e-6, rtol=1e-5), (
            f"Max error: {(approx - exact).abs().max().item():.2e}"
        )

    @pytest.mark.parametrize("order", [8, 12, 16])
    def test_accuracy_negative(self, order: int):
        x = torch.linspace(-10.0, 0.0, 200)
        approx = taylor_exp(x, order=order)
        exact = torch.exp(x)
        assert torch.allclose(approx, exact, atol=1e-6, rtol=1e-5)

    def test_zero(self):
        x = torch.tensor([0.0])
        assert torch.allclose(taylor_exp(x), torch.tensor([1.0]), atol=1e-12)

    def test_one(self):
        x = torch.tensor([1.0])
        result = taylor_exp(x, order=12)
        assert abs(result.item() - np.e) < 1e-6


class TestTaylorLn:
    """Verify taylor_ln matches torch.log across a range of inputs."""

    @pytest.mark.parametrize("order", [8, 12, 16])
    def test_accuracy(self, order: int):
        y = torch.linspace(0.01, 100.0, 300)
        approx = taylor_ln(y, order=order)
        exact = torch.log(y)
        assert torch.allclose(approx, exact, atol=1e-5, rtol=1e-4), (
            f"Max error: {(approx - exact).abs().max().item():.2e}"
        )

    def test_one(self):
        y = torch.tensor([1.0])
        assert torch.allclose(taylor_ln(y), torch.tensor([0.0]), atol=1e-12)

    def test_e(self):
        y = torch.tensor([np.e])
        result = taylor_ln(y, order=16)
        assert abs(result.item() - 1.0) < 1e-10


class TestEML:
    """Verify the combined EML operator."""

    def test_exp_identity(self):
        """eml(x, 1) = exp(x)"""
        x = torch.linspace(-3, 3, 100)
        ones = torch.ones_like(x)
        result = eml(x, ones)
        expected = torch.exp(x)
        assert torch.allclose(result, expected, atol=1e-5, rtol=1e-4)

    def test_e_constant(self):
        """eml(1, 1) = e"""
        x = torch.tensor([1.0])
        y = torch.tensor([1.0])
        result = eml(x, y)
        assert abs(result.item() - np.e) < 1e-5

    def test_gradient_x(self):
        """d/dx eml(x, y) = exp(x)"""
        x = torch.tensor([1.5], requires_grad=True)
        y = torch.tensor([2.0])
        result = eml(x, y)
        result.backward()
        expected_grad = torch.exp(torch.tensor([1.5]))
        assert torch.allclose(x.grad, expected_grad, atol=1e-4)

    def test_gradient_y(self):
        """d/dy eml(x, y) = -1/y"""
        x = torch.tensor([1.0])
        y = torch.tensor([2.0], requires_grad=True)
        result = eml(x, y)
        result.backward()
        expected_grad = torch.tensor([-0.5])
        assert torch.allclose(y.grad, expected_grad, atol=1e-4)


class TestEMLNumpy:
    """Verify the NumPy-only EML."""

    def test_matches_torch(self):
        x = np.linspace(-3, 3, 50)
        y = np.linspace(0.1, 5, 50)
        result_np = eml_numpy(x, y)
        result_torch = eml(torch.tensor(x), torch.tensor(y)).detach().numpy()
        np.testing.assert_allclose(result_np, result_torch, atol=1e-5, rtol=1e-4)

    def test_exp_identity(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.ones(3)
        result = eml_numpy(x, y)
        expected = np.exp(x)
        np.testing.assert_allclose(result, expected, atol=1e-10)


class TestComplexPrimitives:
    """Basic checks for the complex Taylor primitives."""

    def test_exp_real_matches(self):
        z = torch.complex(torch.linspace(-3, 3, 50), torch.zeros(50))
        result = taylor_exp_complex(z)
        expected = torch.exp(z)
        assert torch.allclose(result.real, expected.real, atol=1e-5)
        assert torch.allclose(result.imag, expected.imag, atol=1e-5)

    def test_ln_real_matches(self):
        w = torch.complex(torch.linspace(0.1, 10, 50), torch.zeros(50))
        result = taylor_ln_complex(w)
        expected = torch.log(w)
        assert torch.allclose(result.real, expected.real, atol=1e-4)
        assert torch.allclose(result.imag, expected.imag, atol=1e-4)
