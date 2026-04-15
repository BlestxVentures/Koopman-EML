"""Verify that EML can represent known elementary functions via exact weight settings."""

import math

import numpy as np
import pytest
import torch

from koopman_eml.eml_ops import eml


class TestKnownEMLIdentities:
    """Test the fundamental identities from the Odrzywołek paper."""

    def test_exp_x(self):
        """eml(x, 1) = exp(x)"""
        x = torch.linspace(-5, 5, 100)
        result = eml(x, torch.ones_like(x))
        expected = torch.exp(x)
        assert torch.allclose(result, expected, atol=1e-5)

    def test_e_constant(self):
        """eml(1, 1) = e"""
        result = eml(torch.tensor([1.0]), torch.tensor([1.0]))
        assert abs(result.item() - math.e) < 1e-5

    def test_ln_x(self):
        """ln(x) = eml(1, eml(eml(1, x), 1))  -- depth 3

        This is the fundamental EML identity for natural log.
        """
        x = torch.linspace(0.1, 10.0, 50)
        ones = torch.ones_like(x)

        inner = eml(ones, x)       # eml(1, x) = e - ln(x)
        mid = eml(inner, ones)      # eml(e - ln(x), 1) = exp(e - ln(x))
        result = eml(ones, mid)     # eml(1, exp(e - ln(x))) = e - ln(exp(e - ln(x)))
        expected = torch.log(x)

        assert torch.allclose(result, expected, atol=1e-3), (
            f"Max error: {(result - expected).abs().max().item():.2e}"
        )

    def test_ee_constant(self):
        """eml(eml(1,1), 1) = exp(e) = e^e"""
        one = torch.tensor([1.0])
        e_val = eml(one, one)       # e
        result = eml(e_val, one)    # exp(e)
        expected = math.e ** math.e
        assert abs(result.item() - expected) < 1e-3

    def test_zero_constant(self):
        """0 is derivable from EML using the ln identity:
        We test via: exp(x) - ln(1) = exp(x), and eml(0, 1) should give exp(0) = 1,
        which means we need ln(y)=0 => y=1.  More directly, 0 = ln(1) = eml(1, eml(eml(1,1),1)) - e
        But simpler: eml(x,1) - exp(x) = 0 for any x. Just verify eml(0,1) = 1.
        """
        zero = torch.tensor([0.0])
        one = torch.tensor([1.0])
        result = eml(zero, one)  # exp(0) - ln(1) = 1 - 0 = 1
        assert abs(result.item() - 1.0) < 1e-10

    def test_batch_consistency(self):
        """EML should produce identical results element-wise whether batched or not."""
        x_vals = [0.5, 1.0, 2.0, -1.0]
        y_vals = [1.0, 2.0, 0.5, 3.0]

        x_batch = torch.tensor(x_vals)
        y_batch = torch.tensor(y_vals)
        batch_result = eml(x_batch, y_batch)

        for i, (xi, yi) in enumerate(zip(x_vals, y_vals)):
            single = eml(torch.tensor([xi]), torch.tensor([yi]))
            assert abs(batch_result[i].item() - single.item()) < 1e-10
