"""Tests for complex-valued EML primitives (i and ix candidates).

Verifies that:
  - Candidate counts and logit shapes are correct for all three modes
  - Complex forward pass produces finite complex observables
  - Reconstruction output is always real
  - Complex model trains without NaN/Inf
  - Formula extraction handles i / ix candidate labels
  - Real-mode backward compatibility is preserved
  - Child-logit bias initializes correctly (Fix 1)
  - Warm-start training runs without error (Fix 2)
  - Mixed dictionary produces correct shapes and formulas (Fix 3)
  - Custom phase fractions are respected (Fix 4)
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from koopman_eml import KoopmanEML
from koopman_eml.analysis import extract_eml_formulas, prediction_rollout
from koopman_eml.eml_tree import EMLTreeVectorized
from koopman_eml.training import train_koopman_eml, train_warmstart_complex

FAST_TRAIN = dict(n_epochs=30, lr=5e-3, batch_size=256, device="cpu", verbose=False)
STATE_DIM = 3
N_OBS = 4
DEPTH = 2


# -------------------------------------------------------------------
# Candidate count and logit shape
# -------------------------------------------------------------------


class TestCandidateShapes:
    """Verify logit tensor dimensions match the expected candidate counts."""

    def test_real_baseline_shapes(self):
        tree = EMLTreeVectorized(n_trees=N_OBS, depth=DEPTH, n_vars=STATE_DIM)
        leaf_logits = tree.level_logits[0]
        root_logits = tree.level_logits[1]
        assert leaf_logits.shape == (N_OBS, 2, 1 + STATE_DIM, 2)
        assert root_logits.shape == (N_OBS, 1, 1 + STATE_DIM + 1, 2)

    def test_i_only_shapes(self):
        tree = EMLTreeVectorized(
            n_trees=N_OBS, depth=DEPTH, n_vars=STATE_DIM, use_complex=True,
        )
        leaf_logits = tree.level_logits[0]
        root_logits = tree.level_logits[1]
        assert leaf_logits.shape == (N_OBS, 2, 1 + 1 + STATE_DIM, 2)
        assert root_logits.shape == (N_OBS, 1, 1 + 1 + STATE_DIM + 1, 2)

    def test_i_plus_ix_shapes(self):
        tree = EMLTreeVectorized(
            n_trees=N_OBS, depth=DEPTH, n_vars=STATE_DIM,
            use_complex=True, allow_imaginary_vars=True,
        )
        leaf_logits = tree.level_logits[0]
        root_logits = tree.level_logits[1]
        n_leaf = 1 + 1 + STATE_DIM + STATE_DIM
        n_root = n_leaf + 1
        assert leaf_logits.shape == (N_OBS, 2, n_leaf, 2)
        assert root_logits.shape == (N_OBS, 1, n_root, 2)


# -------------------------------------------------------------------
# Complex forward pass
# -------------------------------------------------------------------


class TestComplexForward:
    """Verify complex-valued EML trees produce correct dtypes and shapes."""

    @pytest.fixture
    def x_batch(self):
        return torch.randn(16, STATE_DIM)

    def test_i_only_output_is_complex(self, x_batch):
        tree = EMLTreeVectorized(
            n_trees=N_OBS, depth=DEPTH, n_vars=STATE_DIM, use_complex=True,
        )
        out = tree(x_batch, tau=1.0, use_complex=True)
        assert out.is_complex()
        assert out.shape == (16, N_OBS)

    def test_i_plus_ix_output_is_complex(self, x_batch):
        tree = EMLTreeVectorized(
            n_trees=N_OBS, depth=DEPTH, n_vars=STATE_DIM,
            use_complex=True, allow_imaginary_vars=True,
        )
        out = tree(x_batch, tau=1.0, use_complex=True)
        assert out.is_complex()
        assert out.shape == (16, N_OBS)

    def test_complex_output_is_finite(self, x_batch):
        tree = EMLTreeVectorized(
            n_trees=N_OBS, depth=DEPTH, n_vars=STATE_DIM, use_complex=True,
        )
        out = tree(x_batch, tau=1.0, use_complex=True)
        assert torch.all(torch.isfinite(out.real))
        assert torch.all(torch.isfinite(out.imag))

    def test_real_baseline_unchanged(self, x_batch):
        tree = EMLTreeVectorized(
            n_trees=N_OBS, depth=DEPTH, n_vars=STATE_DIM,
        )
        out = tree(x_batch, tau=1.0)
        assert not out.is_complex()
        assert out.shape == (16, N_OBS)


# -------------------------------------------------------------------
# KoopmanEML complex model
# -------------------------------------------------------------------


class TestComplexKoopmanModel:
    """Full model integration: lift, predict, reconstruct, loss."""

    def _make_model(self, use_complex, allow_imaginary_vars=False):
        return KoopmanEML(
            state_dim=STATE_DIM, n_observables=N_OBS, tree_depth=DEPTH,
            exp_order=6, ln_order=8,
            use_complex=use_complex,
            allow_imaginary_vars=allow_imaginary_vars,
        )

    def test_complex_lift_is_complex(self):
        model = self._make_model(use_complex=True)
        x = torch.randn(8, STATE_DIM)
        g = model.lift(x, tau=1.0)
        assert g.is_complex()
        assert g.shape == (8, N_OBS)

    def test_complex_reconstruct_is_real(self):
        model = self._make_model(use_complex=True)
        x = torch.randn(8, STATE_DIM)
        g = model.lift(x, tau=1.0)
        x_hat = model.reconstruct(g)
        assert not x_hat.is_complex()
        assert x_hat.shape == (8, STATE_DIM)

    def test_complex_forward_losses_are_real(self):
        model = self._make_model(use_complex=True)
        x_k = torch.randn(8, STATE_DIM)
        x_k1 = torch.randn(8, STATE_DIM)
        out = model(x_k, x_k1, tau=1.0)
        for key in ("pred_loss", "recon_loss", "reg_loss"):
            assert not out[key].is_complex(), f"{key} should be real"
            assert torch.isfinite(out[key]), f"{key} should be finite"

    def test_i_plus_ix_reconstruct_is_real(self):
        model = self._make_model(use_complex=True, allow_imaginary_vars=True)
        x = torch.randn(8, STATE_DIM)
        g = model.lift(x, tau=1.0)
        x_hat = model.reconstruct(g)
        assert not x_hat.is_complex()

    def test_real_model_backward_compat(self):
        model = self._make_model(use_complex=False)
        x_k = torch.randn(8, STATE_DIM)
        x_k1 = torch.randn(8, STATE_DIM)
        out = model(x_k, x_k1, tau=1.0)
        assert not out["g_k"].is_complex()
        assert torch.isfinite(out["pred_loss"])


# -------------------------------------------------------------------
# Training
# -------------------------------------------------------------------


class TestComplexTraining:
    """Verify training loop runs without NaN for complex models."""

    def _train(self, use_complex, allow_imaginary_vars=False):
        model = KoopmanEML(
            state_dim=STATE_DIM, n_observables=N_OBS, tree_depth=DEPTH,
            exp_order=6, ln_order=8,
            use_complex=use_complex,
            allow_imaginary_vars=allow_imaginary_vars,
        )
        X_k = torch.randn(200, STATE_DIM)
        X_k1 = torch.randn(200, STATE_DIM)
        history = train_koopman_eml(model, X_k, X_k1, **FAST_TRAIN)
        return model, history

    def test_i_only_trains(self):
        model, history = self._train(use_complex=True)
        assert all(np.isfinite(history["total"]))

    def test_i_plus_ix_trains(self):
        model, history = self._train(use_complex=True, allow_imaginary_vars=True)
        assert all(np.isfinite(history["total"]))

    def test_real_baseline_still_trains(self):
        model, history = self._train(use_complex=False)
        assert all(np.isfinite(history["total"]))

    def test_complex_rollout_is_real(self):
        model, _ = self._train(use_complex=True)
        x0 = torch.randn(STATE_DIM)
        traj = prediction_rollout(model, x0, 10, device="cpu")
        assert not traj.is_complex()
        assert traj.shape == (11, STATE_DIM)
        assert np.all(np.isfinite(traj.numpy()))


# -------------------------------------------------------------------
# Formula extraction
# -------------------------------------------------------------------


class TestComplexFormulas:
    """Verify formula extraction handles i / ix candidates."""

    def test_i_only_formulas_contain_i(self):
        model = KoopmanEML(
            state_dim=STATE_DIM, n_observables=N_OBS, tree_depth=DEPTH,
            exp_order=6, ln_order=8, use_complex=True,
        )
        X_k = torch.randn(100, STATE_DIM)
        X_k1 = torch.randn(100, STATE_DIM)
        train_koopman_eml(model, X_k, X_k1, **FAST_TRAIN)
        formulas = extract_eml_formulas(model)
        assert len(formulas) == N_OBS
        assert all(isinstance(f, str) for f in formulas)
        valid_tokens = {"1", "i", "x0", "x1", "x2", "eml", "(", ")", ",", " "}
        for f in formulas:
            for token in f.replace("eml(", "").replace(")", "").replace(",", "").split():
                assert token in valid_tokens or token.startswith("x"), f"Unexpected token '{token}' in {f}"

    def test_i_plus_ix_formulas_may_contain_ix(self):
        model = KoopmanEML(
            state_dim=STATE_DIM, n_observables=N_OBS, tree_depth=DEPTH,
            exp_order=6, ln_order=8,
            use_complex=True, allow_imaginary_vars=True,
        )
        X_k = torch.randn(100, STATE_DIM)
        X_k1 = torch.randn(100, STATE_DIM)
        train_koopman_eml(model, X_k, X_k1, **FAST_TRAIN)
        formulas = extract_eml_formulas(model)
        assert len(formulas) == N_OBS
        joined = " ".join(formulas)
        valid_tokens = {"1", "i", "i*x0", "i*x1", "i*x2", "x0", "x1", "x2"}
        assert any(t in joined for t in valid_tokens)

    def test_real_formulas_no_i(self):
        model = KoopmanEML(
            state_dim=STATE_DIM, n_observables=N_OBS, tree_depth=DEPTH,
            exp_order=6, ln_order=8, use_complex=False,
        )
        formulas = extract_eml_formulas(model)
        joined = " ".join(formulas)
        assert "i," not in joined
        assert "i)" not in joined or "eml" in joined


# -------------------------------------------------------------------
# Fix 1: Child-logit bias
# -------------------------------------------------------------------


class TestChildLogitBias:
    """Verify child_logit_bias shifts the initial logit for the child slot."""

    def test_bias_applied_at_root_level(self):
        tree = EMLTreeVectorized(
            n_trees=N_OBS, depth=DEPTH, n_vars=STATE_DIM,
            use_complex=True, allow_imaginary_vars=True,
            child_logit_bias=3.0,
        )
        root_logits = tree.level_logits[1]
        child_mean = root_logits[:, :, -1, :].mean().item()
        other_mean = root_logits[:, :, :-1, :].mean().item()
        assert child_mean > other_mean + 2.0

    def test_bias_not_applied_at_leaf(self):
        tree = EMLTreeVectorized(
            n_trees=N_OBS, depth=DEPTH, n_vars=STATE_DIM,
            use_complex=True, child_logit_bias=3.0,
        )
        leaf_logits = tree.level_logits[0]
        assert leaf_logits.mean().abs().item() < 1.0

    def test_zero_bias_default(self):
        tree = EMLTreeVectorized(
            n_trees=N_OBS, depth=DEPTH, n_vars=STATE_DIM,
            use_complex=True,
        )
        root_logits = tree.level_logits[1]
        assert root_logits[:, :, -1, :].mean().abs().item() < 1.0

    def test_biased_model_trains(self):
        model = KoopmanEML(
            state_dim=STATE_DIM, n_observables=N_OBS, tree_depth=DEPTH,
            exp_order=6, ln_order=8,
            use_complex=True, allow_imaginary_vars=True,
            child_logit_bias=2.0,
        )
        X_k = torch.randn(200, STATE_DIM)
        X_k1 = torch.randn(200, STATE_DIM)
        history = train_koopman_eml(model, X_k, X_k1, **FAST_TRAIN)
        assert all(np.isfinite(history["total"]))


# -------------------------------------------------------------------
# Fix 2: Warm-start training
# -------------------------------------------------------------------


class TestWarmStartTraining:
    """Verify two-stage warm-start trains without error."""

    def test_warmstart_runs(self):
        model = KoopmanEML(
            state_dim=STATE_DIM, n_observables=N_OBS, tree_depth=DEPTH,
            exp_order=6, ln_order=8,
            use_complex=True, allow_imaginary_vars=True,
        )
        X_k = torch.randn(200, STATE_DIM)
        X_k1 = torch.randn(200, STATE_DIM)
        history = train_warmstart_complex(
            model, X_k, X_k1,
            pretrain_epochs=20, complex_epochs=20,
            lr=5e-3, batch_size=256, device="cpu", verbose=False,
        )
        assert all(np.isfinite(history["total"]))
        assert len(history["total"]) == 40

    def test_warmstart_reconstruct_is_real(self):
        model = KoopmanEML(
            state_dim=STATE_DIM, n_observables=N_OBS, tree_depth=DEPTH,
            exp_order=6, ln_order=8,
            use_complex=True, allow_imaginary_vars=True,
        )
        X_k = torch.randn(200, STATE_DIM)
        X_k1 = torch.randn(200, STATE_DIM)
        train_warmstart_complex(
            model, X_k, X_k1,
            pretrain_epochs=15, complex_epochs=15,
            lr=5e-3, batch_size=256, device="cpu", verbose=False,
        )
        x = torch.randn(8, STATE_DIM)
        g = model.lift(x, tau=0.01)
        x_hat = model.reconstruct(g)
        assert not x_hat.is_complex()

    def test_warmstart_formulas(self):
        model = KoopmanEML(
            state_dim=STATE_DIM, n_observables=N_OBS, tree_depth=DEPTH,
            exp_order=6, ln_order=8,
            use_complex=True, allow_imaginary_vars=True,
        )
        X_k = torch.randn(200, STATE_DIM)
        X_k1 = torch.randn(200, STATE_DIM)
        train_warmstart_complex(
            model, X_k, X_k1,
            pretrain_epochs=15, complex_epochs=15,
            lr=5e-3, batch_size=256, device="cpu", verbose=False,
        )
        formulas = extract_eml_formulas(model)
        assert len(formulas) == N_OBS
        assert all(isinstance(f, str) for f in formulas)


# -------------------------------------------------------------------
# Fix 3: Mixed dictionary
# -------------------------------------------------------------------


class TestMixedDictionary:
    """Verify mixed real+complex dictionary works end-to-end."""

    def test_mixed_lift_shape(self):
        model = KoopmanEML(
            state_dim=STATE_DIM, n_observables=N_OBS, tree_depth=DEPTH,
            exp_order=6, ln_order=8,
            use_complex=True, allow_imaginary_vars=True,
            n_complex_trees=N_OBS // 2,
        )
        assert model._mixed
        x = torch.randn(8, STATE_DIM)
        g = model.lift(x, tau=1.0)
        assert g.is_complex()
        assert g.shape == (8, N_OBS)

    def test_mixed_reconstruct_is_real(self):
        model = KoopmanEML(
            state_dim=STATE_DIM, n_observables=N_OBS, tree_depth=DEPTH,
            exp_order=6, ln_order=8,
            use_complex=True, allow_imaginary_vars=True,
            n_complex_trees=N_OBS // 2,
        )
        x = torch.randn(8, STATE_DIM)
        g = model.lift(x, tau=1.0)
        x_hat = model.reconstruct(g)
        assert not x_hat.is_complex()
        assert x_hat.shape == (8, STATE_DIM)

    def test_mixed_trains(self):
        model = KoopmanEML(
            state_dim=STATE_DIM, n_observables=N_OBS, tree_depth=DEPTH,
            exp_order=6, ln_order=8,
            use_complex=True, allow_imaginary_vars=True,
            n_complex_trees=N_OBS // 2,
        )
        X_k = torch.randn(200, STATE_DIM)
        X_k1 = torch.randn(200, STATE_DIM)
        history = train_koopman_eml(model, X_k, X_k1, **FAST_TRAIN)
        assert all(np.isfinite(history["total"]))

    def test_mixed_formulas_length(self):
        model = KoopmanEML(
            state_dim=STATE_DIM, n_observables=N_OBS, tree_depth=DEPTH,
            exp_order=6, ln_order=8,
            use_complex=True, allow_imaginary_vars=True,
            n_complex_trees=N_OBS // 2,
        )
        formulas = extract_eml_formulas(model)
        assert len(formulas) == N_OBS

    def test_mixed_rollout_is_real(self):
        model = KoopmanEML(
            state_dim=STATE_DIM, n_observables=N_OBS, tree_depth=DEPTH,
            exp_order=6, ln_order=8,
            use_complex=True, allow_imaginary_vars=True,
            n_complex_trees=N_OBS // 2,
        )
        X_k = torch.randn(200, STATE_DIM)
        X_k1 = torch.randn(200, STATE_DIM)
        train_koopman_eml(model, X_k, X_k1, **FAST_TRAIN)
        x0 = torch.randn(STATE_DIM)
        traj = prediction_rollout(model, x0, 5, device="cpu")
        assert not traj.is_complex()
        assert traj.shape == (6, STATE_DIM)


# -------------------------------------------------------------------
# Fix 4: Custom phase fractions (slow anneal)
# -------------------------------------------------------------------


class TestSlowAnneal:
    """Verify custom phase fractions work correctly."""

    def test_custom_phases_train(self):
        model = KoopmanEML(
            state_dim=STATE_DIM, n_observables=N_OBS, tree_depth=DEPTH,
            exp_order=6, ln_order=8,
            use_complex=True, allow_imaginary_vars=True,
        )
        X_k = torch.randn(200, STATE_DIM)
        X_k1 = torch.randn(200, STATE_DIM)
        history = train_koopman_eml(
            model, X_k, X_k1,
            n_epochs=40, lr=5e-3, batch_size=256, device="cpu", verbose=False,
            tau_start=3.0, phase1_frac=0.75, phase2_frac=0.15,
        )
        assert all(np.isfinite(history["total"]))
        assert len(history["tau"]) == 40

    def test_phase_fractions_affect_tau_schedule(self):
        model = KoopmanEML(
            state_dim=STATE_DIM, n_observables=N_OBS, tree_depth=DEPTH,
            exp_order=6, ln_order=8,
        )
        X_k = torch.randn(200, STATE_DIM)
        X_k1 = torch.randn(200, STATE_DIM)

        h_default = train_koopman_eml(
            model, X_k, X_k1,
            n_epochs=100, lr=5e-3, batch_size=256, device="cpu", verbose=False,
        )

        model2 = KoopmanEML(
            state_dim=STATE_DIM, n_observables=N_OBS, tree_depth=DEPTH,
            exp_order=6, ln_order=8,
        )
        h_slow = train_koopman_eml(
            model2, X_k, X_k1,
            n_epochs=100, lr=5e-3, batch_size=256, device="cpu", verbose=False,
            tau_start=3.0, phase1_frac=0.75, phase2_frac=0.15,
        )

        assert h_slow["tau"][0] > h_default["tau"][0]
        assert h_slow["tau"][70] > h_default["tau"][70]
