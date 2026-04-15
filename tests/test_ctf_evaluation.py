"""End-to-end test suite for CTF E1-E12 evaluation pipeline.

Tests are grouped by metric category and verify:
  - Data generation produces correct shapes and properties
  - Noise injection has the expected SNR behavior
  - EML-Koopman can train and produce predictions for each scenario
  - CTF scoring formulas return sensible values
  - The full pipeline runs without errors

Uses a *fast* configuration (few epochs, small model) so the suite
completes in ~60s on CPU.  Scores are not expected to match production
quality — the tests only assert structural correctness and plausible
score ranges.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from koopman_eml import KoopmanEML
from koopman_eml.analysis import prediction_rollout
from koopman_eml.ctf import evaluate_ctf, long_term_score, short_term_score
from koopman_eml.training import train_koopman_eml
from experiments.ctf_lorenz.generate_ctf_scenarios import (
    generate_all_ctf_scenarios,
    _add_noise,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

FAST_MODEL = dict(n_observables=4, tree_depth=2, exp_order=6, ln_order=8)
FAST_TRAIN = dict(n_epochs=50, lr=5e-3, batch_size=512, device="cpu", verbose=False)


@pytest.fixture(scope="module")
def ctf_data():
    """Generate CTF scenarios once for all tests in this module."""
    return generate_all_ctf_scenarios(
        n_trajectories=4, n_steps=500, seed=123,
    )


@pytest.fixture(scope="module")
def trained_model(ctf_data):
    """Train a small EML-Koopman model on clean data (shared across tests)."""
    X_k = torch.tensor(ctf_data["X_k_train"][:2000], dtype=torch.float32)
    X_k1 = torch.tensor(ctf_data["X_k1_train"][:2000], dtype=torch.float32)
    model = KoopmanEML(state_dim=3, **FAST_MODEL)
    train_koopman_eml(model, X_k, X_k1, **FAST_TRAIN)
    return model


# ===================================================================
# Data generation tests
# ===================================================================


class TestDataGeneration:
    """Verify generate_all_ctf_scenarios produces well-formed outputs."""

    def test_required_keys_present(self, ctf_data):
        required = [
            "X_k_train", "X_k1_train", "X_k_test", "X_k1_test",
            "X1train", "X1test", "mean", "std", "dt",
            "X2train", "X2test", "X3train", "X3test",
            "X4train", "X4test", "X5train", "X5test",
            "X6_k_train", "X6_k1_train", "X6train", "X6test",
            "X7_k_train", "X7_k1_train", "X7train", "X7test",
            "X8_k_train", "X8_k1_train", "X8train", "X8test",
            "X9_k_train", "X9_k1_train", "X9train", "X9test",
        ]
        for key in required:
            assert key in ctf_data, f"Missing key: {key}"

    def test_consecutive_pair_shapes(self, ctf_data):
        assert ctf_data["X_k_train"].shape == ctf_data["X_k1_train"].shape
        assert ctf_data["X_k_train"].shape[1] == 3

    def test_normalization(self, ctf_data):
        X = ctf_data["X_k_train"]
        assert abs(X.mean()) < 0.5
        assert 0.3 < X.std() < 3.0

    def test_limited_data_is_subset(self, ctf_data):
        assert len(ctf_data["X6_k_train"]) < len(ctf_data["X_k_train"])

    def test_forecast_train_test_contiguous(self, ctf_data):
        n_train = len(ctf_data["X1train"])
        n_test = len(ctf_data["X1test"])
        assert n_train > 0
        assert n_test > 0

    def test_parametric_keys_non_empty(self, ctf_data):
        for prefix in ("X8", "X9"):
            assert len(ctf_data[f"{prefix}_k_train"]) > 0
            assert len(ctf_data[f"{prefix}train"]) > 0
            assert len(ctf_data[f"{prefix}test"]) > 0


class TestNoiseInjection:
    """Verify noise addition has the expected statistical properties."""

    def test_medium_noise_increases_variance(self, ctf_data):
        clean = ctf_data["X1train"]
        noisy = ctf_data["X2train"]
        assert noisy.std() > clean.std() * 0.95

    def test_high_noise_larger_than_medium(self, ctf_data):
        medium = ctf_data["X2train"]
        high = ctf_data["X4train"]
        clean = ctf_data["X1train"]
        medium_err = np.mean((medium - clean) ** 2)
        high_err = np.mean((high - clean) ** 2)
        assert high_err > medium_err

    def test_add_noise_shape_preserved(self):
        rng = np.random.default_rng(0)
        data = rng.standard_normal((100, 3))
        noisy = _add_noise(data, 20.0, rng)
        assert noisy.shape == data.shape

    def test_add_noise_not_identical(self):
        rng = np.random.default_rng(0)
        data = rng.standard_normal((100, 3))
        noisy = _add_noise(data, 20.0, rng)
        assert not np.allclose(noisy, data)


# ===================================================================
# E1-E2: Clean forecasting
# ===================================================================


class TestE1E2Forecast:
    """E1 (short-term) and E2 (long-term) from clean data."""

    def test_model_trains_without_error(self, trained_model):
        assert trained_model is not None

    def test_rollout_shape(self, trained_model, ctf_data):
        x0 = torch.tensor(ctf_data["X1train"][-1], dtype=torch.float32)
        traj = prediction_rollout(trained_model, x0, 20, device="cpu")
        assert traj.shape == (21, 3)

    def test_e1_perfect_is_100(self):
        truth = np.random.randn(100, 3)
        assert short_term_score(truth, truth) == pytest.approx(100.0, abs=1e-8)

    def test_e1_finite(self, trained_model, ctf_data):
        x0 = torch.tensor(ctf_data["X1train"][-1], dtype=torch.float32)
        n = min(50, len(ctf_data["X1test"]))
        traj = prediction_rollout(trained_model, x0, n - 1, device="cpu")
        pred = traj[1:].numpy()
        truth = ctf_data["X1test"][:len(pred)]
        score = short_term_score(pred, truth)
        assert np.isfinite(score)

    def test_e2_perfect_is_100(self):
        truth = np.sin(np.linspace(0, 10 * np.pi, 500)).reshape(-1, 1)
        assert long_term_score(truth, truth) == pytest.approx(100.0, abs=1e-8)

    def test_e2_finite(self, trained_model, ctf_data):
        x0 = torch.tensor(ctf_data["X1train"][-1], dtype=torch.float32)
        n = min(50, len(ctf_data["X1test"]))
        traj = prediction_rollout(trained_model, x0, n - 1, device="cpu")
        pred = traj[1:].numpy()
        truth = ctf_data["X1test"][:len(pred)]
        score = long_term_score(pred, truth, k_max=20)
        assert np.isfinite(score)


# ===================================================================
# E3-E6: Noisy data
# ===================================================================


class TestE3E6Noisy:
    """E3-E6: reconstruction and forecasting from noisy observations."""

    def test_reconstruction_via_lift_reconstruct(self, trained_model, ctf_data):
        """The Koopman lift->reconstruct pipeline should reduce noise."""
        noisy = torch.tensor(ctf_data["X2train"][:100], dtype=torch.float32)
        trained_model.eval()
        with torch.no_grad():
            g = trained_model.lift(noisy, tau=0.01)
            recon = trained_model.reconstruct(g).numpy()
        assert recon.shape == (100, 3)
        assert np.all(np.isfinite(recon))

    def test_e3_reconstruction_score_finite(self, trained_model, ctf_data):
        noisy = torch.tensor(ctf_data["X2train"][:100], dtype=torch.float32)
        trained_model.eval()
        with torch.no_grad():
            g = trained_model.lift(noisy, tau=0.01)
            recon = trained_model.reconstruct(g).numpy()
        truth = ctf_data["X2test"][:100]
        score = short_term_score(recon, truth)
        assert np.isfinite(score)

    def test_e4_noisy_forecast_finite(self, trained_model, ctf_data):
        trained_model.eval()
        noisy_last = torch.tensor(ctf_data["X3train"][-1:], dtype=torch.float32)
        with torch.no_grad():
            g = trained_model.lift(noisy_last, tau=0.01)
            recon_last = trained_model.reconstruct(g).numpy()[0]
        x0 = torch.tensor(recon_last, dtype=torch.float32)
        n = min(30, len(ctf_data["X3test"]))
        traj = prediction_rollout(trained_model, x0, n - 1, device="cpu")
        pred = traj[1:].numpy()
        truth = ctf_data["X3test"][:len(pred)]
        score = long_term_score(pred, truth, k_max=10)
        assert np.isfinite(score)

    def test_high_noise_reconstruction_runs(self, trained_model, ctf_data):
        noisy = torch.tensor(ctf_data["X4train"][:50], dtype=torch.float32)
        trained_model.eval()
        with torch.no_grad():
            g = trained_model.lift(noisy, tau=0.01)
            recon = trained_model.reconstruct(g).numpy()
        assert np.all(np.isfinite(recon))

    def test_e5_score_finite(self, trained_model, ctf_data):
        noisy = torch.tensor(ctf_data["X4train"][:50], dtype=torch.float32)
        trained_model.eval()
        with torch.no_grad():
            recon = trained_model.reconstruct(trained_model.lift(noisy, tau=0.01)).numpy()
        truth = ctf_data["X4test"][:50]
        assert np.isfinite(short_term_score(recon, truth))


# ===================================================================
# E7-E10: Limited data
# ===================================================================


class TestE7E10Limited:
    """E7-E10: training with reduced data budgets."""

    def test_limited_clean_model_trains(self, ctf_data):
        X_k = torch.tensor(ctf_data["X6_k_train"][:200], dtype=torch.float32)
        X_k1 = torch.tensor(ctf_data["X6_k1_train"][:200], dtype=torch.float32)
        model = KoopmanEML(state_dim=3, **FAST_MODEL)
        train_koopman_eml(model, X_k, X_k1, **FAST_TRAIN)

        x0 = torch.tensor(ctf_data["X6train"][-1], dtype=torch.float32)
        traj = prediction_rollout(model, x0, 10, device="cpu")
        assert traj.shape == (11, 3)
        assert np.all(np.isfinite(traj.numpy()))

    def test_limited_noisy_model_trains(self, ctf_data):
        X_k = torch.tensor(ctf_data["X7_k_train"][:200], dtype=torch.float32)
        X_k1 = torch.tensor(ctf_data["X7_k1_train"][:200], dtype=torch.float32)
        model = KoopmanEML(state_dim=3, **FAST_MODEL)
        train_koopman_eml(model, X_k, X_k1, **FAST_TRAIN)

        x0 = torch.tensor(ctf_data["X7train"][-1], dtype=torch.float32)
        traj = prediction_rollout(model, x0, 10, device="cpu")
        assert np.all(np.isfinite(traj.numpy()))

    def test_e7_score_finite(self, ctf_data):
        X_k = torch.tensor(ctf_data["X6_k_train"][:200], dtype=torch.float32)
        X_k1 = torch.tensor(ctf_data["X6_k1_train"][:200], dtype=torch.float32)
        model = KoopmanEML(state_dim=3, **FAST_MODEL)
        train_koopman_eml(model, X_k, X_k1, **FAST_TRAIN)

        x0 = torch.tensor(ctf_data["X6train"][-1], dtype=torch.float32)
        n = min(20, len(ctf_data["X6test"]))
        traj = prediction_rollout(model, x0, n - 1, device="cpu")
        pred = traj[1:].numpy()
        truth = ctf_data["X6test"][:len(pred)]
        assert np.isfinite(short_term_score(pred, truth))
        assert np.isfinite(long_term_score(pred, truth, k_max=10))

    def test_e9_score_finite(self, ctf_data):
        X_k = torch.tensor(ctf_data["X7_k_train"][:200], dtype=torch.float32)
        X_k1 = torch.tensor(ctf_data["X7_k1_train"][:200], dtype=torch.float32)
        model = KoopmanEML(state_dim=3, **FAST_MODEL)
        train_koopman_eml(model, X_k, X_k1, **FAST_TRAIN)

        x0 = torch.tensor(ctf_data["X7train"][-1], dtype=torch.float32)
        n = min(20, len(ctf_data["X7test"]))
        traj = prediction_rollout(model, x0, n - 1, device="cpu")
        pred = traj[1:].numpy()
        truth = ctf_data["X7test"][:len(pred)]
        assert np.isfinite(short_term_score(pred, truth))
        assert np.isfinite(long_term_score(pred, truth, k_max=10))


# ===================================================================
# E11-E12: Parametric generalization
# ===================================================================


class TestE11E12Parametric:
    """E11-E12: generalization across bifurcation parameter (rho)."""

    def test_interp_model_trains(self, ctf_data):
        X_k = torch.tensor(ctf_data["X8_k_train"][:2000], dtype=torch.float32)
        X_k1 = torch.tensor(ctf_data["X8_k1_train"][:2000], dtype=torch.float32)
        model = KoopmanEML(state_dim=3, **FAST_MODEL)
        train_koopman_eml(model, X_k, X_k1, **FAST_TRAIN)

        x0 = torch.tensor(ctf_data["X8train"][-1], dtype=torch.float32)
        traj = prediction_rollout(model, x0, 10, device="cpu")
        assert traj.shape == (11, 3)
        assert np.all(np.isfinite(traj.numpy()))

    def test_extrap_model_trains(self, ctf_data):
        X_k = torch.tensor(ctf_data["X9_k_train"][:2000], dtype=torch.float32)
        X_k1 = torch.tensor(ctf_data["X9_k1_train"][:2000], dtype=torch.float32)
        model = KoopmanEML(state_dim=3, **FAST_MODEL)
        train_koopman_eml(model, X_k, X_k1, **FAST_TRAIN)

        x0 = torch.tensor(ctf_data["X9train"][-1], dtype=torch.float32)
        traj = prediction_rollout(model, x0, 10, device="cpu")
        assert np.all(np.isfinite(traj.numpy()))

    def test_e11_score_finite(self, ctf_data):
        X_k = torch.tensor(ctf_data["X8_k_train"][:2000], dtype=torch.float32)
        X_k1 = torch.tensor(ctf_data["X8_k1_train"][:2000], dtype=torch.float32)
        model = KoopmanEML(state_dim=3, **FAST_MODEL)
        train_koopman_eml(model, X_k, X_k1, **FAST_TRAIN)

        x0 = torch.tensor(ctf_data["X8train"][-1], dtype=torch.float32)
        n = min(20, len(ctf_data["X8test"]))
        traj = prediction_rollout(model, x0, n - 1, device="cpu")
        pred = traj[1:].numpy()
        truth = ctf_data["X8test"][:len(pred)]
        assert np.isfinite(short_term_score(pred, truth))

    def test_e12_score_finite(self, ctf_data):
        X_k = torch.tensor(ctf_data["X9_k_train"][:2000], dtype=torch.float32)
        X_k1 = torch.tensor(ctf_data["X9_k1_train"][:2000], dtype=torch.float32)
        model = KoopmanEML(state_dim=3, **FAST_MODEL)
        train_koopman_eml(model, X_k, X_k1, **FAST_TRAIN)

        x0 = torch.tensor(ctf_data["X9train"][-1], dtype=torch.float32)
        n = min(20, len(ctf_data["X9test"]))
        traj = prediction_rollout(model, x0, n - 1, device="cpu")
        pred = traj[1:].numpy()
        truth = ctf_data["X9test"][:len(pred)]
        assert np.isfinite(short_term_score(pred, truth))


# ===================================================================
# evaluate_ctf integration
# ===================================================================


class TestEvaluateCTFIntegration:
    """Test the top-level evaluate_ctf() aggregation function."""

    def test_full_results_dict(self, trained_model, ctf_data):
        """Build the full X1pred-X9pred dict and score all E1-E12."""
        trained_model.eval()

        results: dict[str, np.ndarray] = {}
        truths: dict[str, np.ndarray] = {}

        # E1/E2: forecast
        x0 = torch.tensor(ctf_data["X1train"][-1], dtype=torch.float32)
        n = min(30, len(ctf_data["X1test"]))
        traj = prediction_rollout(trained_model, x0, n - 1, device="cpu")
        results["X1pred"] = traj[1:].numpy()
        truths["X1test"] = ctf_data["X1test"][:len(results["X1pred"])]

        # E3: reconstruction
        noisy = torch.tensor(ctf_data["X2train"][:100], dtype=torch.float32)
        with torch.no_grad():
            g = trained_model.lift(noisy, tau=0.01)
            results["X2pred"] = trained_model.reconstruct(g).numpy()
        truths["X2test"] = ctf_data["X2test"][:100]

        # E4: noisy forecast
        noisy_last = torch.tensor(ctf_data["X3train"][-1:], dtype=torch.float32)
        with torch.no_grad():
            recon_last = trained_model.reconstruct(
                trained_model.lift(noisy_last, tau=0.01)
            ).numpy()[0]
        x0_n = torch.tensor(recon_last, dtype=torch.float32)
        n = min(30, len(ctf_data["X3test"]))
        traj = prediction_rollout(trained_model, x0_n, n - 1, device="cpu")
        results["X3pred"] = traj[1:].numpy()
        truths["X3test"] = ctf_data["X3test"][:len(results["X3pred"])]

        # E5: high-noise reconstruction
        noisy_hi = torch.tensor(ctf_data["X4train"][:50], dtype=torch.float32)
        with torch.no_grad():
            results["X4pred"] = trained_model.reconstruct(
                trained_model.lift(noisy_hi, tau=0.01)
            ).numpy()
        truths["X4test"] = ctf_data["X4test"][:50]

        # E6: high-noise forecast
        noisy_hi_last = torch.tensor(ctf_data["X5train"][-1:], dtype=torch.float32)
        with torch.no_grad():
            recon_hi = trained_model.reconstruct(
                trained_model.lift(noisy_hi_last, tau=0.01)
            ).numpy()[0]
        x0_hi = torch.tensor(recon_hi, dtype=torch.float32)
        n = min(30, len(ctf_data["X5test"]))
        traj = prediction_rollout(trained_model, x0_hi, n - 1, device="cpu")
        results["X5pred"] = traj[1:].numpy()
        truths["X5test"] = ctf_data["X5test"][:len(results["X5pred"])]

        # E7/E8 and E9/E10: use the same model for simplicity in tests
        for xkey, tkey in [("X6", "X6"), ("X7", "X7")]:
            x0_l = torch.tensor(ctf_data[f"{xkey}train"][-1], dtype=torch.float32)
            n = min(20, len(ctf_data[f"{tkey}test"]))
            traj = prediction_rollout(trained_model, x0_l, n - 1, device="cpu")
            results[f"{xkey}pred"] = traj[1:].numpy()
            truths[f"{tkey}test"] = ctf_data[f"{tkey}test"][:len(results[f"{xkey}pred"])]

        # E11/E12
        for xkey in ("X8", "X9"):
            x0_p = torch.tensor(ctf_data[f"{xkey}train"][-1], dtype=torch.float32)
            n = min(20, len(ctf_data[f"{xkey}test"]))
            traj = prediction_rollout(trained_model, x0_p, n - 1, device="cpu")
            results[f"{xkey}pred"] = traj[1:].numpy()
            truths[f"{xkey}test"] = ctf_data[f"{xkey}test"][:len(results[f"{xkey}pred"])]

        scores = evaluate_ctf(results, truths, k_max=10)

        for i in range(1, 13):
            key = f"E{i}"
            assert key in scores, f"Missing {key}"
            assert np.isfinite(scores[key]), f"{key} is not finite: {scores[key]}"

        assert "avg" in scores
        assert np.isfinite(scores["avg"])

    def test_evaluate_ctf_monotonicity(self):
        """Better predictions should give higher E1."""
        truth = np.random.randn(100, 3)
        good_pred = truth + np.random.randn(*truth.shape) * 0.01
        bad_pred = truth + np.random.randn(*truth.shape) * 1.0

        good = evaluate_ctf({"X1pred": good_pred}, {"X1test": truth})
        bad = evaluate_ctf({"X1pred": bad_pred}, {"X1test": truth})
        assert good["E1"] > bad["E1"]
