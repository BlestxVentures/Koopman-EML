"""Tests for CTF E1-E12 metric computation."""

import numpy as np
import pytest

from koopman_eml.ctf import evaluate_ctf, long_term_score, short_term_score


class TestShortTermScore:
    def test_perfect_prediction(self):
        truth = np.random.randn(100, 3)
        score = short_term_score(truth, truth)
        assert abs(score - 100.0) < 1e-10

    def test_zero_prediction(self):
        truth = np.random.randn(100, 3)
        zeros = np.zeros_like(truth)
        score = short_term_score(zeros, truth)
        assert score == pytest.approx(0.0, abs=1.0)

    def test_k_truncation(self):
        truth = np.random.randn(100, 3)
        score_full = short_term_score(truth, truth)
        score_k10 = short_term_score(truth, truth, k=10)
        assert score_full == pytest.approx(100.0, abs=1e-10)
        assert score_k10 == pytest.approx(100.0, abs=1e-10)


class TestLongTermScore:
    def test_perfect_prediction(self):
        truth = np.sin(np.linspace(0, 10 * np.pi, 1000)).reshape(-1, 1)
        score = long_term_score(truth, truth)
        assert abs(score - 100.0) < 1e-10

    def test_poor_prediction(self):
        truth = np.sin(np.linspace(0, 10 * np.pi, 1000)).reshape(-1, 1)
        noise = np.random.randn(*truth.shape) * 10
        score = long_term_score(noise, truth)
        assert score < 80.0


class TestEvaluateCTF:
    def test_all_scores_present(self):
        results = {f"X{i}pred": np.random.randn(50, 3) for i in range(1, 10)}
        truths = {f"X{i}test": np.random.randn(50, 3) for i in range(1, 10)}
        scores = evaluate_ctf(results, truths)
        for i in range(1, 13):
            assert f"E{i}" in scores
        assert "avg" in scores

    def test_missing_keys_score_zero(self):
        scores = evaluate_ctf({}, {})
        for i in range(1, 13):
            assert scores[f"E{i}"] == 0.0
        assert scores["avg"] == 0.0

    def test_perfect_forecast(self):
        truth = np.random.randn(100, 3)
        results = {"X1pred": truth.copy()}
        truths = {"X1test": truth.copy()}
        scores = evaluate_ctf(results, truths)
        assert scores["E1"] == pytest.approx(100.0, abs=1e-6)
