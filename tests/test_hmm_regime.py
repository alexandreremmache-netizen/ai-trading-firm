"""
Tests for HMM Regime Detection
==============================

Tests for the Hidden Markov Model-based market regime detector.
"""

import pytest
import numpy as np
from datetime import datetime, timezone

from core.hmm_regime import (
    MarketState,
    HMMRegimeResult,
    GaussianHMM,
    HMMRegimeDetector,
    create_hmm_detector,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_returns():
    """Generate sample returns with regime structure."""
    np.random.seed(42)

    # Bull regime: positive mean, low vol
    bull_returns = np.random.normal(0.001, 0.01, 100)

    # Bear regime: negative mean, high vol
    bear_returns = np.random.normal(-0.002, 0.02, 50)

    # Sideways regime: zero mean, moderate vol
    sideways_returns = np.random.normal(0.0, 0.015, 75)

    # Combine into a series
    returns = np.concatenate([bull_returns, bear_returns, sideways_returns, bull_returns[:50]])

    return returns


@pytest.fixture
def hmm_detector():
    """Create HMM regime detector."""
    return create_hmm_detector(n_states=3, min_samples=50)


# ============================================================================
# MARKET STATE TESTS
# ============================================================================

class TestMarketState:
    """Test MarketState enum."""

    def test_all_states_defined(self):
        """Test that all expected states are defined."""
        assert MarketState.BULL.value == "bull"
        assert MarketState.BEAR.value == "bear"
        assert MarketState.SIDEWAYS.value == "sideways"
        assert MarketState.HIGH_VOL.value == "high_vol"
        assert MarketState.LOW_VOL.value == "low_vol"

    def test_state_count(self):
        """Test number of states."""
        assert len(MarketState) == 5


# ============================================================================
# GAUSSIAN HMM TESTS
# ============================================================================

class TestGaussianHMM:
    """Tests for Gaussian HMM implementation."""

    def test_initialization(self):
        """Test HMM initialization."""
        hmm = GaussianHMM(n_states=3)
        assert hmm.n_states == 3
        assert hmm.n_iterations == 100
        assert not hmm._is_fitted

    def test_initialization_custom_params(self):
        """Test HMM with custom parameters."""
        hmm = GaussianHMM(n_states=5, n_iterations=50, tolerance=1e-3)
        assert hmm.n_states == 5
        assert hmm.n_iterations == 50
        assert hmm.tolerance == 1e-3

    def test_fit_basic(self, sample_returns):
        """Test basic fitting."""
        hmm = GaussianHMM(n_states=3)
        hmm.fit(sample_returns)

        assert hmm._is_fitted
        assert hmm.means_ is not None
        assert len(hmm.means_) == 3
        assert hmm.variances_ is not None
        assert len(hmm.variances_) == 3
        assert hmm.transmat_ is not None
        assert hmm.transmat_.shape == (3, 3)

    def test_fit_insufficient_samples(self):
        """Test fitting with insufficient samples."""
        hmm = GaussianHMM(n_states=3)
        with pytest.raises(ValueError, match="Need at least"):
            hmm.fit(np.array([0.01, 0.02, 0.03]))

    def test_predict_before_fit(self):
        """Test prediction before fitting raises error."""
        hmm = GaussianHMM(n_states=3)
        with pytest.raises(RuntimeError, match="must be fitted"):
            hmm.predict(np.array([0.01, 0.02, 0.03]))

    def test_predict_after_fit(self, sample_returns):
        """Test prediction after fitting."""
        hmm = GaussianHMM(n_states=3)
        hmm.fit(sample_returns)

        states = hmm.predict(sample_returns[-50:])
        assert len(states) == 50
        assert all(0 <= s < 3 for s in states)

    def test_predict_proba(self, sample_returns):
        """Test probability prediction."""
        hmm = GaussianHMM(n_states=3)
        hmm.fit(sample_returns)

        probs = hmm.predict_proba(sample_returns[-50:])
        assert probs.shape == (50, 3)

        # Probabilities should sum to 1
        for row in probs:
            assert np.isclose(row.sum(), 1.0, atol=1e-6)

    def test_transition_matrix_valid(self, sample_returns):
        """Test transition matrix is valid stochastic matrix."""
        hmm = GaussianHMM(n_states=3)
        hmm.fit(sample_returns)

        # Each row should sum to 1
        for row in hmm.transmat_:
            assert np.isclose(row.sum(), 1.0, atol=1e-6)

        # All values should be non-negative
        assert np.all(hmm.transmat_ >= 0)

    def test_means_are_sorted(self, sample_returns):
        """Test that means are sorted for interpretability."""
        hmm = GaussianHMM(n_states=3)
        hmm.fit(sample_returns)

        # Means should be sorted low to high
        for i in range(len(hmm.means_) - 1):
            assert hmm.means_[i] <= hmm.means_[i + 1]


# ============================================================================
# HMM REGIME DETECTOR TESTS
# ============================================================================

class TestHMMRegimeDetector:
    """Tests for HMM Regime Detector."""

    def test_initialization(self, hmm_detector):
        """Test detector initialization."""
        assert hmm_detector.n_states == 3
        assert hmm_detector.min_samples == 50
        assert not hmm_detector._is_fitted

    def test_fit(self, hmm_detector, sample_returns):
        """Test fitting the detector."""
        hmm_detector.fit(sample_returns)
        assert hmm_detector._is_fitted

    def test_fit_insufficient_samples(self, hmm_detector):
        """Test fitting with insufficient samples."""
        with pytest.raises(ValueError):
            hmm_detector.fit([0.01, 0.02])

    def test_predict_state(self, hmm_detector, sample_returns):
        """Test state prediction."""
        hmm_detector.fit(sample_returns)
        state = hmm_detector.predict_state()

        assert isinstance(state, MarketState)

    def test_get_regime_probabilities(self, hmm_detector, sample_returns):
        """Test regime probability retrieval."""
        hmm_detector.fit(sample_returns)
        probs = hmm_detector.get_regime_probabilities()

        # Should have probabilities for mapped states
        assert isinstance(probs, dict)
        assert sum(probs.values()) > 0.99  # Should sum to ~1

    def test_get_transition_matrix(self, hmm_detector, sample_returns):
        """Test transition matrix retrieval."""
        hmm_detector.fit(sample_returns)
        trans = hmm_detector.get_transition_matrix()

        assert isinstance(trans, dict)
        # Each row should sum to ~1
        for from_state, to_probs in trans.items():
            total = sum(to_probs.values())
            assert 0.99 <= total <= 1.01

    def test_get_expected_returns_by_regime(self, hmm_detector, sample_returns):
        """Test expected returns by regime."""
        hmm_detector.fit(sample_returns)
        returns = hmm_detector.get_expected_returns_by_regime()

        assert isinstance(returns, dict)
        assert len(returns) > 0

    def test_get_volatility_by_regime(self, hmm_detector, sample_returns):
        """Test volatility by regime."""
        hmm_detector.fit(sample_returns)
        vols = hmm_detector.get_volatility_by_regime()

        assert isinstance(vols, dict)
        assert all(v > 0 for v in vols.values())

    def test_update(self, hmm_detector, sample_returns):
        """Test incremental update."""
        hmm_detector.fit(sample_returns)
        initial_history_len = len(hmm_detector._returns_history)

        hmm_detector.update(0.01)
        # History length stays same if at lookback limit, or increases by 1
        assert len(hmm_detector._returns_history) >= initial_history_len

    def test_analyze(self, hmm_detector, sample_returns):
        """Test full analysis."""
        hmm_detector.fit(sample_returns)
        result = hmm_detector.analyze()

        assert isinstance(result, HMMRegimeResult)
        assert isinstance(result.current_state, MarketState)
        assert 0 <= result.confidence <= 1
        assert isinstance(result.transition_matrix, dict)

    def test_get_regime_signal(self, hmm_detector, sample_returns):
        """Test trading signal generation."""
        hmm_detector.fit(sample_returns)
        signal, rationale = hmm_detector.get_regime_signal()

        assert -1 <= signal <= 1
        assert isinstance(rationale, str)
        assert len(rationale) > 0


# ============================================================================
# HMM REGIME RESULT TESTS
# ============================================================================

class TestHMMRegimeResult:
    """Tests for HMM regime result."""

    def test_to_dict(self, hmm_detector, sample_returns):
        """Test result serialization."""
        hmm_detector.fit(sample_returns)
        result = hmm_detector.analyze()
        result_dict = result.to_dict()

        assert "current_state" in result_dict
        assert "state_probabilities" in result_dict
        assert "expected_return" in result_dict
        assert "expected_volatility" in result_dict
        assert "transition_matrix" in result_dict
        assert "confidence" in result_dict
        assert "timestamp" in result_dict


# ============================================================================
# FACTORY FUNCTION TESTS
# ============================================================================

class TestFactoryFunction:
    """Tests for the factory function."""

    def test_create_default(self):
        """Test creating detector with defaults."""
        detector = create_hmm_detector()
        assert detector.n_states == 3
        assert detector.min_samples == 100

    def test_create_custom(self):
        """Test creating detector with custom params."""
        detector = create_hmm_detector(n_states=5, min_samples=200)
        assert detector.n_states == 5
        assert detector.min_samples == 200


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests for HMM regime detection."""

    def test_full_workflow(self):
        """Test complete workflow from data to signal."""
        # Generate synthetic market data
        np.random.seed(123)
        bull_period = np.random.normal(0.001, 0.008, 150)
        bear_period = np.random.normal(-0.001, 0.015, 100)
        returns = np.concatenate([bull_period, bear_period])

        # Create and fit detector
        detector = create_hmm_detector(n_states=3, min_samples=50)
        detector.fit(returns)

        # Get analysis
        result = detector.analyze()
        assert result is not None

        # Get signal
        signal, rationale = detector.get_regime_signal()
        assert signal is not None

        # Check that we can serialize
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)

    def test_incremental_updates(self):
        """Test detector with incremental updates."""
        np.random.seed(456)
        initial_returns = np.random.normal(0.0005, 0.01, 100)

        detector = create_hmm_detector(n_states=3, min_samples=50)
        detector.fit(initial_returns)

        # Add new observations
        for _ in range(20):
            new_return = np.random.normal(0.001, 0.01)
            detector.update(new_return)

        # Should still work
        state = detector.predict_state()
        assert isinstance(state, MarketState)

    def test_different_n_states(self):
        """Test with different numbers of states."""
        np.random.seed(789)
        returns = np.random.normal(0.0, 0.01, 200)

        for n_states in [2, 3, 5]:
            detector = create_hmm_detector(n_states=n_states, min_samples=50)
            detector.fit(returns)

            result = detector.analyze()
            assert result is not None

            # Number of unique states should be <= n_states
            probs = detector.get_regime_probabilities()
            non_zero_states = sum(1 for p in probs.values() if p > 0.01)
            assert non_zero_states <= n_states
