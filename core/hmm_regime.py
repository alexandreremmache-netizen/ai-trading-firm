"""
Hidden Markov Model Regime Detection
====================================

HMM-based market regime detection for strategy adaptation.

Uses Gaussian Hidden Markov Models to identify market states:
- BULL: Strong upward momentum, low volatility
- BEAR: Strong downward momentum, elevated volatility
- SIDEWAYS: Range-bound, mean-reverting conditions
- HIGH_VOL: Elevated volatility, uncertain direction
- LOW_VOL: Compressed volatility, potential breakout setup

References:
- Rabiner, L.R. (1989) "A Tutorial on Hidden Markov Models"
- Hamilton, J.D. (1989) "Regime Switching Models"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

# NumPy and SciPy are required for HMM calculations
try:
    import numpy as np
    from numpy.typing import NDArray
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None  # type: ignore
    NDArray = Any  # type: ignore

try:
    from scipy import stats
    from scipy.special import logsumexp
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    stats = None  # type: ignore
    logsumexp = None  # type: ignore


logger = logging.getLogger(__name__)


class MarketState(str, Enum):
    """
    Market state classification for HMM regime detection.

    Each state represents a distinct market regime with
    characteristic return and volatility patterns.
    """
    BULL = "bull"           # Positive returns, low-moderate volatility
    BEAR = "bear"           # Negative returns, elevated volatility
    SIDEWAYS = "sideways"   # Near-zero returns, moderate volatility
    HIGH_VOL = "high_vol"   # High volatility, direction uncertain
    LOW_VOL = "low_vol"     # Low volatility, potential compression


@dataclass
class HMMRegimeResult:
    """Result of HMM regime detection."""
    current_state: MarketState
    state_probabilities: dict[MarketState, float]
    expected_return: float
    expected_volatility: float
    transition_matrix: dict[str, dict[str, float]]
    confidence: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "current_state": self.current_state.value,
            "state_probabilities": {k.value: v for k, v in self.state_probabilities.items()},
            "expected_return": self.expected_return,
            "expected_volatility": self.expected_volatility,
            "transition_matrix": self.transition_matrix,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
        }


class GaussianHMM:
    """
    Gaussian Hidden Markov Model implementation.

    Implements the Baum-Welch algorithm for parameter estimation
    and Viterbi algorithm for state sequence decoding.

    This is a simplified implementation suitable for regime detection.
    For production use, consider hmmlearn library.
    """

    def __init__(
        self,
        n_states: int = 3,
        n_iterations: int = 100,
        tolerance: float = 1e-4,
        random_state: int | None = None,
    ):
        """
        Initialize Gaussian HMM.

        Args:
            n_states: Number of hidden states
            n_iterations: Maximum EM iterations
            tolerance: Convergence tolerance for log-likelihood
            random_state: Random seed for reproducibility
        """
        if not HAS_NUMPY or not HAS_SCIPY:
            raise ImportError(
                "NumPy and SciPy are required for HMM. "
                "Install with: pip install numpy scipy"
            )

        self.n_states = n_states
        self.n_iterations = n_iterations
        self.tolerance = tolerance
        self.random_state = random_state

        # Model parameters (initialized in fit)
        self.means_: NDArray | None = None
        self.variances_: NDArray | None = None
        self.transmat_: NDArray | None = None
        self.startprob_: NDArray | None = None

        self._is_fitted = False

    def _initialize_parameters(self, X: NDArray) -> None:
        """Initialize HMM parameters using k-means-like approach."""
        rng = np.random.default_rng(self.random_state)

        n_samples = len(X)

        # Initialize start probabilities uniformly
        self.startprob_ = np.ones(self.n_states) / self.n_states

        # Initialize transition matrix with slight preference for staying
        self.transmat_ = np.ones((self.n_states, self.n_states)) / self.n_states
        for i in range(self.n_states):
            self.transmat_[i, i] += 0.3
        self.transmat_ /= self.transmat_.sum(axis=1, keepdims=True)

        # Initialize means by sorting data into quantiles
        sorted_X = np.sort(X)
        quantile_indices = np.linspace(0, n_samples - 1, self.n_states + 1).astype(int)
        self.means_ = np.array([
            sorted_X[quantile_indices[i]:quantile_indices[i + 1]].mean()
            for i in range(self.n_states)
        ])

        # Sort means for interpretability (low to high)
        self.means_ = np.sort(self.means_)

        # Initialize variances
        overall_var = np.var(X)
        self.variances_ = np.full(self.n_states, overall_var) + rng.uniform(0.001, 0.01, self.n_states)

    def _compute_log_likelihood(self, X: NDArray) -> NDArray:
        """Compute log-likelihood of observations under each state."""
        n_samples = len(X)
        log_likelihood = np.zeros((n_samples, self.n_states))

        for k in range(self.n_states):
            # Gaussian log-likelihood
            log_likelihood[:, k] = stats.norm.logpdf(
                X, loc=self.means_[k], scale=np.sqrt(self.variances_[k])
            )

        return log_likelihood

    def _forward(self, log_likelihood: NDArray) -> tuple[NDArray, float]:
        """
        Forward algorithm (alpha pass).

        Returns log-scaled forward probabilities and total log-likelihood.
        """
        n_samples = log_likelihood.shape[0]
        log_alpha = np.zeros((n_samples, self.n_states))

        # Initialize
        log_alpha[0] = np.log(self.startprob_ + 1e-10) + log_likelihood[0]

        # Forward pass
        log_transmat = np.log(self.transmat_ + 1e-10)
        for t in range(1, n_samples):
            for j in range(self.n_states):
                log_alpha[t, j] = logsumexp(
                    log_alpha[t - 1] + log_transmat[:, j]
                ) + log_likelihood[t, j]

        # Total log-likelihood
        total_log_likelihood = logsumexp(log_alpha[-1])

        return log_alpha, total_log_likelihood

    def _backward(self, log_likelihood: NDArray) -> NDArray:
        """Backward algorithm (beta pass)."""
        n_samples = log_likelihood.shape[0]
        log_beta = np.zeros((n_samples, self.n_states))

        # Initialize (log(1) = 0)
        log_beta[-1] = 0

        # Backward pass
        log_transmat = np.log(self.transmat_ + 1e-10)
        for t in range(n_samples - 2, -1, -1):
            for i in range(self.n_states):
                log_beta[t, i] = logsumexp(
                    log_transmat[i, :] + log_likelihood[t + 1] + log_beta[t + 1]
                )

        return log_beta

    def _compute_posteriors(
        self, log_alpha: NDArray, log_beta: NDArray, log_likelihood: NDArray
    ) -> tuple[NDArray, NDArray]:
        """Compute posterior state probabilities (gamma) and transition posteriors (xi)."""
        n_samples = log_alpha.shape[0]

        # Gamma (state occupation probabilities)
        log_gamma = log_alpha + log_beta
        log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
        gamma = np.exp(log_gamma)

        # Xi (transition probabilities)
        xi = np.zeros((n_samples - 1, self.n_states, self.n_states))
        log_transmat = np.log(self.transmat_ + 1e-10)

        for t in range(n_samples - 1):
            for i in range(self.n_states):
                for j in range(self.n_states):
                    xi[t, i, j] = (
                        log_alpha[t, i] +
                        log_transmat[i, j] +
                        log_likelihood[t + 1, j] +
                        log_beta[t + 1, j]
                    )
            xi[t] -= logsumexp(xi[t])

        xi = np.exp(xi)

        return gamma, xi

    def fit(self, X: NDArray) -> "GaussianHMM":
        """
        Fit HMM parameters using Baum-Welch (EM) algorithm.

        Args:
            X: 1D array of observations (returns)

        Returns:
            self
        """
        X = np.asarray(X).flatten()
        n_samples = len(X)

        if n_samples < self.n_states * 2:
            raise ValueError(
                f"Need at least {self.n_states * 2} samples, got {n_samples}"
            )

        # Initialize parameters
        self._initialize_parameters(X)

        prev_log_likelihood = -np.inf

        for iteration in range(self.n_iterations):
            # E-step
            log_likelihood = self._compute_log_likelihood(X)
            log_alpha, total_log_likelihood = self._forward(log_likelihood)
            log_beta = self._backward(log_likelihood)
            gamma, xi = self._compute_posteriors(log_alpha, log_beta, log_likelihood)

            # Check convergence
            improvement = total_log_likelihood - prev_log_likelihood
            if abs(improvement) < self.tolerance and iteration > 0:
                logger.debug(f"HMM converged at iteration {iteration}")
                break
            prev_log_likelihood = total_log_likelihood

            # M-step: Update parameters
            # Update start probabilities
            self.startprob_ = gamma[0] + 1e-10
            self.startprob_ /= self.startprob_.sum()

            # Update transition matrix
            xi_sum = xi.sum(axis=0)
            self.transmat_ = xi_sum / (xi_sum.sum(axis=1, keepdims=True) + 1e-10)

            # Update means and variances
            gamma_sum = gamma.sum(axis=0) + 1e-10
            for k in range(self.n_states):
                self.means_[k] = np.sum(gamma[:, k] * X) / gamma_sum[k]
                self.variances_[k] = (
                    np.sum(gamma[:, k] * (X - self.means_[k]) ** 2) / gamma_sum[k]
                )
                # Ensure minimum variance
                self.variances_[k] = max(self.variances_[k], 1e-6)

        # Sort states by mean for consistent interpretation
        sort_idx = np.argsort(self.means_)
        self.means_ = self.means_[sort_idx]
        self.variances_ = self.variances_[sort_idx]
        self.transmat_ = self.transmat_[sort_idx][:, sort_idx]
        self.startprob_ = self.startprob_[sort_idx]

        self._is_fitted = True
        logger.info(f"HMM fitted with {self.n_states} states on {n_samples} samples")

        return self

    def predict(self, X: NDArray) -> NDArray:
        """
        Predict most likely state sequence using Viterbi algorithm.

        Args:
            X: 1D array of observations

        Returns:
            Array of state indices
        """
        if not self._is_fitted:
            raise RuntimeError("HMM must be fitted before prediction")

        X = np.asarray(X).flatten()
        n_samples = len(X)

        log_likelihood = self._compute_log_likelihood(X)
        log_transmat = np.log(self.transmat_ + 1e-10)

        # Viterbi algorithm
        viterbi = np.zeros((n_samples, self.n_states))
        backpointer = np.zeros((n_samples, self.n_states), dtype=int)

        # Initialize
        viterbi[0] = np.log(self.startprob_ + 1e-10) + log_likelihood[0]

        # Forward pass
        for t in range(1, n_samples):
            for j in range(self.n_states):
                trans_probs = viterbi[t - 1] + log_transmat[:, j]
                backpointer[t, j] = np.argmax(trans_probs)
                viterbi[t, j] = trans_probs[backpointer[t, j]] + log_likelihood[t, j]

        # Backtrack
        states = np.zeros(n_samples, dtype=int)
        states[-1] = np.argmax(viterbi[-1])
        for t in range(n_samples - 2, -1, -1):
            states[t] = backpointer[t + 1, states[t + 1]]

        return states

    def predict_proba(self, X: NDArray) -> NDArray:
        """
        Predict state probabilities for each observation.

        Args:
            X: 1D array of observations

        Returns:
            Array of shape (n_samples, n_states) with probabilities
        """
        if not self._is_fitted:
            raise RuntimeError("HMM must be fitted before prediction")

        X = np.asarray(X).flatten()

        log_likelihood = self._compute_log_likelihood(X)
        log_alpha, _ = self._forward(log_likelihood)
        log_beta = self._backward(log_likelihood)

        # Posterior probabilities
        log_gamma = log_alpha + log_beta
        log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)

        return np.exp(log_gamma)


class HMMRegimeDetector:
    """
    HMM-based market regime detector for trading strategies.

    Wraps the GaussianHMM with trading-specific functionality:
    - Maps HMM states to interpretable market regimes
    - Provides expected returns and volatility by regime
    - Calculates regime transition probabilities
    - Generates trading signals based on regime
    """

    def __init__(
        self,
        n_states: int = 3,
        min_samples: int = 100,
        lookback_period: int = 252,  # 1 year of daily data
        refit_frequency: int = 20,   # Refit every 20 observations
    ):
        """
        Initialize HMM regime detector.

        Args:
            n_states: Number of hidden states (3 or 5 recommended)
            min_samples: Minimum samples required for fitting
            lookback_period: Rolling window for fitting
            refit_frequency: How often to refit the model
        """
        self.n_states = n_states
        self.min_samples = min_samples
        self.lookback_period = lookback_period
        self.refit_frequency = refit_frequency

        self._hmm: GaussianHMM | None = None
        self._returns_history: list[float] = []
        self._observation_count = 0
        self._is_fitted = False

        # State mapping (will be determined after fitting)
        self._state_mapping: dict[int, MarketState] = {}

        # Performance tracking
        self._regime_returns: dict[MarketState, list[float]] = {
            state: [] for state in MarketState
        }

    def fit(self, returns: NDArray | list[float]) -> "HMMRegimeDetector":
        """
        Train HMM on historical returns.

        Args:
            returns: Array of historical returns

        Returns:
            self
        """
        if not HAS_NUMPY:
            raise ImportError("NumPy is required for HMM fitting")

        returns = np.asarray(returns).flatten()

        if len(returns) < self.min_samples:
            raise ValueError(
                f"Need at least {self.min_samples} samples, got {len(returns)}"
            )

        # Store returns history
        self._returns_history = list(returns[-self.lookback_period:])

        # Fit HMM
        self._hmm = GaussianHMM(n_states=self.n_states)
        self._hmm.fit(returns[-self.lookback_period:])

        # Map states to market regimes based on mean returns
        self._map_states_to_regimes()

        self._is_fitted = True
        logger.info(f"HMM regime detector fitted with {self.n_states} states")

        return self

    def _map_states_to_regimes(self) -> None:
        """Map HMM states to interpretable market regimes."""
        if self._hmm is None or self._hmm.means_ is None:
            return

        means = self._hmm.means_
        variances = self._hmm.variances_
        n_states = self.n_states

        # Sort states by mean return
        state_order = np.argsort(means)

        if n_states == 3:
            # 3-state model: BEAR, SIDEWAYS, BULL
            self._state_mapping = {
                int(state_order[0]): MarketState.BEAR,
                int(state_order[1]): MarketState.SIDEWAYS,
                int(state_order[2]): MarketState.BULL,
            }
        elif n_states == 5:
            # 5-state model: all states
            # Use volatility to distinguish HIGH_VOL and LOW_VOL
            vol_order = np.argsort(variances)

            self._state_mapping = {}
            for i, state_idx in enumerate(state_order):
                if i == 0:
                    self._state_mapping[int(state_idx)] = MarketState.BEAR
                elif i == n_states - 1:
                    self._state_mapping[int(state_idx)] = MarketState.BULL
                else:
                    # Middle states - use volatility
                    if variances[state_idx] == variances[vol_order[-1]]:
                        self._state_mapping[int(state_idx)] = MarketState.HIGH_VOL
                    elif variances[state_idx] == variances[vol_order[0]]:
                        self._state_mapping[int(state_idx)] = MarketState.LOW_VOL
                    else:
                        self._state_mapping[int(state_idx)] = MarketState.SIDEWAYS
        else:
            # Default mapping for other n_states
            for i, state_idx in enumerate(state_order):
                if i == 0:
                    self._state_mapping[int(state_idx)] = MarketState.BEAR
                elif i == n_states - 1:
                    self._state_mapping[int(state_idx)] = MarketState.BULL
                else:
                    self._state_mapping[int(state_idx)] = MarketState.SIDEWAYS

    def update(self, new_return: float) -> None:
        """
        Update model with new return observation.

        Args:
            new_return: Latest return value
        """
        self._returns_history.append(new_return)
        self._observation_count += 1

        # Maintain rolling window
        if len(self._returns_history) > self.lookback_period:
            self._returns_history = self._returns_history[-self.lookback_period:]

        # Periodic refitting
        if (
            self._observation_count % self.refit_frequency == 0
            and len(self._returns_history) >= self.min_samples
        ):
            self._refit()

    def _refit(self) -> None:
        """Refit the HMM on current history."""
        if len(self._returns_history) < self.min_samples:
            return

        try:
            self._hmm = GaussianHMM(n_states=self.n_states)
            self._hmm.fit(np.array(self._returns_history))
            self._map_states_to_regimes()
            logger.debug("HMM regime detector refitted")
        except Exception as e:
            logger.exception(f"Error refitting HMM: {e}")

    def predict_state(self, returns: NDArray | list[float] | None = None) -> MarketState:
        """
        Get current regime prediction.

        Args:
            returns: Optional returns to predict on (uses history if None)

        Returns:
            Predicted market state
        """
        if not self._is_fitted or self._hmm is None:
            raise RuntimeError("Model must be fitted before prediction")

        if returns is None:
            if not self._returns_history:
                raise ValueError("No returns history available")
            returns = np.array(self._returns_history)
        else:
            returns = np.asarray(returns)

        # Get most likely current state
        states = self._hmm.predict(returns)
        current_state_idx = int(states[-1])

        return self._state_mapping.get(current_state_idx, MarketState.SIDEWAYS)

    def get_regime_probabilities(
        self, returns: NDArray | list[float] | None = None
    ) -> dict[MarketState, float]:
        """
        Get probability distribution over regimes.

        Args:
            returns: Optional returns to predict on

        Returns:
            Dictionary mapping each regime to its probability
        """
        if not self._is_fitted or self._hmm is None:
            raise RuntimeError("Model must be fitted before prediction")

        if returns is None:
            if not self._returns_history:
                raise ValueError("No returns history available")
            returns = np.array(self._returns_history)
        else:
            returns = np.asarray(returns)

        # Get state probabilities
        state_probs = self._hmm.predict_proba(returns)
        current_probs = state_probs[-1]

        # Map to regime probabilities
        regime_probs: dict[MarketState, float] = {}
        for state_idx, regime in self._state_mapping.items():
            prob = float(current_probs[state_idx])
            if regime in regime_probs:
                regime_probs[regime] += prob
            else:
                regime_probs[regime] = prob

        # Ensure all regimes have a probability (even if 0)
        for state in MarketState:
            if state not in regime_probs:
                regime_probs[state] = 0.0

        return regime_probs

    def get_transition_matrix(self) -> dict[str, dict[str, float]]:
        """
        Get regime transition probability matrix.

        Returns:
            Nested dict of transition probabilities [from_state][to_state]
        """
        if not self._is_fitted or self._hmm is None:
            raise RuntimeError("Model must be fitted before getting transitions")

        transmat = self._hmm.transmat_

        result: dict[str, dict[str, float]] = {}
        for from_idx, from_regime in self._state_mapping.items():
            from_name = from_regime.value
            if from_name not in result:
                result[from_name] = {}
            for to_idx, to_regime in self._state_mapping.items():
                to_name = to_regime.value
                prob = float(transmat[from_idx, to_idx])
                if to_name in result[from_name]:
                    result[from_name][to_name] += prob
                else:
                    result[from_name][to_name] = prob

        return result

    def get_expected_returns_by_regime(self) -> dict[MarketState, float]:
        """
        Get expected (mean) return for each regime.

        Returns:
            Dictionary mapping regime to expected daily return
        """
        if not self._is_fitted or self._hmm is None:
            raise RuntimeError("Model must be fitted")

        means = self._hmm.means_

        result: dict[MarketState, float] = {}
        for state_idx, regime in self._state_mapping.items():
            result[regime] = float(means[state_idx])

        return result

    def get_volatility_by_regime(self) -> dict[MarketState, float]:
        """
        Get volatility (std dev) for each regime.

        Returns:
            Dictionary mapping regime to daily volatility
        """
        if not self._is_fitted or self._hmm is None:
            raise RuntimeError("Model must be fitted")

        variances = self._hmm.variances_

        result: dict[MarketState, float] = {}
        for state_idx, regime in self._state_mapping.items():
            result[regime] = float(np.sqrt(variances[state_idx]))

        return result

    def analyze(
        self, returns: NDArray | list[float] | None = None
    ) -> HMMRegimeResult:
        """
        Full regime analysis with all metrics.

        Args:
            returns: Optional returns to analyze

        Returns:
            HMMRegimeResult with complete analysis
        """
        current_state = self.predict_state(returns)
        state_probs = self.get_regime_probabilities(returns)

        expected_returns = self.get_expected_returns_by_regime()
        volatility = self.get_volatility_by_regime()

        # Calculate weighted expected return and volatility
        weighted_return = sum(
            state_probs.get(state, 0) * expected_returns.get(state, 0)
            for state in MarketState
        )
        weighted_vol = sum(
            state_probs.get(state, 0) * volatility.get(state, 0)
            for state in MarketState
        )

        # Confidence is the probability of the current state
        confidence = state_probs.get(current_state, 0.5)

        return HMMRegimeResult(
            current_state=current_state,
            state_probabilities=state_probs,
            expected_return=weighted_return,
            expected_volatility=weighted_vol,
            transition_matrix=self.get_transition_matrix(),
            confidence=confidence,
        )

    def get_regime_signal(
        self,
        returns: NDArray | list[float] | None = None,
    ) -> tuple[float, str]:
        """
        Generate trading signal based on regime.

        Args:
            returns: Optional returns to analyze

        Returns:
            (signal_strength, rationale) tuple
            signal_strength: -1 to +1 where positive is bullish
        """
        result = self.analyze(returns)
        state = result.current_state
        confidence = result.confidence

        # Map states to signals
        state_signals = {
            MarketState.BULL: (0.8, "HMM detects bullish regime"),
            MarketState.BEAR: (-0.8, "HMM detects bearish regime"),
            MarketState.SIDEWAYS: (0.0, "HMM detects sideways regime"),
            MarketState.HIGH_VOL: (-0.3, "HMM detects high volatility regime"),
            MarketState.LOW_VOL: (0.2, "HMM detects low volatility (breakout setup)"),
        }

        base_signal, rationale = state_signals.get(
            state, (0.0, "Unknown regime")
        )

        # Scale by confidence
        adjusted_signal = base_signal * confidence

        rationale = f"{rationale} (confidence: {confidence:.0%})"

        return adjusted_signal, rationale


def create_hmm_detector(
    n_states: int = 3,
    min_samples: int = 100,
    lookback_period: int = 252,
) -> HMMRegimeDetector:
    """
    Factory function to create HMM regime detector.

    Args:
        n_states: Number of hidden states (3 or 5)
        min_samples: Minimum samples for fitting
        lookback_period: Rolling window size

    Returns:
        Configured HMMRegimeDetector instance
    """
    return HMMRegimeDetector(
        n_states=n_states,
        min_samples=min_samples,
        lookback_period=lookback_period,
    )
