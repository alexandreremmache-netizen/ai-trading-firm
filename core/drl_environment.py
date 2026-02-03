"""
DRL Environment for CIO Decision Making
=======================================

OpenAI Gym-compatible environment for training DRL agents
to make portfolio allocation decisions based on signal aggregation.

Inspired by:
- TradeMaster (https://github.com/TradeMaster-NTU/TradeMaster)
- attention_drl_trading (https://github.com/iriechie/attention_drl_trading)
- gym-continuousDoubleAuction patterns

Features:
- Observation: Signal strengths, confidences, market state
- Action: Portfolio weights per signal/strategy
- Reward: Risk-adjusted returns (Sharpe-like)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from typing import Any
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    HAS_GYMNASIUM = True
except ImportError:
    try:
        import gym
        from gym import spaces
        HAS_GYMNASIUM = True
    except ImportError:
        HAS_GYMNASIUM = False
        gym = None
        spaces = None

logger = logging.getLogger(__name__)


@dataclass
class SignalObservation:
    """Single signal observation for the environment."""
    agent_name: str
    direction: float  # -1 (SHORT), 0 (FLAT), 1 (LONG)
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0

    def to_array(self) -> np.ndarray:
        """Convert to numpy array [direction, strength, confidence]."""
        return np.array([self.direction, self.strength, self.confidence], dtype=np.float32)


@dataclass
class MarketState:
    """Market state features for observation."""
    vix_level: float = 20.0  # VIX index
    trend_strength: float = 0.0  # -1 to 1 (bearish to bullish)
    volatility_regime: float = 0.5  # 0 to 1 (low to high vol)
    correlation_regime: float = 0.5  # 0 to 1 (low to high correlation)
    momentum_regime: float = 0.0  # -1 to 1

    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([
            self.vix_level / 100.0,  # Normalize VIX
            self.trend_strength,
            self.volatility_regime,
            self.correlation_regime,
            self.momentum_regime,
        ], dtype=np.float32)


@dataclass
class PortfolioState:
    """Portfolio state for observation."""
    cash_ratio: float = 1.0  # Fraction in cash
    exposure: float = 0.0  # Net exposure (-1 to 1)
    leverage: float = 1.0  # Current leverage
    drawdown: float = 0.0  # Current drawdown from peak
    pnl_today: float = 0.0  # Today's P&L

    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([
            self.cash_ratio,
            self.exposure,
            self.leverage / 3.0,  # Normalize assuming max 3x
            self.drawdown,
            np.clip(self.pnl_today / 0.1, -1, 1),  # Normalize P&L
        ], dtype=np.float32)


@dataclass
class DRLEnvConfig:
    """Configuration for the DRL environment."""
    num_signal_agents: int = 8  # Number of signal agents
    num_symbols: int = 10  # Number of tradeable symbols
    max_leverage: float = 2.0
    transaction_cost: float = 0.001  # 10 bps
    reward_scaling: float = 100.0
    risk_free_rate: float = 0.02  # Annual
    lookback_window: int = 20  # Steps of history in observation
    episode_length: int = 252  # Trading days


class CIODecisionEnv:
    """
    Gym Environment for CIO Decision Making.

    The agent learns to allocate portfolio weights based on:
    - Signals from multiple signal agents
    - Market regime indicators
    - Current portfolio state

    Action Space:
        Continuous weights for each signal agent [0, 1] that sum to <= 1
        Remaining weight goes to cash

    Observation Space:
        - Signal features: [direction, strength, confidence] x num_agents
        - Market state: [vix, trend, volatility, correlation, momentum]
        - Portfolio state: [cash, exposure, leverage, drawdown, pnl]
        - Historical returns: lookback_window x 1

    Reward:
        Risk-adjusted returns with transaction cost penalty
    """

    def __init__(self, config: DRLEnvConfig | None = None):
        if not HAS_GYMNASIUM:
            raise ImportError(
                "gymnasium or gym required. Install with: pip install gymnasium"
            )

        self.config = config or DRLEnvConfig()

        # Agent names (will be populated from actual system)
        self.signal_agent_names = [
            "MacroAgent", "StatArbAgent", "MomentumAgent",
            "MarketMakingAgent", "OptionsVolAgent", "SentimentAgent",
            "ChartAnalysisAgent", "ForecastingAgent"
        ][:self.config.num_signal_agents]

        # Observation space dimensions
        self._signal_dim = self.config.num_signal_agents * 3  # dir, str, conf per agent
        self._market_dim = 5  # VIX, trend, vol, corr, momentum
        self._portfolio_dim = 5  # cash, exposure, leverage, dd, pnl
        self._history_dim = self.config.lookback_window
        self._obs_dim = self._signal_dim + self._market_dim + self._portfolio_dim + self._history_dim

        # Define spaces
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._obs_dim,),
            dtype=np.float32
        )

        # Action space: weight for each signal agent (will be normalized)
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.config.num_signal_agents,),
            dtype=np.float32
        )

        # State
        self._current_step = 0
        self._portfolio_value = 1_000_000.0
        self._peak_value = 1_000_000.0
        self._position_weights = np.zeros(self.config.num_signal_agents)
        self._cash_weight = 1.0
        self._returns_history = []

        # Current observations
        self._current_signals: list[SignalObservation] = []
        self._market_state = MarketState()
        self._portfolio_state = PortfolioState()

        # Episode tracking
        self._episode_returns = []
        self._episode_actions = []

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        """Reset the environment to initial state."""
        if seed is not None:
            np.random.seed(seed)

        self._current_step = 0
        self._portfolio_value = 1_000_000.0
        self._peak_value = 1_000_000.0
        self._position_weights = np.zeros(self.config.num_signal_agents)
        self._cash_weight = 1.0
        self._returns_history = [0.0] * self.config.lookback_window

        # Reset episode tracking
        self._episode_returns = []
        self._episode_actions = []

        # Initialize with neutral signals
        self._current_signals = [
            SignalObservation(name, 0.0, 0.0, 0.5)
            for name in self.signal_agent_names
        ]
        self._market_state = MarketState()
        self._portfolio_state = PortfolioState()

        obs = self._get_observation()
        info = {"step": 0}

        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one step in the environment.

        Args:
            action: Portfolio weights for each signal agent [0, 1]

        Returns:
            observation, reward, terminated, truncated, info
        """
        # Normalize action to valid weights
        action = np.clip(action, 0, 1)
        action_sum = action.sum()
        if action_sum > 1.0:
            action = action / action_sum

        # Calculate transaction costs from weight changes
        weight_changes = np.abs(action - self._position_weights)
        transaction_cost = weight_changes.sum() * self.config.transaction_cost

        # Update weights
        old_weights = self._position_weights.copy()
        self._position_weights = action
        self._cash_weight = 1.0 - action.sum()

        # Simulate returns based on signals and weights
        step_return = self._simulate_step_return(action)

        # Apply transaction cost
        net_return = step_return - transaction_cost

        # Update portfolio value
        self._portfolio_value *= (1 + net_return)
        self._peak_value = max(self._peak_value, self._portfolio_value)

        # Update history
        self._returns_history.append(net_return)
        if len(self._returns_history) > self.config.lookback_window:
            self._returns_history.pop(0)

        # Track episode
        self._episode_returns.append(net_return)
        self._episode_actions.append(action.copy())

        # Update portfolio state
        self._update_portfolio_state()

        # Calculate reward (risk-adjusted)
        reward = self._calculate_reward(net_return)

        # Check termination
        self._current_step += 1
        terminated = self._current_step >= self.config.episode_length
        truncated = False

        # Terminate on excessive drawdown
        drawdown = 1 - (self._portfolio_value / self._peak_value)
        if drawdown > 0.20:  # 20% drawdown limit
            terminated = True
            reward -= 10.0  # Penalty for blowing up

        obs = self._get_observation()
        info = {
            "step": self._current_step,
            "portfolio_value": self._portfolio_value,
            "drawdown": drawdown,
            "step_return": net_return,
            "transaction_cost": transaction_cost,
            "weights": action.tolist(),
        }

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Construct observation vector."""
        obs_parts = []

        # Signal features
        for signal in self._current_signals:
            obs_parts.append(signal.to_array())

        # Pad if fewer signals than expected
        while len(obs_parts) < self.config.num_signal_agents:
            obs_parts.append(np.zeros(3, dtype=np.float32))

        # Market state
        obs_parts.append(self._market_state.to_array())

        # Portfolio state
        obs_parts.append(self._portfolio_state.to_array())

        # Historical returns
        history = np.array(self._returns_history[-self.config.lookback_window:], dtype=np.float32)
        if len(history) < self.config.lookback_window:
            history = np.pad(history, (self.config.lookback_window - len(history), 0))
        obs_parts.append(history * 100)  # Scale returns

        return np.concatenate(obs_parts)

    def _simulate_step_return(self, weights: np.ndarray) -> float:
        """
        Simulate step return based on signal-weighted positions.

        In live trading, this would come from actual market moves.
        For training, we simulate based on signal accuracy.
        """
        total_return = 0.0

        for i, (signal, weight) in enumerate(zip(self._current_signals, weights)):
            if weight < 0.01:  # Skip negligible weights
                continue

            # Simulate signal-based return
            # Better signals (higher confidence) have better expected returns
            signal_accuracy = 0.5 + signal.confidence * 0.2  # 50-70% accuracy

            # Random market move
            market_move = np.random.normal(0.0005, 0.02)  # ~0.05% mean, 2% std daily

            # Signal contribution
            if np.random.random() < signal_accuracy:
                # Signal was correct
                signal_return = signal.direction * signal.strength * abs(market_move) * 1.5
            else:
                # Signal was wrong
                signal_return = -signal.direction * signal.strength * abs(market_move)

            total_return += weight * signal_return

        # Cash earns risk-free rate (daily)
        cash_return = self._cash_weight * (self.config.risk_free_rate / 252)
        total_return += cash_return

        return total_return

    def _update_portfolio_state(self) -> None:
        """Update portfolio state based on current positions."""
        self._portfolio_state.cash_ratio = self._cash_weight

        # Net exposure: sum of directional weights
        net_exposure = sum(
            w * s.direction * s.strength
            for w, s in zip(self._position_weights, self._current_signals)
        )
        self._portfolio_state.exposure = np.clip(net_exposure, -1, 1)

        # Leverage: sum of absolute weights
        self._portfolio_state.leverage = self._position_weights.sum()

        # Drawdown
        self._portfolio_state.drawdown = 1 - (self._portfolio_value / self._peak_value)

        # Today's P&L
        if self._episode_returns:
            self._portfolio_state.pnl_today = self._episode_returns[-1]

    def _calculate_reward(self, step_return: float) -> float:
        """
        Calculate reward for the step.

        Uses Sharpe-like reward with drawdown penalty.
        """
        # Base reward: scaled return
        reward = step_return * self.config.reward_scaling

        # Sharpe component (if enough history)
        if len(self._episode_returns) >= 20:
            recent_returns = np.array(self._episode_returns[-20:])
            if recent_returns.std() > 0:
                sharpe = recent_returns.mean() / recent_returns.std() * np.sqrt(252)
                reward += sharpe * 0.1  # Small bonus for good Sharpe

        # Drawdown penalty
        drawdown = self._portfolio_state.drawdown
        if drawdown > 0.05:  # > 5% drawdown
            reward -= drawdown * 5.0

        # Concentration penalty (encourage diversification)
        if self._position_weights.max() > 0.5:
            reward -= 0.1

        return reward

    def update_signals(self, signals: list[SignalObservation]) -> None:
        """Update current signals from live system."""
        self._current_signals = signals
        # Pad if needed
        while len(self._current_signals) < self.config.num_signal_agents:
            self._current_signals.append(
                SignalObservation(f"Agent_{len(self._current_signals)}", 0.0, 0.0, 0.5)
            )

    def update_market_state(self, state: MarketState) -> None:
        """Update market state from live system."""
        self._market_state = state

    def set_portfolio_value(self, value: float) -> None:
        """Set current portfolio value (for live integration)."""
        self._portfolio_value = value
        self._peak_value = max(self._peak_value, value)

    def get_episode_stats(self) -> dict[str, Any]:
        """Get statistics for the current episode."""
        if not self._episode_returns:
            return {}

        returns = np.array(self._episode_returns)
        return {
            "total_return": (self._portfolio_value / 1_000_000.0) - 1,
            "sharpe_ratio": returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0,
            "max_drawdown": self._portfolio_state.drawdown,
            "num_steps": self._current_step,
            "avg_return": returns.mean(),
            "volatility": returns.std() * np.sqrt(252),
        }


def make_cio_env(config: DRLEnvConfig | None = None) -> CIODecisionEnv:
    """Factory function to create CIO environment."""
    return CIODecisionEnv(config)
