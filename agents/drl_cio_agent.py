"""
DRL-Based CIO Agent
===================

Deep Reinforcement Learning CIO Agent using PPO/A2C for
signal aggregation and portfolio allocation decisions.

Inspired by:
- Quant Beckman's DRL approaches (A2C, PPO, TD3)
- TradeMaster framework
- attention_drl_trading for cross-sectional attention

Features:
- PPO/A2C from stable-baselines3 for decision making
- Learns optimal signal weights from experience
- Adapts to market regimes automatically
- Can be trained offline and deployed live
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any
import numpy as np

from core.agent_base import DecisionAgent, AgentConfig
from core.events import (
    Event,
    EventType,
    SignalEvent,
    DecisionEvent,
    SignalDirection,
    OrderSide,
    OrderType,
)
from core.drl_environment import (
    CIODecisionEnv,
    DRLEnvConfig,
    SignalObservation,
    MarketState,
)

# Optional DRL imports
try:
    from stable_baselines3 import PPO, A2C
    from stable_baselines3.common.callbacks import EvalCallback
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    PPO = None
    A2C = None

if TYPE_CHECKING:
    from core.event_bus import EventBus
    from core.logger import AuditLogger


logger = logging.getLogger(__name__)


@dataclass
class DRLCIOConfig:
    """Configuration for DRL CIO Agent."""
    algorithm: str = "PPO"  # "PPO" or "A2C"
    model_path: str | None = None  # Path to pre-trained model
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01  # Entropy coefficient for exploration
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    # Inference settings
    use_deterministic: bool = True  # Deterministic actions in live trading
    fallback_to_heuristic: bool = True  # Use heuristic if model unavailable
    min_conviction_threshold: float = 0.6


class DRLCIOAgent(DecisionAgent):
    """
    Deep Reinforcement Learning CIO Agent.

    Uses PPO or A2C from stable-baselines3 to learn optimal
    signal aggregation and portfolio allocation.

    The agent can be:
    1. Trained offline on historical data
    2. Loaded from a pre-trained model
    3. Used in live trading with deterministic actions

    If no trained model is available, falls back to heuristic
    weighted averaging (like the standard CIO agent).
    """

    def __init__(
        self,
        config: AgentConfig,
        event_bus: EventBus,
        audit_logger: AuditLogger,
    ):
        super().__init__(config, event_bus, audit_logger)

        # DRL configuration
        drl_params = config.parameters.get("drl", {})
        self._drl_config = DRLCIOConfig(
            algorithm=drl_params.get("algorithm", "PPO"),
            model_path=drl_params.get("model_path"),
            learning_rate=drl_params.get("learning_rate", 3e-4),
            n_steps=drl_params.get("n_steps", 2048),
            batch_size=drl_params.get("batch_size", 64),
            gamma=drl_params.get("gamma", 0.99),
            use_deterministic=drl_params.get("use_deterministic", True),
            fallback_to_heuristic=drl_params.get("fallback_to_heuristic", True),
            min_conviction_threshold=drl_params.get("min_conviction_threshold", 0.6),
        )

        # Environment configuration
        env_params = config.parameters.get("environment", {})
        self._env_config = DRLEnvConfig(
            num_signal_agents=env_params.get("num_signal_agents", 8),
            max_leverage=env_params.get("max_leverage", 2.0),
            transaction_cost=env_params.get("transaction_cost", 0.001),
        )

        # Signal agent mapping
        self._signal_agent_names = [
            "MacroAgent", "StatArbAgent", "MomentumAgent",
            "MarketMakingAgent", "OptionsVolAgent", "SentimentAgent",
            "ChartAnalysisAgent", "ForecastingAgent"
        ]

        # Fallback weights (used if no DRL model)
        self._fallback_weights = {
            "MacroAgent": config.parameters.get("signal_weight_macro", 0.10),
            "StatArbAgent": config.parameters.get("signal_weight_stat_arb", 0.20),
            "MomentumAgent": config.parameters.get("signal_weight_momentum", 0.20),
            "MarketMakingAgent": config.parameters.get("signal_weight_market_making", 0.10),
            "OptionsVolAgent": config.parameters.get("signal_weight_options_vol", 0.15),
            "SentimentAgent": config.parameters.get("signal_weight_sentiment", 0.10),
            "ChartAnalysisAgent": config.parameters.get("signal_weight_chart", 0.05),
            "ForecastingAgent": config.parameters.get("signal_weight_forecasting", 0.10),
        }

        # State
        self._model = None
        self._env = None
        self._model_loaded = False
        self._use_drl = HAS_SB3

        # Decision settings
        self._min_conviction = self._drl_config.min_conviction_threshold
        self._base_position_size = config.parameters.get("base_position_size", 100)
        self._max_position_size = config.parameters.get("max_position_size", 1000)
        self._portfolio_value = config.parameters.get("portfolio_value", 1_000_000.0)

        # Current market state
        self._current_market_state = MarketState()

        # Price cache
        self._price_cache: dict[str, float] = {}

        # Performance tracking
        self._decisions_made = 0
        self._drl_decisions = 0
        self._fallback_decisions = 0

    async def initialize(self) -> None:
        """Initialize the DRL CIO agent."""
        logger.info("DRL CIO Agent initializing...")

        if not HAS_SB3:
            logger.warning(
                "stable-baselines3 not installed. "
                "Install with: pip install stable-baselines3. "
                "Falling back to heuristic weights."
            )
            self._use_drl = False
            return

        # Create environment
        self._env = CIODecisionEnv(self._env_config)

        # Load or create model
        if self._drl_config.model_path and Path(self._drl_config.model_path).exists():
            self._load_model(self._drl_config.model_path)
        else:
            self._create_model()

        logger.info(
            f"DRL CIO Agent initialized with {self._drl_config.algorithm}. "
            f"Model loaded: {self._model_loaded}"
        )

    def _create_model(self) -> None:
        """Create a new DRL model."""
        if not HAS_SB3 or self._env is None:
            return

        algorithm = self._drl_config.algorithm.upper()

        if algorithm == "PPO":
            self._model = PPO(
                "MlpPolicy",
                self._env,
                learning_rate=self._drl_config.learning_rate,
                n_steps=self._drl_config.n_steps,
                batch_size=self._drl_config.batch_size,
                n_epochs=self._drl_config.n_epochs,
                gamma=self._drl_config.gamma,
                gae_lambda=self._drl_config.gae_lambda,
                clip_range=self._drl_config.clip_range,
                ent_coef=self._drl_config.ent_coef,
                vf_coef=self._drl_config.vf_coef,
                max_grad_norm=self._drl_config.max_grad_norm,
                verbose=0,
            )
        elif algorithm == "A2C":
            self._model = A2C(
                "MlpPolicy",
                self._env,
                learning_rate=self._drl_config.learning_rate,
                n_steps=self._drl_config.n_steps,
                gamma=self._drl_config.gamma,
                gae_lambda=self._drl_config.gae_lambda,
                ent_coef=self._drl_config.ent_coef,
                vf_coef=self._drl_config.vf_coef,
                max_grad_norm=self._drl_config.max_grad_norm,
                verbose=0,
            )
        else:
            logger.error(f"Unknown algorithm: {algorithm}")
            return

        logger.info(f"Created new {algorithm} model (untrained)")

    def _load_model(self, path: str) -> None:
        """Load a pre-trained model."""
        if not HAS_SB3:
            return

        try:
            algorithm = self._drl_config.algorithm.upper()
            if algorithm == "PPO":
                self._model = PPO.load(path, env=self._env)
            elif algorithm == "A2C":
                self._model = A2C.load(path, env=self._env)

            self._model_loaded = True
            logger.info(f"Loaded pre-trained {algorithm} model from {path}")
        except Exception as e:
            logger.exception(f"Failed to load model from {path}: {e}")
            self._create_model()

    def save_model(self, path: str) -> None:
        """Save the current model."""
        if self._model is not None:
            self._model.save(path)
            logger.info(f"Saved model to {path}")

    def train(self, total_timesteps: int = 100_000) -> dict[str, Any]:
        """
        Train the DRL model.

        Args:
            total_timesteps: Total training steps

        Returns:
            Training statistics
        """
        if not HAS_SB3 or self._model is None:
            logger.error("Cannot train: stable-baselines3 or model not available")
            return {}

        logger.info(f"Starting training for {total_timesteps} timesteps...")

        # Train
        self._model.learn(total_timesteps=total_timesteps)
        self._model_loaded = True

        # Get final stats
        if self._env is not None:
            stats = self._env.get_episode_stats()
        else:
            stats = {}

        logger.info(f"Training complete. Stats: {stats}")
        return stats

    async def process_signals(
        self,
        signals: dict[str, SignalEvent],
        symbol: str,
    ) -> DecisionEvent | None:
        """
        Process aggregated signals and make a decision.

        This is the main decision-making method called by the orchestrator
        after the signal barrier synchronizes.

        Args:
            signals: Dict of agent_name -> SignalEvent
            symbol: Symbol to decide on

        Returns:
            DecisionEvent if action warranted, None otherwise
        """
        self._decisions_made += 1

        # Convert signals to observations
        signal_obs = self._signals_to_observations(signals)

        # Get action from DRL model or fallback
        if self._use_drl and self._model is not None and self._model_loaded:
            weights = self._get_drl_action(signal_obs)
            self._drl_decisions += 1
        else:
            weights = self._get_heuristic_action(signals)
            self._fallback_decisions += 1

        # Aggregate signals with learned weights
        aggregation = self._aggregate_signals(signals, weights)

        # Check conviction threshold
        if aggregation["conviction"] < self._min_conviction:
            logger.debug(
                f"Conviction {aggregation['conviction']:.2f} below threshold "
                f"{self._min_conviction} for {symbol}"
            )
            return None

        # Determine direction
        if aggregation["weighted_strength"] > 0.1:
            direction = SignalDirection.LONG
            side = OrderSide.BUY
        elif aggregation["weighted_strength"] < -0.1:
            direction = SignalDirection.SHORT
            side = OrderSide.SELL
        else:
            return None

        # Calculate position size
        quantity = self._calculate_position_size(
            symbol,
            aggregation["conviction"],
            aggregation["weighted_strength"],
        )

        if quantity <= 0:
            return None

        # Create decision event
        decision = DecisionEvent(
            source_agent=self.name,
            symbol=symbol,
            direction=direction,
            action=side,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            conviction_score=aggregation["conviction"],
            rationale=self._build_rationale(signals, weights, aggregation),
            data_sources=tuple(s.data_sources for s in signals.values()),
            contributing_signals=tuple(signals.keys()),
            metadata={
                "drl_decision": self._use_drl and self._model_loaded,
                "weights": {k: float(v) for k, v in zip(self._signal_agent_names, weights)},
                "weighted_strength": aggregation["weighted_strength"],
            },
        )

        self._audit_logger.log_decision(
            agent_name=self.name,
            decision_id=decision.event_id,
            symbol=symbol,
            action=decision.action.value,
            quantity=quantity,
            rationale=decision.rationale,
            data_sources=list(decision.data_sources),
            contributing_signals=list(decision.contributing_signals),
            conviction_score=decision.conviction_score,
        )

        return decision

    def _signals_to_observations(
        self,
        signals: dict[str, SignalEvent],
    ) -> list[SignalObservation]:
        """Convert SignalEvents to observations for the DRL model."""
        observations = []

        for agent_name in self._signal_agent_names:
            if agent_name in signals:
                signal = signals[agent_name]
                direction_val = {
                    SignalDirection.LONG: 1.0,
                    SignalDirection.SHORT: -1.0,
                    SignalDirection.FLAT: 0.0,
                }.get(signal.direction, 0.0)

                observations.append(SignalObservation(
                    agent_name=agent_name,
                    direction=direction_val,
                    strength=signal.strength,
                    confidence=signal.confidence,
                ))
            else:
                # Missing signal - neutral
                observations.append(SignalObservation(
                    agent_name=agent_name,
                    direction=0.0,
                    strength=0.0,
                    confidence=0.5,
                ))

        return observations

    def _get_drl_action(self, signal_obs: list[SignalObservation]) -> np.ndarray:
        """Get action (weights) from DRL model."""
        if self._env is None or self._model is None:
            return self._get_default_weights()

        # Update environment with current signals
        self._env.update_signals(signal_obs)
        self._env.update_market_state(self._current_market_state)

        # Get observation
        obs = self._env._get_observation()

        # Predict action
        action, _ = self._model.predict(
            obs,
            deterministic=self._drl_config.use_deterministic,
        )

        # Normalize to sum <= 1
        action = np.clip(action, 0, 1)
        if action.sum() > 1.0:
            action = action / action.sum()

        return action

    def _get_heuristic_action(self, signals: dict[str, SignalEvent]) -> np.ndarray:
        """Get heuristic weights (fallback when no DRL model)."""
        weights = []
        for agent_name in self._signal_agent_names:
            base_weight = self._fallback_weights.get(agent_name, 0.1)

            # Adjust by signal confidence if available
            if agent_name in signals:
                confidence = signals[agent_name].confidence
                weight = base_weight * (0.5 + confidence * 0.5)
            else:
                weight = base_weight * 0.5  # Reduce weight for missing signals

            weights.append(weight)

        weights = np.array(weights)

        # Normalize
        if weights.sum() > 0:
            weights = weights / weights.sum()

        return weights

    def _get_default_weights(self) -> np.ndarray:
        """Get default equal weights."""
        n = len(self._signal_agent_names)
        return np.ones(n) / n

    def _aggregate_signals(
        self,
        signals: dict[str, SignalEvent],
        weights: np.ndarray,
    ) -> dict[str, Any]:
        """Aggregate signals using learned weights."""
        weighted_strength = 0.0
        weighted_confidence = 0.0
        total_weight = 0.0

        for i, agent_name in enumerate(self._signal_agent_names):
            if agent_name not in signals:
                continue

            signal = signals[agent_name]
            weight = weights[i]

            if weight < 0.01:
                continue

            direction_val = {
                SignalDirection.LONG: 1.0,
                SignalDirection.SHORT: -1.0,
                SignalDirection.FLAT: 0.0,
            }.get(signal.direction, 0.0)

            weighted_strength += weight * direction_val * signal.strength
            weighted_confidence += weight * signal.confidence
            total_weight += weight

        if total_weight > 0:
            weighted_confidence /= total_weight

        # Conviction = weighted confidence with strength adjustment
        conviction = weighted_confidence * (0.5 + min(abs(weighted_strength), 0.5))

        return {
            "weighted_strength": weighted_strength,
            "weighted_confidence": weighted_confidence,
            "conviction": conviction,
            "total_weight": total_weight,
        }

    def _calculate_position_size(
        self,
        symbol: str,
        conviction: float,
        strength: float,
    ) -> int:
        """Calculate position size based on conviction and strength."""
        # Get price
        price = self._price_cache.get(symbol, 100.0)

        # Base size from portfolio value
        position_value = self._portfolio_value * 0.02 * conviction  # 2% max per position
        base_size = int(position_value / price)

        # Adjust by strength
        size = int(base_size * (0.5 + abs(strength) * 0.5))

        # Apply limits
        size = max(self._base_position_size // 10, size)
        size = min(self._max_position_size, size)

        return size

    def _build_rationale(
        self,
        signals: dict[str, SignalEvent],
        weights: np.ndarray,
        aggregation: dict[str, Any],
    ) -> str:
        """Build rationale string for the decision."""
        parts = [f"DRL CIO Decision (model={'trained' if self._model_loaded else 'heuristic'})"]

        # Top contributing signals
        contributions = []
        for i, agent_name in enumerate(self._signal_agent_names):
            if agent_name in signals and weights[i] > 0.05:
                signal = signals[agent_name]
                contributions.append(
                    f"{agent_name}({signal.direction.value}, "
                    f"w={weights[i]:.2f}, c={signal.confidence:.2f})"
                )

        if contributions:
            parts.append(f"Signals: {', '.join(contributions[:4])}")

        parts.append(
            f"Aggregated: strength={aggregation['weighted_strength']:.2f}, "
            f"conviction={aggregation['conviction']:.2f}"
        )

        return ". ".join(parts)

    def update_market_state(
        self,
        vix: float | None = None,
        trend: float | None = None,
        volatility: float | None = None,
    ) -> None:
        """Update market state for DRL model."""
        if vix is not None:
            self._current_market_state.vix_level = vix
        if trend is not None:
            self._current_market_state.trend_strength = trend
        if volatility is not None:
            self._current_market_state.volatility_regime = volatility

    def update_price(self, symbol: str, price: float) -> None:
        """Update price cache."""
        self._price_cache[symbol] = price

    def get_stats(self) -> dict[str, Any]:
        """Get agent statistics."""
        return {
            "decisions_made": self._decisions_made,
            "drl_decisions": self._drl_decisions,
            "fallback_decisions": self._fallback_decisions,
            "drl_ratio": self._drl_decisions / max(1, self._decisions_made),
            "model_loaded": self._model_loaded,
            "algorithm": self._drl_config.algorithm,
            "use_drl": self._use_drl,
        }

    async def process_event(self, event: Event) -> None:
        """Process incoming events (mainly for price updates)."""
        if event.event_type == EventType.MARKET_DATA:
            if hasattr(event, 'symbol') and hasattr(event, 'last_price'):
                if event.last_price:
                    self.update_price(event.symbol, event.last_price)
