"""
Trading Gym Environment
=======================

OpenAI Gym-style environment for backtesting and training RL agents.
Inspired by TradingGym (https://github.com/Yvictor/TradingGym).

Features:
- Gym-compatible interface (reset, step, render)
- Configurable observation space (OHLCV, indicators)
- Action space: HOLD, BUY, SELL
- Multiple reward functions
- Transaction costs and slippage modeling
- Episode tracking and statistics
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from typing import Any
import numpy as np

logger = logging.getLogger(__name__)


class TradingAction(IntEnum):
    """
    Trading actions.

    Compatible with discrete action spaces in RL frameworks.
    """
    HOLD = 0
    BUY = 1
    SELL = 2


@dataclass
class TradeRecord:
    """Record of a single trade."""
    timestamp: datetime
    action: TradingAction
    price: float
    quantity: int
    position_before: int
    position_after: int
    pnl: float = 0.0
    fees: float = 0.0
    cumulative_pnl: float = 0.0


@dataclass
class GymConfig:
    """Configuration for TradingGym environment."""
    # Data parameters
    observation_window: int = 50  # Number of candles in observation
    step_size: int = 1  # How many candles to advance per step

    # Trading parameters
    initial_capital: float = 100_000.0
    max_position: int = 100  # Max shares to hold
    position_sizing: str = "fixed"  # "fixed", "percentage", "kelly"
    fixed_quantity: int = 10

    # Cost model
    commission_rate: float = 0.001  # 0.1%
    slippage_rate: float = 0.0005  # 0.05%

    # Reward shaping
    reward_function: str = "pnl"  # "pnl", "sharpe", "sortino", "log_return"
    reward_scaling: float = 1.0

    # Episode parameters
    max_steps: int = 0  # 0 = use all data
    random_start: bool = False

    # Features
    feature_columns: list[str] = field(default_factory=lambda: [
        "open", "high", "low", "close", "volume"
    ])
    include_position: bool = True
    include_pnl: bool = True
    normalize_observations: bool = True


@dataclass
class GymState:
    """Internal state of the environment."""
    step_count: int = 0
    current_index: int = 0
    position: int = 0
    entry_price: float = 0.0
    capital: float = 100_000.0
    equity: float = 100_000.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    trades: list[TradeRecord] = field(default_factory=list)
    done: bool = False


class TradingGym:
    """
    OpenAI Gym-style trading environment.

    Provides a standard RL interface for backtesting and training:
    - reset(): Initialize episode, return initial observation
    - step(action): Execute action, return (observation, reward, done, info)
    - render(): Display current state (optional)

    Example usage:
        ```python
        import pandas as pd

        # Load OHLCV data
        df = pd.read_csv("data.csv")

        # Create environment
        env = TradingGym(df, config=GymConfig(
            observation_window=50,
            max_position=100,
            reward_function="sharpe"
        ))

        # Run episode
        obs = env.reset()
        done = False
        while not done:
            action = agent.act(obs)  # Your RL agent
            obs, reward, done, info = env.step(action)

        # Get results
        stats = env.get_episode_stats()
        ```
    """

    def __init__(
        self,
        data: Any,  # pandas DataFrame or numpy array
        config: GymConfig | None = None,
    ):
        """
        Initialize the trading environment.

        Args:
            data: OHLCV data as DataFrame with columns:
                  [open, high, low, close, volume] or similar
            config: Environment configuration
        """
        self._config = config or GymConfig()
        self._data = self._process_data(data)
        self._state = GymState(capital=self._config.initial_capital)

        # Precompute normalized data if needed
        if self._config.normalize_observations:
            self._normalized_data = self._normalize_data(self._data)
        else:
            self._normalized_data = self._data

        # Track episode statistics
        self._episode_returns: list[float] = []

        logger.info(
            f"TradingGym initialized with {len(self._data)} candles, "
            f"observation_window={self._config.observation_window}"
        )

    def _process_data(self, data: Any) -> np.ndarray:
        """Convert input data to numpy array."""
        try:
            import pandas as pd
            if isinstance(data, pd.DataFrame):
                # Extract only the configured feature columns
                columns = []
                for col in self._config.feature_columns:
                    # Case-insensitive column matching
                    matching = [c for c in data.columns if c.lower() == col.lower()]
                    if matching:
                        columns.append(matching[0])
                    elif col.lower() in ["open", "high", "low", "close", "volume"]:
                        # Try standard OHLCV column names
                        for std_name in [col, col.capitalize(), col.upper()]:
                            if std_name in data.columns:
                                columns.append(std_name)
                                break

                if columns:
                    return data[columns].values.astype(np.float32)
                else:
                    # Use all numeric columns
                    return data.select_dtypes(include=[np.number]).values.astype(np.float32)
        except ImportError:
            pass

        # Assume numpy array
        return np.array(data, dtype=np.float32)

    def _normalize_data(self, data: np.ndarray) -> np.ndarray:
        """Normalize data using rolling z-score."""
        window = min(self._config.observation_window, len(data))
        normalized = np.zeros_like(data)

        for i in range(len(data)):
            start_idx = max(0, i - window + 1)
            window_data = data[start_idx:i + 1]

            mean = np.mean(window_data, axis=0)
            std = np.std(window_data, axis=0) + 1e-8  # Avoid division by zero

            normalized[i] = (data[i] - mean) / std

        return normalized

    @property
    def observation_shape(self) -> tuple[int, ...]:
        """Shape of observation array."""
        n_features = self._data.shape[1]
        extra_features = 0
        if self._config.include_position:
            extra_features += 1
        if self._config.include_pnl:
            extra_features += 1
        return (self._config.observation_window, n_features + extra_features)

    @property
    def action_space_size(self) -> int:
        """Number of possible actions."""
        return 3  # HOLD, BUY, SELL

    def reset(self, seed: int | None = None) -> np.ndarray:
        """
        Reset the environment for a new episode.

        Args:
            seed: Random seed for reproducibility

        Returns:
            Initial observation array
        """
        if seed is not None:
            np.random.seed(seed)

        # Determine starting index
        min_start = self._config.observation_window
        max_start = len(self._data) - self._config.max_steps if self._config.max_steps > 0 else min_start

        if self._config.random_start and max_start > min_start:
            start_index = np.random.randint(min_start, max_start)
        else:
            start_index = min_start

        # Initialize state
        self._state = GymState(
            step_count=0,
            current_index=start_index,
            position=0,
            entry_price=0.0,
            capital=self._config.initial_capital,
            equity=self._config.initial_capital,
            realized_pnl=0.0,
            unrealized_pnl=0.0,
            trades=[],
            done=False,
        )

        return self._get_observation()

    def step(self, action: int | TradingAction) -> tuple[np.ndarray, float, bool, dict]:
        """
        Execute one step in the environment.

        Args:
            action: Trading action (0=HOLD, 1=BUY, 2=SELL)

        Returns:
            Tuple of (observation, reward, done, info)
        """
        if self._state.done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        action = TradingAction(action)
        current_price = self._get_current_price()

        # Store state before action for reward calculation
        prev_equity = self._state.equity

        # Execute action
        trade = self._execute_action(action, current_price)

        # Update unrealized PnL
        if self._state.position != 0:
            self._state.unrealized_pnl = (
                (current_price - self._state.entry_price) * self._state.position
            )
        else:
            self._state.unrealized_pnl = 0.0

        # Update equity
        self._state.equity = self._state.capital + self._state.unrealized_pnl

        # Calculate reward
        reward = self._calculate_reward(prev_equity, self._state.equity)

        # Advance to next step
        self._state.step_count += 1
        self._state.current_index += self._config.step_size

        # Check if done
        max_steps_reached = (
            self._config.max_steps > 0 and
            self._state.step_count >= self._config.max_steps
        )
        data_exhausted = (
            self._state.current_index >= len(self._data) - 1
        )
        self._state.done = max_steps_reached or data_exhausted

        # Build info dict
        info = {
            "step": self._state.step_count,
            "price": current_price,
            "position": self._state.position,
            "equity": self._state.equity,
            "realized_pnl": self._state.realized_pnl,
            "unrealized_pnl": self._state.unrealized_pnl,
        }
        if trade:
            info["trade"] = {
                "action": trade.action.name,
                "price": trade.price,
                "quantity": trade.quantity,
                "pnl": trade.pnl,
                "fees": trade.fees,
            }

        return self._get_observation(), reward, self._state.done, info

    def _get_observation(self) -> np.ndarray:
        """Get current observation array."""
        # Get historical window
        start_idx = max(0, self._state.current_index - self._config.observation_window)
        end_idx = self._state.current_index

        if self._config.normalize_observations:
            obs = self._normalized_data[start_idx:end_idx].copy()
        else:
            obs = self._data[start_idx:end_idx].copy()

        # Pad if necessary
        if len(obs) < self._config.observation_window:
            padding = np.zeros((self._config.observation_window - len(obs), obs.shape[1]))
            obs = np.vstack([padding, obs])

        # Add position and PnL features
        extra_features = []
        if self._config.include_position:
            # Normalize position to [-1, 1]
            norm_position = self._state.position / max(1, self._config.max_position)
            extra_features.append(
                np.full((self._config.observation_window, 1), norm_position)
            )
        if self._config.include_pnl:
            # Normalize PnL
            norm_pnl = self._state.unrealized_pnl / max(1, self._config.initial_capital) * 100
            extra_features.append(
                np.full((self._config.observation_window, 1), norm_pnl)
            )

        if extra_features:
            obs = np.hstack([obs] + extra_features)

        return obs.astype(np.float32)

    def _get_current_price(self) -> float:
        """Get current close price."""
        # Assume 'close' is column index 3 (OHLCV order)
        close_idx = 3 if self._data.shape[1] > 3 else 0
        return float(self._data[self._state.current_index, close_idx])

    def _execute_action(
        self,
        action: TradingAction,
        price: float,
    ) -> TradeRecord | None:
        """Execute a trading action."""
        if action == TradingAction.HOLD:
            return None

        # Determine quantity
        quantity = self._config.fixed_quantity

        # Apply slippage
        if action == TradingAction.BUY:
            execution_price = price * (1 + self._config.slippage_rate)
        else:
            execution_price = price * (1 - self._config.slippage_rate)

        # Calculate fees
        fees = abs(quantity * execution_price * self._config.commission_rate)

        # Store state before trade
        position_before = self._state.position

        # Execute trade
        pnl = 0.0

        if action == TradingAction.BUY:
            if self._state.position < self._config.max_position:
                # Open or add to long position
                if self._state.position <= 0:
                    # Closing short or opening long
                    if self._state.position < 0:
                        # Close short position
                        pnl = (self._state.entry_price - execution_price) * abs(self._state.position)
                    self._state.entry_price = execution_price
                else:
                    # Average up
                    total_cost = (
                        self._state.entry_price * self._state.position +
                        execution_price * quantity
                    )
                    self._state.entry_price = total_cost / (self._state.position + quantity)

                self._state.position = min(
                    self._state.position + quantity,
                    self._config.max_position
                )

        elif action == TradingAction.SELL:
            if self._state.position > -self._config.max_position:
                # Close long or open/add to short
                if self._state.position > 0:
                    # Close long position
                    pnl = (execution_price - self._state.entry_price) * min(quantity, self._state.position)
                    self._state.position -= quantity

                    if self._state.position <= 0:
                        # Position closed or reversed
                        if self._state.position < 0:
                            self._state.entry_price = execution_price
                        else:
                            self._state.entry_price = 0.0
                else:
                    # Add to short or open short
                    if self._state.position == 0:
                        self._state.entry_price = execution_price
                    else:
                        # Average short entry
                        total_cost = (
                            self._state.entry_price * abs(self._state.position) +
                            execution_price * quantity
                        )
                        self._state.entry_price = total_cost / (abs(self._state.position) + quantity)

                    self._state.position = max(
                        self._state.position - quantity,
                        -self._config.max_position
                    )

        # Update capital
        pnl -= fees
        self._state.realized_pnl += pnl
        self._state.capital += pnl

        # Record trade
        trade = TradeRecord(
            timestamp=datetime.now(timezone.utc),
            action=action,
            price=execution_price,
            quantity=quantity,
            position_before=position_before,
            position_after=self._state.position,
            pnl=pnl,
            fees=fees,
            cumulative_pnl=self._state.realized_pnl,
        )
        self._state.trades.append(trade)

        return trade

    def _calculate_reward(self, prev_equity: float, curr_equity: float) -> float:
        """Calculate reward based on configured reward function."""
        if self._config.reward_function == "pnl":
            # Simple PnL change
            reward = curr_equity - prev_equity

        elif self._config.reward_function == "log_return":
            # Log return (better for multiplicative growth)
            if prev_equity > 0:
                reward = np.log(curr_equity / prev_equity)
            else:
                reward = 0.0

        elif self._config.reward_function == "sharpe":
            # Approximate Sharpe-like reward
            # Uses running statistics of returns
            ret = (curr_equity - prev_equity) / max(1, prev_equity)
            self._episode_returns.append(ret)

            if len(self._episode_returns) > 1:
                mean_ret = np.mean(self._episode_returns)
                std_ret = np.std(self._episode_returns) + 1e-8
                reward = mean_ret / std_ret
            else:
                reward = ret

        elif self._config.reward_function == "sortino":
            # Sortino-like reward (only penalize downside)
            ret = (curr_equity - prev_equity) / max(1, prev_equity)
            self._episode_returns.append(ret)

            if len(self._episode_returns) > 1:
                mean_ret = np.mean(self._episode_returns)
                downside = [r for r in self._episode_returns if r < 0]
                if downside:
                    downside_std = np.std(downside) + 1e-8
                else:
                    downside_std = 1e-8
                reward = mean_ret / downside_std
            else:
                reward = ret

        else:
            reward = curr_equity - prev_equity

        return float(reward * self._config.reward_scaling)

    def render(self, mode: str = "human") -> str | None:
        """
        Render the current state.

        Args:
            mode: Render mode ("human" for console, "string" for return)

        Returns:
            String representation if mode="string", else None
        """
        current_price = self._get_current_price()

        output = (
            f"Step: {self._state.step_count} | "
            f"Price: ${current_price:.2f} | "
            f"Position: {self._state.position} | "
            f"Equity: ${self._state.equity:.2f} | "
            f"Realized PnL: ${self._state.realized_pnl:.2f} | "
            f"Unrealized PnL: ${self._state.unrealized_pnl:.2f}"
        )

        if mode == "human":
            print(output)
            return None
        else:
            return output

    def get_episode_stats(self) -> dict[str, Any]:
        """Get statistics for the current/completed episode."""
        if not self._state.trades:
            return {
                "total_trades": 0,
                "total_pnl": 0.0,
                "win_rate": 0.0,
                "max_drawdown": 0.0,
            }

        # Calculate statistics
        winning_trades = [t for t in self._state.trades if t.pnl > 0]
        losing_trades = [t for t in self._state.trades if t.pnl < 0]

        total_pnl = sum(t.pnl for t in self._state.trades)
        total_fees = sum(t.fees for t in self._state.trades)

        # Calculate max drawdown
        equity_curve = [self._config.initial_capital]
        for trade in self._state.trades:
            equity_curve.append(equity_curve[-1] + trade.pnl)

        peak = equity_curve[0]
        max_drawdown = 0.0
        for equity in equity_curve:
            peak = max(peak, equity)
            drawdown = (peak - equity) / peak if peak > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)

        return {
            "total_trades": len(self._state.trades),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": len(winning_trades) / len(self._state.trades) if self._state.trades else 0,
            "total_pnl": total_pnl,
            "total_fees": total_fees,
            "net_pnl": total_pnl - total_fees,
            "final_equity": self._state.equity,
            "return_pct": (self._state.equity / self._config.initial_capital - 1) * 100,
            "max_drawdown_pct": max_drawdown * 100,
            "avg_trade_pnl": total_pnl / len(self._state.trades) if self._state.trades else 0,
            "avg_win": sum(t.pnl for t in winning_trades) / len(winning_trades) if winning_trades else 0,
            "avg_loss": sum(t.pnl for t in losing_trades) / len(losing_trades) if losing_trades else 0,
            "profit_factor": (
                abs(sum(t.pnl for t in winning_trades)) /
                max(1, abs(sum(t.pnl for t in losing_trades)))
            ) if losing_trades else float("inf"),
            "steps": self._state.step_count,
        }

    def get_trade_history(self) -> list[dict[str, Any]]:
        """Get all trades from the current episode."""
        return [
            {
                "timestamp": t.timestamp.isoformat(),
                "action": t.action.name,
                "price": t.price,
                "quantity": t.quantity,
                "position_before": t.position_before,
                "position_after": t.position_after,
                "pnl": t.pnl,
                "fees": t.fees,
                "cumulative_pnl": t.cumulative_pnl,
            }
            for t in self._state.trades
        ]

    def close(self) -> None:
        """Clean up resources."""
        pass


def create_gym_from_csv(
    filepath: str,
    config: GymConfig | None = None,
) -> TradingGym:
    """
    Create a TradingGym from a CSV file.

    Args:
        filepath: Path to CSV file with OHLCV data
        config: Environment configuration

    Returns:
        TradingGym instance
    """
    try:
        import pandas as pd
        df = pd.read_csv(filepath)
        return TradingGym(df, config=config)
    except ImportError:
        raise ImportError("pandas is required to load CSV files")


def create_gym_from_dataframe(
    df: Any,  # pandas DataFrame
    config: GymConfig | None = None,
) -> TradingGym:
    """
    Create a TradingGym from a pandas DataFrame.

    Args:
        df: DataFrame with OHLCV data
        config: Environment configuration

    Returns:
        TradingGym instance
    """
    return TradingGym(df, config=config)
