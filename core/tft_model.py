"""
Temporal Fusion Transformer for Price Forecasting
==================================================

Multi-horizon price forecasting using Temporal Fusion Transformer.

Inspired by:
- tft-torch (PlaytikaOSS) - production TFT implementation
- pytorch-forecasting TFT
- imehranasgari/TFT_yfinance (technical indicators integration)

Features:
- Multi-step ahead predictions (1h, 4h, 1d)
- Interpretable attention weights
- Technical indicators as covariates
- Quantile predictions for confidence intervals
- GPU acceleration support
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any
from enum import Enum

# Optional imports
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None

logger = logging.getLogger(__name__)


# Placeholder base class when PyTorch not available
if not HAS_TORCH:
    class _ModulePlaceholder:
        pass
    nn_Module = _ModulePlaceholder
else:
    nn_Module = nn.Module


class ForecastHorizon(Enum):
    """Forecast time horizons."""
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"


@dataclass
class TFTConfig:
    """Configuration for TFT model."""
    # Model architecture
    hidden_size: int = 64
    attention_head_size: int = 4
    num_attention_heads: int = 4
    hidden_continuous_size: int = 32
    dropout: float = 0.1

    # Input configuration
    input_size: int = 20  # Number of input features
    max_encoder_length: int = 60  # Lookback window
    max_prediction_length: int = 24  # Max forecast steps

    # Training
    learning_rate: float = 1e-3
    batch_size: int = 32

    # Quantiles for prediction intervals
    quantiles: list[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])

    # Device
    device: str = "auto"


@dataclass
class TFTForecast:
    """Single forecast from TFT model."""
    symbol: str
    timestamp: datetime
    horizon: ForecastHorizon
    target_time: datetime
    predicted_price: float
    lower_bound: float  # 10th percentile
    upper_bound: float  # 90th percentile
    confidence: float
    attention_weights: dict[str, float] | None = None  # Feature importance

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "horizon": self.horizon.value,
            "target_time": self.target_time.isoformat(),
            "predicted_price": self.predicted_price,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "confidence": self.confidence,
            "attention_weights": self.attention_weights,
        }


class GatedLinearUnit(nn_Module):
    """Gated Linear Unit for TFT."""

    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.1):
        if not HAS_TORCH:
            return
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(input_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.sigmoid(self.fc2(x)) * self.fc1(x))


class GatedResidualNetwork(nn_Module):
    """Gated Residual Network for TFT."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int | None = None,
        dropout: float = 0.1,
        context_size: int | None = None,
    ):
        if not HAS_TORCH:
            return
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size or input_size
        self.context_size = context_size

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.elu = nn.ELU()

        if context_size is not None:
            self.context_fc = nn.Linear(context_size, hidden_size, bias=False)

        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.glu = GatedLinearUnit(hidden_size, self.output_size, dropout)

        if input_size != self.output_size:
            self.skip_layer = nn.Linear(input_size, self.output_size)
        else:
            self.skip_layer = None

        self.layer_norm = nn.LayerNorm(self.output_size)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Skip connection
        if self.skip_layer is not None:
            skip = self.skip_layer(x)
        else:
            skip = x

        # Main path
        hidden = self.fc1(x)
        if context is not None and self.context_size is not None:
            hidden = hidden + self.context_fc(context)
        hidden = self.elu(hidden)
        hidden = self.fc2(hidden)
        hidden = self.glu(hidden)

        # Residual connection and layer norm
        return self.layer_norm(skip + hidden)


class VariableSelectionNetwork(nn_Module):
    """Variable Selection Network for TFT."""

    def __init__(
        self,
        input_sizes: list[int],
        hidden_size: int,
        dropout: float = 0.1,
        context_size: int | None = None,
    ):
        if not HAS_TORCH:
            return
        super().__init__()
        self.hidden_size = hidden_size
        self.num_inputs = len(input_sizes)

        # Individual variable GRNs
        self.var_grns = nn.ModuleList([
            GatedResidualNetwork(
                input_size=size,
                hidden_size=hidden_size,
                output_size=hidden_size,
                dropout=dropout,
            )
            for size in input_sizes
        ])

        # Softmax for variable weights
        self.flattened_grn = GatedResidualNetwork(
            input_size=hidden_size * self.num_inputs,
            hidden_size=hidden_size,
            output_size=self.num_inputs,
            dropout=dropout,
            context_size=context_size,
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self,
        inputs: list[torch.Tensor],
        context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Process each variable
        var_outputs = []
        for i, (inp, grn) in enumerate(zip(inputs, self.var_grns)):
            var_outputs.append(grn(inp))

        # Stack for attention
        var_outputs = torch.stack(var_outputs, dim=-2)  # [batch, time, num_vars, hidden]

        # Calculate variable weights
        flattened = var_outputs.flatten(start_dim=-2)  # [batch, time, num_vars * hidden]
        weights = self.flattened_grn(flattened, context)
        weights = self.softmax(weights)  # [batch, time, num_vars]

        # Weighted sum
        weights_expanded = weights.unsqueeze(-1)  # [batch, time, num_vars, 1]
        combined = (var_outputs * weights_expanded).sum(dim=-2)  # [batch, time, hidden]

        return combined, weights


class InterpretableMultiHeadAttention(nn_Module):
    """Interpretable Multi-Head Attention for TFT."""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        if not HAS_TORCH:
            return
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads

        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)
        self.out_linear = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_size ** 0.5

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = query.size(0)

        # Linear projections
        q = self.q_linear(query)
        k = self.k_linear(key)
        v = self.v_linear(value)

        # Reshape for multi-head attention
        q = q.view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        context = torch.matmul(attn_weights, v)

        # Reshape and project
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_size)
        output = self.out_linear(context)

        # Average attention weights across heads for interpretability
        avg_attn = attn_weights.mean(dim=1)

        return output, avg_attn


class TemporalFusionTransformer(nn_Module):
    """
    Temporal Fusion Transformer for time series forecasting.

    Simplified implementation focusing on:
    - Variable selection for interpretability
    - Multi-head attention for temporal patterns
    - Quantile outputs for prediction intervals
    """

    def __init__(self, config: TFTConfig):
        if not HAS_TORCH:
            return
        super().__init__()
        self.config = config

        # Input embedding
        self.input_embedding = nn.Linear(config.input_size, config.hidden_size)

        # Variable selection (simplified - treats all as continuous)
        self.encoder_vsn = GatedResidualNetwork(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            dropout=config.dropout,
        )

        # LSTM encoder
        self.encoder_lstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=2,
            dropout=config.dropout,
            batch_first=True,
        )

        # Gated skip connection
        self.gate_encoder = GatedLinearUnit(
            config.hidden_size, config.hidden_size, config.dropout
        )

        # Static enrichment
        self.static_enrichment = GatedResidualNetwork(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            dropout=config.dropout,
        )

        # Temporal self-attention
        self.self_attention = InterpretableMultiHeadAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.dropout,
        )

        # Position-wise feed-forward
        self.positionwise_grn = GatedResidualNetwork(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            dropout=config.dropout,
        )

        # Output layers for each quantile
        self.output_layers = nn.ModuleList([
            nn.Linear(config.hidden_size, config.max_prediction_length)
            for _ in config.quantiles
        ])

        # Layer norms
        self.layer_norm1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor [batch, seq_len, input_size]

        Returns:
            predictions: [batch, num_quantiles, prediction_length]
            attention_weights: [batch, seq_len, seq_len]
        """
        batch_size = x.size(0)

        # Input embedding
        embedded = self.input_embedding(x)

        # Variable selection (simplified)
        selected = self.encoder_vsn(embedded)

        # LSTM encoding
        lstm_out, _ = self.encoder_lstm(selected)

        # Gated skip connection
        gated = self.gate_encoder(lstm_out)
        enriched = self.layer_norm1(selected + gated)

        # Static enrichment
        enriched = self.static_enrichment(enriched)

        # Self-attention
        attn_out, attn_weights = self.self_attention(
            enriched, enriched, enriched
        )
        attn_out = self.layer_norm2(enriched + attn_out)

        # Position-wise feed-forward
        output = self.positionwise_grn(attn_out)

        # Get last timestep for prediction
        last_hidden = output[:, -1, :]  # [batch, hidden]

        # Quantile predictions
        quantile_preds = []
        for layer in self.output_layers:
            pred = layer(last_hidden)  # [batch, prediction_length]
            quantile_preds.append(pred)

        predictions = torch.stack(quantile_preds, dim=1)  # [batch, num_quantiles, pred_len]

        return predictions, attn_weights


class TFTForecaster:
    """
    High-level TFT Forecaster for price prediction.

    Wraps the TFT model with data preprocessing, training, and inference.

    Example:
        forecaster = TFTForecaster()
        await forecaster.initialize()

        # Add price data
        forecaster.update_price("AAPL", 150.0, indicators)

        # Get forecast
        forecast = await forecaster.predict("AAPL", ForecastHorizon.HOUR_4)
    """

    def __init__(self, config: TFTConfig | None = None):
        self._config = config or TFTConfig()
        self._model: TemporalFusionTransformer | None = None
        self._device = None
        self._initialized = False

        # Price history per symbol
        self._price_history: dict[str, list[dict]] = {}
        self._max_history = self._config.max_encoder_length + 10

        # Feature names for interpretability
        self._feature_names = [
            "close", "open", "high", "low", "volume",
            "rsi", "macd", "macd_signal", "bb_upper", "bb_lower",
            "atr", "obv", "sma_20", "ema_12", "ema_26",
            "momentum", "roc", "williams_r", "stoch_k", "stoch_d"
        ]

        # Normalization stats
        self._means: dict[str, np.ndarray] = {}
        self._stds: dict[str, np.ndarray] = {}

    async def initialize(self) -> bool:
        """Initialize the TFT model."""
        if not HAS_TORCH:
            logger.warning(
                "PyTorch not installed. Install with: pip install torch"
            )
            return False

        try:
            # Determine device
            if self._config.device == "auto":
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self._device = self._config.device

            # Create model
            self._model = TemporalFusionTransformer(self._config)
            self._model = self._model.to(self._device)
            self._model.eval()

            self._initialized = True
            logger.info(f"TFT Forecaster initialized on {self._device}")
            return True

        except Exception as e:
            logger.exception(f"Failed to initialize TFT: {e}")
            return False

    def update_price(
        self,
        symbol: str,
        price: float,
        indicators: dict[str, float] | None = None,
        timestamp: datetime | None = None,
    ) -> None:
        """
        Update price history for a symbol.

        Args:
            symbol: Stock symbol
            price: Current price
            indicators: Technical indicators dict
            timestamp: Timestamp (defaults to now)
        """
        if symbol not in self._price_history:
            self._price_history[symbol] = []

        timestamp = timestamp or datetime.now(timezone.utc)

        # Build feature dict
        features = {
            "timestamp": timestamp,
            "close": price,
            "open": indicators.get("open", price) if indicators else price,
            "high": indicators.get("high", price) if indicators else price,
            "low": indicators.get("low", price) if indicators else price,
            "volume": indicators.get("volume", 0) if indicators else 0,
        }

        # Add technical indicators
        if indicators:
            for name in self._feature_names[5:]:  # Skip OHLCV
                features[name] = indicators.get(name, 0.0)
        else:
            for name in self._feature_names[5:]:
                features[name] = 0.0

        self._price_history[symbol].append(features)

        # Trim history
        if len(self._price_history[symbol]) > self._max_history:
            self._price_history[symbol] = self._price_history[symbol][-self._max_history:]

    def _prepare_input(self, symbol: str) -> torch.Tensor | None:
        """Prepare input tensor from price history."""
        if symbol not in self._price_history:
            return None

        history = self._price_history[symbol]
        if len(history) < self._config.max_encoder_length:
            return None

        # Extract features
        features = []
        for record in history[-self._config.max_encoder_length:]:
            row = [record.get(name, 0.0) for name in self._feature_names]
            features.append(row)

        features = np.array(features, dtype=np.float32)

        # Normalize
        if symbol not in self._means:
            self._means[symbol] = features.mean(axis=0)
            self._stds[symbol] = features.std(axis=0) + 1e-8

        features = (features - self._means[symbol]) / self._stds[symbol]

        # Convert to tensor
        tensor = torch.tensor(features, dtype=torch.float32)
        tensor = tensor.unsqueeze(0)  # Add batch dimension

        return tensor.to(self._device)

    async def predict(
        self,
        symbol: str,
        horizon: ForecastHorizon = ForecastHorizon.HOUR_4,
    ) -> TFTForecast | None:
        """
        Generate price forecast for a symbol.

        Args:
            symbol: Stock symbol
            horizon: Forecast horizon

        Returns:
            TFTForecast or None if insufficient data
        """
        if not self._initialized or self._model is None:
            logger.warning("TFT not initialized")
            return None

        # Prepare input
        input_tensor = self._prepare_input(symbol)
        if input_tensor is None:
            logger.debug(f"Insufficient history for {symbol}")
            return None

        # Get current price for denormalization
        current_price = self._price_history[symbol][-1]["close"]

        # Inference
        try:
            with torch.no_grad():
                predictions, attention = self._model(input_tensor)

            # predictions shape: [1, num_quantiles, prediction_length]
            predictions = predictions.cpu().numpy()[0]
            attention = attention.cpu().numpy()[0]

            # Map horizon to prediction index
            horizon_map = {
                ForecastHorizon.HOUR_1: 0,
                ForecastHorizon.HOUR_4: 3,
                ForecastHorizon.DAY_1: 23,
            }
            pred_idx = min(horizon_map.get(horizon, 3), predictions.shape[1] - 1)

            # Get quantile predictions
            lower = predictions[0, pred_idx]  # 10th percentile
            median = predictions[1, pred_idx]  # 50th percentile (point estimate)
            upper = predictions[2, pred_idx]  # 90th percentile

            # Denormalize (simplified - assumes close is feature 0)
            if symbol in self._means:
                mean_close = self._means[symbol][0]
                std_close = self._stds[symbol][0]
                lower = lower * std_close + mean_close
                median = median * std_close + mean_close
                upper = upper * std_close + mean_close
            else:
                # Relative prediction
                lower = current_price * (1 + lower * 0.1)
                median = current_price * (1 + median * 0.1)
                upper = current_price * (1 + upper * 0.1)

            # Calculate confidence from prediction interval width
            interval_width = (upper - lower) / current_price
            confidence = max(0.3, min(0.9, 1.0 - interval_width * 5))

            # Calculate target time
            horizon_deltas = {
                ForecastHorizon.HOUR_1: timedelta(hours=1),
                ForecastHorizon.HOUR_4: timedelta(hours=4),
                ForecastHorizon.DAY_1: timedelta(days=1),
            }
            target_time = datetime.now(timezone.utc) + horizon_deltas[horizon]

            # Extract attention weights for interpretability
            # Average attention over time steps
            avg_attention = attention.mean(axis=0)
            attention_weights = {
                f"t-{i}": float(avg_attention[i])
                for i in range(min(10, len(avg_attention)))
            }

            return TFTForecast(
                symbol=symbol,
                timestamp=datetime.now(timezone.utc),
                horizon=horizon,
                target_time=target_time,
                predicted_price=float(median),
                lower_bound=float(lower),
                upper_bound=float(upper),
                confidence=confidence,
                attention_weights=attention_weights,
            )

        except Exception as e:
            logger.exception(f"TFT prediction error: {e}")
            return None

    def get_feature_importance(self, symbol: str) -> dict[str, float] | None:
        """Get feature importance from attention weights."""
        # This would require tracking attention during training
        # Simplified: return uniform importance
        return {name: 1.0 / len(self._feature_names) for name in self._feature_names}

    def save_model(self, path: str) -> None:
        """Save model weights."""
        if self._model is not None:
            torch.save(self._model.state_dict(), path)
            logger.info(f"Saved TFT model to {path}")

    def load_model(self, path: str) -> bool:
        """Load model weights."""
        if not HAS_TORCH or self._model is None:
            return False

        try:
            self._model.load_state_dict(torch.load(path, map_location=self._device))
            self._model.eval()
            logger.info(f"Loaded TFT model from {path}")
            return True
        except Exception as e:
            logger.exception(f"Failed to load model: {e}")
            return False

    @property
    def is_available(self) -> bool:
        """Check if TFT is available."""
        return self._initialized

    def get_stats(self) -> dict[str, Any]:
        """Get forecaster statistics."""
        return {
            "initialized": self._initialized,
            "device": self._device,
            "symbols_tracked": list(self._price_history.keys()),
            "history_lengths": {
                s: len(h) for s, h in self._price_history.items()
            },
            "config": {
                "hidden_size": self._config.hidden_size,
                "max_encoder_length": self._config.max_encoder_length,
                "max_prediction_length": self._config.max_prediction_length,
                "quantiles": self._config.quantiles,
            },
        }
