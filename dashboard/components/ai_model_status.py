"""
AI Model Status Tracker
========================

Real-time AI/ML model status and performance tracking component for the trading system dashboard.

Tracks all AI/ML models including:
- DRL CIO Agent (PPO/A2C)
- TFT Forecaster (Temporal Fusion Transformer)
- FinBERT Sentiment Analyzer
- LLM Clients (Claude, GPT, etc.)

Features:
- Model availability and health checks
- Inference metrics tracking (latency, throughput)
- Model-specific performance metrics
- Fallback status monitoring
- Thread-safe with asyncio locks
- WebSocket-ready export to dict
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    pass


logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Classification of AI/ML model types."""
    DRL = "drl"                # Deep Reinforcement Learning (PPO, A2C, TD3)
    TFT = "tft"                # Temporal Fusion Transformer
    FINBERT = "finbert"        # FinBERT Sentiment
    LLM = "llm"                # Large Language Models (Claude, GPT)
    TRANSFORMER = "transformer" # Other transformer models
    ENSEMBLE = "ensemble"      # Ensemble models


class ModelStatus(Enum):
    """Status of an AI/ML model."""
    LOADED = "loaded"              # Model is loaded and ready
    TRAINING = "training"          # Model is being trained
    ERROR = "error"                # Model encountered an error
    NOT_AVAILABLE = "not_available" # Model/dependencies not available
    LOADING = "loading"            # Model is being loaded
    UNLOADED = "unloaded"          # Model is not loaded
    FALLBACK = "fallback"          # Using fallback/heuristic mode


# =============================================================================
# Model-Specific Metrics Dataclasses
# =============================================================================

@dataclass
class DRLModelMetrics:
    """
    Performance metrics specific to DRL models (PPO/A2C/TD3).

    Tracks training progress and policy quality metrics.
    """
    algorithm: str = "PPO"          # PPO, A2C, TD3, SAC
    episode_reward: float = 0.0     # Latest episode reward
    episode_reward_avg: float = 0.0 # Moving average of episode rewards
    episode_count: int = 0          # Total episodes trained
    policy_loss: float = 0.0        # Policy network loss
    value_loss: float = 0.0         # Value network loss
    entropy: float = 0.0            # Policy entropy (exploration)
    kl_divergence: float = 0.0      # KL divergence (PPO specific)
    explained_variance: float = 0.0 # Explained variance of value function
    total_timesteps: int = 0        # Total training timesteps
    decisions_made: int = 0         # Decisions made in live trading
    drl_decisions: int = 0          # Decisions using DRL (vs fallback)
    fallback_decisions: int = 0     # Decisions using heuristic fallback

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for WebSocket streaming."""
        return {
            "algorithm": self.algorithm,
            "episode_reward": round(self.episode_reward, 4),
            "episode_reward_avg": round(self.episode_reward_avg, 4),
            "episode_count": self.episode_count,
            "policy_loss": round(self.policy_loss, 6),
            "value_loss": round(self.value_loss, 6),
            "entropy": round(self.entropy, 4),
            "kl_divergence": round(self.kl_divergence, 6),
            "explained_variance": round(self.explained_variance, 4),
            "total_timesteps": self.total_timesteps,
            "decisions_made": self.decisions_made,
            "drl_decisions": self.drl_decisions,
            "fallback_decisions": self.fallback_decisions,
            "drl_ratio": round(self.drl_decisions / max(1, self.decisions_made), 4),
        }


@dataclass
class TFTModelMetrics:
    """
    Performance metrics specific to TFT forecaster.

    Tracks forecast accuracy across different horizons.
    """
    mape: float = 0.0               # Mean Absolute Percentage Error
    mae: float = 0.0                # Mean Absolute Error
    rmse: float = 0.0               # Root Mean Square Error
    forecast_horizon: str = "4h"    # Current forecast horizon
    symbols_tracked: int = 0        # Number of symbols being tracked
    forecasts_generated: int = 0    # Total forecasts generated
    avg_confidence: float = 0.0     # Average forecast confidence
    calibration_score: float = 0.0  # Prediction interval calibration
    # Per-horizon metrics
    mape_1h: float = 0.0
    mape_4h: float = 0.0
    mape_1d: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for WebSocket streaming."""
        return {
            "mape": round(self.mape, 4),
            "mae": round(self.mae, 4),
            "rmse": round(self.rmse, 4),
            "forecast_horizon": self.forecast_horizon,
            "symbols_tracked": self.symbols_tracked,
            "forecasts_generated": self.forecasts_generated,
            "avg_confidence": round(self.avg_confidence, 4),
            "calibration_score": round(self.calibration_score, 4),
            "per_horizon": {
                "1h": {"mape": round(self.mape_1h, 4)},
                "4h": {"mape": round(self.mape_4h, 4)},
                "1d": {"mape": round(self.mape_1d, 4)},
            },
        }


@dataclass
class FinBERTMetrics:
    """
    Performance metrics specific to FinBERT sentiment analyzer.

    Tracks sentiment analysis accuracy and throughput.
    """
    accuracy: float = 0.0           # Classification accuracy
    f1_score: float = 0.0           # F1 score
    precision: float = 0.0          # Precision
    recall: float = 0.0             # Recall
    inference_speed: float = 0.0    # Texts per second
    cache_hit_rate: float = 0.0     # Cache efficiency
    total_analyzed: int = 0         # Total texts analyzed
    positive_ratio: float = 0.0     # Ratio of positive sentiments
    negative_ratio: float = 0.0     # Ratio of negative sentiments
    neutral_ratio: float = 0.0      # Ratio of neutral sentiments
    device: str = "cpu"             # Running on CPU/CUDA

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for WebSocket streaming."""
        return {
            "accuracy": round(self.accuracy, 4),
            "f1_score": round(self.f1_score, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "inference_speed": round(self.inference_speed, 2),
            "cache_hit_rate": round(self.cache_hit_rate, 4),
            "total_analyzed": self.total_analyzed,
            "sentiment_distribution": {
                "positive": round(self.positive_ratio, 4),
                "negative": round(self.negative_ratio, 4),
                "neutral": round(self.neutral_ratio, 4),
            },
            "device": self.device,
        }


@dataclass
class LLMMetrics:
    """
    Performance metrics specific to LLM clients.

    Tracks API usage, costs, and response quality.
    """
    provider: str = "anthropic"     # anthropic, openai, etc.
    model_name: str = ""            # claude-3-opus, gpt-4, etc.
    total_requests: int = 0         # Total API requests
    successful_requests: int = 0    # Successful requests
    failed_requests: int = 0        # Failed requests
    total_tokens_in: int = 0        # Input tokens used
    total_tokens_out: int = 0       # Output tokens generated
    estimated_cost_usd: float = 0.0 # Estimated API cost
    avg_response_quality: float = 0.0  # Quality score (0-1)
    rate_limit_hits: int = 0        # Rate limit encounters

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for WebSocket streaming."""
        return {
            "provider": self.provider,
            "model_name": self.model_name,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": round(
                self.successful_requests / max(1, self.total_requests), 4
            ),
            "total_tokens_in": self.total_tokens_in,
            "total_tokens_out": self.total_tokens_out,
            "estimated_cost_usd": round(self.estimated_cost_usd, 4),
            "avg_response_quality": round(self.avg_response_quality, 4),
            "rate_limit_hits": self.rate_limit_hits,
        }


# =============================================================================
# Model Info Dataclass
# =============================================================================

@dataclass
class ModelInfo:
    """
    Comprehensive information about an AI/ML model.

    Captures status, version, performance metrics, and health data.
    """
    model_name: str
    model_type: ModelType
    version: str = "1.0.0"
    status: ModelStatus = ModelStatus.NOT_AVAILABLE
    last_inference_time: datetime | None = None
    inference_count: int = 0
    avg_inference_latency_ms: float = 0.0
    max_inference_latency_ms: float = 0.0
    min_inference_latency_ms: float = float('inf')
    accuracy_metrics: dict[str, float] = field(default_factory=dict)
    # Model-specific metrics
    drl_metrics: DRLModelMetrics | None = None
    tft_metrics: TFTModelMetrics | None = None
    finbert_metrics: FinBERTMetrics | None = None
    llm_metrics: LLMMetrics | None = None
    # Health and availability
    is_available: bool = False
    fallback_active: bool = False
    error_message: str | None = None
    error_count: int = 0
    last_error_time: datetime | None = None
    loaded_at: datetime | None = None
    # Internal tracking
    _latency_samples: deque = field(default_factory=lambda: deque(maxlen=100))

    def __post_init__(self):
        """Initialize deques if they were not set."""
        if not isinstance(self._latency_samples, deque):
            object.__setattr__(self, '_latency_samples', deque(maxlen=100))
        # Initialize model-specific metrics based on type
        if self.model_type == ModelType.DRL and self.drl_metrics is None:
            self.drl_metrics = DRLModelMetrics()
        elif self.model_type == ModelType.TFT and self.tft_metrics is None:
            self.tft_metrics = TFTModelMetrics()
        elif self.model_type == ModelType.FINBERT and self.finbert_metrics is None:
            self.finbert_metrics = FinBERTMetrics()
        elif self.model_type == ModelType.LLM and self.llm_metrics is None:
            self.llm_metrics = LLMMetrics()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for WebSocket streaming."""
        result = {
            "model_name": self.model_name,
            "model_type": self.model_type.value,
            "version": self.version,
            "status": self.status.value,
            "last_inference_time": (
                self.last_inference_time.isoformat()
                if self.last_inference_time else None
            ),
            "inference_count": self.inference_count,
            "avg_inference_latency_ms": round(self.avg_inference_latency_ms, 2),
            "max_inference_latency_ms": round(self.max_inference_latency_ms, 2),
            "min_inference_latency_ms": (
                round(self.min_inference_latency_ms, 2)
                if self.min_inference_latency_ms != float('inf') else None
            ),
            "accuracy_metrics": self.accuracy_metrics,
            "is_available": self.is_available,
            "fallback_active": self.fallback_active,
            "error_message": self.error_message,
            "error_count": self.error_count,
            "last_error_time": (
                self.last_error_time.isoformat()
                if self.last_error_time else None
            ),
            "loaded_at": self.loaded_at.isoformat() if self.loaded_at else None,
            "health_score": self._calculate_health_score(),
        }

        # Add model-specific metrics
        if self.drl_metrics is not None:
            result["drl_metrics"] = self.drl_metrics.to_dict()
        if self.tft_metrics is not None:
            result["tft_metrics"] = self.tft_metrics.to_dict()
        if self.finbert_metrics is not None:
            result["finbert_metrics"] = self.finbert_metrics.to_dict()
        if self.llm_metrics is not None:
            result["llm_metrics"] = self.llm_metrics.to_dict()

        return result

    def _calculate_health_score(self) -> float:
        """Calculate model health score (0-100)."""
        score = 100.0

        # Penalty for error status
        if self.status == ModelStatus.ERROR:
            score -= 50.0
        elif self.status == ModelStatus.NOT_AVAILABLE:
            score -= 30.0
        elif self.status == ModelStatus.FALLBACK:
            score -= 20.0
        elif self.status == ModelStatus.LOADING:
            score -= 10.0

        # Penalty for errors
        if self.error_count > 0:
            score -= min(20.0, self.error_count * 2.0)

        # Penalty for high latency (over 1 second average)
        if self.avg_inference_latency_ms > 1000:
            score -= min(15.0, (self.avg_inference_latency_ms - 1000) / 100)

        # Penalty for no recent activity (over 1 hour)
        if self.last_inference_time:
            idle_seconds = (datetime.now(timezone.utc) - self.last_inference_time).total_seconds()
            if idle_seconds > 3600:
                score -= min(10.0, idle_seconds / 3600)

        # Bonus for availability
        if self.is_available and self.status == ModelStatus.LOADED:
            score += 5.0

        return max(0.0, min(100.0, score))


# =============================================================================
# System-Wide Model Health Metrics
# =============================================================================

@dataclass
class ModelSystemHealth:
    """
    Aggregated health metrics across all AI/ML models.

    Provides system-wide model health overview for monitoring.
    """
    total_models: int = 0
    loaded_models: int = 0
    error_models: int = 0
    fallback_models: int = 0
    unavailable_models: int = 0
    avg_health_score: float = 100.0
    min_health_score: float = 100.0
    min_health_model: str | None = None
    total_inferences: int = 0
    total_errors: int = 0
    system_health: str = "healthy"  # healthy, degraded, critical
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for WebSocket streaming."""
        return {
            "total_models": self.total_models,
            "loaded_models": self.loaded_models,
            "error_models": self.error_models,
            "fallback_models": self.fallback_models,
            "unavailable_models": self.unavailable_models,
            "avg_health_score": round(self.avg_health_score, 1),
            "min_health_score": round(self.min_health_score, 1),
            "min_health_model": self.min_health_model,
            "total_inferences": self.total_inferences,
            "total_errors": self.total_errors,
            "system_health": self.system_health,
            "last_updated": self.last_updated.isoformat(),
        }


# =============================================================================
# AI Model Tracker Class
# =============================================================================

class AIModelTracker:
    """
    Tracks and monitors the status of all AI/ML models in the trading system.

    Provides real-time visibility into model health, performance, and availability
    for the trading system dashboard.

    Usage:
        tracker = AIModelTracker()

        # Register models at startup
        tracker.register_model("DRL_CIO", ModelType.DRL, version="1.0.0")
        tracker.register_model("TFT_Forecaster", ModelType.TFT)
        tracker.register_model("FinBERT_Sentiment", ModelType.FINBERT)

        # Update status when model is loaded
        await tracker.update_status("DRL_CIO", ModelStatus.LOADED)

        # Record inference
        await tracker.record_inference("DRL_CIO", latency_ms=15.5)

        # Update model-specific metrics
        await tracker.update_drl_metrics(
            "DRL_CIO",
            episode_reward=125.5,
            policy_loss=0.0023,
        )

        # Get model status
        model_info = await tracker.get_model_info("DRL_CIO")

        # Get all models for dashboard
        all_models = await tracker.get_all_models()

        # Check model health
        health = await tracker.get_model_health("DRL_CIO")

        # Export for WebSocket streaming
        data = tracker.to_dict()

    Thread Safety:
        All public methods acquire an asyncio lock before modifying state,
        making the tracker safe for concurrent use.
    """

    # Health score thresholds
    HEALTH_CRITICAL_THRESHOLD = 50.0
    HEALTH_DEGRADED_THRESHOLD = 75.0

    def __init__(self):
        """Initialize the AI model tracker."""
        self._models: dict[str, ModelInfo] = {}
        self._lock = asyncio.Lock()

        # Predefined models based on trading system architecture
        self._expected_models = [
            ("DRL_CIO_PPO", ModelType.DRL, "PPO-based CIO decision agent"),
            ("DRL_CIO_A2C", ModelType.DRL, "A2C-based CIO decision agent"),
            ("TFT_Forecaster", ModelType.TFT, "Price forecasting model"),
            ("FinBERT_Sentiment", ModelType.FINBERT, "Financial sentiment analyzer"),
            ("Claude_Client", ModelType.LLM, "Claude API client"),
            ("GPT_Client", ModelType.LLM, "OpenAI GPT client"),
        ]

        logger.info("AIModelTracker initialized")

    def register_model(
        self,
        model_name: str,
        model_type: ModelType,
        version: str = "1.0.0",
        description: str | None = None,
    ) -> ModelInfo:
        """
        Register an AI/ML model for tracking.

        Args:
            model_name: Unique name for the model
            model_type: Type of the model (DRL, TFT, FinBERT, LLM)
            version: Model version string
            description: Optional description

        Returns:
            ModelInfo for the registered model
        """
        if model_name not in self._models:
            model_info = ModelInfo(
                model_name=model_name,
                model_type=model_type,
                version=version,
                status=ModelStatus.NOT_AVAILABLE,
            )
            self._models[model_name] = model_info
            logger.info(
                f"Registered model: {model_name} (type={model_type.value}, v={version})"
            )
        else:
            model_info = self._models[model_name]
            model_info.version = version

        return model_info

    async def update_status(
        self,
        model_name: str,
        status: ModelStatus,
        error_message: str | None = None,
    ) -> bool:
        """
        Update the status of a model.

        Args:
            model_name: Name of the model
            status: New status
            error_message: Error message if status is ERROR

        Returns:
            True if updated successfully, False if model not found
        """
        async with self._lock:
            if model_name not in self._models:
                logger.warning(f"Model not found: {model_name}")
                return False

            model = self._models[model_name]
            now = datetime.now(timezone.utc)
            old_status = model.status

            model.status = status

            # Handle status-specific updates
            if status == ModelStatus.LOADED:
                model.is_available = True
                model.fallback_active = False
                model.loaded_at = now
                model.error_message = None
            elif status == ModelStatus.ERROR:
                model.error_count += 1
                model.error_message = error_message
                model.last_error_time = now
            elif status == ModelStatus.FALLBACK:
                model.is_available = True
                model.fallback_active = True
            elif status == ModelStatus.NOT_AVAILABLE:
                model.is_available = False
            elif status == ModelStatus.UNLOADED:
                model.is_available = False
                model.loaded_at = None

            logger.debug(
                f"Model {model_name} status updated: {old_status.value} -> {status.value}"
            )

            return True

    async def record_inference(
        self,
        model_name: str,
        latency_ms: float,
        success: bool = True,
    ) -> bool:
        """
        Record an inference by a model.

        Args:
            model_name: Name of the model
            latency_ms: Inference latency in milliseconds
            success: Whether the inference was successful

        Returns:
            True if recorded successfully, False if model not found
        """
        async with self._lock:
            if model_name not in self._models:
                return False

            model = self._models[model_name]
            now = datetime.now(timezone.utc)

            model.inference_count += 1
            model.last_inference_time = now

            # Update latency statistics
            model._latency_samples.append(latency_ms)
            samples = list(model._latency_samples)

            if samples:
                model.avg_inference_latency_ms = sum(samples) / len(samples)
                model.max_inference_latency_ms = max(
                    model.max_inference_latency_ms, latency_ms
                )
                model.min_inference_latency_ms = min(
                    model.min_inference_latency_ms, latency_ms
                )

            # Record error if not successful
            if not success:
                model.error_count += 1
                model.last_error_time = now

            return True

    async def update_drl_metrics(
        self,
        model_name: str,
        algorithm: str | None = None,
        episode_reward: float | None = None,
        policy_loss: float | None = None,
        value_loss: float | None = None,
        entropy: float | None = None,
        kl_divergence: float | None = None,
        explained_variance: float | None = None,
        total_timesteps: int | None = None,
        decisions_made: int | None = None,
        drl_decisions: int | None = None,
        fallback_decisions: int | None = None,
    ) -> bool:
        """
        Update DRL-specific metrics for a model.

        Args:
            model_name: Name of the DRL model
            algorithm: Algorithm name (PPO, A2C, TD3)
            episode_reward: Latest episode reward
            policy_loss: Policy network loss
            value_loss: Value network loss
            entropy: Policy entropy
            kl_divergence: KL divergence (PPO)
            explained_variance: Value function explained variance
            total_timesteps: Total training timesteps
            decisions_made: Total decisions made
            drl_decisions: Decisions using DRL model
            fallback_decisions: Decisions using heuristic

        Returns:
            True if updated successfully
        """
        async with self._lock:
            if model_name not in self._models:
                return False

            model = self._models[model_name]

            if model.drl_metrics is None:
                model.drl_metrics = DRLModelMetrics()

            metrics = model.drl_metrics

            if algorithm is not None:
                metrics.algorithm = algorithm
            if episode_reward is not None:
                metrics.episode_reward = episode_reward
                metrics.episode_count += 1
                # Update moving average (exponential)
                alpha = 0.1
                metrics.episode_reward_avg = (
                    alpha * episode_reward + (1 - alpha) * metrics.episode_reward_avg
                )
            if policy_loss is not None:
                metrics.policy_loss = policy_loss
            if value_loss is not None:
                metrics.value_loss = value_loss
            if entropy is not None:
                metrics.entropy = entropy
            if kl_divergence is not None:
                metrics.kl_divergence = kl_divergence
            if explained_variance is not None:
                metrics.explained_variance = explained_variance
            if total_timesteps is not None:
                metrics.total_timesteps = total_timesteps
            if decisions_made is not None:
                metrics.decisions_made = decisions_made
            if drl_decisions is not None:
                metrics.drl_decisions = drl_decisions
            if fallback_decisions is not None:
                metrics.fallback_decisions = fallback_decisions

            return True

    async def update_tft_metrics(
        self,
        model_name: str,
        mape: float | None = None,
        mae: float | None = None,
        rmse: float | None = None,
        forecast_horizon: str | None = None,
        symbols_tracked: int | None = None,
        forecasts_generated: int | None = None,
        avg_confidence: float | None = None,
        calibration_score: float | None = None,
        mape_1h: float | None = None,
        mape_4h: float | None = None,
        mape_1d: float | None = None,
    ) -> bool:
        """
        Update TFT-specific metrics for a model.

        Args:
            model_name: Name of the TFT model
            mape: Mean Absolute Percentage Error
            mae: Mean Absolute Error
            rmse: Root Mean Square Error
            forecast_horizon: Current forecast horizon
            symbols_tracked: Number of symbols tracked
            forecasts_generated: Total forecasts generated
            avg_confidence: Average forecast confidence
            calibration_score: Prediction interval calibration
            mape_1h: MAPE for 1-hour horizon
            mape_4h: MAPE for 4-hour horizon
            mape_1d: MAPE for 1-day horizon

        Returns:
            True if updated successfully
        """
        async with self._lock:
            if model_name not in self._models:
                return False

            model = self._models[model_name]

            if model.tft_metrics is None:
                model.tft_metrics = TFTModelMetrics()

            metrics = model.tft_metrics

            if mape is not None:
                metrics.mape = mape
            if mae is not None:
                metrics.mae = mae
            if rmse is not None:
                metrics.rmse = rmse
            if forecast_horizon is not None:
                metrics.forecast_horizon = forecast_horizon
            if symbols_tracked is not None:
                metrics.symbols_tracked = symbols_tracked
            if forecasts_generated is not None:
                metrics.forecasts_generated = forecasts_generated
            if avg_confidence is not None:
                metrics.avg_confidence = avg_confidence
            if calibration_score is not None:
                metrics.calibration_score = calibration_score
            if mape_1h is not None:
                metrics.mape_1h = mape_1h
            if mape_4h is not None:
                metrics.mape_4h = mape_4h
            if mape_1d is not None:
                metrics.mape_1d = mape_1d

            return True

    async def update_finbert_metrics(
        self,
        model_name: str,
        accuracy: float | None = None,
        f1_score: float | None = None,
        precision: float | None = None,
        recall: float | None = None,
        inference_speed: float | None = None,
        cache_hit_rate: float | None = None,
        total_analyzed: int | None = None,
        positive_ratio: float | None = None,
        negative_ratio: float | None = None,
        neutral_ratio: float | None = None,
        device: str | None = None,
    ) -> bool:
        """
        Update FinBERT-specific metrics for a model.

        Args:
            model_name: Name of the FinBERT model
            accuracy: Classification accuracy
            f1_score: F1 score
            precision: Precision score
            recall: Recall score
            inference_speed: Texts per second
            cache_hit_rate: Cache efficiency
            total_analyzed: Total texts analyzed
            positive_ratio: Ratio of positive sentiments
            negative_ratio: Ratio of negative sentiments
            neutral_ratio: Ratio of neutral sentiments
            device: Device (cpu/cuda)

        Returns:
            True if updated successfully
        """
        async with self._lock:
            if model_name not in self._models:
                return False

            model = self._models[model_name]

            if model.finbert_metrics is None:
                model.finbert_metrics = FinBERTMetrics()

            metrics = model.finbert_metrics

            if accuracy is not None:
                metrics.accuracy = accuracy
            if f1_score is not None:
                metrics.f1_score = f1_score
            if precision is not None:
                metrics.precision = precision
            if recall is not None:
                metrics.recall = recall
            if inference_speed is not None:
                metrics.inference_speed = inference_speed
            if cache_hit_rate is not None:
                metrics.cache_hit_rate = cache_hit_rate
            if total_analyzed is not None:
                metrics.total_analyzed = total_analyzed
            if positive_ratio is not None:
                metrics.positive_ratio = positive_ratio
            if negative_ratio is not None:
                metrics.negative_ratio = negative_ratio
            if neutral_ratio is not None:
                metrics.neutral_ratio = neutral_ratio
            if device is not None:
                metrics.device = device

            return True

    async def update_llm_metrics(
        self,
        model_name: str,
        provider: str | None = None,
        llm_model_name: str | None = None,
        total_requests: int | None = None,
        successful_requests: int | None = None,
        failed_requests: int | None = None,
        total_tokens_in: int | None = None,
        total_tokens_out: int | None = None,
        estimated_cost_usd: float | None = None,
        avg_response_quality: float | None = None,
        rate_limit_hits: int | None = None,
    ) -> bool:
        """
        Update LLM-specific metrics for a model.

        Args:
            model_name: Name of the LLM client
            provider: Provider name (anthropic, openai)
            llm_model_name: Actual model name (claude-3-opus, gpt-4)
            total_requests: Total API requests
            successful_requests: Successful requests
            failed_requests: Failed requests
            total_tokens_in: Input tokens used
            total_tokens_out: Output tokens generated
            estimated_cost_usd: Estimated API cost
            avg_response_quality: Quality score (0-1)
            rate_limit_hits: Rate limit encounters

        Returns:
            True if updated successfully
        """
        async with self._lock:
            if model_name not in self._models:
                return False

            model = self._models[model_name]

            if model.llm_metrics is None:
                model.llm_metrics = LLMMetrics()

            metrics = model.llm_metrics

            if provider is not None:
                metrics.provider = provider
            if llm_model_name is not None:
                metrics.model_name = llm_model_name
            if total_requests is not None:
                metrics.total_requests = total_requests
            if successful_requests is not None:
                metrics.successful_requests = successful_requests
            if failed_requests is not None:
                metrics.failed_requests = failed_requests
            if total_tokens_in is not None:
                metrics.total_tokens_in = total_tokens_in
            if total_tokens_out is not None:
                metrics.total_tokens_out = total_tokens_out
            if estimated_cost_usd is not None:
                metrics.estimated_cost_usd = estimated_cost_usd
            if avg_response_quality is not None:
                metrics.avg_response_quality = avg_response_quality
            if rate_limit_hits is not None:
                metrics.rate_limit_hits = rate_limit_hits

            return True

    async def get_model_info(self, model_name: str) -> ModelInfo | None:
        """
        Get information about a specific model.

        Args:
            model_name: Name of the model

        Returns:
            ModelInfo or None if not found
        """
        async with self._lock:
            return self._models.get(model_name)

    async def get_all_models(self) -> dict[str, ModelInfo]:
        """
        Get information about all registered models.

        Returns:
            Dictionary mapping model names to ModelInfo
        """
        async with self._lock:
            return dict(self._models)

    async def get_models_by_type(self, model_type: ModelType) -> list[ModelInfo]:
        """
        Get all models of a specific type.

        Args:
            model_type: Type of models to retrieve

        Returns:
            List of ModelInfo for matching models
        """
        async with self._lock:
            return [
                model for model in self._models.values()
                if model.model_type == model_type
            ]

    async def get_models_by_status(self, status: ModelStatus) -> list[ModelInfo]:
        """
        Get all models with a specific status.

        Args:
            status: Status to filter by

        Returns:
            List of ModelInfo for matching models
        """
        async with self._lock:
            return [
                model for model in self._models.values()
                if model.status == status
            ]

    async def get_model_health(self, model_name: str) -> float:
        """
        Get the health score for a specific model.

        Args:
            model_name: Name of the model

        Returns:
            Health score (0-100) or 0 if model not found
        """
        async with self._lock:
            model = self._models.get(model_name)
            if model:
                return model._calculate_health_score()
            return 0.0

    async def get_system_health(self) -> ModelSystemHealth:
        """
        Get aggregated health metrics for all models.

        Returns:
            ModelSystemHealth with system-wide health overview
        """
        async with self._lock:
            health = ModelSystemHealth(
                total_models=len(self._models),
                last_updated=datetime.now(timezone.utc),
            )

            if not self._models:
                return health

            health_scores = []
            min_health = 100.0
            min_health_model = None

            for model in self._models.values():
                score = model._calculate_health_score()
                health_scores.append(score)

                # Track minimum health
                if score < min_health:
                    min_health = score
                    min_health_model = model.model_name

                # Count by status
                if model.status == ModelStatus.LOADED:
                    health.loaded_models += 1
                elif model.status == ModelStatus.ERROR:
                    health.error_models += 1
                elif model.status == ModelStatus.FALLBACK:
                    health.fallback_models += 1
                elif model.status == ModelStatus.NOT_AVAILABLE:
                    health.unavailable_models += 1

                # Aggregate metrics
                health.total_inferences += model.inference_count
                health.total_errors += model.error_count

            # Calculate averages
            if health_scores:
                health.avg_health_score = sum(health_scores) / len(health_scores)
                health.min_health_score = min_health
                health.min_health_model = min_health_model

            # Determine system health status
            if health.avg_health_score >= self.HEALTH_DEGRADED_THRESHOLD:
                health.system_health = "healthy"
            elif health.avg_health_score >= self.HEALTH_CRITICAL_THRESHOLD:
                health.system_health = "degraded"
            else:
                health.system_health = "critical"

            # Override to degraded if any model in error
            if health.error_models > 0:
                if health.system_health == "healthy":
                    health.system_health = "degraded"

            return health

    async def check_model_availability(self, model_name: str) -> tuple[bool, str]:
        """
        Check if a model is available and ready for inference.

        Args:
            model_name: Name of the model

        Returns:
            Tuple of (is_available, status_message)
        """
        async with self._lock:
            if model_name not in self._models:
                return False, f"Model '{model_name}' not registered"

            model = self._models[model_name]

            if model.status == ModelStatus.LOADED:
                return True, "Model loaded and ready"
            elif model.status == ModelStatus.FALLBACK:
                return True, "Using fallback/heuristic mode"
            elif model.status == ModelStatus.LOADING:
                return False, "Model is loading"
            elif model.status == ModelStatus.TRAINING:
                return False, "Model is training"
            elif model.status == ModelStatus.ERROR:
                return False, f"Model error: {model.error_message or 'Unknown error'}"
            elif model.status == ModelStatus.NOT_AVAILABLE:
                return False, "Model/dependencies not available"
            else:
                return False, f"Model status: {model.status.value}"

    async def get_available_models(self) -> list[ModelInfo]:
        """
        Get all models that are available for inference.

        Returns:
            List of ModelInfo for available models
        """
        async with self._lock:
            return [
                model for model in self._models.values()
                if model.is_available
            ]

    async def get_unhealthy_models(
        self,
        threshold: float = HEALTH_DEGRADED_THRESHOLD,
    ) -> list[ModelInfo]:
        """
        Get models with health score below threshold.

        Args:
            threshold: Health score threshold (default 75.0)

        Returns:
            List of ModelInfo for unhealthy models, sorted by health score
        """
        async with self._lock:
            unhealthy = []
            for model in self._models.values():
                score = model._calculate_health_score()
                if score < threshold:
                    unhealthy.append(model)
            return sorted(unhealthy, key=lambda m: m._calculate_health_score())

    async def reset_model_errors(self, model_name: str) -> bool:
        """
        Reset error counts for a model.

        Args:
            model_name: Name of the model

        Returns:
            True if reset successful, False if model not found
        """
        async with self._lock:
            model = self._models.get(model_name)
            if model:
                model.error_count = 0
                model.error_message = None
                model.last_error_time = None
                logger.info(f"Reset errors for model: {model_name}")
                return True
            return False

    def to_dict(self) -> dict[str, Any]:
        """
        Export tracker state to dictionary for WebSocket streaming.

        Note: This is synchronous for compatibility with simple JSON serialization.
        For async contexts, use to_dict_async().

        Returns:
            Complete tracker state as dict
        """
        # Calculate system health synchronously
        health_scores = []
        metrics = {
            "total_models": len(self._models),
            "loaded_models": 0,
            "error_models": 0,
            "fallback_models": 0,
            "unavailable_models": 0,
            "total_inferences": 0,
            "total_errors": 0,
        }

        for model in self._models.values():
            score = model._calculate_health_score()
            health_scores.append(score)
            metrics["total_inferences"] += model.inference_count
            metrics["total_errors"] += model.error_count

            if model.status == ModelStatus.LOADED:
                metrics["loaded_models"] += 1
            elif model.status == ModelStatus.ERROR:
                metrics["error_models"] += 1
            elif model.status == ModelStatus.FALLBACK:
                metrics["fallback_models"] += 1
            elif model.status == ModelStatus.NOT_AVAILABLE:
                metrics["unavailable_models"] += 1

        avg_health = sum(health_scores) / len(health_scores) if health_scores else 100.0
        min_health = min(health_scores) if health_scores else 100.0

        # Determine system health
        if avg_health >= self.HEALTH_DEGRADED_THRESHOLD:
            system_health = "healthy"
        elif avg_health >= self.HEALTH_CRITICAL_THRESHOLD:
            system_health = "degraded"
        else:
            system_health = "critical"

        if metrics["error_models"] > 0 and system_health == "healthy":
            system_health = "degraded"

        return {
            "models": {
                name: model.to_dict()
                for name, model in self._models.items()
            },
            "by_type": {
                model_type.value: [
                    model.to_dict()
                    for model in self._models.values()
                    if model.model_type == model_type
                ]
                for model_type in ModelType
            },
            "system_health": {
                "status": system_health,
                "avg_health_score": round(avg_health, 1),
                "min_health_score": round(min_health, 1),
                **metrics,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    async def to_dict_async(self) -> dict[str, Any]:
        """
        Async version of to_dict with proper locking.

        Returns:
            Complete tracker state as dict
        """
        async with self._lock:
            return self.to_dict()

    @property
    def model_count(self) -> int:
        """Get the number of registered models."""
        return len(self._models)

    @property
    def available_model_count(self) -> int:
        """Get the number of available models."""
        return sum(1 for model in self._models.values() if model.is_available)

    @property
    def registered_model_names(self) -> list[str]:
        """Get list of all registered model names."""
        return list(self._models.keys())


# =============================================================================
# Factory Functions for Model Status Cards
# =============================================================================

def create_drl_model_card(
    model_name: str,
    algorithm: str = "PPO",
    version: str = "1.0.0",
) -> ModelInfo:
    """
    Create a ModelInfo for a DRL model (for status card display).

    Args:
        model_name: Name for the model
        algorithm: DRL algorithm (PPO, A2C, TD3)
        version: Model version

    Returns:
        Configured ModelInfo for DRL model
    """
    model = ModelInfo(
        model_name=model_name,
        model_type=ModelType.DRL,
        version=version,
        status=ModelStatus.NOT_AVAILABLE,
    )
    model.drl_metrics = DRLModelMetrics(algorithm=algorithm)
    return model


def create_tft_model_card(
    model_name: str = "TFT_Forecaster",
    version: str = "1.0.0",
) -> ModelInfo:
    """
    Create a ModelInfo for a TFT model (for status card display).

    Args:
        model_name: Name for the model
        version: Model version

    Returns:
        Configured ModelInfo for TFT model
    """
    model = ModelInfo(
        model_name=model_name,
        model_type=ModelType.TFT,
        version=version,
        status=ModelStatus.NOT_AVAILABLE,
    )
    model.tft_metrics = TFTModelMetrics()
    return model


def create_finbert_model_card(
    model_name: str = "FinBERT_Sentiment",
    version: str = "1.0.0",
) -> ModelInfo:
    """
    Create a ModelInfo for a FinBERT model (for status card display).

    Args:
        model_name: Name for the model
        version: Model version

    Returns:
        Configured ModelInfo for FinBERT model
    """
    model = ModelInfo(
        model_name=model_name,
        model_type=ModelType.FINBERT,
        version=version,
        status=ModelStatus.NOT_AVAILABLE,
    )
    model.finbert_metrics = FinBERTMetrics()
    return model


def create_llm_model_card(
    model_name: str,
    provider: str = "anthropic",
    llm_model_name: str = "",
    version: str = "1.0.0",
) -> ModelInfo:
    """
    Create a ModelInfo for an LLM client (for status card display).

    Args:
        model_name: Name for the model/client
        provider: Provider name (anthropic, openai)
        llm_model_name: Actual model name (claude-3-opus, gpt-4)
        version: Client version

    Returns:
        Configured ModelInfo for LLM model
    """
    model = ModelInfo(
        model_name=model_name,
        model_type=ModelType.LLM,
        version=version,
        status=ModelStatus.NOT_AVAILABLE,
    )
    model.llm_metrics = LLMMetrics(provider=provider, model_name=llm_model_name)
    return model
