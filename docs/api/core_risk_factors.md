# risk_factors

**Path**: `C:\Users\Alexa\ai-trading-firm\core\risk_factors.py`

## Overview

Risk Factor Decomposition Module
================================

Risk factor analysis and decomposition (Issue #R12).

Features:
- Multi-factor risk model (market, size, value, momentum, quality)
- Factor exposure calculation
- Risk contribution attribution
- Factor covariance estimation
- Systematic vs idiosyncratic risk separation

## Classes

### RiskFactor

**Inherits from**: str, Enum

Standard risk factors.

### FactorExposure

Single factor exposure measurement.

### PositionFactorExposures

Factor exposures for a single position.

#### Methods

##### `def get_exposure(self, factor: RiskFactor) -> float`

Get exposure to a specific factor.

### PortfolioFactorDecomposition

Full portfolio factor decomposition.

#### Methods

##### `def get_systematic_pct(self) -> float`

Get percentage of risk that is systematic.

##### `def get_top_factors(self, n: int) -> list[tuple[RiskFactor, float]]`

Get top N factors by contribution.

##### `def to_dict(self) -> dict`

Convert to dictionary.

### FactorModel

Multi-factor risk model.

Implements factor-based risk decomposition.

#### Methods

##### `def __init__(self, factors: , lookback_days: int, min_observations: int)`

##### `def update_factor_returns(self, factor_returns: dict[RiskFactor, float]) -> None`

Update factor return history.

##### `def update_position_returns(self, position_returns: dict[str, float]) -> None`

Update position return history.

##### `def estimate_factor_covariance(self) -> dict[tuple[RiskFactor, RiskFactor], float]`

Estimate factor covariance matrix.

##### `def calculate_position_exposures(self, symbol: str) -> PositionFactorExposures`

Calculate factor exposures for a single position.

Uses OLS regression of position returns on factor returns.

##### `def decompose_portfolio(self, positions: dict[str, float]) -> PortfolioFactorDecomposition`

Decompose portfolio risk by factor.

Args:
    positions: Dictionary of symbol to portfolio weight

Returns:
    PortfolioFactorDecomposition with full analysis

### RiskContribution

Risk contribution by source.

### RiskContributionAnalyzer

Analyzes risk contributions from different sources.

Supports position-level, factor-level, and strategy-level attribution.

#### Methods

##### `def __init__(self, factor_model: FactorModel)`

##### `def analyze_position_contributions(self, positions: dict[str, float], position_vols: dict[str, float], correlation_matrix: dict[tuple[str, str], float]) -> list[RiskContribution]`

Analyze risk contribution by position.

Uses Euler decomposition for additive risk attribution.

##### `def analyze_factor_contributions(self, decomposition: PortfolioFactorDecomposition) -> list[RiskContribution]`

Analyze risk contribution by factor.

Based on factor decomposition results.

### FactorTilt

Intentional factor tilt relative to benchmark.

### ActiveRiskDecomposer

Decomposes active risk (tracking error) by factor.

Useful for understanding sources of benchmark-relative risk.

#### Methods

##### `def __init__(self, factor_model: FactorModel)`

##### `def decompose_active_risk(self, portfolio_positions: dict[str, float], benchmark_positions: dict[str, float]) -> tuple[float, list[FactorTilt]]`

Decompose tracking error by factor tilt.

Returns:
    - Tracking error (annualized volatility)
    - List of factor tilts with contributions
