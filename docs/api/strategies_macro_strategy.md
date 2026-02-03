# macro_strategy

**Path**: `C:\Users\Alexa\ai-trading-firm\strategies\macro_strategy.py`

## Overview

Macro Strategy
==============

Implements macroeconomic signal generation logic.

MATURITY: ALPHA
---------------
Status: Basic framework - placeholder implementation
- [x] Regime classification (expansion, slowdown, recession, recovery)
- [x] Risk allocation by regime
- [x] Sector rotation signals
- [x] VIX term structure analysis
- [ ] Hidden Markov Model for regime detection (TODO)
- [ ] Leading economic indicators integration (TODO)
- [ ] Cross-asset correlation analysis (TODO)
- [ ] Credit cycle indicators (TODO)

Production Readiness:
- Unit tests: Minimal
- Backtesting: Not performed
- Data sources: Not integrated (needs external macro data)

WARNING: DO NOT USE IN PRODUCTION
- Regime detection is overly simplified
- Requires external macro data feeds (not implemented)
- Sector signals are static, not data-driven
- Consider this a conceptual framework only

## Classes

### MacroRegime

**Inherits from**: Enum

Macroeconomic regime classification.

### MacroIndicators

Container for macro indicators.

### MacroStrategy

Macro Strategy Implementation.

Analyzes macroeconomic indicators to determine:
1. Current economic regime
2. Risk-on/risk-off positioning
3. Sector rotation signals

TODO: Implement actual models:
- Yield curve analysis
- Credit cycle indicators
- Leading economic indicators
- Cross-asset correlations

#### Methods

##### `def __init__(self, config: dict[str, Any])`

##### `def update_indicator(self, name: str, value: float) -> None`

Update a macro indicator value.

##### `def analyze_regime(self) -> MacroRegime`

Analyze current macro regime.

TODO: Implement proper regime detection:
- Hidden Markov Model
- Rule-based classification
- Machine learning classifier

##### `def get_risk_allocation(self) -> float`

Get recommended risk allocation based on regime.

Returns a factor 0.0 to 1.0 for risk scaling.

##### `def get_sector_signals(self) -> dict[str, float]`

Get sector rotation signals based on regime.

Returns dict of sector -> signal strength (-1 to 1).

TODO: Implement proper sector rotation model.

##### `def analyze_vix_term_structure(self, vix_spot: float, vix_futures: list[float]) -> dict[str, Any]`

Analyze VIX term structure for signals.

TODO: Implement VIX term structure analysis:
- Contango vs backwardation
- Roll yield estimation
- Vol regime detection
