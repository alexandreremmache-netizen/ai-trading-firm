# market_making_strategy

**Path**: `C:\Users\Alexa\ai-trading-firm\strategies\market_making_strategy.py`

## Overview

Market Making Strategy
======================

Implements market making logic (IB-compatible latency).

MATURITY: ALPHA
---------------
Status: Basic framework - NOT ready for production
- [x] Fair value estimation (size-weighted mid)
- [x] Basic spread calculation (A-S inspired)
- [x] Inventory management (skew adjustment)
- [x] Quote generation
- [ ] Avellaneda-Stoikov full model (TODO)
- [ ] Order arrival rate estimation (TODO)
- [ ] Kyle's lambda (adverse selection) (TODO)
- [ ] VPIN (order flow toxicity) (TODO)

Production Readiness:
- Unit tests: Minimal
- Backtesting: Not performed
- Live testing: Not performed

WARNING: DO NOT USE IN PRODUCTION
- This is a simplified placeholder implementation
- Real market making requires:
  * Sub-100ms latency (not achievable with IB)
  * Sophisticated inventory models
  * Adverse selection detection
  * Fee structure optimization

NOTE: Not HFT - operates at 100ms+ latencies due to IB constraints.

## Classes

### Quote

A bid/ask quote.

### MarketMakingSignal

Market making signal output.

### MarketMakingStrategy

Market Making Strategy Implementation.

Implements:
1. Fair value estimation
2. Optimal spread calculation
3. Inventory management
4. Adverse selection avoidance

NOT HFT - All quotes refreshed at IB-compatible intervals (100ms+).

TODO: Implement proper models:
- Avellaneda-Stoikov optimal market making
- Kyle's lambda for adverse selection
- Order flow toxicity (VPIN)

#### Methods

##### `def __init__(self, config: dict[str, Any])`

##### `def estimate_fair_value(self, bid: float, ask: float, bid_size: int, ask_size: int) -> float`

Estimate fair value from order book.

Uses size-weighted mid price.

TODO: Implement more sophisticated fair value:
- Micro-price
- Order flow imbalance
- Volume-weighted price

##### `def calculate_optimal_spread(self, volatility: float, inventory: int, time_horizon: float) -> float`

Calculate optimal spread using Avellaneda-Stoikov framework.

s* = gamma * sigma^2 * T + (2/gamma) * ln(1 + gamma/k)

Simplified version - full implementation would need order arrival rate.

TODO: Implement full A-S model with:
- Order arrival rate estimation
- Inventory penalty
- Terminal inventory constraint

##### `def calculate_quote_skew(self, inventory: int) -> float`

Calculate quote skew based on inventory.

Positive skew = higher ask (want to sell)
Negative skew = lower bid (want to buy)

##### `def generate_quotes(self, symbol: str, fair_value: float, volatility: float, inventory: int, bid_size: int, ask_size: int) -> MarketMakingSignal`

Generate bid/ask quotes.

##### `def should_quote(self, current_spread: float, our_spread: float, inventory: int) -> bool`

Determine if we should be quoting.

Don't quote if:
- Spread is too tight (can't make money)
- Inventory is at limit
- Market is too fast (adverse selection)
