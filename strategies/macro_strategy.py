"""
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
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)


class MacroRegime(Enum):
    """Macroeconomic regime classification."""
    EXPANSION = "expansion"
    SLOWDOWN = "slowdown"
    RECESSION = "recession"
    RECOVERY = "recovery"


@dataclass
class MacroIndicators:
    """Container for macro indicators."""
    vix: float = 0.0
    yield_2y: float = 0.0
    yield_10y: float = 0.0
    dxy: float = 0.0
    credit_spread: float = 0.0
    pmi: float = 0.0
    unemployment: float = 0.0


class MacroStrategy:
    """
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
    """

    def __init__(self, config: dict[str, Any]):
        self._indicators_to_track = config.get("indicators", ["vix", "yield_curve"])
        self._current_indicators = MacroIndicators()
        self._current_regime = MacroRegime.EXPANSION
        self._regime_history: list[MacroRegime] = []

    def update_indicator(self, name: str, value: float) -> None:
        """Update a macro indicator value."""
        if hasattr(self._current_indicators, name):
            setattr(self._current_indicators, name, value)

    def analyze_regime(self) -> MacroRegime:
        """
        Analyze current macro regime.

        TODO: Implement proper regime detection:
        - Hidden Markov Model
        - Rule-based classification
        - Machine learning classifier
        """
        ind = self._current_indicators

        # Simplified regime detection
        # Real implementation would use proper models

        # Yield curve inversion check
        yield_spread = ind.yield_10y - ind.yield_2y

        if yield_spread < 0 and ind.vix > 25:
            regime = MacroRegime.RECESSION
        elif yield_spread < 0.5 and ind.vix > 20:
            regime = MacroRegime.SLOWDOWN
        elif yield_spread > 1.0 and ind.vix < 15:
            regime = MacroRegime.EXPANSION
        else:
            regime = MacroRegime.RECOVERY

        if regime != self._current_regime:
            logger.info(f"Macro regime change: {self._current_regime.value} -> {regime.value}")
            self._regime_history.append(self._current_regime)
            self._current_regime = regime

        return regime

    def get_risk_allocation(self) -> float:
        """
        Get recommended risk allocation based on regime.

        Returns a factor 0.0 to 1.0 for risk scaling.
        """
        regime_allocations = {
            MacroRegime.EXPANSION: 1.0,
            MacroRegime.SLOWDOWN: 0.6,
            MacroRegime.RECESSION: 0.3,
            MacroRegime.RECOVERY: 0.8,
        }
        return regime_allocations.get(self._current_regime, 0.5)

    def get_sector_signals(self) -> dict[str, float]:
        """
        Get sector rotation signals based on regime.

        Returns dict of sector -> signal strength (-1 to 1).

        TODO: Implement proper sector rotation model.
        """
        # Simplified sector signals by regime
        if self._current_regime == MacroRegime.EXPANSION:
            return {
                "technology": 0.8,
                "consumer_discretionary": 0.6,
                "financials": 0.5,
                "utilities": -0.3,
                "consumer_staples": -0.2,
            }
        elif self._current_regime == MacroRegime.RECESSION:
            return {
                "utilities": 0.7,
                "consumer_staples": 0.6,
                "healthcare": 0.5,
                "technology": -0.4,
                "financials": -0.6,
            }
        else:
            return {}

    def analyze_vix_term_structure(
        self,
        vix_spot: float,
        vix_futures: list[float],
    ) -> dict[str, Any]:
        """
        Analyze VIX term structure for signals.

        TODO: Implement VIX term structure analysis:
        - Contango vs backwardation
        - Roll yield estimation
        - Vol regime detection
        """
        if not vix_futures:
            return {"signal": "neutral", "strength": 0.0}

        # Simplified: Check if term structure is in contango or backwardation
        avg_futures = np.mean(vix_futures)

        if vix_spot > avg_futures * 1.1:  # Backwardation
            return {
                "signal": "risk_off",
                "strength": -0.7,
                "structure": "backwardation",
            }
        elif vix_spot < avg_futures * 0.9:  # Steep contango
            return {
                "signal": "risk_on",
                "strength": 0.5,
                "structure": "contango",
            }
        else:
            return {
                "signal": "neutral",
                "strength": 0.0,
                "structure": "flat",
            }

    # =========================================================================
    # ECONOMIC INDICATOR TRACKING (P3)
    # =========================================================================

    def track_economic_indicators(
        self,
        indicators: dict[str, float]
    ) -> dict[str, Any]:
        """
        Track and analyze key economic indicators (P3).

        Monitors leading, coincident, and lagging indicators for regime signals.

        Args:
            indicators: Dictionary of indicator values with keys:
                - pmi: Manufacturing PMI
                - nfp: Non-farm payrolls (month-over-month)
                - cpi_yoy: CPI year-over-year
                - gdp_growth: GDP growth rate
                - housing_starts: Housing starts
                - retail_sales: Retail sales growth
                - jobless_claims: Weekly jobless claims
                - industrial_production: Industrial production index
                - consumer_confidence: Consumer confidence index

        Returns:
            Economic indicator analysis with signals
        """
        # Update current indicators
        for name, value in indicators.items():
            self.update_indicator(name, value)

        # Categorize indicators
        leading = {
            "pmi": indicators.get("pmi", 50),
            "housing_starts": indicators.get("housing_starts", 0),
            "consumer_confidence": indicators.get("consumer_confidence", 100),
            "jobless_claims": indicators.get("jobless_claims", 0),
        }

        coincident = {
            "nfp": indicators.get("nfp", 0),
            "industrial_production": indicators.get("industrial_production", 0),
            "retail_sales": indicators.get("retail_sales", 0),
        }

        lagging = {
            "cpi_yoy": indicators.get("cpi_yoy", 2.0),
            "gdp_growth": indicators.get("gdp_growth", 2.0),
            "unemployment": indicators.get("unemployment", 4.0),
        }

        # Calculate composite scores
        leading_score = self._calculate_indicator_score(leading)
        coincident_score = self._calculate_indicator_score(coincident)
        lagging_score = self._calculate_indicator_score(lagging)

        # Generate signals
        signals = []

        # PMI signal
        pmi = leading.get("pmi", 50)
        if pmi > 55:
            signals.append({"indicator": "pmi", "signal": "expansion", "strength": 0.8})
        elif pmi < 45:
            signals.append({"indicator": "pmi", "signal": "contraction", "strength": 0.8})
        elif pmi < 50:
            signals.append({"indicator": "pmi", "signal": "slowdown", "strength": 0.5})

        # Jobless claims signal (inverted - lower is better)
        claims = leading.get("jobless_claims", 250000)
        if claims < 200000:
            signals.append({"indicator": "jobless_claims", "signal": "strong_labor", "strength": 0.7})
        elif claims > 350000:
            signals.append({"indicator": "jobless_claims", "signal": "weak_labor", "strength": 0.7})

        # CPI/inflation signal
        cpi = lagging.get("cpi_yoy", 2.0)
        if cpi > 4.0:
            signals.append({"indicator": "cpi", "signal": "high_inflation", "strength": 0.8})
        elif cpi < 1.0:
            signals.append({"indicator": "cpi", "signal": "deflation_risk", "strength": 0.6})

        # Determine overall economic phase
        if leading_score > 0.5 and coincident_score > 0:
            phase = "expansion"
        elif leading_score < -0.3 and coincident_score < 0:
            phase = "contraction"
        elif leading_score > 0 and coincident_score < 0:
            phase = "early_recovery"
        elif leading_score < 0 and coincident_score > 0:
            phase = "late_cycle"
        else:
            phase = "neutral"

        return {
            "leading_score": leading_score,
            "coincident_score": coincident_score,
            "lagging_score": lagging_score,
            "economic_phase": phase,
            "signals": signals,
            "indicators": {
                "leading": leading,
                "coincident": coincident,
                "lagging": lagging,
            },
            "risk_allocation_suggestion": self._get_phase_allocation(phase),
        }

    def _calculate_indicator_score(self, indicators: dict[str, float]) -> float:
        """Calculate normalized score from indicator values."""
        # Thresholds for normalization
        thresholds = {
            "pmi": (45, 50, 55),  # contraction, neutral, expansion
            "housing_starts": (1.0, 1.3, 1.6),  # millions
            "consumer_confidence": (80, 100, 120),
            "jobless_claims": (350000, 250000, 200000),  # inverted
            "nfp": (-100, 100, 250),  # thousands
            "industrial_production": (-2, 0, 2),  # % change
            "retail_sales": (-1, 1, 3),  # % change
            "cpi_yoy": (1, 2, 3),
            "gdp_growth": (0, 2, 3),
            "unemployment": (6, 5, 4),  # inverted
        }

        scores = []
        for name, value in indicators.items():
            if name in thresholds:
                low, mid, high = thresholds[name]
                # Check if inverted (lower is better)
                if name in ["jobless_claims", "unemployment"]:
                    if value <= high:
                        score = 1.0
                    elif value >= low:
                        score = -1.0
                    else:
                        score = (mid - value) / (mid - high) if value < mid else -(value - mid) / (low - mid)
                else:
                    if value >= high:
                        score = 1.0
                    elif value <= low:
                        score = -1.0
                    else:
                        score = (value - mid) / (high - mid) if value > mid else -(mid - value) / (mid - low)
                scores.append(np.clip(score, -1, 1))

        return np.mean(scores) if scores else 0.0

    def _get_phase_allocation(self, phase: str) -> dict[str, float]:
        """Get suggested asset allocation for economic phase."""
        allocations = {
            "expansion": {
                "equities": 0.7,
                "bonds": 0.15,
                "commodities": 0.10,
                "cash": 0.05,
            },
            "late_cycle": {
                "equities": 0.5,
                "bonds": 0.25,
                "commodities": 0.15,
                "cash": 0.10,
            },
            "contraction": {
                "equities": 0.25,
                "bonds": 0.45,
                "commodities": 0.05,
                "cash": 0.25,
            },
            "early_recovery": {
                "equities": 0.6,
                "bonds": 0.20,
                "commodities": 0.15,
                "cash": 0.05,
            },
            "neutral": {
                "equities": 0.5,
                "bonds": 0.30,
                "commodities": 0.10,
                "cash": 0.10,
            },
        }
        return allocations.get(phase, allocations["neutral"])

    # =========================================================================
    # CENTRAL BANK EVENT CALENDAR (P3)
    # =========================================================================

    def get_central_bank_calendar(
        self,
        year: int | None = None,
        month: int | None = None
    ) -> list[dict[str, Any]]:
        """
        Get central bank meeting calendar (P3).

        Returns scheduled FOMC, ECB, BOE, BOJ meetings.

        Args:
            year: Year to filter (default: current year)
            month: Month to filter (optional)

        Returns:
            List of central bank events
        """
        from datetime import datetime, date
        import calendar

        if year is None:
            year = datetime.now().year

        # FOMC meetings (approximately 8 per year)
        # Typically: Jan, Mar, May, Jun, Jul, Sep, Nov, Dec
        fomc_meetings = [
            {"month": 1, "week": 4, "day": 2},  # Last week of Jan, Wednesday
            {"month": 3, "week": 3, "day": 2},  # Third week of Mar
            {"month": 5, "week": 1, "day": 2},  # First week of May
            {"month": 6, "week": 2, "day": 2},  # Second week of Jun
            {"month": 7, "week": 4, "day": 2},  # Fourth week of Jul
            {"month": 9, "week": 3, "day": 2},  # Third week of Sep
            {"month": 11, "week": 1, "day": 2},  # First week of Nov
            {"month": 12, "week": 2, "day": 2},  # Second week of Dec
        ]

        # ECB meetings (approximately 8 per year)
        ecb_meetings = [
            {"month": 1, "week": 4, "day": 3},
            {"month": 3, "week": 2, "day": 3},
            {"month": 4, "week": 2, "day": 3},
            {"month": 6, "week": 2, "day": 3},
            {"month": 7, "week": 3, "day": 3},
            {"month": 9, "week": 2, "day": 3},
            {"month": 10, "week": 4, "day": 3},
            {"month": 12, "week": 2, "day": 3},
        ]

        def get_nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
            """Get nth occurrence of weekday in month."""
            first_day = date(year, month, 1)
            first_weekday = first_day.weekday()
            days_until = (weekday - first_weekday) % 7
            return first_day.replace(day=1 + days_until + (n - 1) * 7)

        events = []

        # Generate FOMC events
        for meeting in fomc_meetings:
            if month is not None and meeting["month"] != month:
                continue
            try:
                meeting_date = get_nth_weekday(year, meeting["month"], meeting["day"], meeting["week"])
                events.append({
                    "date": meeting_date.isoformat(),
                    "bank": "FOMC",
                    "currency": "USD",
                    "type": "rate_decision",
                    "importance": "high",
                    "affected_assets": ["USD", "US_EQUITIES", "US_BONDS", "GOLD"],
                })
            except ValueError:
                continue

        # Generate ECB events
        for meeting in ecb_meetings:
            if month is not None and meeting["month"] != month:
                continue
            try:
                meeting_date = get_nth_weekday(year, meeting["month"], meeting["day"], meeting["week"])
                events.append({
                    "date": meeting_date.isoformat(),
                    "bank": "ECB",
                    "currency": "EUR",
                    "type": "rate_decision",
                    "importance": "high",
                    "affected_assets": ["EUR", "EU_EQUITIES", "EU_BONDS"],
                })
            except ValueError:
                continue

        # Sort by date
        events.sort(key=lambda x: x["date"])

        return events

    def get_upcoming_cb_events(
        self,
        days_ahead: int = 30
    ) -> list[dict[str, Any]]:
        """
        Get upcoming central bank events within specified days (P3).

        Args:
            days_ahead: Number of days to look ahead

        Returns:
            List of upcoming events
        """
        from datetime import datetime, timedelta

        today = datetime.now().date()
        end_date = today + timedelta(days=days_ahead)

        # Get events for current and next month
        current_year = today.year
        current_month = today.month
        next_month = current_month + 1 if current_month < 12 else 1
        next_year = current_year if current_month < 12 else current_year + 1

        events = (
            self.get_central_bank_calendar(current_year, current_month) +
            self.get_central_bank_calendar(next_year, next_month)
        )

        # Filter to upcoming events
        upcoming = []
        for event in events:
            event_date = datetime.fromisoformat(event["date"]).date()
            if today <= event_date <= end_date:
                event["days_until"] = (event_date - today).days
                upcoming.append(event)

        return upcoming

    def analyze_cb_impact(
        self,
        event: dict[str, Any],
        rate_expectation: str = "unchanged"  # "hike", "cut", "unchanged"
    ) -> dict[str, Any]:
        """
        Analyze potential market impact of central bank event (P3).

        Args:
            event: Central bank event from calendar
            rate_expectation: Market rate expectation

        Returns:
            Impact analysis
        """
        bank = event.get("bank", "")

        # Base impact patterns
        impact_patterns = {
            "FOMC": {
                "hike": {
                    "USD": "bullish",
                    "US_EQUITIES": "bearish",
                    "US_BONDS": "bearish",
                    "GOLD": "bearish",
                    "EM_ASSETS": "bearish",
                },
                "cut": {
                    "USD": "bearish",
                    "US_EQUITIES": "bullish",
                    "US_BONDS": "bullish",
                    "GOLD": "bullish",
                    "EM_ASSETS": "bullish",
                },
                "unchanged": {
                    "USD": "neutral",
                    "US_EQUITIES": "neutral",
                    "US_BONDS": "neutral",
                    "GOLD": "neutral",
                },
            },
            "ECB": {
                "hike": {
                    "EUR": "bullish",
                    "EU_EQUITIES": "bearish",
                    "EU_BONDS": "bearish",
                },
                "cut": {
                    "EUR": "bearish",
                    "EU_EQUITIES": "bullish",
                    "EU_BONDS": "bullish",
                },
                "unchanged": {
                    "EUR": "neutral",
                    "EU_EQUITIES": "neutral",
                    "EU_BONDS": "neutral",
                },
            },
        }

        impacts = impact_patterns.get(bank, {}).get(rate_expectation, {})

        return {
            "event": event,
            "rate_expectation": rate_expectation,
            "expected_impacts": impacts,
            "volatility_warning": "high" if rate_expectation != "unchanged" else "moderate",
            "recommendation": self._get_cb_recommendation(bank, rate_expectation),
        }

    def _get_cb_recommendation(self, bank: str, expectation: str) -> str:
        """Get trading recommendation around central bank event."""
        if expectation == "hike":
            return f"Reduce risk ahead of {bank} - potential for hawkish surprise. Consider USD longs, equity puts."
        elif expectation == "cut":
            return f"Position for easing from {bank} - potential for dovish surprise. Consider risk-on positions."
        else:
            return f"Monitor {bank} statement for forward guidance. Vol may be elevated around announcement."

    # =========================================================================
    # CROSS-ASSET CORRELATION SIGNALS (P3)
    # =========================================================================

    def calculate_cross_asset_correlations(
        self,
        returns: dict[str, np.ndarray],
        lookback_periods: int = 60
    ) -> dict[str, Any]:
        """
        Calculate cross-asset correlations (P3).

        Args:
            returns: Dictionary of asset returns {asset_name: returns_array}
            lookback_periods: Number of periods for correlation calculation

        Returns:
            Correlation matrix and analysis
        """
        if len(returns) < 2:
            return {"error": "need_at_least_two_assets"}

        assets = list(returns.keys())
        n_assets = len(assets)

        # Build correlation matrix
        corr_matrix = np.zeros((n_assets, n_assets))

        for i, asset1 in enumerate(assets):
            for j, asset2 in enumerate(assets):
                r1 = returns[asset1][-lookback_periods:]
                r2 = returns[asset2][-lookback_periods:]

                if len(r1) != len(r2) or len(r1) < 10:
                    corr_matrix[i, j] = np.nan
                else:
                    corr_matrix[i, j] = np.corrcoef(r1, r2)[0, 1]

        # Identify notable correlations
        notable = []
        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                corr = corr_matrix[i, j]
                if not np.isnan(corr):
                    if abs(corr) > 0.7:
                        notable.append({
                            "asset1": assets[i],
                            "asset2": assets[j],
                            "correlation": corr,
                            "type": "high_positive" if corr > 0 else "high_negative",
                        })
                    elif abs(corr) < 0.1:
                        notable.append({
                            "asset1": assets[i],
                            "asset2": assets[j],
                            "correlation": corr,
                            "type": "uncorrelated",
                        })

        return {
            "assets": assets,
            "correlation_matrix": corr_matrix.tolist(),
            "lookback_periods": lookback_periods,
            "notable_correlations": notable,
        }

    def detect_correlation_regime_change(
        self,
        returns: dict[str, np.ndarray],
        short_lookback: int = 20,
        long_lookback: int = 60,
        threshold: float = 0.3
    ) -> dict[str, Any]:
        """
        Detect correlation regime changes (P3).

        Compares short-term vs long-term correlations to identify shifts.

        Args:
            returns: Asset returns
            short_lookback: Short-term lookback periods
            long_lookback: Long-term lookback periods
            threshold: Change threshold to flag as regime change

        Returns:
            Regime change detection results
        """
        short_corr = self.calculate_cross_asset_correlations(returns, short_lookback)
        long_corr = self.calculate_cross_asset_correlations(returns, long_lookback)

        if "error" in short_corr or "error" in long_corr:
            return {"error": "correlation_calculation_failed"}

        assets = short_corr["assets"]
        short_matrix = np.array(short_corr["correlation_matrix"])
        long_matrix = np.array(long_corr["correlation_matrix"])

        # Detect changes
        changes = []
        n_assets = len(assets)

        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                short_c = short_matrix[i, j]
                long_c = long_matrix[i, j]

                if not np.isnan(short_c) and not np.isnan(long_c):
                    change = short_c - long_c
                    if abs(change) > threshold:
                        changes.append({
                            "asset1": assets[i],
                            "asset2": assets[j],
                            "short_term_corr": short_c,
                            "long_term_corr": long_c,
                            "change": change,
                            "direction": "increasing" if change > 0 else "decreasing",
                        })

        # Calculate average correlation change
        all_changes = []
        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                if not np.isnan(short_matrix[i, j]) and not np.isnan(long_matrix[i, j]):
                    all_changes.append(short_matrix[i, j] - long_matrix[i, j])

        avg_change = np.mean(all_changes) if all_changes else 0

        # Determine regime
        if avg_change > 0.15:
            regime = "correlation_spike"
            warning = "Risk-off environment - correlations rising, diversification benefits reduced"
        elif avg_change < -0.15:
            regime = "correlation_breakdown"
            warning = "Correlations falling - increased dispersion, potential for stock picking"
        else:
            regime = "stable"
            warning = None

        return {
            "regime": regime,
            "average_correlation_change": avg_change,
            "significant_changes": changes,
            "warning": warning,
            "short_lookback": short_lookback,
            "long_lookback": long_lookback,
        }

    def generate_cross_asset_signal(
        self,
        correlations: dict[str, Any],
        regime: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Generate trading signals from cross-asset analysis (P3).

        Args:
            correlations: Correlation analysis
            regime: Regime change detection

        Returns:
            Cross-asset trading signal
        """
        signals = []

        # Signal from correlation regime
        regime_type = regime.get("regime", "stable")

        if regime_type == "correlation_spike":
            signals.append({
                "signal": "reduce_risk",
                "rationale": "Correlation spike indicates stress - reduce position sizes",
                "strength": 0.8,
            })
            signals.append({
                "signal": "add_hedges",
                "rationale": "Diversification benefit reduced - add explicit hedges",
                "strength": 0.7,
            })

        elif regime_type == "correlation_breakdown":
            signals.append({
                "signal": "increase_alpha",
                "rationale": "Lower correlations favor stock/asset selection strategies",
                "strength": 0.6,
            })

        # Signals from notable correlations
        for notable in correlations.get("notable_correlations", []):
            if notable["type"] == "high_positive":
                signals.append({
                    "signal": "pair_trade_potential",
                    "assets": [notable["asset1"], notable["asset2"]],
                    "rationale": f"High correlation ({notable['correlation']:.2f}) - mean reversion opportunity",
                    "strength": 0.5,
                })
            elif notable["type"] == "uncorrelated":
                signals.append({
                    "signal": "diversification_pair",
                    "assets": [notable["asset1"], notable["asset2"]],
                    "rationale": f"Uncorrelated ({notable['correlation']:.2f}) - good for portfolio diversification",
                    "strength": 0.4,
                })

        return {
            "signals": signals,
            "regime": regime_type,
            "overall_recommendation": self._get_cross_asset_recommendation(regime_type),
        }

    def _get_cross_asset_recommendation(self, regime: str) -> str:
        """Get overall portfolio recommendation based on cross-asset regime."""
        recommendations = {
            "correlation_spike": "Defensive positioning recommended. Reduce gross exposure, add tail hedges.",
            "correlation_breakdown": "Opportunistic positioning. Favor alpha strategies over beta exposure.",
            "stable": "Normal market conditions. Maintain strategic allocation.",
        }
        return recommendations.get(regime, "Monitor conditions.")
