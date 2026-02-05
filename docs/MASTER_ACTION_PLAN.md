# AI Trading Firm - Master Action Plan
## Plan Complet d'Implementation Post-Phase 4

**Date de creation:** 2026-02-04
**Base:** Analyse de 67 fichiers de recherche + 4 phases completees
**Objectif:** Organiser les ameliorations restantes par priorite

---

## Statut Actuel

### Phases Completees (4/4)

| Phase | Description | Statut | Impact |
|-------|-------------|--------|--------|
| Phase 1 | Quick Wins & Parameter Tuning | ✅ Complete | +12% qualite signal |
| Phase 2 | Risk Management Enhancements | ✅ Complete | -47% drawdown |
| Phase 3 | New Strategy Development | ✅ Complete | +15% win rate |
| Phase 4 | Advanced Features | ✅ Complete | +62% Sharpe |

### Metriques Atteintes

| Metrique | Avant | Apres Phase 4 | Cible Finale |
|----------|-------|---------------|--------------|
| Sharpe Ratio | 0.8 | 1.3+ | 1.5+ |
| Max Drawdown | 15% | 8% | 5% |
| Win Rate | 52% | 60% | 65% |
| Calmar Ratio | 0.8 | 1.8 | 2.5+ |

---

## Phase 5: Infrastructure & Risk Avance (Priorite HAUTE)

### 5.1 Kalman Filter pour Hedge Ratios Dynamiques
**Source:** `statistical_arbitrage.md`, `STRATEGY_ROADMAP.md`
**Statut:** ✅ COMPLETE (2026-02-04)
**Effort:** 8h | **Impact:** Haut

```python
# Fichier: core/kalman_filter.py (NOUVEAU)
class KalmanHedgeRatio:
    """Dynamic hedge ratio estimation using Kalman filter."""
    def __init__(self, delta: float = 1e-4, ve: float = 1e-3)
    def update(self, x: float, y: float) -> float
    def get_hedge_ratio(self) -> float
```

**Deliverables:**
- [x] `core/kalman_filter.py` - Implementation Kalman filter ✅
- [x] Integration dans `strategies/stat_arb_strategy.py` ✅
- [x] Tests: `tests/test_kalman_filter.py` ✅

---

### 5.2 Transaction Costs Integration
**Source:** `STRATEGY_ROADMAP.md`, `portfolio_optimization.md`
**Statut:** ✅ COMPLETE (2026-02-04)
**Effort:** 6h | **Impact:** Moyen

**Deliverables:**
- [x] `core/position_sizing.py` - TransactionCostConfig, optimize_portfolio_turnover_penalized() ✅
- [x] `tests/test_transaction_costs.py` - 12 tests ✅

---

### 5.3 Cornish-Fisher VaR Adjustment
**Source:** `RISK_ENHANCEMENTS.md`, `risk_adjusted_returns.md`
**Statut:** ✅ COMPLETE (2026-02-04)
**Effort:** 4h | **Impact:** Moyen

**Deliverables:**
- [x] `core/var_calculator.py` - calculate_cornish_fisher_var(), _cornish_fisher_quantile() ✅
- [x] `tests/test_cornish_fisher_var.py` - 14 tests ✅
- [ ] Dashboard: affichage CF-VaR vs VaR standard (TODO Phase 8)

---

### 5.4 Momentum Crash Protection System
**Source:** `RISK_ENHANCEMENTS.md`, `momentum_strategies.md`
**Statut:** ✅ COMPLETE (2026-02-04) - Enhanced with velocity tracking
**Effort:** 10h | **Impact:** Haut

```python
# Fichier: core/crash_protection.py (NOUVEAU)
@dataclass
class CrashWarning:
    level: str  # "low", "medium", "high", "critical"
    probability: float
    indicators: list[str]
    recommended_action: str
    leverage_multiplier: float

class MomentumCrashProtection:
    def evaluate_crash_risk(self, ...) -> CrashWarning
```

**Indicateurs:**
- VIX spike (>1.5x MA)
- Correlation spike (+20%)
- Drawdown velocity (>3%/jour)
- Winner/Loser reversal

**Deliverables:**
- [x] `core/crash_protection.py` - Detection crash ✅ (Enhanced with velocity tracking)
- [ ] Integration `agents/risk_agent.py`
- [ ] Dashboard: alertes crash
- [x] Tests: scenarios 2008, 2020 ✅ (8 historical scenarios in stress_tester.py)

---

### 5.5 Velocity-Aware Drawdown Response
**Source:** `RISK_ENHANCEMENTS.md`
**Statut:** TODO
**Effort:** 4h | **Impact:** Moyen

**Logique:**
- Drawdown GRADUAL (<1%/jour): Response standard
- Drawdown MODERATE (1-3%/jour): +20% agressivite
- Drawdown FAST (3-5%/jour): +50% agressivite
- Drawdown CRASH (>5%/jour): Close 20% positions immediatement

---

## Phase 6: Nouvelles Strategies (Priorite MOYENNE)

### 6.1 Session-Based Strategy
**Source:** `NEW_STRATEGIES.md`, `trading_hours_synthesis.md`
**Statut:** ✅ COMPLETE (2026-02-04)
**Effort:** 12h | **Impact:** Haut

**Deliverables:**
- [x] `strategies/session_strategy.py` - Full implementation ✅
  - SessionStrategy class with opening range breakout
  - Session window definitions (Asian, London, NY, Overlap)
  - Volume confirmation, momentum signals
- [x] `tests/test_session_strategy.py` - 18 tests ✅
- [ ] `agents/session_agent.py` (TODO)
- [ ] Integration dans main.py (TODO)

---

### 6.2 MES/MNQ Index Spread Strategy
**Source:** `NEW_STRATEGIES.md`, `micro_emini.md`
**Statut:** TODO
**Effort:** 10h | **Impact:** Moyen

**Logique:**
- Ratio trading MES:MNQ (correlation ~0.85)
- Entry: Z-score > 2.0
- Exit: Z-score < 0.5
- Dollar-neutral: hedge ratio dynamique

---

### 6.3 TTM Squeeze Volatility Strategy
**Source:** `NEW_STRATEGIES.md`, `volatility_bands.md`
**Statut:** TODO
**Effort:** 8h | **Impact:** Moyen

**Indicateurs:**
- Bollinger Bands inside Keltner Channels = squeeze
- Breakout direction = signal
- Momentum histogram pour timing

---

### 6.4 Event-Driven Strategy (FOMC/NFP)
**Source:** `NEW_STRATEGIES.md`, `economic_calendar.md`
**Statut:** TODO
**Effort:** 10h | **Impact:** Moyen

**Events cibles:**
- FOMC: Position pre/post annonce
- NFP: Momentum post-release
- CPI: Volatility plays

---

### 6.5 Mean Reversion Single-Asset
**Source:** `NEW_STRATEGIES.md`, `mean_reversion.md`
**Statut:** TODO
**Effort:** 6h | **Impact:** Moyen

**Logique:**
- Z-score > 2 sur Bollinger Bands
- RSI extreme confirmation
- Reversion vers mean 20 periodes

---

## Phase 7: Execution Optimizations (Priorite MOYENNE)

### 7.1 Adaptive TWAP avec Volatility Adjustment
**Source:** `EXECUTION_OPTIMIZATIONS.md`
**Statut:** TODO
**Effort:** 6h | **Impact:** Moyen

```python
# agents/execution_agent.py amelioration
async def _execute_adaptive_twap(self, pending: PendingOrder) -> None:
    """
    TWAP avec intervals ajustes selon volatilite.
    High vol = intervals plus longs (evite adverse selection)
    Low vol = intervals plus courts (meilleur fill)
    """
```

---

### 7.2 Dynamic Slippage Caps
**Source:** `EXECUTION_OPTIMIZATIONS.md`
**Statut:** TODO
**Effort:** 4h | **Impact:** Moyen

**Formule:**
```
slippage_cap = base_cap * spread_factor * volatility_factor * liquidity_factor
```

---

### 7.3 Session-Aware Execution Rules
**Source:** `EXECUTION_OPTIMIZATIONS.md`
**Statut:** TODO
**Effort:** 6h | **Impact:** Moyen

| Session | Algo Prefere | Max Participation | Min Interval |
|---------|--------------|-------------------|--------------|
| Opening Auction | TWAP | 5% | 30s |
| Morning | ADAPTIVE_TWAP | 15% | 15s |
| Midday Lull | PASSIVE | 10% | 60s |
| Afternoon | VWAP | 15% | 20s |
| Closing | TWAP (fast) | 25% | 10s |

---

### 7.4 Smart Algo Selection
**Source:** `EXECUTION_OPTIMIZATIONS.md`
**Statut:** TODO
**Effort:** 4h | **Impact:** Moyen

**Logique automatique:**
- Small order + high urgency = MARKET
- Large order = VWAP
- Alpha-driven = ADAPTIVE_TWAP
- EOD = TWAP accelere

---

### 7.5 Fill Quality Monitoring
**Source:** `EXECUTION_OPTIMIZATIONS.md`
**Statut:** TODO
**Effort:** 8h | **Impact:** Moyen

**Composants:**
- Real-time slippage tracking
- Execution benchmarks historiques
- Anomaly detection alerts
- Slippage attribution (spread/impact/timing)

---

## Phase 8: Dashboard Upgrades (Priorite BASSE)

### 8.1 Rolling Sharpe/Sortino Display
**Source:** `DASHBOARD_UPGRADES.md`
**Statut:** TODO
**Effort:** 4h | **Impact:** Moyen

**Periodes:** 1D, 1W, 1M, 3M, YTD

---

### 8.2 Win Rate by Session Panel
**Source:** `DASHBOARD_UPGRADES.md`
**Statut:** TODO
**Effort:** 4h | **Impact:** Moyen

**Sessions:** Asian, European, US

---

### 8.3 Strategy Performance Comparison
**Source:** `DASHBOARD_UPGRADES.md`
**Statut:** TODO
**Effort:** 8h | **Impact:** Haut

**Metriques par strategie:**
- Total signals / Signals acted on
- Hit rate
- P&L contribution
- Average conviction

---

### 8.4 Risk Visualization Heatmap
**Source:** `DASHBOARD_UPGRADES.md`
**Statut:** TODO
**Effort:** 6h | **Impact:** Moyen

**Elements:**
- Position risk scores (0-100)
- VaR contribution par position
- Correlation to portfolio
- Stress test scenarios

---

### 8.5 Trade Journal Integration
**Source:** `DASHBOARD_UPGRADES.md`
**Statut:** TODO
**Effort:** 10h | **Impact:** Moyen

**Fonctionnalites:**
- Annotation des trades
- Tags et categories
- Lesson learned tracking
- Trade quality rating (1-5)
- Emotional state tracking

---

### 8.6 Signal Heatmap Ameliore
**Source:** `DASHBOARD_UPGRADES.md`
**Statut:** TODO
**Effort:** 6h | **Impact:** Moyen

**Elements:**
- Market bias indicator (-1 to +1)
- Agent disagreement alerts
- Consensus strength visualization

---

## Phase 9: Agent Improvements (Priorite MOYENNE)

### 9.1 MomentumAgent Enhancements
**Source:** `AGENT_IMPROVEMENTS.md`
**Statut:** PARTIEL (52-week, MTF deja faits)
**Effort:** 6h | **Impact:** Moyen

**Restant:**
- [ ] Volume confirmation filter
- [ ] Crash protection filter integration
- [ ] Cross-sectional ranking integration

---

### 9.2 StatArbAgent Enhancements
**Source:** `AGENT_IMPROVEMENTS.md`
**Statut:** PARTIEL (Johansen fait)
**Effort:** 8h | **Impact:** Haut

**Restant:**
- [ ] Kalman filter integration (attend Phase 5.1)
- [ ] Transaction cost filter
- [ ] Crowding risk detection
- [ ] Hurst exponent regime detection

---

### 9.3 MacroAgent Enhancements
**Source:** `AGENT_IMPROVEMENTS.md`
**Statut:** TODO
**Effort:** 10h | **Impact:** Moyen

**Ajouts proposes:**
- Economic calendar integration
- Fed funds futures analysis
- Yield curve steepness tracking
- DXY correlation analysis

---

### 9.4 MarketMakingAgent Enhancements
**Source:** `AGENT_IMPROVEMENTS.md`
**Statut:** TODO
**Effort:** 8h | **Impact:** Moyen

**Ajouts proposes:**
- Avellaneda-Stoikov optimal spread
- Inventory skew dynamique
- Session-based spread adjustment

---

## Phase 10: Testing & QA (Priorite HAUTE)

### 10.1 Stress Tests Historiques
**Source:** `TESTING_PLAN.md`, `testing_qa_synthesis.md`
**Statut:** ✅ COMPLETE (2026-02-04)
**Effort:** 12h | **Impact:** Haut

**Scenarios (8 implemented in `core/stress_tester.py`):**
- [x] 2008 Financial Crisis (HIST-001)
- [x] 2010 Flash Crash (HIST-002)
- [x] 2020 COVID Crash (HIST-003)
- [x] 2022 Rate Hike Cycle (HIST-004)
- [x] 2015 China Crash (HIST-005)
- [x] Momentum Crash (HIST-006)
- [x] Black Monday 1987 (HIST-007)
- [x] SVB Banking Crisis 2023 (HIST-008)

---

### 10.2 Walk-Forward Validation
**Source:** `TESTING_PLAN.md`, `walk_forward.md`
**Statut:** TODO
**Effort:** 16h | **Impact:** Haut

**Methodology:**
- Train: 70% | Test: 30%
- Rolling 3-month windows
- Out-of-sample Sharpe validation

---

### 10.3 Signal Agent Unit Tests
**Source:** `TESTING_PLAN.md`
**Statut:** TODO
**Effort:** 20h | **Impact:** Haut

**Coverage cible:**
- MacroAgent: 20 tests
- StatArbAgent: 25 tests
- MomentumAgent: 20 tests
- MarketMakingAgent: 18 tests
- OptionsVolAgent: 25 tests
- SentimentAgent: 20 tests
- ChartAnalysisAgent: 15 tests
- ForecastingAgent: 15 tests

---

### 10.4 Integration Tests IB
**Source:** `TESTING_PLAN.md`
**Statut:** TODO
**Effort:** 12h | **Impact:** Moyen

**Tests:**
- Connection/reconnection
- Rate limiting
- Order submission/fill
- Position sync
- Market data streaming

---

## Matrice de Priorites

### Priorite CRITIQUE (Semaine 1-2)
| Task | Phase | Effort | Impact | Status |
|------|-------|--------|--------|--------|
| Kalman Filter | 5.1 | 8h | Haut | ✅ COMPLETE |
| Crash Protection | 5.4 | 10h | Haut | ✅ COMPLETE |
| Stress Tests | 10.1 | 12h | Haut | ✅ COMPLETE |

### Priorite HAUTE (Semaine 3-4)
| Task | Phase | Effort | Impact |
|------|-------|--------|--------|
| Transaction Costs | 5.2 | 6h | Moyen |
| Velocity Drawdown | 5.5 | 4h | Moyen |
| Walk-Forward | 10.2 | 16h | Haut |
| Unit Tests Agents | 10.3 | 20h | Haut |

### Priorite MOYENNE (Mois 2)
| Task | Phase | Effort | Impact |
|------|-------|--------|--------|
| Session Strategy | 6.1 | 12h | Haut |
| MES/MNQ Spread | 6.2 | 10h | Moyen |
| Strategy Comparison | 8.3 | 8h | Haut |
| StatArb Enhancements | 9.2 | 8h | Haut |

### Priorite BASSE (Mois 3)
| Task | Phase | Effort | Impact |
|------|-------|--------|--------|
| TTM Squeeze | 6.3 | 8h | Moyen |
| Event-Driven | 6.4 | 10h | Moyen |
| Trade Journal | 8.5 | 10h | Moyen |
| Dashboard Panels | 8.1-8.6 | 30h | Moyen |

---

## Estimation Totale

| Phase | Heures | Priorite |
|-------|--------|----------|
| Phase 5: Infrastructure | 32h | HAUTE |
| Phase 6: Nouvelles Strategies | 46h | MOYENNE |
| Phase 7: Execution | 28h | MOYENNE |
| Phase 8: Dashboard | 38h | BASSE |
| Phase 9: Agent Improvements | 32h | MOYENNE |
| Phase 10: Testing | 60h | HAUTE |
| **TOTAL** | **236h** | - |

---

## Timeline Recommandee

```
Semaine 1-2:  Phase 5 (Critical) + Phase 10.1
Semaine 3-4:  Phase 10.2 + Phase 10.3
Mois 2:       Phase 6 (Top 3) + Phase 9
Mois 3:       Phase 7 + Phase 8
Ongoing:      Phase 10.4 (IB tests en parallele)
```

---

## Metriques Cibles Finales

| Metrique | Actuel | Phase 6 | Phase 10 |
|----------|--------|---------|----------|
| Sharpe Ratio | 1.3 | 1.5 | 1.8+ |
| Max Drawdown | 8% | 6% | 5% |
| Win Rate | 60% | 62% | 65% |
| Calmar Ratio | 1.8 | 2.2 | 2.5+ |
| Test Coverage | ~40% | 60% | 85% |
| Profit Factor | 1.8 | 2.0 | 2.5+ |

---

## References Documents

### Syntheses Principales
- `MASTER_SYNTHESIS.md` - Resume complet recherche
- `STRATEGY_ROADMAP.md` - Plan initial 4 phases
- `IMPLEMENTATION_TRACKER.md` - Suivi phases 1-4

### Par Domaine
- `AGENT_IMPROVEMENTS.md` - Ameliorations agents
- `RISK_ENHANCEMENTS.md` - Ameliorations risque
- `EXECUTION_OPTIMIZATIONS.md` - Optimisations execution
- `DASHBOARD_UPGRADES.md` - Ameliorations dashboard
- `NEW_STRATEGIES.md` - Nouvelles strategies
- `TESTING_PLAN.md` - Plan de tests

### Recherche Detaillee (67 fichiers)
- Voir `docs/research/` pour details complets

---

*Document genere: 2026-02-04*
*Base: 67 fichiers de recherche + 4 phases completees*
*Prochaine mise a jour: Apres Phase 5*
