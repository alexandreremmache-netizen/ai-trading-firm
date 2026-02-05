# AI Trading Firm - Constitution & Technical Reference

## Vision

Le systeme emule une societe de trading de type hedge fund inspiree des fonds multi-strategies du monde reel.
Ce n'est **PAS** un jouet, **PAS** un chatbot, et **PAS** un agent autonome unique.

---

## Principes Fondamentaux

### Architecture Multi-Agents

- Chaque agent a une responsabilite unique et bien definie
- Pas d'agents omniscients ou generalistes
- Separation claire : Generation de signaux → Prise de decision → Validation risque/conformite → Execution
- **Une et une seule autorite de decision : l'agent CIO**

### Modele d'Execution

- Pilote par evenements (evenements de marche, ticks programmes, declencheurs d'actualites)
- **Pas de boucles infinies**
- **Pas de polling continu**
- Comportement deterministe et reproductible

### Regles de Concurrence

- Les agents de signaux s'executent en **parallele** (fan-out)
- La decision du CIO uniquement **apres barriere de synchronisation** (fan-in)
- Risk, Compliance, Execution s'executent **sequentiellement**
- Les timeouts et la tolerance aux pannes sont **obligatoires**

---

## Architecture du Systeme

```
┌─────────────────────────────────────────────────────────────┐
│                     MARKET DATA (IB)                        │
└─────────────────────────┬───────────────────────────────────┘
                          │ MarketDataEvent
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              SIGNAL AGENTS (parallel fan-out)               │
│  [Macro] [StatArb] [Momentum] [MarketMaking]                │
│  [Session] [IndexSpread] [TTMSqueeze] [EventDriven]         │
│  [MeanReversion] [MACDv] [Sentiment*] [ChartAnalysis*]      │
│  [Forecast*] (* = LLM agents, disabled by default)          │
└─────────────────────────┬───────────────────────────────────┘
                          │ SignalEvent (barrier sync)
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    CIO AGENT (single)                       │
│              THE decision-making authority                  │
└─────────────────────────┬───────────────────────────────────┘
                          │ DecisionEvent
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    RISK AGENT                               │
│   Kill-switch, VaR, Crash Protection, position limits       │
└─────────────────────────┬───────────────────────────────────┘
                          │ ValidatedDecisionEvent
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                 COMPLIANCE AGENT (EU/AMF)                   │
│        Blackout, MNPI, restricted instruments              │
└─────────────────────────┬───────────────────────────────────┘
                          │ ValidatedDecisionEvent
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                  EXECUTION AGENT                            │
│    Adaptive TWAP/VWAP, Smart Algo Selection, Fill Quality   │
└─────────────────────────┬───────────────────────────────────┘
                          │ OrderEvent
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                  INTERACTIVE BROKERS                        │
└─────────────────────────────────────────────────────────────┘
```

---

## Architecture Invariants (CRITICAL)

**These constraints MUST NEVER be violated:**

### 1. DecisionMode vs EventBus Decoupling
- `DecisionMode` (NORMAL/DEFENSIVE/EMERGENCY/HUMAN_CIO) affects CIO's trading behavior
- `DecisionMode` does NOT modify the EventBus quorum mechanism
- Barrier release is governed ONLY by `quorum_config`, not by decision mode

### 2. Critical Agent Invariant
- Barrier MUST NOT release if ANY CRITICAL agent is missing, regardless of quorum percentage
- MacroAgent and RiskAgent are CRITICAL by default
- Use `SignalBarrier.is_valid()` to check before making decisions

### 3. Capital Allocation Governor Caller
- ONLY `CIOAgent` calls `CapitalAllocationGovernor`
- RiskAgent and PositionSizer query but do NOT modify allocations
- Governor is read-only for all other components

### 4. Position State Source of Truth
- **Interactive Brokers is the authoritative source of truth for positions**
- Internal tracking is a cache for performance
- `ReconciliationAgent` detects and reports discrepancies
- Never auto-correct without explicit human approval

### 5. Audit Ledger Timestamps
- MiFID II requires UTC timestamps with millisecond precision
- All `ImmutableAuditLedger` entries use `datetime.now(timezone.utc).isoformat()`
- NEVER use local time or naive datetime objects

### 6. Risk Counter Persistence
- Weekly loss counters and rolling drawdown are persisted via `StatePersistence`
- On restart, load counters from persisted state before processing
- HARD stops (weekly, rolling) require manual override - no auto-reset

---

## Structure du Projet

```
ai-trading-firm/
├── main.py                 # Orchestrateur principal
├── config.yaml             # Configuration complete
├── config.simple.yaml      # Configuration simplifiee
│
├── agents/                 # Agents de trading (16 agents)
│   ├── cio_agent.py        # Chief Investment Officer
│   ├── risk_agent.py       # Risk + Crash Protection
│   ├── compliance_agent.py # EU/AMF compliance
│   ├── execution_agent.py  # Execution TWAP/VWAP
│   ├── macro_agent.py      # Macro + HMM regime
│   ├── stat_arb_agent.py   # Pairs trading
│   ├── momentum_agent.py   # Momentum/tendance
│   ├── market_making_agent.py # Avellaneda-Stoikov MM
│   ├── session_agent.py    # NEW: Session-based trading
│   ├── index_spread_agent.py  # NEW: MES/MNQ spreads
│   ├── ttm_squeeze_agent.py   # NEW: TTM Squeeze
│   ├── event_driven_agent.py  # NEW: FOMC/NFP events
│   ├── mean_reversion_agent.py # NEW: Mean reversion
│   ├── macdv_agent.py      # NEW: MACD-v (Charles H. Dow Award 2022)
│   ├── sentiment_agent.py  # LLM sentiment (disabled by default)
│   ├── chart_analysis_agent.py # Claude Vision (disabled)
│   └── forecasting_agent.py   # LLM forecasting (disabled)
│
├── strategies/             # Logique de strategies (12 strategies)
│   ├── stat_arb_strategy.py    # Pairs + Johansen cointegration
│   ├── momentum_strategy.py    # MA, RSI, MACD, Dual Momentum, MTF
│   ├── macro_strategy.py       # Indicateurs macro
│   ├── market_making_strategy.py # Avellaneda-Stoikov
│   ├── session_strategy.py     # NEW: Opening Range Breakout
│   ├── index_spread_strategy.py # NEW: MES/MNQ spread trading
│   ├── ttm_squeeze_strategy.py  # NEW: BB inside KC
│   ├── event_driven_strategy.py # NEW: Economic events
│   ├── mean_reversion_strategy.py # NEW: RSI/BB/Z-score
│   ├── macdv_strategy.py       # NEW: MACD-v (Charles H. Dow Award 2022)
│   ├── ichimoku_strategy.py    # NEW: Ichimoku Cloud
│   └── seasonality.py          # Patterns saisonniers
│
├── core/                   # Infrastructure core (~75 modules)
│   ├── event_bus.py        # Bus d'evenements + quorum barrier sync
│   ├── broker.py           # Interactive Brokers + rate limiter
│   ├── var_calculator.py   # VaR + Cornish-Fisher + Regime-conditional
│   ├── crash_protection.py # Enhanced Crash Protection (velocity-aware)
│   ├── position_sizing.py  # Kelly, HRP, Resampled Efficiency
│   ├── black_litterman.py  # Black-Litterman optimization
│   ├── kalman_filter.py    # Dynamic hedge ratios
│   ├── hmm_regime.py       # Hidden Markov Model regime
│   ├── yield_curve.py      # Yield curve analysis
│   ├── dxy_analyzer.py     # Dollar index analysis
│   ├── volume_indicators.py # VWMA, VWAP, OBV, Volume Profile
│   ├── execution_optimizer.py # Adaptive TWAP, Smart Algo
│   ├── walk_forward.py     # Walk-forward validation
│   ├── pair_screener.py    # Automated pair discovery
│   ├── session_risk_manager.py # Session-based risk
│   ├── signal_quality.py   # Signal Quality Scorer (6 dimensions)
│   ├── historical_warmup.py # IB historical data warmup
│   ├── economic_calendar.py # FOMC/NFP/CPI event calendar
│   ├── async_signal_cache.py # NEW: Out-of-band agent signal cache
│   ├── state_persistence.py  # NEW: CIO state persistence (hot restart)
│   ├── ib_failure_simulator.py # NEW: IB failure scenario testing
│   ├── capital_allocation_governor.py # NEW: Regime-based capital allocation
│   ├── immutable_ledger.py # NEW: Hash-chained audit ledger (MiFID II)
│   └── ...
│
├── dashboard/              # Dashboard temps reel
│   ├── server.py           # FastAPI + WebSocket (port 8081)
│   ├── templates/index.html # UI avec 6 panels analytics
│   └── components/
│       └── advanced_analytics.py # Rolling Sharpe, Session Perf, etc.
│
├── tests/                  # Tests (1042 tests)
│   ├── test_cio_agent.py        # 72 tests
│   ├── test_risk_agent.py       # 68 tests
│   ├── test_walk_forward.py     # 58 tests
│   ├── test_integration_full.py # 39 tests
│   └── ...
│
└── docs/                   # Documentation
    └── IMPLEMENTATION_TRACKER.md
```

---

## Agents

### Signal Agents (Parallele - Fan-out)

| Agent | Responsabilite | Maturite | API Tokens |
|-------|---------------|----------|------------|
| `MacroAgent` | Macro, HMM regime detection | BETA | Non |
| `StatArbAgent` | Pairs trading, Johansen cointegration | BETA | Non |
| `MomentumAgent` | MA, RSI, MACD, Dual Momentum, MTF | BETA | Non |
| `MarketMakingAgent` | Avellaneda-Stoikov optimal MM | BETA | Non |
| `SessionAgent` | Opening Range Breakout, session momentum | BETA | Non |
| `IndexSpreadAgent` | MES/MNQ spread trading, z-score | BETA | Non |
| `TTMSqueezeAgent` | BB inside KC, squeeze release | BETA | Non |
| `EventDrivenAgent` | FOMC/NFP/CPI event trading | BETA | Non |
| `MeanReversionAgent` | RSI extremes, BB touches, z-score | BETA | Non |
| `MACDvAgent` | MACD-v volatility-normalized (Dow Award 2022) | BETA | Non |
| `SentimentAgent` | LLM news sentiment | BETA | **Oui** (desactive) |
| `ChartAnalysisAgent` | Claude Vision patterns | BETA | **Oui** (desactive) |
| `ForecastingAgent` | LLM price forecasting | BETA | **Oui** (desactive) |

### Decision Agent

| Agent | Responsabilite |
|-------|---------------|
| `CIOAgent` | **SEULE autorite de decision** - Agregation, Kelly, VIX regime weights, Decision Modes |

**Decision Modes (NEW - Phase 10):**
- `NORMAL`: Full strategy suite, normal position sizing
- `DEFENSIVE`: 50% position cap, conservative strategies only
- `EMERGENCY`: 25% position cap, core strategies only (Macro, Momentum, StatArb)
- `HUMAN_CIO`: Queue all decisions for human approval with configurable timeout

### Supporting Agents

| Agent | Responsabilite |
|-------|---------------|
| `ReconciliationAgent` | Position reconciliation between internal state and IB (NEW) |

### Validation & Execution Agents

| Agent | Responsabilite |
|-------|---------------|
| `RiskAgent` | VaR, Crash Protection, kill-switch, position limits, Weekly/Rolling drawdown stops (NEW) |
| `ComplianceAgent` | EU/AMF, blackout, MNPI, LEI validation |
| `ExecutionAgent` | Adaptive TWAP/VWAP, Smart Algo Selection, Zombie order management (NEW) |

---

## Fonctionnalites Implementees (Phases 1-8)

### Phase 1-4: Core Enhancements
- Oscillator parameters par asset class
- Stop-loss ATR-based
- VIX contrarian signals
- Regime-conditional VaR
- HRP (Hierarchical Risk Parity)
- 52-week High/Low momentum
- Dual Momentum (Antonacci)
- Johansen cointegration test
- Black-Litterman optimization
- Multi-timeframe analysis
- Resampled Efficiency (Michaud)

### Phase 5: Infrastructure
- Kalman filter pour hedge ratios dynamiques
- Enhanced Crash Protection (velocity-aware)
- 8 scenarios de stress historiques (2008, COVID, etc.)
- Transaction cost integration
- Cornish-Fisher VaR adjustment

### Phase 6: Nouvelles Strategies
- Session-based strategy (Opening Range Breakout)
- Index Spread (MES/MNQ pairs)
- TTM Squeeze (volatility breakout)
- Event-Driven (FOMC, NFP, CPI)
- Mean Reversion (RSI, BB, Z-score)

### Phase 7: Execution Optimizations
- Adaptive TWAP avec ajustement volatilite
- Dynamic Slippage Caps
- Session-Aware Execution Rules
- Smart Algo Selection
- Fill Quality Monitoring

### Phase 8: Dashboard Upgrades
- Rolling Sharpe/Sortino display
- Correlation heatmap temps reel
- Win rate by session panel
- Strategy comparison view
- Signal consensus panel
- Risk heatmap
- **Agent toggle on/off depuis le dashboard**

### Phase 9: Win Rate Optimization
- **Signal Quality Scoring** - Filtre les signaux de mauvaise qualite (6 dimensions, 0-100)
- **R-Multiple Tracking** - Gestion de position basee sur le ratio risque/reward
- **Memory Leak Fixes** - Correction de 20+ listes non bornees dans 8 agents
- Historical Warmup - Chargement de 100 barres historiques au demarrage
- Economic Calendar - 61 evenements economiques pre-charges (FOMC, NFP, CPI, etc.)

### Phase 10: Architecture Improvements (NEW - Feb 2026)
- **Quorum-Based Barrier** - 80% agent quorum with 100ms fast path, agent criticality levels
- **Per-Agent Timeouts** - Individual timeouts with fallback signals and confidence decay
- **Async Signal Cache** - Out-of-band processing for heavy agents (HMM, LLM)
- **CIO State Persistence** - Hot restart with atomic writes and backup rotation
- **Enhanced Correlation Filter** - Monthly rolling correlation, weight halving at >0.95
- **Reconciliation Agent** - Position reconciliation between theoretical and IB
- **Zombie Order Management** - 30s timeout with automatic cancellation
- **Local LLM Support** - Ollama and llama.cpp backends for free sentiment analysis
- **Weekly/Rolling Drawdown Stops** - HARD stops with manual override requirement
- **Decision Modes** - NORMAL, DEFENSIVE, EMERGENCY, HUMAN_CIO with position caps
- **Human-in-the-Loop Mode** - Queue decisions for human approval with timeout
- **Capital Allocation Governor** - Regime-based budgets and drawdown reduction
- **Immutable Audit Ledger** - Hash-chained logging for MiFID II compliance
- **IB Failure Simulator** - Test scenarios for zombie orders, missing fills, disconnects
- **Regime/Session Attribution** - Track entry/exit regime and session for closed positions

### Infrastructure Avancee
- HMM regime detection (Hidden Markov Model)
- Yield curve analysis (2s10s, recession probability)
- DXY analyzer (Dollar index correlations)
- Volume indicators (VWMA, VWAP, OBV, Volume Profile)
- Ichimoku Cloud strategy
- Walk-forward validation framework
- Avellaneda-Stoikov market making
- **Signal Quality Scorer** (core/signal_quality.py)
- **Historical Warmup** (core/historical_warmup.py)
- **Economic Calendar** (core/economic_calendar.py)
- **Async Signal Cache** (core/async_signal_cache.py) - NEW
- **State Persistence** (core/state_persistence.py) - NEW
- **Capital Allocation Governor** (core/capital_allocation_governor.py) - NEW
- **Immutable Audit Ledger** (core/immutable_ledger.py) - NEW
- **IB Failure Simulator** (core/ib_failure_simulator.py) - NEW

---

## Configuration

### Parametres Critiques

```yaml
firm:
  mode: paper          # OBLIGATOIRE: "paper" ou "live"

broker:
  port: 4002           # Paper: 4002/7497, Live: 4001/7496

# Agents LLM desactives par defaut (economie de tokens)
agents:
  sentiment:
    enabled: false     # Uses Claude API tokens
  chart_analysis:
    enabled: false     # Uses Claude API tokens
  forecasting:
    enabled: false     # Uses Claude API tokens

# Historical Warmup (Phase 9)
warmup:
  enabled: true
  bars_to_fetch: 100
  bar_size: "1 min"

# Signal Quality Scoring (Phase 9)
cio:
  signal_quality:
    enabled: true
    min_total_score: 50.0   # Reject signals < 50/100
    min_volume_score: 5.0
    min_trend_score: 5.0

# Phase 10 New Features
risk_limits:
  max_weekly_loss_pct: 5.0           # Weekly HARD stop
  rolling_drawdown_days: 5           # 5-day rolling window
  max_rolling_drawdown_pct: 4.0      # Rolling HARD stop
  kill_switch_manual_override_required: true

event_bus:
  quorum_threshold: 0.8              # 80% of agents for release
  fast_path_timeout_ms: 100          # Quick consensus path
  agent_criticality:
    MacroAgent: critical             # Must respond
    MomentumAgent: high
    SentimentAgent: normal

reconciliation:
  enabled: true
  interval_seconds: 60               # Check every minute
  auto_correct: false                # Require manual approval

llm:
  backend: "ollama"                  # or "claude", "llama_cpp"
  ollama:
    url: "http://localhost:11434"
    model: "llama3:8b"
```

### Dashboard Agent Toggle

Les agents peuvent etre actives/desactives depuis le dashboard:
- Toggle switch on/off pour chaque agent
- Badge "LLM" pour les agents consommant des tokens
- Mise a jour en temps reel via WebSocket
- API: `POST /api/agent/toggle?agent_name=X&enabled=true`

---

## Tests

```bash
# Tous les tests (1042)
python -m pytest tests/ -v

# Tests rapides
python -m pytest tests/ -q

# Tests specifiques
python -m pytest tests/test_walk_forward.py -v
python -m pytest tests/test_integration_full.py -v
```

### Couverture Tests

| Module | Tests |
|--------|-------|
| CIO Agent | 72 |
| Risk Agent | 68 |
| Walk-forward Validation | 58 |
| Integration Tests | 39 |
| Dashboard Integration | 34 |
| Ichimoku Strategy | 36 |
| Volume Indicators | 41 |
| HMM Regime | 28 |
| Yield Curve | 35 |
| DXY Analyzer | 43 |
| Avellaneda-Stoikov | 38 |
| Session Strategy | 18 |
| Index Spread | 27 |
| TTM Squeeze | 21 |
| Event-Driven | 28 |
| Mean Reversion | 27 |
| Advanced Analytics | 29 |
| **TOTAL** | **1042** |

---

## Dashboard Temps Reel

- `http://localhost:8081` - Dashboard de monitoring

### Onglets
1. **Performance** - P&L, Sharpe, Drawdown, Rolling Metrics
2. **Analytics** - Session Performance, Strategy Comparison, Signal Consensus, Correlation Heatmap, Risk Heatmap
3. **Positions** - Open positions avec TP/SL
4. **Closed** - Historique positions fermees
5. **Signals** - Signaux temps reel

### Fonctionnalites
- Toggle on/off agents depuis le dashboard
- Badge LLM pour agents consommant des tokens
- Correlation heatmap temps reel
- Win rate par session (Asian, London, NY, Overlap)
- Signal consensus avec alerte disagreement
- Kill Switch dans la navbar

---

## Nouveaux Modules (Fevrier 2026)

### core/hmm_regime.py
```python
from core.hmm_regime import HMMRegimeDetector, create_hmm_detector
detector = create_hmm_detector(n_states=3)
detector.fit(returns)
regime = detector.predict_state(recent_returns)
```

### core/yield_curve.py
```python
from core.yield_curve import YieldCurveAnalyzer
analyzer = YieldCurveAnalyzer()
spread_2s10s = analyzer.calculate_2s10s_spread(y2, y10)
recession_prob = analyzer.get_recession_probability()
```

### core/dxy_analyzer.py
```python
from core.dxy_analyzer import DXYAnalyzer
analyzer = DXYAnalyzer()
analyzer.update(dxy_price)
signal = analyzer.get_signal_for_symbol("GC")  # Gold signal vs DXY
```

### strategies/ichimoku_strategy.py
```python
from strategies.ichimoku_strategy import IchimokuStrategy
strategy = IchimokuStrategy()
signal = strategy.generate_signal(symbol, highs, lows, closes)
```

### core/walk_forward.py
```python
from core.walk_forward import WalkForwardValidator, WalkForwardConfig
config = WalkForwardConfig(train_period_days=252, test_period_days=63)
validator = WalkForwardValidator(config, strategy_factory)
result = validator.run(price_data, start_date, end_date)
```

### core/signal_quality.py (NEW - Phase 9)
```python
from core.signal_quality import SignalQualityScorer, create_signal_quality_scorer

# Create scorer with custom thresholds
scorer = create_signal_quality_scorer({
    "min_total_score": 50.0,    # Reject signals below 50/100
    "min_volume_score": 5.0,    # Minimum volume confirmation
    "min_trend_score": 5.0,     # Minimum trend alignment
})

# Validate a signal
result = scorer.validate_signal(
    signal=signal_event,
    market_data={"prices": [...], "volumes": [...], "adx": 25.0},
    support_levels=[100.0, 95.0],
    resistance_levels=[110.0, 115.0],
    other_signals=[...],  # For confluence check
)

if result.is_valid:
    print(f"Signal quality: {result.total_score}/100 ({result.tier.value})")
else:
    print(f"Rejected: {result.rejection_reasons}")
```

### core/historical_warmup.py (NEW - Phase 9)
```python
from core.historical_warmup import HistoricalWarmup, create_historical_warmup

# Create warmup handler
warmup = create_historical_warmup(broker, event_bus, {
    "bars_to_fetch": 100,
    "bar_size": "1 min",
    "max_concurrent": 5,
})

# Warmup symbols on startup
results = await warmup.warmup_symbols(["MES", "MNQ", "GC", "CL"])
print(f"Warmed up {sum(results.values())} bars")
```

### core/economic_calendar.py (NEW - Phase 9)
```python
from core.economic_calendar import EconomicCalendar, create_economic_calendar

# Create calendar with 61 pre-loaded events
calendar = create_economic_calendar()
await calendar.initialize()

# Get upcoming events
events = calendar.get_upcoming_events(hours=24)
for event in events:
    print(f"{event.event_type}: {event.timestamp}")
```

### core/capital_allocation_governor.py (NEW - Phase 10)
```python
from core.capital_allocation_governor import CapitalAllocationGovernor, create_default_governor

# Create governor with default allocations
governor = create_default_governor(total_capital=1_000_000.0)

# Update market regime - allocations auto-adjust
allocations = governor.update_regime(MarketRegime.RISK_OFF)

# Update drawdown - allocations reduce automatically
allocations = governor.update_drawdown(0.05)  # 5% drawdown

# Check if strategy can allocate capital
can_alloc, max_pct, reason = governor.can_allocate("MomentumAgent", 5.0)
```

### core/immutable_ledger.py (NEW - Phase 10)
```python
from core.immutable_ledger import ImmutableAuditLedger, create_audit_ledger

# Create hash-chained audit ledger
ledger = create_audit_ledger(storage_path="logs/audit_ledger")

# Append events (cryptographically linked)
entry = ledger.append(
    event_type="decision",
    source_agent="CIOAgent",
    event_data={"symbol": "MES", "action": "BUY", "quantity": 5}
)

# Verify chain integrity
is_valid, invalid_seqs = ledger.verify_chain()

# Export compliance report
report = ledger.export_compliance_report(start_date, end_date, "reports/compliance.json")
```

### core/ib_failure_simulator.py (NEW - Phase 10)
```python
from core.ib_failure_simulator import IBFailureSimulator, FailureScenario

# Create simulator
simulator = IBFailureSimulator(seed=42)

# Enable specific failure scenarios
simulator.enable_scenario(FailureScenario.ORDER_ACCEPTED_NO_FILL, probability=0.1)
simulator.enable_scenario(FailureScenario.FILL_WITHOUT_CALLBACK, probability=0.05)

# Simulate zombie order (order sits in "working" forever)
await simulator.simulate_order_accepted_no_fill(order_id, symbol, action, qty, price, on_status)

# Get statistics
stats = simulator.get_statistics()
```

### CIO Agent Decision Modes (NEW - Phase 10)
```python
from agents.cio_agent import CIOAgent, DecisionMode

# Set decision mode
cio.set_decision_mode(DecisionMode.DEFENSIVE, reason="high_volatility")

# Enable emergency mode (auto-triggered at 5% drawdown)
cio.set_decision_mode_override(DecisionMode.EMERGENCY, reason="manual_override")

# Enable human-in-the-loop mode
cio.enable_human_cio_mode(reason="critical_market_event")

# Get pending decisions awaiting approval
pending = cio.get_pending_human_decisions()

# Approve or reject
cio.approve_human_decision(decision_id, approved_by="trader1", modified_quantity=3)
cio.reject_human_decision(decision_id, rejected_by="trader1", reason="too_risky")
```

---

## R-Multiple Position Management (Phase 9)

Le CIO Agent utilise maintenant le R-Multiple pour gerer les positions:

```python
# TrackedPosition avec R-Multiple
position.initial_risk   # Distance entry -> stop-loss (1R)
position.r_multiple     # PnL / initial_risk

# Regles de gestion automatique:
# - Profit partiel a 2R
# - Exit complet a 3R si conviction baisse
# - Stats trackees: r_multiple_exits_2r, r_multiple_exits_3r
```

---

## Problemes Connus et Solutions

| Probleme | Cause | Solution |
|----------|-------|----------|
| Agents LLM consomment trop de tokens | Enabled by default | Desactives par defaut dans config.yaml |
| Dashboard tests failing | System Started alert | Tests mis a jour pour en tenir compte |
| VaR crash "shapes not aligned" | Arrays numpy vides | Validation shapes dans var_calculator.py |
| Memory leaks en production | Listes non bornees | Remplace par `deque(maxlen=N)` dans 8 agents |
| Signal barrier timeout 5 agents | Pas de signal pendant warmup | Ajout `_emit_warmup_heartbeat()` |
| EventDrivenAgent sans events | Pas de calendrier | Economic Calendar avec 61 events pre-charges |
| IB rate limit au demarrage | 44 symboles souscrits simultanément | Progressive subscription avec delai 1.5s |
| Deque slice error dashboard | `deque[-N:]` invalide en Python | Conversion `list(deque)[-N:]` dans risk_agent |
| Risk limits default 2.0x | Pas de `limits` dans get_status() | Ajout section `limits` dans RiskAgent.get_status() |
| Entry price = notional | IB retourne avg_cost × multiplier | Division par multiplier dans get_positions_async() |
| Micro futures manquants | MES/MNQ/MYM/M2K pas dans CONTRACT_SPECS | Ajout des 4 micro futures avec multiplicateurs |
| /api/status positions vide | Utilisait get_positions() sync | Changé pour get_positions_async() |

---

## Statut Production

**PRODUCTION-READY pour paper trading**

- 1127 tests passent (100% coverage sur modules critiques)
- 15 agents de signal
- 12 strategies
- Dashboard avec analytics avances
- Walk-forward validation framework
- **MiFID II Audit Trail ACTIF** - ImmutableAuditLedger integre dans main.py

Pour live trading:
- [ ] LEI valide et enregistre GLEIF
- [ ] Tests bout-en-bout avec IB connecte
- [ ] Validation compliance officer
- [ ] ANTHROPIC_API_KEY si agents LLM actives

---

## Expert Review V2 - Corrections (2026-02-05)

### Corrections Critiques Appliquees

| Domaine | Corrections |
|---------|-------------|
| **Dashboard/UX** | WebSocket error handling, agent toggle debounce, kill switch confirmation |
| **Compliance MiFID II** | Audit ledger integre dans main.py, flush on shutdown |
| **Risk Management** | Weekly loss dedup, VaR division/zero protection, risk limits dans get_status() |
| **Trading Systems** | Kill-switch TWAP/VWAP bypass fix, fill overflow race condition |
| **Python Architecture** | Input validation, deque slice bugs (5 fixes dans risk_agent.py) |
| **Infrastructure** | Progressive market data subscription (evite IB rate limit) |
| **Dashboard/Positions** | Entry price affiche prix reel (divise par multiplier), micro futures ajoutes |

### Audit Ledger Integration (CRITIQUE)

L'audit ledger est maintenant **ACTIF** dans le pipeline de production:

```python
# main.py - Integration complete
from core.immutable_ledger import create_audit_ledger

# Dans TradingFirmOrchestrator.initialize():
self._audit_ledger = create_audit_ledger(storage_path="logs/audit_ledger")
self._event_bus.set_audit_ledger(self._audit_ledger)

# Dans stop():
self._audit_ledger.flush_to_disk()  # Garantit persistence
```

Tous les evenements (decisions, orders, fills, kill switch) sont enregistres dans un ledger immutable avec chaine de hachage SHA-256.

### Progressive Market Data Subscription

Pour eviter le rate limit IB (60 requetes/10 min), les subscriptions de marche sont ajoutees progressivement:

```python
# data/market_data.py
subscription_delay_seconds: float = 1.5  # Delai entre chaque subscription

# Subscriptions en arriere-plan - systeme demarre immediatement
self._subscription_task = asyncio.create_task(self._progressive_subscribe())

# Monitoring
status = market_data_manager.get_subscription_status()
# {"subscribed_symbols": 37, "total_symbols": 44, "progress_pct": 84.1}
```

### Deque Slice Fixes (risk_agent.py)

Python `deque` ne supporte pas le slicing `[-N:]`. Corrections:
- `list(self._returns_history)[-252:]` au lieu de `self._returns_history[-252:]`
- Suppression des slicing inutiles quand deque a deja `maxlen`

---

*Document mis a jour: 2026-02-05*
*Total tests: 1127*
*Total lignes de code: ~32,000+*
*Phase 10: Architecture Improvements + Expert Review Fixes*
*Risk Score: 9.5/10 | Compliance Score: 85/100*
