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
│  [Macro] [StatArb] [Momentum] [MarketMaking] [OptionsVol]   │
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
│     Kill-switch, VaR, position/leverage limits             │
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
│      TWAP/VWAP algorithms - ONLY one sending orders        │
└─────────────────────────┬───────────────────────────────────┘
                          │ OrderEvent
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                  INTERACTIVE BROKERS                        │
└─────────────────────────────────────────────────────────────┘
```

---

## Structure du Projet

```
ai-trading-firm/
├── main.py                 # Orchestrateur principal (TradingFirmOrchestrator)
├── config.yaml             # Configuration complete (451 lignes)
├── config.simple.yaml      # Configuration simplifiee (50 lignes)
│
├── agents/                 # Agents de trading
│   ├── cio_agent.py        # Chief Investment Officer - SEULE autorite de decision
│   ├── risk_agent.py       # Validation des limites de risque + kill-switch
│   ├── compliance_agent.py # Conformite EU/AMF + LEI validation
│   ├── execution_agent.py  # Execution TWAP/VWAP - SEUL envoi d'ordres
│   ├── surveillance_agent.py         # MAR 2014/596/EU - detection abus
│   ├── transaction_reporting_agent.py # ESMA RTS 22/23
│   ├── macro_agent.py      # Signal agent - macro/sentiment
│   ├── stat_arb_agent.py   # Signal agent - arbitrage statistique
│   ├── momentum_agent.py   # Signal agent - momentum/tendance
│   ├── market_making_agent.py # Signal agent - market making
│   ├── options_vol_agent.py   # Signal agent - volatilite options
│   └── sentiment_agent.py  # Signal agent - LLM news sentiment (Claude/GPT)
│
├── strategies/             # Logique de strategies (utilisee par les agents)
│   ├── stat_arb_strategy.py    # Pairs trading, spreads commodities (BETA)
│   ├── momentum_strategy.py    # MA, RSI, MACD + stop-loss ATR (BETA)
│   ├── options_vol_strategy.py # Vol surface, Greeks (BETA)
│   ├── macro_strategy.py       # Indicateurs macro (ALPHA)
│   ├── market_making_strategy.py # Spreads, inventory (ALPHA)
│   └── seasonality.py          # Patterns saisonniers
│
├── core/                   # Infrastructure core
│   ├── event_bus.py        # Bus d'evenements central + barrier sync
│   ├── events.py           # Types d'evenements (Signal, Decision, Fill...)
│   ├── broker.py           # Integration Interactive Brokers + rate limiter
│   ├── agent_base.py       # Classes de base des agents
│   ├── agent_factory.py    # Factory SOLID pour creation agents
│   ├── dependency_injection.py # Container DI
│   ├── logger.py           # Audit logger (decisions, trades)
│   ├── llm_client.py       # Client async LLM (Anthropic/OpenAI)
│   ├── var_calculator.py   # VaR parametrique/historique/Monte Carlo
│   ├── stress_tester.py    # Scenarios de stress
│   ├── correlation_manager.py # Gestion correlations
│   ├── position_sizing.py  # Kelly criterion
│   ├── attribution.py      # TWR/MWR performance
│   ├── best_execution.py   # Analyse execution
│   ├── circuit_breaker.py  # Protection broker
│   ├── health_check.py     # Endpoints /health /ready /alive
│   ├── notifications.py    # Alertes compliance/risque
│   ├── config_validator.py # Validation config au demarrage
│   └── ...                 # ~50 autres modules
│
├── data/                   # Gestion donnees marche
│   └── market_data.py      # Manager donnees temps reel
│
├── tests/                  # Tests (299 tests)
│   ├── test_cio_agent.py        # 40 tests CIO
│   ├── test_risk_agent.py       # 35 tests Risk
│   ├── test_execution_agent.py  # 44 tests Execution
│   ├── test_compliance_agent.py # 51 tests Compliance
│   ├── test_refactoring.py      # 33 tests SOLID
│   └── ...
│
├── logs/                   # Logs et audit trail
│   ├── audit.jsonl         # Audit complet
│   ├── trades.jsonl        # Historique trades
│   ├── decisions.jsonl     # Historique decisions
│   └── system.log          # Logs systeme
│
├── scripts/
│   └── generate_docs.py    # Generateur documentation
│
└── docs/                   # Documentation generee
```

---

## Agents

### Signal Agents (Parallele - Fan-out)

| Agent | Responsabilite | Maturite |
|-------|---------------|----------|
| `MacroAgent` | Analyse macro, sentiment, indicateurs economiques | ALPHA |
| `StatArbAgent` | Pairs trading, spreads commodities, cointegration | BETA |
| `MomentumAgent` | MA crossovers, RSI, MACD, tendances | BETA |
| `MarketMakingAgent` | Spreads, gestion inventory | ALPHA |
| `OptionsVolAgent` | Surface de volatilite, Greeks, term structure | BETA |
| `SentimentAgent` | Analyse sentiment news via LLM (Claude/GPT) | BETA |

### Decision Agent

| Agent | Responsabilite |
|-------|---------------|
| `CIOAgent` | **SEULE autorite de decision** - Agregation signaux, Kelly sizing, poids dynamiques |

### Validation Agents (Sequentiel)

| Agent | Responsabilite |
|-------|---------------|
| `RiskAgent` | Limites position/secteur/leverage, VaR, drawdown, kill-switch |
| `ComplianceAgent` | EU/AMF, blackout, MNPI, LEI validation, SSR |

### Execution Agent

| Agent | Responsabilite |
|-------|---------------|
| `ExecutionAgent` | **SEUL** a envoyer des ordres, TWAP/VWAP, pre-trade checks |

### Compliance Agents Specialises

| Agent | Responsabilite |
|-------|---------------|
| `SurveillanceAgent` | MAR 2014/596/EU - wash trading, spoofing, layering |
| `TransactionReportingAgent` | ESMA RTS 22/23 - reporting transactions |

---

## Configuration

### Fichiers de Configuration

- `config.yaml` - Configuration complete avec toutes les options
- `config.simple.yaml` - Configuration simplifiee pour demarrage rapide

### Parametres Critiques

```yaml
firm:
  mode: paper          # OBLIGATOIRE: "paper" ou "live"

broker:
  port: 7497           # Paper: 7497/4002, Live: 7496/4001
  environment: paper   # DOIT correspondre au port

compliance:
  firm_lei: "..."      # OBLIGATOIRE pour MiFID II (20 caracteres ISO 17442)

risk:
  max_position_size_pct: 5.0    # Max 5% par position
  max_leverage: 2.0             # Max 2x leverage
  max_daily_loss_pct: 3.0       # Kill-switch a -3%
  max_drawdown_pct: 10.0        # Kill-switch a -10%
```

### Univers d'Instruments Supportes

| Type | Instruments | Exchange | Heures |
|------|-------------|----------|--------|
| **Equities** | AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA, JPM, V, JNJ | SMART | 09:30-16:00 ET |
| **ETFs** | SPY, QQQ, IWM, DIA, GLD, SLV, TLT, XLF, XLE, VXX | SMART | 09:30-16:00 ET |
| **E-mini Futures** | ES, NQ, YM, RTY, CL, GC, SI | CME/COMEX/NYMEX | ~23h/jour |
| **Micro Futures** | MES, MNQ, MYM, M2K, MCL, MGC, SIL | CME/COMEX/NYMEX | ~23h/jour |
| **Forex** | EUR/USD, GBP/USD, USD/JPY, USD/CHF, AUD/USD, USD/CAD | IDEALPRO | 24h/5j |

**Detection automatique du type de contrat** dans `broker.py`:
- Forex: `Forex('EURUSD', exchange='IDEALPRO')`
- Futures: `Future(symbol, exchange, lastTradeDateOrContractMonth)`
- Stocks/ETFs: `Stock(symbol, 'SMART', 'USD')`

---

## Contraintes

### Legales & Conformite

- Conception compatible **EU / AMF**
- **Pas de delit d'initie**
- **Pas de donnees illegales, divulguees ou privilegiees**
- Toutes les decisions journalisees avec: horodatage, sources, justification, agent responsable
- **LEI obligatoire** (validation ISO 17442 au demarrage)

### Courtier & Acces au Marche

- **Interactive Brokers (IB)** exclusivement pour: donnees marche, etat portefeuille, execution
- **Paper trading** est le mode par defaut
- Rate limiting IB: 60 req/10min, 15s entre requetes identiques

### Techniques

- Les agents sont **sans etat** autant que possible
- **Pas de strategies auto-modifiantes**
- **Pas de mise a l'echelle autonome du capital**
- **Pas de prise de decision cachee**
- Le systeme doit etre **testable, observable et auditable**

### Explicitement Hors Perimetre

- Trading Haute Frequence (HFT)
- Manipulation de marche
- Evolution des strategies sans validation humaine
- Logique de decision boite noire

---

## Lancement

### Prerequisites

```bash
pip install ib_insync numpy scipy pandas pyyaml aiohttp nest_asyncio
```

### Demarrage

```bash
# 1. Lancer IB Gateway ou TWS avec API active
# 2. Configurer config.yaml avec LEI valide
# 3. Lancer le systeme

python main.py
# ou avec config specifique
python main.py --config config.simple.yaml
```

### Endpoints Monitoring

- `http://localhost:8080/health` - Etat global
- `http://localhost:8080/ready` - Pret a trader
- `http://localhost:8080/alive` - Liveness check

---

## Tests

```bash
# Tous les tests (299)
python -m pytest tests/ -v

# Tests rapides
python -m pytest tests/ -q

# Tests specifiques
python -m pytest tests/test_cio_agent.py -v
python -m pytest tests/test_risk_agent.py -v
```

---

## Scores Qualite (Expert Review)

| Domaine | Score | Statut |
|---------|-------|--------|
| Architecture | 7.5/10 | Event-driven, separation claire |
| Quant/Math | 7.2/10 | VaR, Kelly, correlations |
| Risk Management | 7.5/10 | Kill-switch, tiered drawdown |
| CIO Decision | 6.5/10 | Signal aggregation fonctionnel |
| Compliance | 7.0/10 | LEI validation, EU/AMF |
| Error Handling | 4.0/10 | logger.exception() ajoute |
| Tests | 8.0/10 | 299 tests (vs 10% initial) |
| Infrastructure | 6.0/10 | Health checks, backup |

### Corrections Majeures Appliquees

- IB Rate Limiter (60 req/10min)
- Paper/Live protection explicite
- LEI validation obligatoire au demarrage
- logger.exception() sur tous les catch critiques
- Kelly sizing avec minimum 50 trades
- Pre-trade checks (spread, volume)
- Stop-loss ATR-based
- TWR/MWR performance attribution
- EventBus health check + cleanup handlers
- 203 nouveaux tests
- **Heures de marche par type**: Forex 24/5, Futures ~23h/jour, Equities 09:30-16:00
- **Micro contrats**: MES, MNQ, MYM, M2K, MCL, MGC, SIL ajoutes
- **Detection auto type contrat**: Forex/Future/Stock dans broker.py
- **Fix bug None*int**: Protection `price = ... or 100.0` dans compliance

---

## Conventions de Code

### Imports

```python
from __future__ import annotations
import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
```

### Logging

```python
logger = logging.getLogger(__name__)

# Pour les erreurs avec stack trace (OBLIGATOIRE dans catch)
logger.exception(f"Error description: {e}")

# Pour les infos
logger.info(f"Event: {details}")
```

### Audit (OBLIGATOIRE pour toute decision)

```python
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
```

### Timestamps

```python
# TOUJOURS UTC
from datetime import datetime, timezone
now = datetime.now(timezone.utc)
```

---

## Statut Production

**PRODUCTION-READY pour paper trading**

Pour live trading avec capital reel, verifier:
- [ ] LEI valide et enregistre GLEIF
- [ ] Tests bout-en-bout avec IB connecte
- [ ] Backup automatique configure
- [ ] Monitoring/alertes operationnels
- [ ] Validation compliance officer
