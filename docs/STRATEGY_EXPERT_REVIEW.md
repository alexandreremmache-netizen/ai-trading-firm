# Expert Review: Strategy & Pipeline Audit
**Date**: 2026-02-06
**Context**: Portfolio -45%. System trades micro futures (MES, MNQ, M2K, MYM, MCL, MGC) on 1-min bars, intraday only.
**Agents**: 6 expert reviewers ran in parallel across all strategies, agents, CIO and Risk pipeline.

---

## EXECUTIVE SUMMARY

| Category | CRITICAL | HIGH | MEDIUM | LOW | Total |
|----------|----------|------|--------|-----|-------|
| CIO & Risk Pipeline | 4 | 6 | 6 | 1 | 17 |
| StatArb & Index Spread | 3 | 8 | 6 | 1 | 18 |
| Market Making & TTM Squeeze | 4 | 5 | 6 | 2 | 17 |
| Event-Driven & Macro | 4 | 8 | 7 | 3 | 22 |
| Mean Reversion & MACD-v | 1 | 1 | 1 | 1 | 4 |
| Momentum & Session | 5 | 5 | 2 | 0 | 12 |
| **TOTAL** | **21** | **33** | **28** | **8** | **90** |

### Root Cause of 45% Loss (Causal Chain)
1. CIO position sizing ignores futures CONTRACT_SPECS multiplier -> 20x+ over-leverage
2. base_position_size=100 treated as contracts (not dollars) -> 60 MNQ = $2.28M on $100K
3. Minimum order = 10 contracts blocks proper 1-3 contract trades
4. Position limit 1.0% rejects ALL futures (1 MES = $26K notional > $1K limit)
5. Defensive mode blocks ALL orders including exits -> deadlock
6. _no_new_longs blocks BUY to close shorts -> can't deleverage

---

## PHASE 1: IMMEDIATE FIXES (Unblock deadlock + prevent recurrence)

### FIX-01: CIO Position Sizing - Add Futures Multiplier [CRITICAL]
**File**: `agents/cio_agent.py` lines 3153, 3218
**Problem**: `size = int(position_value / estimated_price)` ignores multiplier
**Fix**: Import CONTRACT_SPECS, divide by `price * multiplier`

### FIX-02: RiskAgent Defensive Mode - Allow Exit Orders [CRITICAL]
**File**: `agents/risk_agent.py` lines 703-710
**Problem**: Defensive mode blocks ALL orders, message says "only exits" but rejects everything
**Fix**: Check if order reduces existing position before rejecting

### FIX-03: RiskAgent _no_new_longs - Allow BUY to Close Shorts [CRITICAL]
**File**: `agents/risk_agent.py` lines 712-719
**Problem**: BUY to cover short rejected when _no_new_longs=True
**Fix**: Check current position quantity, allow if closing a short

### FIX-04: CIO Leverage Guard - Use Config, Not Hardcoded 1.5x [HIGH]
**File**: `agents/cio_agent.py` line 2066
**Problem**: Hardcoded `if lev > 1.5` ignores `max_leverage: 3.0` in config
**Fix**: Read from config parameters

### FIX-05: Config Position Limit 1% -> 5% [HIGH]
**File**: `config.yaml` lines 124-125
**Problem**: 1% = $1K max, but 1 MES = $26K notional. NO futures can pass.
**Fix**: `max_position_pct: 5.0`, `max_position_size_pct: 5.0`

### FIX-06: Minimum Order 10 -> 1 for Futures [HIGH]
**File**: `agents/cio_agent.py` lines 3170, 3244; `agents/risk_agent.py` line 900
**Problem**: Min 10 contracts = $380K MNQ notional. Proper 1-3 contracts blocked.
**Fix**: Min 1 for futures (check CONTRACT_SPECS.multiplier > 1)

### FIX-07: CIO Close Orders - Verify Against Broker [HIGH]
**File**: `agents/cio_agent.py` line 1569
**Problem**: Uses stale TrackedPosition quantity (100 when IB has 5)
**Fix**: Check broker portfolio before creating close decision

### FIX-08: CIO TrackedPosition Signed/Unsigned Fix [CRITICAL]
**File**: `agents/cio_agent.py` line 1136
**Problem**: Sync assigns raw IB signed qty (-5) but TrackedPosition expects unsigned
**Fix**: `tracked.quantity = abs(pos.quantity)`

### FIX-09: CIO pnl_dollar - Add Multiplier [MEDIUM]
**File**: `agents/cio_agent.py` lines 220-225
**Problem**: PnL calculation ignores futures multiplier, understates losses 2-5x
**Fix**: Add `contract_multiplier` attribute to TrackedPosition

### FIX-10: base_position_size=100 -> Dollar-Based [HIGH]
**File**: `config.yaml` line 880, `agents/cio_agent.py` line 3218
**Problem**: Conviction sizing: 100 * 0.75 * 0.80 = 60 contracts of MNQ = $2.28M
**Fix**: Interpret as dollar amount, divide by notional per contract

---

## PHASE 2: STRATEGY FIXES (Signal quality)

### FIX-11: Remove Stock/ETF Pairs from StatArb Config [CRITICAL]
**File**: `config.yaml` lines 577-591; `agents/stat_arb_agent.py` lines 146-151
**Problem**: AAPL:MSFT, GOOGL:META, SPY:QQQ pairs active in futures-only system
**Fix**: Keep only MES:MNQ, MGC:MCL, add MYM:M2K

### FIX-12: TTM Squeeze SQUEEZE_FIRING Unhandled [CRITICAL]
**File**: `agents/ttm_squeeze_agent.py` lines 187-215
**Problem**: SQUEEZE_FIRING falls into else branch, signal silently discarded. Agent NEVER generates signals.
**Fix**: Handle SQUEEZE_FIRING same as SQUEEZE_OFF

### FIX-13: TTM Config Key Mismatch [CRITICAL]
**File**: `agents/ttm_squeeze_agent.py` lines 87-93
**Problem**: Agent passes bb_length/kc_length/kc_mult, strategy expects bb_period/kc_period/kc_atr_mult
**Fix**: Align key names

### FIX-14: Market Making Agent Doesn't Use A-S Strategy [CRITICAL]
**File**: `agents/market_making_agent.py` lines 218-242
**Problem**: Naive formula `base_spread + vol*10 + inventory*5` instead of Avellaneda-Stoikov
**Fix**: Import and call strategy's calculate_optimal_spread()

### FIX-15: Micro-Price Formula Inverted [HIGH]
**File**: `strategies/market_making_strategy.py` lines 644-646
**Problem**: `bid*bid_size + ask*ask_size` should be `bid*ask_size + ask*bid_size`
**Fix**: Swap weights

### FIX-16: EventDrivenAgent Surprise=0.0 Hardcoded [CRITICAL]
**File**: `agents/event_driven_agent.py` line 176
**Problem**: Post-event signal always receives surprise=0.0, making _get_post_event_signal dead code
**Fix**: Compute surprise from price action

### FIX-17: EventDrivenAgent Barrier-Silent Returns [CRITICAL]
**File**: `agents/event_driven_agent.py` lines 137,142,155,172,220,225
**Problem**: 5 of 6 return paths emit no heartbeat -> agent missing from barrier
**Fix**: Add heartbeat emission to all return paths

### FIX-18: Event Sensitive Assets Use Full-Size Symbols [CRITICAL]
**File**: `strategies/event_driven_strategy.py` lines 128-156
**Problem**: Lists "ES","NQ","GC","CL" but system trades MES,MNQ,MGC,MCL
**Fix**: Add micro futures to sensitive_assets

### FIX-19: Mean Reversion Agent RSI Parameters Wrong [CRITICAL]
**File**: `agents/mean_reversion_agent.py` lines 78-80
**Problem**: Agent defaults rsi_period=14, oversold=30, overbought=70 instead of Connors RSI(2) 5/95
**Fix**: Change defaults to period=2, thresholds 5/95

### FIX-20: MACD-v Agent min_confidence Blocks Signals [HIGH]
**File**: `agents/macdv_agent.py` line 128
**Problem**: Agent requires 0.75 but strategy base confidence is 0.6. Most signals rejected.
**Fix**: Lower to 0.5

### FIX-21: Session Strategy breakout_threshold_atr=0.5 [CRITICAL]
**File**: `strategies/session_strategy.py` line 178
**Problem**: 0.5 ATR = massive false breakouts. Research says 1.5-2.0 ATR.
**Fix**: Default 1.5

### FIX-22: Momentum Strategy slow_period=30 [HIGH]
**File**: `strategies/momentum_strategy.py` line 128
**Problem**: 30-period MA too short for 1-min bars, causes whipsaws
**Fix**: Default 50 (per CLAUDE.md Phase 11)

### FIX-23: Index Spread Config Key Mismatch [CRITICAL]
**File**: `agents/index_spread_agent.py` lines 92-111
**Problem**: Agent passes entry_zscore/exit_zscore/lookback, strategy expects zscore_entry/zscore_exit/lookback_days
**Fix**: Align key names

### FIX-24: Index Spread Z-Score Days vs Bars [CRITICAL]
**File**: `strategies/index_spread_strategy.py` lines 159, 376
**Problem**: lookback_days=60 used as bar count. 60 bars = 1 hour, not 60 days.
**Fix**: Convert days to bars or use bar-based lookback

### FIX-25: StatArb Min Data Threshold 10 Bars [HIGH]
**File**: `agents/stat_arb_agent.py` lines 279-282
**Problem**: Starts generating signals after only 10 minutes. Hedge ratio on 10 points is noise.
**Fix**: Minimum 120 bars (2 hours)

### FIX-26: IndexSpreadAgent Drops FLAT Exit Signals [MEDIUM]
**File**: `agents/index_spread_agent.py` lines 244-251
**Problem**: Strategy returns "flat" but agent only checks for "exit". Open positions never closed.
**Fix**: Add `"flat"` to the elif condition

### FIX-27: Pair Signal Symbol "MES:MNQ" Unexecutable [HIGH]
**File**: `agents/stat_arb_agent.py` lines 370-385
**Problem**: CIO sends order for symbol "MES:MNQ" which IB rejects
**Fix**: Emit two separate signals, one per leg

### FIX-28: Price Arrays Desynchronized Between Legs [HIGH]
**File**: `agents/stat_arb_agent.py` lines 200-212
**Problem**: Each leg updates independently, spread computed on misaligned data
**Fix**: Only compute when BOTH legs updated

### FIX-29: HMM Blocks Event Loop 500-2000ms [HIGH]
**File**: `agents/macro_agent.py` lines 729-737
**Problem**: Synchronous HMM fit() blocks async loop, causes barrier timeouts
**Fix**: Run in asyncio.to_thread()

### FIX-30: MacroAgent Signals SPY But System Trades MES [MEDIUM]
**File**: `agents/macro_agent.py` lines 397, 559, 618, 682, 802
**Problem**: All macro signals use symbol="SPY", system has no SPY->MES mapping
**Fix**: Emit with symbol="MES"

### FIX-31: Momentum Confidence Variable Overwritten [HIGH]
**File**: `strategies/momentum_strategy.py` line 2020
**Problem**: All confidence adjustments from lines 1982-2018 are lost at line 2020
**Fix**: Incorporate prior confidence value

### FIX-32: TTM Momentum Missing linreg Smoothing [CRITICAL]
**File**: `strategies/ttm_squeeze_strategy.py` lines 291-297
**Problem**: Raw momentum without linreg = noisy direction flips on 1-min bars
**Fix**: Apply linear regression smoothing per Carter's original

### FIX-33: TTM ATR Calculation Wrong [HIGH]
**File**: `agents/ttm_squeeze_agent.py` lines 270-274
**Problem**: Missing |L-prev_close| component, shape mismatch in np.maximum
**Fix**: Use proper 3-component True Range

### FIX-34: EventDrivenAgent ATR = 1.5% of Price [HIGH]
**File**: `agents/event_driven_agent.py` line 186
**Problem**: For MES at 5000, ATR estimate = 75 points. Real 1-min ATR = 2-5 points. Stops 15-50x too wide.
**Fix**: Use 0.1% for 1-min ATR estimate

### FIX-35: MacroStrategy R:R Float Precision Not Fixed [HIGH]
**File**: `strategies/macro_strategy.py` lines 357-358
**Problem**: Agent was fixed (round to 2 dp) but strategy still has raw division
**Fix**: Add `round(reward_risk_ratio, 2)`

### FIX-36: MACD-v Neutral Counter Per-Tick Not Per-Bar [HIGH]
**File**: `strategies/macdv_strategy.py` lines 707-712
**Problem**: _bars_in_neutral increments per market data event, not per bar
**Fix**: Track last update timestamp, increment only every 60s

### FIX-37: Dollar-Neutral Sizing Wrong [HIGH]
**File**: `strategies/index_spread_strategy.py` lines 491-512
**Problem**: Adding notionals doesn't create dollar-neutral. Ratios meaningless for execution.
**Fix**: Use contract-count-based neutrality

### FIX-38: Economic Calendar Race Condition [CRITICAL]
**File**: `main.py` lines 1257-1258
**Problem**: Calendar init is fire-and-forget async task, agent starts with empty events
**Fix**: await the initialization

---

## PHASE 3: TUNING

### FIX-39: Consensus Threshold 55% -> 40%
### FIX-40: Conviction Threshold 0.70 -> 0.55
### FIX-41: Confidence * Strength Double-Penalty
### FIX-42: M2K/MYM Missing from EventDrivenAgent Sensitivity Map
### FIX-43: MacroAgent Confidence Not Calibrated (0.55-0.60 for SMA cross)
### FIX-44: Pre-Event Window 24h -> 2h for Intraday
### FIX-45: min_surprise_std 3.0 -> 1.5

---

## STATISTICS

- **Total Issues Found**: 90
- **21 CRITICAL**: System cannot function correctly
- **33 HIGH**: Causes significant losses or missed trades
- **28 MEDIUM**: Suboptimal behavior
- **8 LOW**: Code quality / minor

### Strategies That NEVER Generate Valid Signals
1. **TTM Squeeze**: SQUEEZE_FIRING unhandled + config key mismatch = zero signals ever
2. **Market Making**: A-S strategy never imported, naive formula used
3. **Event-Driven**: surprise=0.0 hardcoded, post-event always returns None
4. **Index Spread**: Config key mismatch = all params default, FLAT exits dropped

### Key Config Issues
| Parameter | Current | Correct | Impact |
|-----------|---------|---------|--------|
| max_position_pct | 1.0% | 5.0% | Blocks ALL futures |
| base_position_size | 100 (contracts) | 5000 (dollars) | 60 MNQ = $2.28M |
| min_order | 10 | 1 (futures) | Proper 1-3 contracts rejected |
| max_leverage | 1.5 (CIO hardcoded) | 3.0 (config) | Deadlock at 2.55x |
| breakout_threshold_atr | 0.5 | 1.5 | Massive false breakouts |
| rsi_period (MR agent) | 14 | 2 | Wrong strategy entirely |
| min_confidence (MACD-v) | 0.75 | 0.50 | Most signals rejected |
