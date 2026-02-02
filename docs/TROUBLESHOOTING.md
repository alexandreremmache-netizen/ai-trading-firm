# Troubleshooting Guide

## Overview

This guide provides solutions to common issues encountered when running the AI Trading Firm system.

---

## Connection Issues

### Cannot Connect to IB Gateway/TWS

**Symptoms**:
- "Connection refused" error
- Timeout during connection
- System runs in simulation mode

**Diagnostic Steps**:

1. **Verify IB Gateway/TWS is running**
   ```bash
   # Windows
   tasklist | findstr "ibgateway\|tws"

   # Linux/Mac
   ps aux | grep -i "ibgateway\|tws"
   ```

2. **Check port is listening**
   ```bash
   # Windows
   netstat -an | findstr "4002\|7497"

   # Linux/Mac
   netstat -an | grep "4002\|7497"
   ```

3. **Verify API is enabled in IB**
   - Open IB Gateway/TWS
   - Go to: Edit > Global Configuration > API > Settings
   - Ensure "Enable ActiveX and Socket Clients" is checked
   - Verify socket port matches your configuration

**Solutions**:

| Cause | Solution |
|-------|----------|
| IB not running | Start IB Gateway or TWS |
| Wrong port | Update config.yaml to match IB port |
| API disabled | Enable API in IB settings |
| Firewall blocking | Allow port in firewall |
| IP not trusted | Add 127.0.0.1 to trusted IPs in IB |

### Connection Drops Frequently

**Symptoms**:
- Reconnecting messages in logs
- Intermittent market data
- Orders fail sporadically

**Solutions**:

1. **Check network stability**
   ```bash
   ping api.ibkr.com
   ```

2. **Increase timeout**
   ```yaml
   broker:
     timeout_seconds: 60  # Increase from default 30
   ```

3. **Enable auto-reconnect**
   ```yaml
   broker:
     auto_reconnect: true
     max_reconnect_attempts: 10
   ```

4. **Check IB Gateway idle settings**
   - IB Gateway may disconnect after idle period
   - Disable auto-logoff in settings

---

## Market Data Issues

### No Market Data Received

**Symptoms**:
- "Using simulated market data" message
- No signals generated
- Prices show as 0

**Diagnostic Steps**:

1. **Check market data subscription**
   ```python
   # Enable delayed data if no subscription
   broker:
     use_delayed_data: true
   ```

2. **Verify instruments are valid**
   - Check symbol spelling
   - Verify exchange code
   - Confirm currency

3. **Check market hours**
   - Equities: 9:30 AM - 4:00 PM ET
   - Futures: Nearly 24/7 (check specific product)
   - Forex: Sunday 5 PM - Friday 5 PM ET

**Solutions**:

| Cause | Solution |
|-------|----------|
| No subscription | Enable delayed data |
| Invalid symbol | Verify symbol/exchange/currency |
| Market closed | Wait for market hours |
| Data type wrong | Call `request_market_data_type(3)` |

### Stale Market Data

**Symptoms**:
- Warning about stale data
- Old timestamps on market data
- Decisions based on outdated prices

**Solutions**:

1. **Check staleness configuration**
   ```yaml
   broker:
     staleness_warning_seconds: 5.0
     staleness_critical_seconds: 30.0
   ```

2. **Monitor data freshness**
   - Check logs for staleness warnings
   - Verify network latency
   - Consider different data source

---

## Order Execution Issues

### Orders Rejected

**Symptoms**:
- "Order rejected" in logs
- Validation failed messages
- Risk check failures

**Common Rejection Reasons**:

| Rejection | Cause | Solution |
|-----------|-------|----------|
| Position size | Exceeds 5% limit | Reduce quantity |
| Sector exposure | Exceeds 20% limit | Diversify |
| Leverage | Exceeds 2x limit | Reduce exposure |
| Rate limit | Too many orders | Wait between orders |
| Blackout period | Compliance restriction | Wait for blackout end |
| Kill switch | Trading halted | Address halt condition |

### Orders Not Filled

**Symptoms**:
- Orders remain pending
- Partial fills only
- Execution takes too long

**Solutions**:

1. **Check order type**
   - Limit orders may not fill if price moves
   - Consider market orders for urgent execution

2. **Review execution algorithm**
   ```yaml
   agents:
     execution:
       default_algo: "TWAP"
       slice_interval_seconds: 30  # Reduce for faster execution
   ```

3. **Check liquidity**
   - Large orders may need more time
   - Check average daily volume

### High Slippage

**Symptoms**:
- Fill prices far from expected
- Large market impact
- Performance degradation

**Solutions**:

1. **Use algorithmic execution**
   ```yaml
   agents:
     execution:
       default_algo: "TWAP"  # Or "VWAP"
       max_slippage_bps: 50
   ```

2. **Reduce position sizes**
   - Large orders move markets
   - Split into smaller pieces

3. **Trade more liquid instruments**
   - Check ADV (average daily volume)
   - Avoid low-liquidity securities

---

## Agent Issues

### Agent Not Starting

**Symptoms**:
- Agent missing from status
- "Agent disabled" message
- Incomplete initialization

**Solutions**:

1. **Check agent is enabled**
   ```yaml
   agents:
     macro:
       enabled: true  # Ensure true, not false
   ```

2. **Check for initialization errors**
   - Review logs for exceptions
   - Verify dependencies installed

3. **Check event bus connection**
   - Verify event bus started first
   - Check subscription succeeded

### Agent Timeout

**Symptoms**:
- "Agent timed out" in logs
- Events not processed
- High latency

**Solutions**:

1. **Increase timeout**
   ```python
   AgentConfig(
       timeout_seconds=60.0,  # Increase from 30
   )
   ```

2. **Optimize processing**
   - Profile slow operations
   - Consider async operations

3. **Check system resources**
   - CPU utilization
   - Memory usage
   - Disk I/O

### Signal Barrier Timeout

**Symptoms**:
- "Signal barrier timeout" message
- Missing signals from agents
- CIO decisions delayed

**Solutions**:

1. **Increase barrier timeout**
   ```yaml
   event_bus:
     sync_barrier_timeout_seconds: 15.0  # Increase from 10
   ```

2. **Check slow agents**
   - Review per-agent timing
   - Optimize signal generation

3. **Verify all agents registered**
   - Check startup logs
   - Confirm agent count

---

## Risk and Compliance Issues

### Kill Switch Activated

**Symptoms**:
- "KILL SWITCH ACTIVATED" message
- All trading halted
- Emergency shutdown

**Diagnostic Steps**:

1. **Check trigger reason in logs**
   ```
   EMERGENCY SHUTDOWN TRIGGERED
   Reason: Daily loss limit exceeded (-3.5%)
   ```

2. **Review risk state**
   - Check daily P&L
   - Check drawdown
   - Review VaR

**Recovery Steps**:

1. Stop the system
2. Analyze what caused the loss
3. Address root cause
4. Restart with appropriate risk limits

### Compliance Rejection

**Symptoms**:
- "BLACKOUT_PERIOD" rejection
- "RESTRICTED_INSTRUMENT" rejection
- Decisions not reaching execution

**Solutions**:

| Rejection Code | Cause | Solution |
|----------------|-------|----------|
| BLACKOUT_PERIOD | Earnings blackout | Wait for blackout end |
| RESTRICTED_INSTRUMENT | Banned security | Remove from universe |
| MARKET_CLOSED | Outside hours | Trade during market hours |
| INVALID_LEI | Bad LEI in config | Get valid LEI from GLEIF |

### VaR Calculation Errors

**Symptoms**:
- VaR shows as 0 or NaN
- Risk validation fails
- Unexpected risk values

**Solutions**:

1. **Check historical data**
   - Need minimum history for VaR
   - Verify data quality

2. **Review portfolio weights**
   - Weights should sum to ~1
   - No negative weights (unless shorting)

3. **Check covariance matrix**
   - Must be positive definite
   - Verify sufficient observations

---

## Performance Issues

### High Event Queue

**Symptoms**:
- Backpressure warnings
- Events dropped
- Processing delays

**Solutions**:

1. **Increase queue size** (short-term)
   ```yaml
   event_bus:
     max_queue_size: 20000
   ```

2. **Reduce event rate** (long-term)
   - Fewer instruments
   - Slower tick rate

3. **Optimize handlers**
   - Profile slow handlers
   - Use async processing

### Memory Usage High

**Symptoms**:
- Memory warnings
- System slowdown
- Out of memory errors

**Solutions**:

1. **Limit history size**
   ```python
   self._max_history = 5000  # Reduce from 10000
   ```

2. **Reduce event persistence**
   ```yaml
   event_bus:
     persistence_enabled: false
   ```

3. **Restart periodically**
   - Daily restart during off-hours
   - Clear accumulated state

### Slow Startup

**Symptoms**:
- Initialization takes minutes
- Timeout during startup
- Incomplete initialization

**Solutions**:

1. **Skip startup stress tests**
   ```yaml
   stress_testing:
     run_on_startup: false
   ```

2. **Reduce instrument universe**
   - Start with fewer symbols
   - Add incrementally

3. **Check broker connection**
   - Increase connection timeout
   - Verify network latency

---

## Logging Issues

### Missing Audit Logs

**Symptoms**:
- audit.jsonl empty or missing
- Compliance gaps
- Missing trade records

**Solutions**:

1. **Check log directory exists**
   ```bash
   mkdir -p logs
   ```

2. **Verify log configuration**
   ```yaml
   logging:
     audit_file: "logs/audit.jsonl"
   ```

3. **Check file permissions**
   - Write access to logs directory
   - Disk space available

### Log File Too Large

**Symptoms**:
- Disk space warnings
- Slow log writes
- Performance degradation

**Solutions**:

1. **Implement log rotation**
   - Configure external log rotation
   - Archive old logs

2. **Reduce log level**
   ```yaml
   logging:
     level: "WARNING"  # Instead of "DEBUG"
   ```

3. **Archive old logs**
   - Keep 7 years for compliance
   - Compress older files

---

## Quick Reference

### Emergency Procedures

| Situation | Action |
|-----------|--------|
| Kill switch activated | Stop, analyze, fix, restart |
| Connection lost | Check IB, wait for reconnect |
| High losses | Stop trading, review positions |
| System unresponsive | Force restart (Ctrl+C twice) |

### Key Log Locations

| Log | Location | Purpose |
|-----|----------|---------|
| System | logs/system.log | General logs |
| Audit | logs/audit.jsonl | Compliance |
| Trades | logs/trades.jsonl | Execution |
| Decisions | logs/decisions.jsonl | CIO decisions |

### Health Check Commands

```bash
# Check system health
curl http://localhost:8080/health

# Check readiness
curl http://localhost:8080/ready

# Check liveness
curl http://localhost:8080/live
```

### Support Information

When reporting issues, include:
1. Error message (exact text)
2. Relevant log entries
3. Configuration (sanitized)
4. Steps to reproduce
5. System information (OS, Python version)
