# Operational Runbooks

**Last Updated**: 2026-02-02
**Audience**: System operators, on-call engineers

---

## Table of Contents

1. [Overview](#overview)
2. [Daily Operations](#daily-operations)
3. [Startup Procedures](#startup-procedures)
4. [Shutdown Procedures](#shutdown-procedures)
5. [Incident Response](#incident-response)
6. [Common Issues](#common-issues)
7. [Monitoring](#monitoring)
8. [Backup & Recovery](#backup--recovery)
9. [Emergency Procedures](#emergency-procedures)

---

## Overview

This document provides step-by-step operational procedures for managing the AI Trading Firm system. Follow these runbooks for routine operations and incident response.

### System Components

| Component | Purpose | Critical Level |
|-----------|---------|----------------|
| IB Gateway | Broker connection | CRITICAL |
| Event Bus | Message routing | CRITICAL |
| CIO Agent | Trading decisions | CRITICAL |
| Risk Agent | Risk validation | CRITICAL |
| Compliance Agent | Regulatory checks | HIGH |
| Execution Agent | Order execution | HIGH |
| Health Check Server | Monitoring | MEDIUM |

### Contact Information

| Role | Contact | Escalation Time |
|------|---------|-----------------|
| On-Call Engineer | oncall@company.com | Immediate |
| Risk Manager | risk@company.com | 15 minutes |
| Compliance Officer | compliance@company.com | 30 minutes |
| CTO | cto@company.com | 1 hour |

---

## Daily Operations

### Pre-Market Checklist (T-30 minutes)

```
[ ] 1. Verify IB Gateway connection
      $ python scripts/check_ib_connection.py
      Expected: "Connection: OK"

[ ] 2. Check system health
      $ curl http://localhost:8080/health
      Expected: All components "ok"

[ ] 3. Verify risk limits are loaded
      $ curl http://localhost:8080/api/risk/limits
      Expected: Current limits displayed

[ ] 4. Check overnight positions
      $ python scripts/show_positions.py
      Expected: Positions match expected

[ ] 5. Verify compliance calendar
      $ python scripts/check_compliance_deadlines.py
      Expected: No overdue deadlines

[ ] 6. Review overnight alerts
      $ cat logs/alerts_$(date +%Y%m%d).log
      Action: Address any CRITICAL alerts
```

### Market Hours Monitoring

```
Frequency: Every 15 minutes during market hours

[ ] Check portfolio VaR vs limits
    $ curl http://localhost:8080/api/risk/var
    Alert if: utilization > 80%

[ ] Monitor execution quality
    $ curl http://localhost:8080/api/execution/stats
    Alert if: slippage > 10 bps average

[ ] Check for compliance alerts
    $ curl http://localhost:8080/api/compliance/alerts
    Action: Escalate any new alerts
```

### End-of-Day Checklist

```
[ ] 1. Review daily P&L
      $ python scripts/daily_pnl_report.py
      Action: Escalate if loss > daily limit

[ ] 2. Generate risk report
      $ python scripts/generate_risk_report.py
      Output: reports/risk_$(date +%Y%m%d).pdf

[ ] 3. Verify transaction reporting
      $ python scripts/check_emir_reporting.py
      Expected: All trades reported

[ ] 4. Archive logs
      $ python scripts/archive_logs.py
      Expected: Logs compressed and stored

[ ] 5. Backup critical data
      $ python scripts/backup_positions.py
      Expected: Backup completed

[ ] 6. Generate compliance summary
      $ python scripts/compliance_summary.py
      Output: reports/compliance_$(date +%Y%m%d).pdf
```

---

## Startup Procedures

### Standard Startup

```bash
#!/bin/bash
# startup.sh

echo "=== AI Trading System Startup ==="

# Step 1: Start IB Gateway (manual step)
echo "1. Ensure IB Gateway is running and logged in"
read -p "Press enter when IB Gateway is ready..."

# Step 2: Verify connection
echo "2. Verifying IB connection..."
python scripts/check_ib_connection.py
if [ $? -ne 0 ]; then
    echo "ERROR: IB connection failed"
    exit 1
fi

# Step 3: Load configuration
echo "3. Loading configuration..."
python scripts/validate_config.py
if [ $? -ne 0 ]; then
    echo "ERROR: Configuration validation failed"
    exit 1
fi

# Step 4: Start trading system
echo "4. Starting trading system..."
python main.py --mode paper &
TRADING_PID=$!
echo "Trading system PID: $TRADING_PID"

# Step 5: Wait for health check
echo "5. Waiting for system to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:8080/ready | grep -q "ready"; then
        echo "System is ready"
        break
    fi
    sleep 1
done

# Step 6: Verify all agents running
echo "6. Verifying agent status..."
curl http://localhost:8080/api/agents/status

echo "=== Startup Complete ==="
```

### Startup After Incident

Additional steps after an incident:

```bash
# Step A: Clear any stale locks
rm -f /tmp/trading_system.lock

# Step B: Reconcile orders with broker
python scripts/reconcile_orders.py

# Step C: Verify position consistency
python scripts/verify_positions.py

# Step D: Review any pending compliance alerts
python scripts/review_pending_alerts.py
```

---

## Shutdown Procedures

### Graceful Shutdown

```bash
#!/bin/bash
# shutdown.sh

echo "=== AI Trading System Shutdown ==="

# Step 1: Signal graceful shutdown
echo "1. Initiating graceful shutdown..."
curl -X POST http://localhost:8080/api/system/shutdown

# Step 2: Wait for pending orders
echo "2. Waiting for pending orders to complete..."
for i in {1..60}; do
    PENDING=$(curl -s http://localhost:8080/api/orders/pending | jq '.count')
    if [ "$PENDING" -eq 0 ]; then
        echo "All orders completed"
        break
    fi
    echo "Pending orders: $PENDING"
    sleep 5
done

# Step 3: Verify clean state
echo "3. Verifying clean shutdown..."
curl http://localhost:8080/api/system/state

# Step 4: Stop the process
echo "4. Stopping process..."
kill $(cat /var/run/trading_system.pid)

# Step 5: Archive session data
echo "5. Archiving session data..."
python scripts/archive_session.py

echo "=== Shutdown Complete ==="
```

### Emergency Shutdown (Kill Switch)

```bash
#!/bin/bash
# emergency_shutdown.sh

echo "!!! EMERGENCY SHUTDOWN !!!"

# Immediately halt all trading
curl -X POST http://localhost:8080/api/system/kill_switch

# Cancel all open orders
python scripts/cancel_all_orders.py --force

# Log emergency shutdown
echo "$(date): Emergency shutdown initiated by $USER" >> /var/log/emergency.log

# Notify stakeholders
python scripts/notify_emergency.py --message "Emergency shutdown initiated"

# Force kill if needed
sleep 10
pkill -9 -f "python main.py"
```

---

## Incident Response

### Severity Levels

| Level | Description | Response Time | Examples |
|-------|-------------|---------------|----------|
| SEV1 | Critical | 5 minutes | Kill switch activated, major loss |
| SEV2 | High | 15 minutes | Risk limit breach, execution failure |
| SEV3 | Medium | 1 hour | Performance degradation, minor alerts |
| SEV4 | Low | 4 hours | Non-critical warnings |

### SEV1 Response Procedure

```
1. IMMEDIATE: Activate kill switch if not already active
   $ curl -X POST http://localhost:8080/api/system/kill_switch

2. ASSESS: Identify the scope of the incident
   - What triggered the incident?
   - What positions are affected?
   - What is the current P&L impact?

3. COMMUNICATE: Notify stakeholders
   - On-call engineer
   - Risk manager
   - Compliance officer
   - Management (if required)

4. CONTAIN: Prevent further damage
   - Cancel pending orders
   - Halt strategy execution
   - Freeze position changes

5. INVESTIGATE: Gather evidence
   - Collect relevant logs
   - Document timeline
   - Identify root cause

6. RESOLVE: Fix the issue
   - Apply fix or workaround
   - Verify fix is effective
   - Prepare for restart

7. RECOVER: Restore normal operations
   - Follow startup procedure
   - Monitor closely for recurrence
   - Update stakeholders

8. REVIEW: Post-incident analysis
   - Schedule post-mortem
   - Document lessons learned
   - Implement preventive measures
```

---

## Common Issues

### Issue: IB Gateway Disconnection

**Symptoms:**
- Health check shows broker "disconnected"
- Orders not executing
- No market data updates

**Resolution:**
```bash
# Step 1: Check IB Gateway process
ps aux | grep ibgateway

# Step 2: If not running, restart IB Gateway manually
# (Log into IB Gateway application)

# Step 3: Verify reconnection
python scripts/check_ib_connection.py

# Step 4: Reconcile orders after reconnection
python scripts/reconcile_orders.py
```

### Issue: Risk Limit Breach

**Symptoms:**
- Alert: "VaR limit exceeded"
- Trading halted for affected strategy

**Resolution:**
```bash
# Step 1: Check current VaR
curl http://localhost:8080/api/risk/var

# Step 2: Review positions
curl http://localhost:8080/api/positions

# Step 3: Identify contributing positions
python scripts/var_decomposition.py

# Step 4: Reduce risk if needed
python scripts/reduce_position.py --symbol XXXX --target 0.5

# Step 5: Reset alert after resolution
curl -X POST http://localhost:8080/api/risk/reset_alert
```

### Issue: Event Bus Backlog

**Symptoms:**
- Delayed signal processing
- High memory usage
- Slow system response

**Resolution:**
```bash
# Step 1: Check backlog size
curl http://localhost:8080/api/eventbus/stats

# Step 2: If backlog > 1000, consider clearing
curl -X POST http://localhost:8080/api/eventbus/clear_stale

# Step 3: Identify slow consumers
curl http://localhost:8080/api/eventbus/consumers

# Step 4: Restart slow agent if needed
curl -X POST http://localhost:8080/api/agents/restart/slow_agent
```

### Issue: Compliance Alert

**Symptoms:**
- Alert from compliance agent
- Potential regulatory concern

**Resolution:**
```bash
# Step 1: Review alert details
curl http://localhost:8080/api/compliance/alerts

# Step 2: Assess severity and type
# If MAR violation suspected:
python scripts/review_mar_alert.py --alert_id XXXX

# Step 3: Notify compliance officer immediately
python scripts/notify_compliance.py --alert_id XXXX

# Step 4: Document all actions taken
python scripts/log_compliance_action.py --alert_id XXXX --action "Reviewed"

# Step 5: Follow compliance officer instructions
```

---

## Monitoring

### Key Metrics Dashboard

| Metric | Normal Range | Warning | Critical |
|--------|--------------|---------|----------|
| Portfolio VaR | < 80% limit | 80-90% | > 90% |
| Leverage | < 1.5x | 1.5-1.8x | > 1.8x |
| Event Bus Latency | < 10ms | 10-50ms | > 50ms |
| Fill Rate | > 95% | 90-95% | < 90% |
| Slippage | < 5 bps | 5-10 bps | > 10 bps |

### Alert Configuration

```yaml
# alerts.yaml
alerts:
  var_warning:
    metric: portfolio_var_utilization
    threshold: 0.8
    severity: WARNING
    notify: [oncall]

  var_critical:
    metric: portfolio_var_utilization
    threshold: 0.9
    severity: CRITICAL
    notify: [oncall, risk_manager]
    action: reduce_positions

  broker_disconnect:
    metric: broker_connected
    threshold: false
    severity: CRITICAL
    notify: [oncall]
    action: halt_trading
```

### Log Locations

| Log | Location | Retention |
|-----|----------|-----------|
| Application | /var/log/trading/app.log | 30 days |
| Audit | /var/log/trading/audit.log | 7 years |
| Error | /var/log/trading/error.log | 90 days |
| Performance | /var/log/trading/perf.log | 7 days |

---

## Backup & Recovery

### Backup Schedule

| Data | Frequency | Retention | Location |
|------|-----------|-----------|----------|
| Positions | Every 15 min | 30 days | S3 |
| Configuration | Daily | 90 days | S3 |
| Audit Logs | Daily | 7 years | S3 Glacier |
| Database | Hourly | 7 days | RDS Snapshot |

### Restore Procedure

```bash
# Restore positions from backup
python scripts/restore_positions.py \
    --backup-date 2026-02-01 \
    --backup-time 16:00

# Verify restoration
python scripts/verify_positions.py --expected-count 50

# Reconcile with broker
python scripts/reconcile_orders.py
```

---

## Emergency Procedures

### Kill Switch Activation

```bash
# Activate kill switch
curl -X POST http://localhost:8080/api/system/kill_switch \
    -H "Authorization: Bearer $ADMIN_TOKEN" \
    -d '{"reason": "Manual activation", "operator": "john.doe"}'

# Verify activation
curl http://localhost:8080/api/system/status
# Expected: {"kill_switch": true, "trading_enabled": false}
```

### Kill Switch Deactivation

```bash
# Requires two-person authorization
# Person 1: Generate unlock request
curl -X POST http://localhost:8080/api/system/unlock_request \
    -d '{"requestor": "john.doe"}'

# Person 2: Approve unlock
curl -X POST http://localhost:8080/api/system/unlock_approve \
    -d '{"approver": "jane.smith", "request_id": "REQ-123"}'

# System automatically deactivates after dual approval
```

### Data Center Failover

```bash
# Step 1: Verify DR site is ready
ssh dr-site "python scripts/check_dr_ready.py"

# Step 2: Update DNS
aws route53 change-resource-record-sets ...

# Step 3: Start system at DR site
ssh dr-site "bash startup.sh"

# Step 4: Verify DR system health
curl https://dr-site.company.com:8080/health
```

---

## Appendix

### Useful Commands

```bash
# Check all agent statuses
curl http://localhost:8080/api/agents/status

# Get current positions
curl http://localhost:8080/api/positions

# Get today's trades
curl http://localhost:8080/api/trades?date=today

# Check pending orders
curl http://localhost:8080/api/orders/pending

# Get risk metrics
curl http://localhost:8080/api/risk/metrics

# Get compliance status
curl http://localhost:8080/api/compliance/status
```

### Emergency Contacts

| Role | Name | Phone | Email |
|------|------|-------|-------|
| Primary On-Call | Rotates | +1-XXX-XXX-XXXX | oncall@company.com |
| Risk Manager | John Smith | +1-XXX-XXX-XXXX | risk@company.com |
| Compliance | Jane Doe | +1-XXX-XXX-XXXX | compliance@company.com |
| CTO | Bob Johnson | +1-XXX-XXX-XXXX | cto@company.com |

---

*Review and update this runbook quarterly or after any significant incident.*
