#!/usr/bin/env python3
"""
Test script to verify dashboard-EventBus integration.

This script:
1. Creates an EventBus
2. Creates a DashboardServer with that EventBus
3. Publishes test events
4. Verifies the dashboard state receives them

Run: python test_dashboard_integration.py
"""

import asyncio
import sys
from datetime import datetime, timezone

# Add project root to path
sys.path.insert(0, ".")

from core.event_bus import EventBus
from core.events import (
    EventType, SignalEvent, SignalDirection,
    DecisionEvent, OrderSide, MarketDataEvent,
)
from dashboard.server import create_dashboard_server, DashboardServer


async def test_dashboard_event_integration():
    """Test that dashboard receives events from EventBus."""
    print("=" * 60)
    print("DASHBOARD-EVENTBUS INTEGRATION TEST")
    print("=" * 60)

    # 1. Create EventBus
    print("\n1. Creating EventBus...")
    event_bus = EventBus(
        max_queue_size=1000,
        signal_timeout=5.0,
        barrier_timeout=10.0,
    )
    print("   EventBus created")

    # 2. Create DashboardServer with EventBus
    print("\n2. Creating DashboardServer with shared EventBus...")
    dashboard_server = create_dashboard_server(
        event_bus=event_bus,
        host="127.0.0.1",
        port=8081,
    )
    print("   DashboardServer created")

    # 3. Initialize dashboard state (this subscribes to events)
    print("\n3. Initializing dashboard state...")
    await dashboard_server.state.initialize()
    print("   Dashboard state initialized and subscribed to events")

    # 4. Start EventBus in background
    print("\n4. Starting EventBus...")
    event_bus_task = asyncio.create_task(event_bus.start())
    await asyncio.sleep(0.1)  # Let it start
    print(f"   EventBus running: {event_bus.is_running}")

    # 5. Publish test events
    print("\n5. Publishing test events...")

    # Market data event
    market_event = MarketDataEvent(
        symbol="AAPL",
        bid=150.00,
        ask=150.05,
        last=150.02,
        volume=1000000,
        source_agent="MarketDataManager",
    )
    await event_bus.publish(market_event)
    print("   Published MarketDataEvent for AAPL")

    # Signal event
    signal_event = SignalEvent(
        symbol="AAPL",
        direction=SignalDirection.LONG,
        confidence=0.85,
        strength=0.7,
        rationale="Strong momentum detected",
        source_agent="MomentumAgent",
    )
    await event_bus.publish(signal_event)
    print("   Published SignalEvent from MomentumAgent")

    # Decision event
    decision_event = DecisionEvent(
        symbol="AAPL",
        action=OrderSide.BUY,
        quantity=100,
        conviction_score=0.82,
        rationale="Consensus buy signal",
        source_agent="CIOAgent",
        data_sources=("momentum", "macro"),
        contributing_signals=("MomentumAgent",),
    )
    await event_bus.publish(decision_event)
    print("   Published DecisionEvent from CIOAgent")

    # Give time for events to be processed
    await asyncio.sleep(0.5)

    # 6. Check dashboard state
    print("\n6. Checking dashboard state...")

    events = dashboard_server.state.get_events()
    agents = dashboard_server.state.get_agents()
    signals = dashboard_server.state.get_signals()
    decisions = dashboard_server.state.get_decisions()

    print(f"   Events received: {len(events)}")
    print(f"   Agents tracked: {len(agents)}")
    print(f"   Signals stored: {len(signals)}")
    print(f"   Decisions stored: {len(decisions)}")

    # 7. Verify results
    print("\n7. Verification results:")
    success = True

    if len(events) >= 3:
        print("   [PASS] Events received by dashboard")
    else:
        print(f"   [FAIL] Expected >= 3 events, got {len(events)}")
        success = False

    if len(agents) >= 2:
        print("   [PASS] Agents tracked by dashboard")
        for agent in agents:
            print(f"         - {agent['name']} ({agent['type']}): {agent['status']}")
    else:
        print(f"   [FAIL] Expected >= 2 agents, got {len(agents)}")
        success = False

    if len(signals) >= 1:
        print("   [PASS] Signals stored in dashboard")
        for sig in signals:
            print(f"         - {sig['symbol']} {sig['direction']} by {sig['agent']}")
    else:
        print(f"   [FAIL] Expected >= 1 signal, got {len(signals)}")
        success = False

    if len(decisions) >= 1:
        print("   [PASS] Decisions stored in dashboard")
        for dec in decisions:
            print(f"         - {dec['symbol']} {dec['direction']} x{dec['quantity']}")
    else:
        print(f"   [FAIL] Expected >= 1 decision, got {len(decisions)}")
        success = False

    # 8. Cleanup
    print("\n8. Cleaning up...")
    await event_bus.stop()
    print("   EventBus stopped")

    # 9. Final result
    print("\n" + "=" * 60)
    if success:
        print("RESULT: ALL TESTS PASSED")
        print("Dashboard successfully receives events from EventBus!")
    else:
        print("RESULT: SOME TESTS FAILED")
        print("Check the output above for details.")
    print("=" * 60)

    return success


if __name__ == "__main__":
    result = asyncio.run(test_dashboard_event_integration())
    sys.exit(0 if result else 1)
