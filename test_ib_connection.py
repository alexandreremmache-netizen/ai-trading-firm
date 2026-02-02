#!/usr/bin/env python3
"""
Interactive Brokers Connection Test
====================================

Run this script to test your IB connection.

Prerequisites:
1. TWS or IB Gateway must be running
2. API connections must be enabled:
   - TWS: Edit > Global Configuration > API > Settings
   - Enable "Enable ActiveX and Socket Clients"
   - Set Socket port to 7497 (paper) or 7496 (live)
   - Check "Allow connections from localhost only"

Usage:
    python test_ib_connection.py
"""

import asyncio
import logging
from core.broker import IBBroker, BrokerConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


async def test_connection():
    """Test basic IB connection."""
    print("=" * 60)
    print("AI TRADING FIRM - IB CONNECTION TEST")
    print("=" * 60)

    # Create broker with paper trading config
    config = BrokerConfig(
        host="127.0.0.1",
        port=4002,  # IB Gateway Paper Trading
        client_id=99,  # Use different ID to avoid conflicts
        timeout_seconds=10.0,
        readonly=True,  # Safe mode for testing
    )

    broker = IBBroker(config)

    print(f"\nConnecting to IB at {config.host}:{config.port}...")

    connected = await broker.connect()

    if not connected:
        print("\n[FAILED] Connection FAILED")
        print("\nTroubleshooting:")
        print("1. Ensure TWS or IB Gateway is running")
        print("2. Check API settings in TWS: Edit > Global Configuration > API > Settings")
        print("3. Verify port number (7497 for TWS Paper, 4002 for Gateway Paper)")
        print("4. Make sure 'Enable ActiveX and Socket Clients' is checked")
        return False

    print(f"\n[OK] Connected successfully!")
    print(f"   Account: {broker.account_id}")

    # Test portfolio state
    print("\n[INFO] Fetching portfolio state...")
    try:
        portfolio = await broker.get_portfolio_state()
        print(f"   Net Liquidation: ${portfolio.net_liquidation:,.2f}")
        print(f"   Total Cash: ${portfolio.total_cash:,.2f}")
        print(f"   Buying Power: ${portfolio.buying_power:,.2f}")
        print(f"   Daily P&L: ${portfolio.daily_pnl:,.2f}")
        print(f"   Positions: {len(portfolio.positions)}")

        if portfolio.positions:
            print("\n   Current Positions:")
            for symbol, pos in portfolio.positions.items():
                print(f"   - {symbol}: {pos.quantity} shares @ ${pos.avg_cost:.2f} "
                      f"(P&L: ${pos.unrealized_pnl:,.2f})")
    except Exception as e:
        print(f"   [WARN] Error getting portfolio: {e}")

    # Test market data subscription
    print("\n[INFO] Testing market data subscription (AAPL)...")
    try:
        # Use delayed data (free, no subscription needed)
        await broker.request_market_data_type(3)  # 3 = Delayed

        subscribed = await broker.subscribe_market_data("AAPL")
        if subscribed:
            print("   [OK] Subscribed to AAPL market data")
            await asyncio.sleep(2)  # Wait for data
            await broker.unsubscribe_market_data("AAPL")
            print("   [OK] Unsubscribed from AAPL")
        else:
            print("   [WARN] Failed to subscribe to market data")
    except Exception as e:
        print(f"   [WARN] Error with market data: {e}")

    # Test historical data
    print("\n[INFO] Testing historical data (SPY, 1 day)...")
    try:
        bars = await broker.get_historical_data(
            symbol="SPY",
            duration="1 D",
            bar_size="5 mins",
        )
        if bars:
            print(f"   [OK] Retrieved {len(bars)} bars")
            if bars:
                last_bar = bars[-1]
                print(f"   Last bar: {last_bar['date']} - "
                      f"O:{last_bar['open']:.2f} H:{last_bar['high']:.2f} "
                      f"L:{last_bar['low']:.2f} C:{last_bar['close']:.2f}")
        else:
            print("   [WARN] No historical data returned")
    except Exception as e:
        print(f"   [WARN] Error getting historical data: {e}")

    # Disconnect
    print("\n[INFO] Disconnecting...")
    await broker.disconnect()
    print("   [OK] Disconnected")

    print("\n" + "=" * 60)
    print("TEST COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print("\nYour IB connection is working. You can now run the trading system.")

    return True


async def test_order_preview():
    """Test order creation (preview only, no actual execution)."""
    print("\n" + "=" * 60)
    print("ORDER PREVIEW TEST (No actual orders)")
    print("=" * 60)

    from core.events import OrderEvent, OrderSide, OrderType

    # Create a sample order event
    order = OrderEvent(
        source_agent="test",
        decision_id="test-decision-001",
        validation_id="test-validation-001",
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=10,
        order_type=OrderType.LIMIT,
        limit_price=150.00,
    )

    print(f"\nSample Order Preview:")
    print(f"  Symbol: {order.symbol}")
    print(f"  Side: {order.side.value}")
    print(f"  Quantity: {order.quantity}")
    print(f"  Type: {order.order_type.value}")
    print(f"  Limit Price: ${order.limit_price}")
    print(f"  Event ID: {order.event_id}")

    print("\n[WARN] This was just a preview - no order was sent to IB")


if __name__ == "__main__":
    print("\n")
    success = asyncio.run(test_connection())

    if success:
        asyncio.run(test_order_preview())
