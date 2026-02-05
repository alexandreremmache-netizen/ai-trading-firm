#!/usr/bin/env python3
"""
Close All Positions Script
==========================

Closes all open positions in Interactive Brokers by sending market orders.

Usage:
    python scripts/close_all_positions.py --dry-run    # Preview what will be closed
    python scripts/close_all_positions.py --execute    # Actually close positions
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timezone

# Add project root to path
sys.path.insert(0, '.')

from ib_insync import IB, Stock, Future, Forex, MarketOrder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


# Contract creation helpers
def create_contract(symbol: str, exchange: str = "SMART"):
    """Create appropriate contract based on symbol."""
    # Forex pairs
    forex_pairs = ["EUR", "GBP", "JPY", "CHF", "AUD", "CAD", "NZD"]
    if symbol in forex_pairs or symbol.endswith("USD"):
        # Convert to proper forex symbol
        if symbol in forex_pairs:
            pair = f"{symbol}USD"
        else:
            pair = symbol
        return Forex(pair[:3] + pair[3:], exchange="IDEALPRO")

    # Futures
    futures = {
        "ES": ("ES", "CME"),
        "NQ": ("NQ", "CME"),
        "YM": ("CBOT"),
        "RTY": ("RTY", "CME"),
        "MES": ("MES", "CME"),
        "MNQ": ("MNQ", "CME"),
        "MYM": ("MYM", "CBOT"),
        "M2K": ("M2K", "CME"),
        "CL": ("CL", "NYMEX"),
        "GC": ("GC", "COMEX"),
        "SI": ("SI", "COMEX"),
        "ZC": ("ZC", "CBOT"),
        "ZW": ("ZW", "CBOT"),
        "ZS": ("ZS", "CBOT"),
        "MCL": ("MCL", "NYMEX"),
        "MGC": ("MGC", "COMEX"),
        "ZF": ("ZF", "CBOT"),
    }

    if symbol in futures:
        info = futures[symbol]
        exch = info[1] if len(info) > 1 else info[0]
        # Use front month (placeholder - IB will resolve)
        return Future(symbol, exchange=exch, lastTradeDateOrContractMonth='')

    # Default to stock
    return Stock(symbol, "SMART", "USD")


async def close_all_positions(execute: bool = False):
    """Close all open positions."""
    ib = IB()

    try:
        # Connect to IB
        logger.info("Connecting to Interactive Brokers...")
        await ib.connectAsync('127.0.0.1', 4002, clientId=99)
        logger.info("Connected to IB")

        # Get all positions
        positions = ib.positions()

        if not positions:
            logger.info("No open positions found")
            return

        logger.info(f"Found {len(positions)} open positions")
        print("\n" + "="*60)
        print("POSITIONS TO CLOSE")
        print("="*60)

        orders_to_place = []

        for pos in positions:
            symbol = pos.contract.symbol
            qty = pos.position
            avg_cost = pos.avgCost

            if qty == 0:
                continue

            # Determine action to close
            action = "SELL" if qty > 0 else "BUY"
            close_qty = abs(qty)
            side = "LONG" if qty > 0 else "SHORT"

            print(f"  {symbol:10} | {qty:>6} ({side:5}) | Avg: ${avg_cost:.2f} | Action: {action} {close_qty}")

            orders_to_place.append({
                "contract": pos.contract,
                "action": action,
                "quantity": close_qty,
                "symbol": symbol,
            })

        print("="*60)

        if not execute:
            print("\n[DRY RUN] No orders placed. Use --execute to close positions.")
            return

        # Place market orders to close each position
        print(f"\nPlacing {len(orders_to_place)} market orders to close positions...")

        for order_info in orders_to_place:
            contract = order_info["contract"]
            action = order_info["action"]
            qty = order_info["quantity"]
            symbol = order_info["symbol"]

            # Fix contract for proper routing
            # For stocks/ETFs: use SMART routing to avoid direct route fees
            # For futures: ensure exchange is set
            if contract.secType == 'STK':
                contract = Stock(
                    symbol=contract.symbol,
                    exchange='SMART',
                    currency=contract.currency or 'USD'
                )
                # Qualify to get conId
                await ib.qualifyContractsAsync(contract)
            elif contract.secType == 'FUT':
                # Futures need exchange - use original or lookup
                if not contract.exchange:
                    futures_exchanges = {
                        'ES': 'CME', 'NQ': 'CME', 'YM': 'CBOT', 'RTY': 'CME',
                        'MES': 'CME', 'MNQ': 'CME', 'MYM': 'CBOT', 'M2K': 'CME',
                        'CL': 'NYMEX', 'GC': 'COMEX', 'SI': 'COMEX',
                        'ZC': 'CBOT', 'ZW': 'CBOT', 'ZS': 'CBOT', 'ZM': 'CBOT',
                        'MCL': 'NYMEX', 'MGC': 'COMEX', 'ZF': 'CBOT',
                    }
                    contract.exchange = futures_exchanges.get(symbol, 'SMART')

            # Create market order
            order = MarketOrder(action, qty)

            logger.info(f"Placing {action} {qty} {symbol} on {contract.exchange}...")
            trade = ib.placeOrder(contract, order)

            # Wait for fill (with timeout)
            timeout = 30
            start = datetime.now(timezone.utc)
            while not trade.isDone():
                await asyncio.sleep(0.5)
                elapsed = (datetime.now(timezone.utc) - start).total_seconds()
                if elapsed > timeout:
                    logger.warning(f"Order timeout for {symbol}")
                    break

            if trade.orderStatus.status == 'Filled':
                fill_price = trade.orderStatus.avgFillPrice
                logger.info(f"  Filled {symbol} @ ${fill_price:.2f}")
            else:
                logger.warning(f"  Order status: {trade.orderStatus.status}")

        print("\n" + "="*60)
        print("ALL POSITIONS CLOSED")
        print("="*60)

    except Exception as e:
        logger.error(f"Error: {e}")
        raise
    finally:
        ib.disconnect()
        logger.info("Disconnected from IB")


def main():
    parser = argparse.ArgumentParser(description="Close all open positions in IB")
    parser.add_argument('--dry-run', action='store_true', default=True,
                        help='Preview what will be closed (default)')
    parser.add_argument('--execute', action='store_true',
                        help='Actually close positions')

    args = parser.parse_args()

    execute = args.execute

    if execute:
        print("\n" + "!"*60)
        print("WARNING: This will CLOSE ALL POSITIONS with MARKET ORDERS!")
        print("!"*60)
        confirm = input("\nType 'YES' to confirm: ")
        if confirm != 'YES':
            print("Aborted.")
            return

    asyncio.run(close_all_positions(execute=execute))


if __name__ == "__main__":
    main()
