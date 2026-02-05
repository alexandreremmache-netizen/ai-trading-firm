#!/usr/bin/env python3
"""
Place Stop-Loss Orders for Existing Positions
==============================================

This script calculates and places stop-loss orders on IB for all existing
positions that don't have protective stops.

Usage:
    python scripts/place_stop_orders.py [--dry-run] [--stop-pct 2.0]

Options:
    --dry-run       Show what would be done without placing orders
    --stop-pct      Stop-loss percentage (default: 2.0%)
    --atr-mult      ATR multiplier for dynamic stops (default: 2.5)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ib_insync import IB, Stock, Future, Forex, StopOrder, Contract
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


class StopOrderPlacer:
    """Places stop-loss orders for existing positions on IB."""

    def __init__(
        self,
        config_path: str = "config.yaml",
        dry_run: bool = True,
        stop_pct: float = 2.0,
        atr_multiplier: float = 2.5,
    ):
        self.config_path = config_path
        self.dry_run = dry_run
        self.stop_pct = stop_pct
        self.atr_multiplier = atr_multiplier
        self.ib = IB()
        self.config = self._load_config()

    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)

    async def connect(self) -> bool:
        """Connect to IB Gateway/TWS."""
        broker_config = self.config.get("broker", {})
        host = broker_config.get("host", "127.0.0.1")
        port = broker_config.get("port", 7497)
        client_id = broker_config.get("client_id", 1) + 100  # Use different client ID

        try:
            await self.ib.connectAsync(host, port, clientId=client_id)
            logger.info(f"Connected to IB at {host}:{port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to IB: {e}")
            return False

    def disconnect(self):
        """Disconnect from IB."""
        if self.ib.isConnected():
            self.ib.disconnect()
            logger.info("Disconnected from IB")

    def _create_contract(self, symbol: str) -> Contract | None:
        """Create IB contract from symbol."""
        # Forex pairs
        forex_symbols = {"EUR", "GBP", "JPY", "CHF", "AUD", "CAD", "NZD"}
        if symbol in forex_symbols:
            return Forex(f"{symbol}USD" if symbol != "JPY" else "USDJPY")

        # Futures
        futures_symbols = {
            "ES", "NQ", "YM", "RTY",  # E-mini
            "MES", "MNQ", "MYM", "M2K",  # Micro index
            "CL", "MCL", "NG", "RB", "HO",  # Energy
            "GC", "MGC", "SI", "SIL", "PL", "HG",  # Metals
            "ZC", "ZW", "ZS", "ZM", "ZL",  # Agriculture
            "ZB", "ZN", "ZF",  # Bonds
        }
        if symbol in futures_symbols:
            # Get exchange mapping
            exchange_map = {
                "ES": "CME", "NQ": "CME", "RTY": "CME",
                "MES": "CME", "MNQ": "CME", "M2K": "CME",
                "YM": "CBOT", "MYM": "CBOT",
                "CL": "NYMEX", "MCL": "NYMEX", "NG": "NYMEX", "RB": "NYMEX", "HO": "NYMEX",
                "GC": "COMEX", "MGC": "COMEX", "SI": "COMEX", "SIL": "COMEX", "HG": "COMEX",
                "PL": "NYMEX",
                "ZC": "CBOT", "ZW": "CBOT", "ZS": "CBOT", "ZM": "CBOT", "ZL": "CBOT",
                "ZB": "CBOT", "ZN": "CBOT", "ZF": "CBOT",
            }
            exchange = exchange_map.get(symbol, "CME")
            return Future(symbol=symbol, exchange=exchange)

        # Default to stock
        return Stock(symbol, "SMART", "USD")

    def _calculate_stop_price(
        self,
        current_price: float,
        is_long: bool,
        atr: float | None = None,
    ) -> float:
        """Calculate stop-loss price."""
        if atr and atr > 0:
            # ATR-based stop
            stop_distance = atr * self.atr_multiplier
        else:
            # Percentage-based stop
            stop_distance = current_price * (self.stop_pct / 100)

        if is_long:
            stop_price = current_price - stop_distance
        else:
            stop_price = current_price + stop_distance

        # Round to 2 decimal places
        return round(stop_price, 2)

    async def get_existing_orders(self) -> dict[str, list]:
        """Get existing open orders grouped by symbol."""
        orders = self.ib.openOrders()
        orders_by_symbol: dict[str, list] = {}

        for order in orders:
            trade = self.ib.trades()
            for t in trade:
                if t.order.orderId == order.orderId:
                    symbol = t.contract.symbol
                    if symbol not in orders_by_symbol:
                        orders_by_symbol[symbol] = []
                    orders_by_symbol[symbol].append({
                        "order_id": order.orderId,
                        "order_type": order.orderType,
                        "action": order.action,
                        "quantity": order.totalQuantity,
                        "aux_price": getattr(order, "auxPrice", None),
                    })
                    break

        return orders_by_symbol

    async def place_stop_orders(self) -> dict:
        """Place stop-loss orders for all positions without stops."""
        results = {
            "processed": 0,
            "placed": 0,
            "skipped": 0,
            "errors": 0,
            "details": [],
        }

        # Get current positions
        positions = self.ib.positions()
        if not positions:
            logger.info("No positions found")
            return results

        logger.info(f"Found {len(positions)} positions")

        # Get existing orders to avoid duplicates
        existing_orders = await self.get_existing_orders()

        for pos in positions:
            symbol = pos.contract.symbol
            quantity = pos.position
            avg_cost = pos.avgCost
            results["processed"] += 1

            if quantity == 0:
                continue

            is_long = quantity > 0
            abs_qty = abs(quantity)

            # Check if stop order already exists
            symbol_orders = existing_orders.get(symbol, [])
            has_stop = any(
                o["order_type"] == "STP" and
                o["action"] == ("SELL" if is_long else "BUY")
                for o in symbol_orders
            )

            if has_stop:
                logger.info(f"  {symbol}: Stop order already exists, skipping")
                results["skipped"] += 1
                results["details"].append({
                    "symbol": symbol,
                    "status": "skipped",
                    "reason": "Stop order already exists",
                })
                continue

            # Get current market price
            contract = pos.contract
            try:
                await self.ib.qualifyContractsAsync(contract)
            except Exception as e:
                logger.warning(f"  {symbol}: Could not qualify contract: {e}")
                current_price = avg_cost

            try:
                ticker = self.ib.reqMktData(contract, "", False, False)
                await asyncio.sleep(2)  # Wait for data

                current_price = ticker.last or ticker.close or avg_cost
                if current_price <= 0:
                    current_price = avg_cost

                self.ib.cancelMktData(contract)
            except Exception as e:
                logger.warning(f"  {symbol}: Could not get market data: {e}")
                current_price = avg_cost

            # Calculate stop price
            stop_price = self._calculate_stop_price(current_price, is_long)

            # Determine order action (opposite of position)
            action = "SELL" if is_long else "BUY"

            detail = {
                "symbol": symbol,
                "position": quantity,
                "is_long": is_long,
                "current_price": current_price,
                "stop_price": stop_price,
                "action": action,
            }

            if self.dry_run:
                logger.info(
                    f"  [DRY-RUN] {symbol}: Would place {action} STOP @ {stop_price:.2f} "
                    f"for {abs_qty} units (current: {current_price:.2f})"
                )
                detail["status"] = "dry_run"
                results["details"].append(detail)
            else:
                try:
                    # Create and place stop order
                    stop_order = StopOrder(
                        action=action,
                        totalQuantity=abs_qty,
                        stopPrice=stop_price,
                        tif="GTC",  # Good Till Cancelled
                    )

                    trade = self.ib.placeOrder(contract, stop_order)
                    await asyncio.sleep(0.5)  # Wait for order acknowledgment

                    logger.info(
                        f"  {symbol}: Placed {action} STOP @ {stop_price:.2f} "
                        f"for {abs_qty} units (Order ID: {trade.order.orderId})"
                    )

                    detail["status"] = "placed"
                    detail["order_id"] = trade.order.orderId
                    results["placed"] += 1

                except Exception as e:
                    logger.error(f"  {symbol}: Failed to place stop order: {e}")
                    detail["status"] = "error"
                    detail["error"] = str(e)
                    results["errors"] += 1

                results["details"].append(detail)

        return results


async def main():
    parser = argparse.ArgumentParser(
        description="Place stop-loss orders for existing positions"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Show what would be done without placing orders (default: True)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually place the orders (overrides --dry-run)",
    )
    parser.add_argument(
        "--stop-pct",
        type=float,
        default=2.0,
        help="Stop-loss percentage (default: 2.0%%)",
    )
    parser.add_argument(
        "--atr-mult",
        type=float,
        default=2.5,
        help="ATR multiplier for dynamic stops (default: 2.5)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file",
    )

    args = parser.parse_args()

    dry_run = not args.execute

    logger.info("=" * 60)
    logger.info("STOP-LOSS ORDER PLACEMENT SCRIPT")
    logger.info("=" * 60)
    logger.info(f"Mode: {'DRY-RUN (no orders will be placed)' if dry_run else 'LIVE (orders will be placed)'}")
    logger.info(f"Stop percentage: {args.stop_pct}%")
    logger.info(f"ATR multiplier: {args.atr_mult}")
    logger.info("=" * 60)

    placer = StopOrderPlacer(
        config_path=args.config,
        dry_run=dry_run,
        stop_pct=args.stop_pct,
        atr_multiplier=args.atr_mult,
    )

    if not await placer.connect():
        logger.error("Could not connect to IB. Make sure TWS/Gateway is running.")
        return 1

    try:
        results = await placer.place_stop_orders()

        logger.info("=" * 60)
        logger.info("RESULTS")
        logger.info("=" * 60)
        logger.info(f"Positions processed: {results['processed']}")
        logger.info(f"Stop orders placed:  {results['placed']}")
        logger.info(f"Skipped (existing):  {results['skipped']}")
        logger.info(f"Errors:              {results['errors']}")

        if dry_run and results['processed'] > 0:
            logger.info("")
            logger.info("To actually place orders, run with --execute flag:")
            logger.info("  python scripts/place_stop_orders.py --execute")

        return 0 if results['errors'] == 0 else 1

    finally:
        placer.disconnect()


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
