"""
Contract Specifications
=======================

Futures contract specifications for proper position sizing, margin calculation,
and P&L computation. This module provides institutional-grade contract data.

Based on CME/CBOT/NYMEX/COMEX specifications.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from functools import lru_cache
from typing import Any


logger = logging.getLogger(__name__)


class AssetClass(Enum):
    """Asset class categorization for futures."""
    INDEX = "index"
    ENERGY = "energy"
    PRECIOUS_METALS = "precious_metals"
    BASE_METALS = "base_metals"
    AGRICULTURE = "agriculture"
    BONDS = "bonds"
    FOREX = "forex"


class Exchange(Enum):
    """Exchange identifiers."""
    CME = "CME"
    CBOT = "CBOT"
    NYMEX = "NYMEX"
    COMEX = "COMEX"
    GLOBEX = "GLOBEX"
    IDEALPRO = "IDEALPRO"  # IB forex ECN


@dataclass(frozen=True)
class ContractSpec:
    """
    Futures contract specification.

    All values are per standard contract unless otherwise noted.
    """
    symbol: str
    name: str
    asset_class: AssetClass
    exchange: Exchange
    currency: str

    # Contract sizing
    multiplier: float  # Dollar value per point
    tick_size: float  # Minimum price increment
    tick_value: float  # Dollar value per tick

    # Margins (approximate, varies by broker/clearinghouse)
    initial_margin: float  # Initial margin requirement
    maintenance_margin: float  # Maintenance margin requirement

    # Trading specifications
    trading_hours: str  # Description of trading hours
    contract_months: tuple[str, ...]  # Standard contract months
    last_trading_day: str  # Description of last trading day

    # Settlement
    settlement_type: str  # "cash" or "physical"

    # Position limits (if applicable)
    position_limit: int | None = None

    # Average daily volume (for liquidity checks)
    avg_daily_volume: int = 100000  # Default estimate

    # Point value for quick calculations
    @property
    def point_value(self) -> float:
        """Dollar value of one point move."""
        return self.multiplier

    def calculate_tick_value(self) -> float:
        """Calculate tick value from multiplier and tick size."""
        return self.multiplier * self.tick_size

    def notional_value(self, price: float, contracts: int = 1) -> float:
        """Calculate notional value of position."""
        return price * self.multiplier * contracts

    def margin_required(self, contracts: int, is_initial: bool = True) -> float:
        """Calculate margin required for position."""
        margin = self.initial_margin if is_initial else self.maintenance_margin
        return margin * abs(contracts)

    def pnl_per_tick(self, contracts: int) -> float:
        """Calculate P&L per tick for given number of contracts."""
        return self.tick_value * contracts

    def price_to_ticks(self, price_change: float) -> int:
        """Convert price change to number of ticks."""
        return int(round(price_change / self.tick_size))

    def ticks_to_price(self, ticks: int) -> float:
        """Convert number of ticks to price change."""
        return ticks * self.tick_size


# =============================================================================
# CONTRACT SPECIFICATIONS DATABASE
# =============================================================================

CONTRACT_SPECS: dict[str, ContractSpec] = {
    # =========================================================================
    # INDEX FUTURES
    # =========================================================================
    "ES": ContractSpec(
        symbol="ES",
        name="E-mini S&P 500",
        asset_class=AssetClass.INDEX,
        exchange=Exchange.CME,
        currency="USD",
        multiplier=50.0,
        tick_size=0.25,
        tick_value=12.50,
        initial_margin=12_650.0,
        maintenance_margin=11_500.0,
        trading_hours="Sun-Fri 6:00pm-5:00pm ET",
        contract_months=("H", "M", "U", "Z"),  # Mar, Jun, Sep, Dec
        last_trading_day="3rd Friday of contract month",
        settlement_type="cash",
        avg_daily_volume=1_500_000,
    ),

    "NQ": ContractSpec(
        symbol="NQ",
        name="E-mini NASDAQ 100",
        asset_class=AssetClass.INDEX,
        exchange=Exchange.CME,
        currency="USD",
        multiplier=20.0,
        tick_size=0.25,
        tick_value=5.0,
        initial_margin=17_600.0,
        maintenance_margin=16_000.0,
        trading_hours="Sun-Fri 6:00pm-5:00pm ET",
        contract_months=("H", "M", "U", "Z"),
        last_trading_day="3rd Friday of contract month",
        settlement_type="cash",
        avg_daily_volume=800_000,
    ),

    "YM": ContractSpec(
        symbol="YM",
        name="E-mini Dow",
        asset_class=AssetClass.INDEX,
        exchange=Exchange.CBOT,
        currency="USD",
        multiplier=5.0,
        tick_size=1.0,
        tick_value=5.0,
        initial_margin=9_350.0,
        maintenance_margin=8_500.0,
        trading_hours="Sun-Fri 6:00pm-5:00pm ET",
        contract_months=("H", "M", "U", "Z"),
        last_trading_day="3rd Friday of contract month",
        settlement_type="cash",
        avg_daily_volume=150_000,
    ),

    "RTY": ContractSpec(
        symbol="RTY",
        name="E-mini Russell 2000",
        asset_class=AssetClass.INDEX,
        exchange=Exchange.CME,
        currency="USD",
        multiplier=50.0,
        tick_size=0.1,
        tick_value=5.0,
        initial_margin=7_150.0,
        maintenance_margin=6_500.0,
        trading_hours="Sun-Fri 6:00pm-5:00pm ET",
        contract_months=("H", "M", "U", "Z"),
        last_trading_day="3rd Friday of contract month",
        settlement_type="cash",
        avg_daily_volume=250_000,
    ),

    # =========================================================================
    # ENERGY FUTURES
    # =========================================================================
    "CL": ContractSpec(
        symbol="CL",
        name="Crude Oil WTI",
        asset_class=AssetClass.ENERGY,
        exchange=Exchange.NYMEX,
        currency="USD",
        multiplier=1000.0,  # 1,000 barrels
        tick_size=0.01,
        tick_value=10.0,
        initial_margin=6_600.0,
        maintenance_margin=6_000.0,
        trading_hours="Sun-Fri 6:00pm-5:00pm ET",
        contract_months=("F", "G", "H", "J", "K", "M", "N", "Q", "U", "V", "X", "Z"),
        last_trading_day="3 business days before 25th of month prior to delivery",
        settlement_type="physical",
        avg_daily_volume=500_000,
    ),

    "NG": ContractSpec(
        symbol="NG",
        name="Natural Gas",
        asset_class=AssetClass.ENERGY,
        exchange=Exchange.NYMEX,
        currency="USD",
        multiplier=10000.0,  # 10,000 mmBtu
        tick_size=0.001,
        tick_value=10.0,
        initial_margin=4_400.0,
        maintenance_margin=4_000.0,
        trading_hours="Sun-Fri 6:00pm-5:00pm ET",
        contract_months=("F", "G", "H", "J", "K", "M", "N", "Q", "U", "V", "X", "Z"),
        last_trading_day="3 business days before 1st of delivery month",
        settlement_type="physical",
        avg_daily_volume=400_000,
    ),

    "RB": ContractSpec(
        symbol="RB",
        name="RBOB Gasoline",
        asset_class=AssetClass.ENERGY,
        exchange=Exchange.NYMEX,
        currency="USD",
        multiplier=42000.0,  # 42,000 gallons
        tick_size=0.0001,
        tick_value=4.20,
        initial_margin=6_050.0,
        maintenance_margin=5_500.0,
        trading_hours="Sun-Fri 6:00pm-5:00pm ET",
        contract_months=("F", "G", "H", "J", "K", "M", "N", "Q", "U", "V", "X", "Z"),
        last_trading_day="Last business day of month prior to delivery",
        settlement_type="physical",
        avg_daily_volume=100_000,
    ),

    "HO": ContractSpec(
        symbol="HO",
        name="Heating Oil",
        asset_class=AssetClass.ENERGY,
        exchange=Exchange.NYMEX,
        currency="USD",
        multiplier=42000.0,  # 42,000 gallons
        tick_size=0.0001,
        tick_value=4.20,
        initial_margin=6_050.0,
        maintenance_margin=5_500.0,
        trading_hours="Sun-Fri 6:00pm-5:00pm ET",
        contract_months=("F", "G", "H", "J", "K", "M", "N", "Q", "U", "V", "X", "Z"),
        last_trading_day="Last business day of month prior to delivery",
        settlement_type="physical",
        avg_daily_volume=100_000,
    ),

    # =========================================================================
    # PRECIOUS METALS FUTURES
    # =========================================================================
    "GC": ContractSpec(
        symbol="GC",
        name="Gold",
        asset_class=AssetClass.PRECIOUS_METALS,
        exchange=Exchange.COMEX,
        currency="USD",
        multiplier=100.0,  # 100 troy ounces
        tick_size=0.10,
        tick_value=10.0,
        initial_margin=9_900.0,
        maintenance_margin=9_000.0,
        trading_hours="Sun-Fri 6:00pm-5:00pm ET",
        contract_months=("G", "J", "M", "Q", "V", "Z"),  # Feb, Apr, Jun, Aug, Oct, Dec
        last_trading_day="3rd to last business day of delivery month",
        settlement_type="physical",
        avg_daily_volume=250_000,
    ),

    "SI": ContractSpec(
        symbol="SI",
        name="Silver",
        asset_class=AssetClass.PRECIOUS_METALS,
        exchange=Exchange.COMEX,
        currency="USD",
        multiplier=5000.0,  # 5,000 troy ounces
        tick_size=0.005,
        tick_value=25.0,
        initial_margin=11_550.0,
        maintenance_margin=10_500.0,
        trading_hours="Sun-Fri 6:00pm-5:00pm ET",
        contract_months=("H", "K", "N", "U", "Z"),  # Mar, May, Jul, Sep, Dec
        last_trading_day="3rd to last business day of delivery month",
        settlement_type="physical",
        avg_daily_volume=100_000,
    ),

    "PL": ContractSpec(
        symbol="PL",
        name="Platinum",
        asset_class=AssetClass.PRECIOUS_METALS,
        exchange=Exchange.NYMEX,
        currency="USD",
        multiplier=50.0,  # 50 troy ounces
        tick_size=0.10,
        tick_value=5.0,
        initial_margin=4_950.0,
        maintenance_margin=4_500.0,
        trading_hours="Sun-Fri 6:00pm-5:00pm ET",
        contract_months=("F", "J", "N", "V"),  # Jan, Apr, Jul, Oct
        last_trading_day="3rd to last business day of delivery month",
        settlement_type="physical",
        avg_daily_volume=20_000,
    ),

    "HG": ContractSpec(
        symbol="HG",
        name="Copper",
        asset_class=AssetClass.BASE_METALS,
        exchange=Exchange.COMEX,
        currency="USD",
        multiplier=25000.0,  # 25,000 pounds
        tick_size=0.0005,
        tick_value=12.50,
        initial_margin=6_600.0,
        maintenance_margin=6_000.0,
        trading_hours="Sun-Fri 6:00pm-5:00pm ET",
        contract_months=("H", "K", "N", "U", "Z"),  # Mar, May, Jul, Sep, Dec
        last_trading_day="3rd to last business day of delivery month",
        settlement_type="physical",
        avg_daily_volume=80_000,
    ),

    # =========================================================================
    # AGRICULTURE FUTURES
    # =========================================================================
    "ZC": ContractSpec(
        symbol="ZC",
        name="Corn",
        asset_class=AssetClass.AGRICULTURE,
        exchange=Exchange.CBOT,
        currency="USD",
        multiplier=5000.0,  # 5,000 bushels
        tick_size=0.0025,  # 1/4 cent per bushel = $0.0025 (prices in dollars)
        tick_value=12.50,  # 5000 * 0.0025 = $12.50 per tick
        initial_margin=1_650.0,
        maintenance_margin=1_500.0,
        trading_hours="Sun-Fri 7:00pm-7:45am, 8:30am-1:20pm CT",
        contract_months=("H", "K", "N", "U", "Z"),  # Mar, May, Jul, Sep, Dec
        last_trading_day="Business day prior to 15th of contract month",
        settlement_type="physical",
        avg_daily_volume=300_000,
    ),

    "ZW": ContractSpec(
        symbol="ZW",
        name="Wheat",
        asset_class=AssetClass.AGRICULTURE,
        exchange=Exchange.CBOT,
        currency="USD",
        multiplier=5000.0,  # 5,000 bushels
        tick_size=0.0025,  # 1/4 cent per bushel = $0.0025 (prices in dollars)
        tick_value=12.50,  # 5000 * 0.0025 = $12.50 per tick
        initial_margin=2_200.0,
        maintenance_margin=2_000.0,
        trading_hours="Sun-Fri 7:00pm-7:45am, 8:30am-1:20pm CT",
        contract_months=("H", "K", "N", "U", "Z"),  # Mar, May, Jul, Sep, Dec
        last_trading_day="Business day prior to 15th of contract month",
        settlement_type="physical",
        avg_daily_volume=150_000,
    ),

    "ZS": ContractSpec(
        symbol="ZS",
        name="Soybeans",
        asset_class=AssetClass.AGRICULTURE,
        exchange=Exchange.CBOT,
        currency="USD",
        multiplier=5000.0,  # 5,000 bushels
        tick_size=0.0025,  # 1/4 cent per bushel = $0.0025 (prices in dollars)
        tick_value=12.50,  # 5000 * 0.0025 = $12.50 per tick
        initial_margin=3_300.0,
        maintenance_margin=3_000.0,
        trading_hours="Sun-Fri 7:00pm-7:45am, 8:30am-1:20pm CT",
        contract_months=("F", "H", "K", "N", "Q", "U", "X"),  # Jan, Mar, May, Jul, Aug, Sep, Nov
        last_trading_day="Business day prior to 15th of contract month",
        settlement_type="physical",
        avg_daily_volume=200_000,
    ),

    "ZM": ContractSpec(
        symbol="ZM",
        name="Soybean Meal",
        asset_class=AssetClass.AGRICULTURE,
        exchange=Exchange.CBOT,
        currency="USD",
        multiplier=100.0,  # 100 short tons
        tick_size=0.10,
        tick_value=10.0,
        initial_margin=2_750.0,
        maintenance_margin=2_500.0,
        trading_hours="Sun-Fri 7:00pm-7:45am, 8:30am-1:20pm CT",
        contract_months=("F", "H", "K", "N", "Q", "U", "V", "Z"),  # Jan, Mar, May, Jul, Aug, Sep, Oct, Dec
        last_trading_day="Business day prior to 15th of contract month",
        settlement_type="physical",
        avg_daily_volume=100_000,
    ),

    "ZL": ContractSpec(
        symbol="ZL",
        name="Soybean Oil",
        asset_class=AssetClass.AGRICULTURE,
        exchange=Exchange.CBOT,
        currency="USD",
        multiplier=60000.0,  # 60,000 pounds
        tick_size=0.0001,  # $0.0001 per pound (0.01 cents per pound per CME)
        tick_value=6.0,  # 60,000 * 0.0001 = $6.00 per tick
        initial_margin=2_420.0,
        maintenance_margin=2_200.0,
        trading_hours="Sun-Fri 7:00pm-7:45am, 8:30am-1:20pm CT",
        contract_months=("F", "H", "K", "N", "Q", "U", "V", "Z"),
        last_trading_day="Business day prior to 15th of contract month",
        settlement_type="physical",
        avg_daily_volume=80_000,
    ),

    # =========================================================================
    # BOND FUTURES
    # =========================================================================
    "ZB": ContractSpec(
        symbol="ZB",
        name="30-Year Treasury Bond",
        asset_class=AssetClass.BONDS,
        exchange=Exchange.CBOT,
        currency="USD",
        multiplier=1000.0,  # $100,000 face value / 100
        tick_size=0.03125,  # 1/32 of a point
        tick_value=31.25,
        initial_margin=4_400.0,
        maintenance_margin=4_000.0,
        trading_hours="Sun-Fri 6:00pm-5:00pm ET",
        contract_months=("H", "M", "U", "Z"),  # Mar, Jun, Sep, Dec
        last_trading_day="7 business days before last business day of delivery month",
        settlement_type="physical",
        avg_daily_volume=400_000,
    ),

    "ZN": ContractSpec(
        symbol="ZN",
        name="10-Year Treasury Note",
        asset_class=AssetClass.BONDS,
        exchange=Exchange.CBOT,
        currency="USD",
        multiplier=1000.0,  # $100,000 face value / 100
        tick_size=0.015625,  # 1/64 of a point
        tick_value=15.625,
        initial_margin=2_090.0,
        maintenance_margin=1_900.0,
        trading_hours="Sun-Fri 6:00pm-5:00pm ET",
        contract_months=("H", "M", "U", "Z"),
        last_trading_day="7 business days before last business day of delivery month",
        settlement_type="physical",
        avg_daily_volume=1_500_000,
    ),

    "ZF": ContractSpec(
        symbol="ZF",
        name="5-Year Treasury Note",
        asset_class=AssetClass.BONDS,
        exchange=Exchange.CBOT,
        currency="USD",
        multiplier=1000.0,  # $100,000 face value / 100
        tick_size=0.0078125,  # 1/128 of a point
        tick_value=7.8125,
        initial_margin=1_210.0,
        maintenance_margin=1_100.0,
        trading_hours="Sun-Fri 6:00pm-5:00pm ET",
        contract_months=("H", "M", "U", "Z"),
        last_trading_day="Last business day of delivery month",
        settlement_type="physical",
        avg_daily_volume=600_000,
    ),

    # =========================================================================
    # FOREX (IB IDEALPRO Cash FX)
    # =========================================================================
    # Note: Forex trading on IB IDEALPRO uses standard lots (100,000 base currency)
    # Margin requirements are typically 2-5% depending on pair and account type

    "EUR": ContractSpec(
        symbol="EUR",
        name="Euro (EUR/USD)",
        asset_class=AssetClass.FOREX,
        exchange=Exchange.IDEALPRO,
        currency="USD",
        multiplier=100_000.0,  # Standard lot = 100,000 EUR
        tick_size=0.00005,  # Half pip (0.5 pips)
        tick_value=5.0,  # $5 per half pip for standard lot
        initial_margin=3_000.0,  # ~3% of notional (varies)
        maintenance_margin=2_500.0,
        trading_hours="Sun 5:00pm - Fri 5:00pm ET",
        contract_months=("SPOT",),  # Spot FX, no delivery months
        last_trading_day="N/A - Spot FX",
        settlement_type="cash",
        avg_daily_volume=500_000_000,  # FX is extremely liquid
    ),

    "GBP": ContractSpec(
        symbol="GBP",
        name="British Pound (GBP/USD)",
        asset_class=AssetClass.FOREX,
        exchange=Exchange.IDEALPRO,
        currency="USD",
        multiplier=100_000.0,  # Standard lot = 100,000 GBP
        tick_size=0.00005,  # Half pip
        tick_value=5.0,
        initial_margin=3_500.0,  # Slightly higher margin for GBP
        maintenance_margin=3_000.0,
        trading_hours="Sun 5:00pm - Fri 5:00pm ET",
        contract_months=("SPOT",),
        last_trading_day="N/A - Spot FX",
        settlement_type="cash",
        avg_daily_volume=300_000_000,
    ),

    "JPY": ContractSpec(
        symbol="JPY",
        name="Japanese Yen (USD/JPY)",
        asset_class=AssetClass.FOREX,
        exchange=Exchange.IDEALPRO,
        currency="JPY",  # Quote currency is JPY
        multiplier=100_000.0,  # Standard lot = 100,000 USD
        tick_size=0.005,  # Half pip for JPY pairs (0.5 pips)
        tick_value=500.0,  # Tick value in JPY
        initial_margin=2_500.0,  # In USD equivalent
        maintenance_margin=2_000.0,
        trading_hours="Sun 5:00pm - Fri 5:00pm ET",
        contract_months=("SPOT",),
        last_trading_day="N/A - Spot FX",
        settlement_type="cash",
        avg_daily_volume=400_000_000,
    ),

    "CHF": ContractSpec(
        symbol="CHF",
        name="Swiss Franc (USD/CHF)",
        asset_class=AssetClass.FOREX,
        exchange=Exchange.IDEALPRO,
        currency="CHF",  # Quote currency is CHF
        multiplier=100_000.0,  # Standard lot = 100,000 USD
        tick_size=0.00005,  # Half pip
        tick_value=5.0,  # Approximate tick value in CHF
        initial_margin=3_000.0,
        maintenance_margin=2_500.0,
        trading_hours="Sun 5:00pm - Fri 5:00pm ET",
        contract_months=("SPOT",),
        last_trading_day="N/A - Spot FX",
        settlement_type="cash",
        avg_daily_volume=200_000_000,
    ),

    "AUD": ContractSpec(
        symbol="AUD",
        name="Australian Dollar (AUD/USD)",
        asset_class=AssetClass.FOREX,
        exchange=Exchange.IDEALPRO,
        currency="USD",
        multiplier=100_000.0,  # Standard lot = 100,000 AUD
        tick_size=0.00005,  # Half pip
        tick_value=5.0,
        initial_margin=2_500.0,
        maintenance_margin=2_000.0,
        trading_hours="Sun 5:00pm - Fri 5:00pm ET",
        contract_months=("SPOT",),
        last_trading_day="N/A - Spot FX",
        settlement_type="cash",
        avg_daily_volume=150_000_000,
    ),

    "CAD": ContractSpec(
        symbol="CAD",
        name="Canadian Dollar (USD/CAD)",
        asset_class=AssetClass.FOREX,
        exchange=Exchange.IDEALPRO,
        currency="CAD",  # Quote currency is CAD
        multiplier=100_000.0,  # Standard lot = 100,000 USD
        tick_size=0.00005,  # Half pip
        tick_value=5.0,  # Approximate tick value in CAD
        initial_margin=2_500.0,
        maintenance_margin=2_000.0,
        trading_hours="Sun 5:00pm - Fri 5:00pm ET",
        contract_months=("SPOT",),
        last_trading_day="N/A - Spot FX",
        settlement_type="cash",
        avg_daily_volume=180_000_000,
    ),

    "NZD": ContractSpec(
        symbol="NZD",
        name="New Zealand Dollar (NZD/USD)",
        asset_class=AssetClass.FOREX,
        exchange=Exchange.IDEALPRO,
        currency="USD",
        multiplier=100_000.0,  # Standard lot = 100,000 NZD
        tick_size=0.00005,  # Half pip
        tick_value=5.0,
        initial_margin=2_200.0,
        maintenance_margin=1_800.0,
        trading_hours="Sun 5:00pm - Fri 5:00pm ET",
        contract_months=("SPOT",),
        last_trading_day="N/A - Spot FX",
        settlement_type="cash",
        avg_daily_volume=50_000_000,
    ),

    # =========================================================================
    # FOREX CROSS PAIRS (Non-USD pairs)
    # =========================================================================
    # Note: Cross pairs are synthetic in some systems, traded directly on IDEALPRO

    "EURGBP": ContractSpec(
        symbol="EURGBP",
        name="Euro / British Pound",
        asset_class=AssetClass.FOREX,
        exchange=Exchange.IDEALPRO,
        currency="GBP",  # Quote currency is GBP
        multiplier=100_000.0,  # Standard lot = 100,000 EUR
        tick_size=0.00005,  # Half pip
        tick_value=5.0,  # Approximate tick value in GBP
        initial_margin=3_200.0,
        maintenance_margin=2_800.0,
        trading_hours="Sun 5:00pm - Fri 5:00pm ET",
        contract_months=("SPOT",),
        last_trading_day="N/A - Spot FX",
        settlement_type="cash",
        avg_daily_volume=120_000_000,
    ),

    "EURJPY": ContractSpec(
        symbol="EURJPY",
        name="Euro / Japanese Yen",
        asset_class=AssetClass.FOREX,
        exchange=Exchange.IDEALPRO,
        currency="JPY",  # Quote currency is JPY
        multiplier=100_000.0,  # Standard lot = 100,000 EUR
        tick_size=0.005,  # Half pip for JPY pairs
        tick_value=500.0,  # Tick value in JPY
        initial_margin=3_000.0,
        maintenance_margin=2_500.0,
        trading_hours="Sun 5:00pm - Fri 5:00pm ET",
        contract_months=("SPOT",),
        last_trading_day="N/A - Spot FX",
        settlement_type="cash",
        avg_daily_volume=150_000_000,
    ),

    "GBPJPY": ContractSpec(
        symbol="GBPJPY",
        name="British Pound / Japanese Yen",
        asset_class=AssetClass.FOREX,
        exchange=Exchange.IDEALPRO,
        currency="JPY",  # Quote currency is JPY
        multiplier=100_000.0,  # Standard lot = 100,000 GBP
        tick_size=0.005,  # Half pip for JPY pairs
        tick_value=500.0,  # Tick value in JPY
        initial_margin=4_000.0,  # Higher margin due to volatility
        maintenance_margin=3_500.0,
        trading_hours="Sun 5:00pm - Fri 5:00pm ET",
        contract_months=("SPOT",),
        last_trading_day="N/A - Spot FX",
        settlement_type="cash",
        avg_daily_volume=100_000_000,
    ),

    "EURCHF": ContractSpec(
        symbol="EURCHF",
        name="Euro / Swiss Franc",
        asset_class=AssetClass.FOREX,
        exchange=Exchange.IDEALPRO,
        currency="CHF",  # Quote currency is CHF
        multiplier=100_000.0,  # Standard lot = 100,000 EUR
        tick_size=0.00005,  # Half pip
        tick_value=5.0,  # Tick value in CHF
        initial_margin=2_800.0,
        maintenance_margin=2_400.0,
        trading_hours="Sun 5:00pm - Fri 5:00pm ET",
        contract_months=("SPOT",),
        last_trading_day="N/A - Spot FX",
        settlement_type="cash",
        avg_daily_volume=80_000_000,
    ),

    "AUDJPY": ContractSpec(
        symbol="AUDJPY",
        name="Australian Dollar / Japanese Yen",
        asset_class=AssetClass.FOREX,
        exchange=Exchange.IDEALPRO,
        currency="JPY",  # Quote currency is JPY
        multiplier=100_000.0,  # Standard lot = 100,000 AUD
        tick_size=0.005,  # Half pip for JPY pairs
        tick_value=500.0,  # Tick value in JPY
        initial_margin=2_800.0,
        maintenance_margin=2_400.0,
        trading_hours="Sun 5:00pm - Fri 5:00pm ET",
        contract_months=("SPOT",),
        last_trading_day="N/A - Spot FX",
        settlement_type="cash",
        avg_daily_volume=60_000_000,
    ),

    "NZDJPY": ContractSpec(
        symbol="NZDJPY",
        name="New Zealand Dollar / Japanese Yen",
        asset_class=AssetClass.FOREX,
        exchange=Exchange.IDEALPRO,
        currency="JPY",  # Quote currency is JPY
        multiplier=100_000.0,  # Standard lot = 100,000 NZD
        tick_size=0.005,  # Half pip for JPY pairs
        tick_value=500.0,  # Tick value in JPY
        initial_margin=2_500.0,
        maintenance_margin=2_100.0,
        trading_hours="Sun 5:00pm - Fri 5:00pm ET",
        contract_months=("SPOT",),
        last_trading_day="N/A - Spot FX",
        settlement_type="cash",
        avg_daily_volume=30_000_000,
    ),

    "GBPCHF": ContractSpec(
        symbol="GBPCHF",
        name="British Pound / Swiss Franc",
        asset_class=AssetClass.FOREX,
        exchange=Exchange.IDEALPRO,
        currency="CHF",  # Quote currency is CHF
        multiplier=100_000.0,  # Standard lot = 100,000 GBP
        tick_size=0.00005,  # Half pip
        tick_value=5.0,  # Tick value in CHF
        initial_margin=3_500.0,
        maintenance_margin=3_000.0,
        trading_hours="Sun 5:00pm - Fri 5:00pm ET",
        contract_months=("SPOT",),
        last_trading_day="N/A - Spot FX",
        settlement_type="cash",
        avg_daily_volume=40_000_000,
    ),

    "AUDNZD": ContractSpec(
        symbol="AUDNZD",
        name="Australian Dollar / New Zealand Dollar",
        asset_class=AssetClass.FOREX,
        exchange=Exchange.IDEALPRO,
        currency="NZD",  # Quote currency is NZD
        multiplier=100_000.0,  # Standard lot = 100,000 AUD
        tick_size=0.00005,  # Half pip
        tick_value=5.0,  # Tick value in NZD
        initial_margin=2_200.0,
        maintenance_margin=1_800.0,
        trading_hours="Sun 5:00pm - Fri 5:00pm ET",
        contract_months=("SPOT",),
        last_trading_day="N/A - Spot FX",
        settlement_type="cash",
        avg_daily_volume=25_000_000,
    ),

    "EURAUD": ContractSpec(
        symbol="EURAUD",
        name="Euro / Australian Dollar",
        asset_class=AssetClass.FOREX,
        exchange=Exchange.IDEALPRO,
        currency="AUD",  # Quote currency is AUD
        multiplier=100_000.0,  # Standard lot = 100,000 EUR
        tick_size=0.00005,  # Half pip
        tick_value=5.0,  # Tick value in AUD
        initial_margin=3_200.0,
        maintenance_margin=2_800.0,
        trading_hours="Sun 5:00pm - Fri 5:00pm ET",
        contract_months=("SPOT",),
        last_trading_day="N/A - Spot FX",
        settlement_type="cash",
        avg_daily_volume=50_000_000,
    ),

    "GBPAUD": ContractSpec(
        symbol="GBPAUD",
        name="British Pound / Australian Dollar",
        asset_class=AssetClass.FOREX,
        exchange=Exchange.IDEALPRO,
        currency="AUD",  # Quote currency is AUD
        multiplier=100_000.0,  # Standard lot = 100,000 GBP
        tick_size=0.00005,  # Half pip
        tick_value=5.0,  # Tick value in AUD
        initial_margin=3_800.0,  # Higher margin due to volatility
        maintenance_margin=3_300.0,
        trading_hours="Sun 5:00pm - Fri 5:00pm ET",
        contract_months=("SPOT",),
        last_trading_day="N/A - Spot FX",
        settlement_type="cash",
        avg_daily_volume=35_000_000,
    ),

    "CADJPY": ContractSpec(
        symbol="CADJPY",
        name="Canadian Dollar / Japanese Yen",
        asset_class=AssetClass.FOREX,
        exchange=Exchange.IDEALPRO,
        currency="JPY",  # Quote currency is JPY
        multiplier=100_000.0,  # Standard lot = 100,000 CAD
        tick_size=0.005,  # Half pip for JPY pairs
        tick_value=500.0,  # Tick value in JPY
        initial_margin=2_600.0,
        maintenance_margin=2_200.0,
        trading_hours="Sun 5:00pm - Fri 5:00pm ET",
        contract_months=("SPOT",),
        last_trading_day="N/A - Spot FX",
        settlement_type="cash",
        avg_daily_volume=45_000_000,
    ),

    "CHFJPY": ContractSpec(
        symbol="CHFJPY",
        name="Swiss Franc / Japanese Yen",
        asset_class=AssetClass.FOREX,
        exchange=Exchange.IDEALPRO,
        currency="JPY",  # Quote currency is JPY
        multiplier=100_000.0,  # Standard lot = 100,000 CHF
        tick_size=0.005,  # Half pip for JPY pairs
        tick_value=500.0,  # Tick value in JPY
        initial_margin=2_800.0,
        maintenance_margin=2_400.0,
        trading_hours="Sun 5:00pm - Fri 5:00pm ET",
        contract_months=("SPOT",),
        last_trading_day="N/A - Spot FX",
        settlement_type="cash",
        avg_daily_volume=35_000_000,
    ),

    "EURCAD": ContractSpec(
        symbol="EURCAD",
        name="Euro / Canadian Dollar",
        asset_class=AssetClass.FOREX,
        exchange=Exchange.IDEALPRO,
        currency="CAD",  # Quote currency is CAD
        multiplier=100_000.0,  # Standard lot = 100,000 EUR
        tick_size=0.00005,  # Half pip
        tick_value=5.0,  # Tick value in CAD
        initial_margin=3_000.0,
        maintenance_margin=2_600.0,
        trading_hours="Sun 5:00pm - Fri 5:00pm ET",
        contract_months=("SPOT",),
        last_trading_day="N/A - Spot FX",
        settlement_type="cash",
        avg_daily_volume=55_000_000,
    ),

    "GBPCAD": ContractSpec(
        symbol="GBPCAD",
        name="British Pound / Canadian Dollar",
        asset_class=AssetClass.FOREX,
        exchange=Exchange.IDEALPRO,
        currency="CAD",  # Quote currency is CAD
        multiplier=100_000.0,  # Standard lot = 100,000 GBP
        tick_size=0.00005,  # Half pip
        tick_value=5.0,  # Tick value in CAD
        initial_margin=3_600.0,
        maintenance_margin=3_100.0,
        trading_hours="Sun 5:00pm - Fri 5:00pm ET",
        contract_months=("SPOT",),
        last_trading_day="N/A - Spot FX",
        settlement_type="cash",
        avg_daily_volume=40_000_000,
    ),

    "AUDCAD": ContractSpec(
        symbol="AUDCAD",
        name="Australian Dollar / Canadian Dollar",
        asset_class=AssetClass.FOREX,
        exchange=Exchange.IDEALPRO,
        currency="CAD",  # Quote currency is CAD
        multiplier=100_000.0,  # Standard lot = 100,000 AUD
        tick_size=0.00005,  # Half pip
        tick_value=5.0,  # Tick value in CAD
        initial_margin=2_400.0,
        maintenance_margin=2_000.0,
        trading_hours="Sun 5:00pm - Fri 5:00pm ET",
        contract_months=("SPOT",),
        last_trading_day="N/A - Spot FX",
        settlement_type="cash",
        avg_daily_volume=20_000_000,
    ),

    "AUDCHF": ContractSpec(
        symbol="AUDCHF",
        name="Australian Dollar / Swiss Franc",
        asset_class=AssetClass.FOREX,
        exchange=Exchange.IDEALPRO,
        currency="CHF",  # Quote currency is CHF
        multiplier=100_000.0,  # Standard lot = 100,000 AUD
        tick_size=0.00005,  # Half pip
        tick_value=5.0,  # Tick value in CHF
        initial_margin=2_600.0,
        maintenance_margin=2_200.0,
        trading_hours="Sun 5:00pm - Fri 5:00pm ET",
        contract_months=("SPOT",),
        last_trading_day="N/A - Spot FX",
        settlement_type="cash",
        avg_daily_volume=18_000_000,
    ),

    "EURNZD": ContractSpec(
        symbol="EURNZD",
        name="Euro / New Zealand Dollar",
        asset_class=AssetClass.FOREX,
        exchange=Exchange.IDEALPRO,
        currency="NZD",  # Quote currency is NZD
        multiplier=100_000.0,  # Standard lot = 100,000 EUR
        tick_size=0.00005,  # Half pip
        tick_value=5.0,  # Tick value in NZD
        initial_margin=3_400.0,
        maintenance_margin=2_900.0,
        trading_hours="Sun 5:00pm - Fri 5:00pm ET",
        contract_months=("SPOT",),
        last_trading_day="N/A - Spot FX",
        settlement_type="cash",
        avg_daily_volume=25_000_000,
    ),

    "GBPNZD": ContractSpec(
        symbol="GBPNZD",
        name="British Pound / New Zealand Dollar",
        asset_class=AssetClass.FOREX,
        exchange=Exchange.IDEALPRO,
        currency="NZD",  # Quote currency is NZD
        multiplier=100_000.0,  # Standard lot = 100,000 GBP
        tick_size=0.00005,  # Half pip
        tick_value=5.0,  # Tick value in NZD
        initial_margin=4_000.0,  # Higher margin due to volatility
        maintenance_margin=3_500.0,
        trading_hours="Sun 5:00pm - Fri 5:00pm ET",
        contract_months=("SPOT",),
        last_trading_day="N/A - Spot FX",
        settlement_type="cash",
        avg_daily_volume=22_000_000,
    ),
}


# =============================================================================
# CURRENCY CONVERSION
# =============================================================================

class CurrencyConverter:
    """
    Currency conversion for P&L calculation.

    Handles conversion of P&L from non-USD quote currencies to USD
    for proper portfolio aggregation.
    """

    def __init__(self):
        """Initialize with default exchange rates (should be updated with live rates)."""
        # Default rates (USD per 1 unit of foreign currency)
        # These should be updated with live market data
        self._rates_to_usd: dict[str, float] = {
            "USD": 1.0,
            "EUR": 1.08,  # EUR/USD
            "GBP": 1.26,  # GBP/USD
            "JPY": 0.0067,  # 1 JPY in USD (approx 150 JPY/USD)
            "CHF": 1.12,  # CHF/USD
            "AUD": 0.65,  # AUD/USD
            "CAD": 0.74,  # CAD/USD
            "NZD": 0.61,  # NZD/USD
        }
        self._last_update: datetime | None = None

    def update_rate(self, currency: str, rate_to_usd: float) -> None:
        """
        Update exchange rate for a currency.

        Args:
            currency: Currency code (e.g., "EUR", "JPY")
            rate_to_usd: How many USD per 1 unit of currency
        """
        self._rates_to_usd[currency.upper()] = rate_to_usd
        self._last_update = datetime.now(timezone.utc)

    def update_rates_from_market_data(self, fx_prices: dict[str, float]) -> None:
        """
        Update rates from FX market data.

        Args:
            fx_prices: Dictionary of pair -> price (e.g., {"EURUSD": 1.08, "USDJPY": 150.0})
        """
        for pair, price in fx_prices.items():
            pair = pair.upper().replace("/", "")
            if len(pair) == 6:
                base = pair[:3]
                quote = pair[3:]

                if quote == "USD":
                    # Direct quote: BASE/USD
                    self._rates_to_usd[base] = price
                elif base == "USD":
                    # Indirect quote: USD/QUOTE
                    self._rates_to_usd[quote] = 1.0 / price

        self._last_update = datetime.now(timezone.utc)

    def get_rate(self, currency: str) -> float:
        """
        Get exchange rate (USD per 1 unit of currency).

        Args:
            currency: Currency code

        Returns:
            Exchange rate to USD, or 1.0 if unknown
        """
        return self._rates_to_usd.get(currency.upper(), 1.0)

    def convert_to_usd(self, amount: float, from_currency: str) -> float:
        """
        Convert amount from foreign currency to USD.

        Args:
            amount: Amount in foreign currency
            from_currency: Source currency code

        Returns:
            Equivalent amount in USD
        """
        rate = self.get_rate(from_currency)
        return amount * rate

    def convert_pnl_to_usd(
        self,
        pnl: float,
        quote_currency: str,
        symbol: str | None = None
    ) -> float:
        """
        Convert P&L from quote currency to USD.

        Special handling for JPY pairs where tick values are in JPY.

        Args:
            pnl: P&L in quote currency
            quote_currency: Quote currency of the pair
            symbol: Optional symbol for logging

        Returns:
            P&L in USD
        """
        if quote_currency == "USD":
            return pnl

        usd_pnl = self.convert_to_usd(pnl, quote_currency)

        if symbol:
            logger.debug(
                f"P&L conversion for {symbol}: {pnl:.2f} {quote_currency} = "
                f"{usd_pnl:.2f} USD (rate: {self.get_rate(quote_currency):.6f})"
            )

        return usd_pnl

    def calculate_fx_position_value_usd(
        self,
        base_currency: str,
        quote_currency: str,
        base_amount: float,
        price: float
    ) -> float:
        """
        Calculate USD value of an FX position.

        Args:
            base_currency: Base currency (e.g., "EUR" in EUR/USD)
            quote_currency: Quote currency (e.g., "USD" in EUR/USD)
            base_amount: Amount of base currency
            price: Current price (quote per base)

        Returns:
            Position value in USD
        """
        # Value in quote currency
        quote_value = base_amount * price

        # Convert to USD if needed
        if quote_currency == "USD":
            return quote_value
        else:
            return self.convert_to_usd(quote_value, quote_currency)

    def get_tick_value_usd(self, spec: ContractSpec) -> float:
        """
        Get tick value in USD for a contract.

        Args:
            spec: Contract specification

        Returns:
            Tick value in USD
        """
        if spec.currency == "USD":
            return spec.tick_value

        return self.convert_to_usd(spec.tick_value, spec.currency)

    def get_fx_pip_value_usd(
        self,
        pair: str,
        lot_size: int = 100_000,
        current_price: float | None = None
    ) -> float:
        """
        Calculate FX pip value in USD (#X2).

        Properly handles JPY pairs where pip = 0.01 (vs 0.0001 for other pairs)
        and converts native pip value to USD.

        Args:
            pair: FX pair (e.g., "USDJPY", "EURUSD")
            lot_size: Position size in base currency units (default: 100,000 = 1 standard lot)
            current_price: Current price (required for XXX/USD pairs, optional for USD/XXX)

        Returns:
            Value of 1 pip move in USD

        Examples:
            USDJPY @ 150.00:
                - 1 pip = 0.01 (JPY pairs)
                - Pip value = 100,000 * 0.01 = 1,000 JPY
                - In USD = 1,000 / 150 = $6.67

            EURUSD @ 1.0800:
                - 1 pip = 0.0001 (standard pairs)
                - Pip value = 100,000 * 0.0001 = 10 USD
                - In USD = $10.00 (already in USD)
        """
        pair = pair.upper().replace("/", "")
        if len(pair) != 6:
            logger.warning(f"Invalid FX pair format: {pair}")
            return 0.0

        base = pair[:3]
        quote = pair[3:]

        # Determine pip size based on whether it's a JPY pair
        is_jpy_pair = quote == "JPY"
        pip_size = 0.01 if is_jpy_pair else 0.0001

        # Calculate pip value in quote currency
        pip_value_quote = lot_size * pip_size

        # Convert to USD
        if quote == "USD":
            # Direct quote: pip value already in USD
            return pip_value_quote

        elif is_jpy_pair:
            # JPY quote: need to convert JPY to USD
            # If we have current price (XXX/JPY), use it for more accuracy
            if current_price and current_price > 0:
                # current_price is in JPY per 1 base, so JPY to USD = 1/price
                jpy_to_usd = 1.0 / current_price
                pip_value_usd = pip_value_quote * jpy_to_usd
            else:
                # Use stored rate
                pip_value_usd = self.convert_to_usd(pip_value_quote, "JPY")

            return pip_value_usd

        else:
            # Other quote currencies (EUR, GBP, etc.)
            if current_price and current_price > 0 and base == "USD":
                # USD/XXX pair - price is in quote per USD
                quote_to_usd = 1.0 / current_price
                pip_value_usd = pip_value_quote * quote_to_usd
            else:
                # Use stored rate for quote currency
                pip_value_usd = self.convert_to_usd(pip_value_quote, quote)

            return pip_value_usd

    def get_fx_tick_value_usd(
        self,
        pair: str,
        lot_size: int = 100_000,
        current_price: float | None = None
    ) -> float:
        """
        Calculate FX tick value in USD (half pip for JPY pairs).

        Most FX brokers quote JPY pairs in 0.5 pip increments (0.005),
        so tick value = pip value / 2 for JPY pairs.

        Args:
            pair: FX pair
            lot_size: Position size
            current_price: Current price for accurate conversion

        Returns:
            Tick value in USD
        """
        pair = pair.upper().replace("/", "")
        quote = pair[3:] if len(pair) == 6 else ""
        is_jpy_pair = quote == "JPY"

        pip_value = self.get_fx_pip_value_usd(pair, lot_size, current_price)

        # For JPY pairs, tick = half pip
        if is_jpy_pair:
            return pip_value / 2.0

        return pip_value

    def get_status(self) -> dict[str, Any]:
        """Get converter status."""
        return {
            "currencies_tracked": len(self._rates_to_usd),
            "rates": dict(self._rates_to_usd),
            "last_update": self._last_update.isoformat() if self._last_update else None,
        }


# Global currency converter instance
_currency_converter: CurrencyConverter | None = None


def get_currency_converter() -> CurrencyConverter:
    """Get or create the global currency converter instance."""
    global _currency_converter
    if _currency_converter is None:
        _currency_converter = CurrencyConverter()
    return _currency_converter


# =============================================================================
# FX ROLLOVER/SWAP RATES
# =============================================================================

@dataclass
class FXSwapRate:
    """FX swap/rollover rate for overnight positions."""
    pair: str  # e.g., "EURUSD"
    long_rate: float  # Points per night for long position (positive = earn, negative = pay)
    short_rate: float  # Points per night for short position
    last_update: datetime | None = None
    source: str = "default"

    def get_rollover_cost(self, position_size: float, is_long: bool) -> float:
        """
        Calculate rollover cost/credit for a position.

        Args:
            position_size: Position size in lots
            is_long: True if long position, False if short

        Returns:
            Rollover amount (positive = credit, negative = cost)
        """
        rate = self.long_rate if is_long else self.short_rate
        return position_size * rate


class FXRolloverManager:
    """
    Manages FX rollover/swap rates for overnight positions.

    FX positions held overnight are subject to rollover (swap) rates based on
    the interest rate differential between the two currencies. This manager
    tracks swap rates and calculates rollover costs/credits.

    Note: Wednesday rollovers typically include 3 days (Wed-Thu-Fri-Sat-Sun)
    to account for T+2 settlement.
    """

    def __init__(self):
        """Initialize with default swap rates."""
        # Default swap rates (in points/pips)
        # These are illustrative - real rates vary by broker and market conditions
        self._swap_rates: dict[str, FXSwapRate] = {
            "EURUSD": FXSwapRate("EURUSD", long_rate=-0.5, short_rate=-0.2),
            "GBPUSD": FXSwapRate("GBPUSD", long_rate=-0.6, short_rate=-0.1),
            "USDJPY": FXSwapRate("USDJPY", long_rate=0.3, short_rate=-0.8),
            "USDCHF": FXSwapRate("USDCHF", long_rate=0.2, short_rate=-0.7),
            "AUDUSD": FXSwapRate("AUDUSD", long_rate=-0.4, short_rate=-0.3),
            "USDCAD": FXSwapRate("USDCAD", long_rate=0.1, short_rate=-0.6),
            "NZDUSD": FXSwapRate("NZDUSD", long_rate=-0.3, short_rate=-0.4),
            # Cross pairs
            "EURGBP": FXSwapRate("EURGBP", long_rate=-0.2, short_rate=-0.3),
            "EURJPY": FXSwapRate("EURJPY", long_rate=0.4, short_rate=-0.9),
            "GBPJPY": FXSwapRate("GBPJPY", long_rate=0.5, short_rate=-1.0),
            "EURCHF": FXSwapRate("EURCHF", long_rate=-0.1, short_rate=-0.4),
            "AUDJPY": FXSwapRate("AUDJPY", long_rate=0.3, short_rate=-0.7),
            "CADJPY": FXSwapRate("CADJPY", long_rate=0.2, short_rate=-0.6),
        }

        # Track rollover history
        self._rollover_history: list[dict[str, Any]] = []

    def update_swap_rate(
        self,
        pair: str,
        long_rate: float,
        short_rate: float,
        source: str = "broker"
    ) -> None:
        """
        Update swap rate for a currency pair.

        Args:
            pair: Currency pair (e.g., "EURUSD")
            long_rate: Rate for long positions
            short_rate: Rate for short positions
            source: Source of the rate data
        """
        pair = pair.upper().replace("/", "")
        self._swap_rates[pair] = FXSwapRate(
            pair=pair,
            long_rate=long_rate,
            short_rate=short_rate,
            last_update=datetime.now(timezone.utc),
            source=source,
        )
        logger.debug(f"Updated swap rate for {pair}: long={long_rate}, short={short_rate}")

    def get_swap_rate(self, pair: str) -> FXSwapRate | None:
        """Get swap rate for a pair."""
        pair = pair.upper().replace("/", "")
        return self._swap_rates.get(pair)

    def calculate_rollover(
        self,
        pair: str,
        position_size: float,
        is_long: bool,
        nights: int = 1
    ) -> float:
        """
        Calculate rollover cost/credit for a position.

        Args:
            pair: Currency pair
            position_size: Position size in lots
            is_long: True if long, False if short
            nights: Number of nights (usually 1, except Wednesday = 3)

        Returns:
            Rollover amount in pips/points
        """
        swap_rate = self.get_swap_rate(pair)
        if swap_rate is None:
            logger.warning(f"No swap rate for {pair}, returning 0")
            return 0.0

        return swap_rate.get_rollover_cost(position_size, is_long) * nights

    def calculate_rollover_usd(
        self,
        pair: str,
        position_size_lots: float,
        is_long: bool,
        nights: int = 1
    ) -> float:
        """
        Calculate rollover cost/credit in USD.

        Args:
            pair: Currency pair
            position_size_lots: Position size in lots
            is_long: True if long, False if short
            nights: Number of nights

        Returns:
            Rollover amount in USD
        """
        pair = pair.upper().replace("/", "")
        spec = CONTRACT_SPECS.get(pair[:3]) or CONTRACT_SPECS.get(pair)

        # Get pip value
        pip_value = 10.0  # Default for most pairs
        if pair.endswith("JPY"):
            pip_value = 1000.0  # JPY pairs have larger pip value

        # Get swap in pips
        swap_pips = self.calculate_rollover(pair, position_size_lots, is_long, nights)

        # Convert to USD
        swap_value = swap_pips * pip_value / 10000  # Normalize

        # Convert to USD if not already
        converter = get_currency_converter()
        quote_currency = pair[3:6] if len(pair) >= 6 else "USD"

        if quote_currency != "USD":
            swap_value = converter.convert_to_usd(swap_value, quote_currency)

        return swap_value

    def is_triple_swap_day(self) -> bool:
        """
        Check if today is a triple swap day (Wednesday for T+2 settlement).

        Returns:
            True if today's rollover includes 3 days
        """
        return datetime.now(timezone.utc).weekday() == 2  # Wednesday

    def get_rollover_nights(self) -> int:
        """Get number of nights for today's rollover."""
        if self.is_triple_swap_day():
            return 3
        return 1

    def record_rollover(
        self,
        pair: str,
        position_size: float,
        is_long: bool,
        rollover_amount: float,
        nights: int
    ) -> None:
        """
        Record a rollover event for tracking.

        Args:
            pair: Currency pair
            position_size: Position size
            is_long: Position direction
            rollover_amount: Amount credited/charged
            nights: Number of nights
        """
        self._rollover_history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pair": pair,
            "position_size": position_size,
            "direction": "long" if is_long else "short",
            "rollover_amount": rollover_amount,
            "nights": nights,
        })

    def get_total_rollovers(
        self,
        start_date: datetime | None = None
    ) -> float:
        """
        Get total rollovers since start date.

        Args:
            start_date: Optional start date filter

        Returns:
            Total rollover amount
        """
        total = 0.0
        for record in self._rollover_history:
            if start_date:
                record_date = datetime.fromisoformat(record["timestamp"])
                if record_date < start_date:
                    continue
            total += record["rollover_amount"]
        return total

    def get_status(self) -> dict[str, Any]:
        """Get manager status."""
        return {
            "pairs_tracked": len(self._swap_rates),
            "is_triple_swap_day": self.is_triple_swap_day(),
            "rollover_nights_today": self.get_rollover_nights(),
            "total_rollover_records": len(self._rollover_history),
        }


# Global FX rollover manager instance
_fx_rollover_manager: FXRolloverManager | None = None


def get_fx_rollover_manager() -> FXRolloverManager:
    """Get or create the global FX rollover manager."""
    global _fx_rollover_manager
    if _fx_rollover_manager is None:
        _fx_rollover_manager = FXRolloverManager()
    return _fx_rollover_manager


# =============================================================================
# CONTRACT SPECS MANAGER
# =============================================================================

class ContractSpecsManager:
    """
    Centralized manager for contract specifications.

    Provides:
    - Contract spec lookup with caching
    - Validation of contract symbols
    - Margin and position value calculations
    - Asset class grouping
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize contract specs manager.

        Args:
            config: Optional configuration with margin buffer settings
        """
        self._config = config or {}
        self._margin_buffer_pct = self._config.get("margin_buffer_pct", 10.0)
        self._cache: dict[str, ContractSpec] = {}
        self._last_update = datetime.now(timezone.utc)

        # Initialize cache with all specs
        for symbol, spec in CONTRACT_SPECS.items():
            self._cache[symbol] = spec

        logger.info(f"ContractSpecsManager initialized with {len(self._cache)} contracts")

    @lru_cache(maxsize=100)
    def get_spec(self, symbol: str) -> ContractSpec | None:
        """
        Get contract specification by symbol.

        Args:
            symbol: Contract symbol (e.g., "ES", "CL", "GC")

        Returns:
            ContractSpec if found, None otherwise
        """
        # Normalize symbol (uppercase, strip whitespace)
        symbol = symbol.upper().strip()

        # Handle continuous contract notation (e.g., "ESZ4" -> "ES")
        base_symbol = self._extract_base_symbol(symbol)

        return self._cache.get(base_symbol)

    def _extract_base_symbol(self, symbol: str) -> str:
        """
        Extract base symbol from contract notation.

        Examples:
            "ESZ4" -> "ES"
            "CLF25" -> "CL"
            "ES" -> "ES"
        """
        # Common futures symbols are 2-3 characters
        for length in [2, 3]:
            if len(symbol) >= length:
                base = symbol[:length]
                if base in CONTRACT_SPECS:
                    return base

        return symbol

    def is_valid_symbol(self, symbol: str) -> bool:
        """Check if symbol is a valid futures contract."""
        return self.get_spec(symbol) is not None

    def get_all_symbols(self) -> list[str]:
        """Get list of all available contract symbols."""
        return list(CONTRACT_SPECS.keys())

    def get_symbols_by_asset_class(self, asset_class: AssetClass) -> list[str]:
        """Get all symbols for a given asset class."""
        return [
            symbol for symbol, spec in CONTRACT_SPECS.items()
            if spec.asset_class == asset_class
        ]

    def get_symbols_by_exchange(self, exchange: Exchange) -> list[str]:
        """Get all symbols traded on a given exchange."""
        return [
            symbol for symbol, spec in CONTRACT_SPECS.items()
            if spec.exchange == exchange
        ]

    def calculate_position_value(
        self,
        symbol: str,
        price: float,
        contracts: int
    ) -> float | None:
        """
        Calculate notional value of a position.

        Args:
            symbol: Contract symbol
            price: Current price
            contracts: Number of contracts (positive=long, negative=short)

        Returns:
            Notional value in USD, or None if symbol not found
        """
        spec = self.get_spec(symbol)
        if spec is None:
            logger.warning(f"Unknown contract symbol: {symbol}")
            return None

        return spec.notional_value(price, abs(contracts))

    def calculate_margin_requirement(
        self,
        symbol: str,
        contracts: int,
        is_initial: bool = True
    ) -> float | None:
        """
        Calculate margin requirement for a position.

        Includes configured margin buffer and cross-currency conversion (#X4).

        Args:
            symbol: Contract symbol
            contracts: Number of contracts
            is_initial: True for initial margin, False for maintenance

        Returns:
            Margin required in USD, or None if symbol not found
        """
        spec = self.get_spec(symbol)
        if spec is None:
            logger.warning(f"Unknown contract symbol: {symbol}")
            return None

        base_margin = spec.margin_required(contracts, is_initial)
        buffer_multiplier = 1 + (self._margin_buffer_pct / 100)
        margin_native = base_margin * buffer_multiplier

        # Convert to USD if margin is in another currency (#X4)
        if spec.currency != "USD":
            converter = get_currency_converter()
            margin_usd = converter.convert_to_usd(margin_native, spec.currency)
            logger.debug(
                f"Margin conversion for {symbol}: {margin_native:.2f} {spec.currency} -> {margin_usd:.2f} USD"
            )
            return margin_usd

        return margin_native

    def calculate_portfolio_margin(
        self,
        positions: dict[str, int],
        is_initial: bool = True,
        use_netting: bool = True
    ) -> tuple[float, dict[str, float]]:
        """
        Calculate total portfolio margin with cross-currency conversion (#X4).

        Args:
            positions: Dictionary of symbol -> quantity
            is_initial: True for initial margin
            use_netting: Apply cross-margin benefits for correlated positions

        Returns:
            Tuple of (total_margin_usd, margin_by_symbol)
        """
        margin_by_symbol: dict[str, float] = {}
        total_margin = 0.0

        for symbol, quantity in positions.items():
            if quantity == 0:
                continue

            margin = self.calculate_margin(symbol, quantity, is_initial)
            if margin is not None:
                margin_by_symbol[symbol] = margin
                total_margin += margin

        # Apply cross-margin netting if enabled (simplified)
        # In production, this would use a proper portfolio margin model
        if use_netting and len(margin_by_symbol) > 1:
            # Simple 10% reduction for diversified portfolios
            netting_benefit = total_margin * 0.10
            total_margin -= netting_benefit
            logger.debug(f"Cross-margin netting benefit: {netting_benefit:.2f} USD")

        return total_margin, margin_by_symbol

    def calculate_pnl(
        self,
        symbol: str,
        entry_price: float,
        exit_price: float,
        contracts: int,
        convert_to_usd: bool = True
    ) -> float | None:
        """
        Calculate P&L for a trade.

        Args:
            symbol: Contract symbol
            entry_price: Entry price
            exit_price: Exit price
            contracts: Number of contracts (positive=long, negative=short)
            convert_to_usd: Whether to convert non-USD P&L to USD

        Returns:
            P&L in USD (if convert_to_usd=True) or native currency, or None if symbol not found
        """
        spec = self.get_spec(symbol)
        if spec is None:
            logger.warning(f"Unknown contract symbol: {symbol}")
            return None

        price_diff = exit_price - entry_price
        pnl_native = price_diff * spec.multiplier * contracts

        if convert_to_usd and spec.currency != "USD":
            converter = get_currency_converter()
            return converter.convert_pnl_to_usd(pnl_native, spec.currency, symbol)

        return pnl_native

    def calculate_pnl_in_ticks(
        self,
        symbol: str,
        entry_price: float,
        exit_price: float,
        contracts: int,
        convert_to_usd: bool = True
    ) -> tuple[float, int] | None:
        """
        Calculate P&L for a trade with tick breakdown.

        Args:
            symbol: Contract symbol
            entry_price: Entry price
            exit_price: Exit price
            contracts: Number of contracts
            convert_to_usd: Whether to convert to USD

        Returns:
            Tuple of (P&L in currency, number of ticks), or None if symbol not found
        """
        spec = self.get_spec(symbol)
        if spec is None:
            return None

        ticks = spec.price_to_ticks(exit_price - entry_price)
        pnl_native = ticks * spec.tick_value * contracts

        if convert_to_usd and spec.currency != "USD":
            converter = get_currency_converter()
            pnl = converter.convert_pnl_to_usd(pnl_native, spec.currency, symbol)
        else:
            pnl = pnl_native

        return pnl, ticks

    def get_tick_value_usd(self, symbol: str) -> float | None:
        """
        Get tick value in USD for a symbol.

        Handles currency conversion for non-USD quoted contracts.

        Args:
            symbol: Contract symbol

        Returns:
            Tick value in USD, or None if symbol not found
        """
        spec = self.get_spec(symbol)
        if spec is None:
            return None

        if spec.currency == "USD":
            return spec.tick_value

        converter = get_currency_converter()
        return converter.get_tick_value_usd(spec)

    def round_to_tick(self, symbol: str, price: float) -> float | None:
        """
        Round price to nearest valid tick.

        Args:
            symbol: Contract symbol
            price: Price to round

        Returns:
            Rounded price, or None if symbol not found
        """
        spec = self.get_spec(symbol)
        if spec is None:
            return None

        ticks = round(price / spec.tick_size)
        return ticks * spec.tick_size

    def get_liquidity_tier(self, symbol: str) -> str:
        """
        Get liquidity tier for a contract.

        Returns: "high", "medium", or "low"
        """
        spec = self.get_spec(symbol)
        if spec is None:
            return "unknown"

        adv = spec.avg_daily_volume

        if adv >= 500_000:
            return "high"
        elif adv >= 100_000:
            return "medium"
        else:
            return "low"

    def get_summary(self) -> dict[str, Any]:
        """Get summary of all contract specifications."""
        by_asset_class = {}
        for asset_class in AssetClass:
            symbols = self.get_symbols_by_asset_class(asset_class)
            if symbols:
                by_asset_class[asset_class.value] = symbols

        return {
            "total_contracts": len(CONTRACT_SPECS),
            "by_asset_class": by_asset_class,
            "margin_buffer_pct": self._margin_buffer_pct,
            "last_update": self._last_update.isoformat(),
        }

    def to_dataframe(self):
        """
        Export all contract specs to a pandas DataFrame.

        Returns:
            DataFrame with contract specifications
        """
        try:
            import pandas as pd
        except ImportError:
            logger.error("pandas required for to_dataframe()")
            return None

        data = []
        for symbol, spec in CONTRACT_SPECS.items():
            data.append({
                "symbol": spec.symbol,
                "name": spec.name,
                "asset_class": spec.asset_class.value,
                "exchange": spec.exchange.value,
                "multiplier": spec.multiplier,
                "tick_size": spec.tick_size,
                "tick_value": spec.tick_value,
                "initial_margin": spec.initial_margin,
                "maintenance_margin": spec.maintenance_margin,
                "avg_daily_volume": spec.avg_daily_volume,
                "settlement_type": spec.settlement_type,
            })

        return pd.DataFrame(data)
