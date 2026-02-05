"""
Test Fixtures Package
=====================

Contains utilities for generating synthetic test data for trading strategy testing.
"""

from tests.fixtures.test_data_generator import (
    generate_price_series,
    generate_ohlcv_data,
    generate_correlated_series,
    generate_regime_switching_data,
    generate_crisis_scenario,
    SyntheticDataConfig,
    OHLCVData,
    CrisisScenario,
)

__all__ = [
    "generate_price_series",
    "generate_ohlcv_data",
    "generate_correlated_series",
    "generate_regime_switching_data",
    "generate_crisis_scenario",
    "SyntheticDataConfig",
    "OHLCVData",
    "CrisisScenario",
]
