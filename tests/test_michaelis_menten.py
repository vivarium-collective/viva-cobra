"""Michaelis-Menten kinetics: v = Vmax * S / (Km + S)."""
from __future__ import annotations

import pytest

from viva_cobra import mm_uptake_rates


def test_mm_rate_matches_formula():
    # Km = 0.5, Vmax = 10, S = 0.5  ->  10 * 0.5 / (0.5 + 0.5) = 5.0
    rates = mm_uptake_rates({"glc": 0.5}, {"glc": (0.5, 10.0)}, ["glc"])
    assert rates["glc"] == pytest.approx(5.0, rel=1e-6)


def test_mm_saturates_toward_vmax():
    # S >> Km  ->  v -> Vmax
    rates = mm_uptake_rates({"glc": 1e6}, {"glc": (0.5, 10.0)}, ["glc"])
    assert rates["glc"] == pytest.approx(10.0, rel=1e-4)


def test_mm_zero_at_zero_and_negative_concentration():
    rates = mm_uptake_rates({"glc": 0.0, "ac": -3.0}, {"glc": (0.5, 10.0), "ac": (0.5, 2.0)}, ["glc", "ac"])
    assert rates["glc"] == 0.0
    assert rates["ac"] == 0.0


def test_mm_missing_substrate_is_zero():
    rates = mm_uptake_rates({}, {"glc": (0.5, 10.0)}, ["glc"])
    assert rates["glc"] == 0.0
