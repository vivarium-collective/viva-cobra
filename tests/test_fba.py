"""COBRApy FBA helpers: model loading, static bounds, and uptake -> exchange bounds."""
from __future__ import annotations

import pytest

from viva_cobra.processes.fba import _apply_bounds, _load_fba_model, apply_uptake_bounds


def test_textbook_model_loads_and_grows():
    model = _load_fba_model("textbook", {})
    sol = model.optimize()
    assert sol.status == "optimal"
    # Known textbook E. coli core aerobic growth rate.
    assert sol.objective_value == pytest.approx(0.8739, abs=1e-3)


def test_static_bounds_applied():
    model = _load_fba_model("textbook", {"EX_o2_e": {"lower": -20.0, "upper": 1000.0}, "ATPM": {"lower": 1.0, "upper": 1.0}})
    assert model.reactions.get_by_id("EX_o2_e").lower_bound == -20.0
    assert model.reactions.get_by_id("ATPM").lower_bound == 1.0
    assert model.reactions.get_by_id("ATPM").upper_bound == 1.0


def test_uptake_bounds_set_exchange_lower_bound():
    model = _load_fba_model("textbook", {})
    apply_uptake_bounds(model, {"glucose": 8.0}, {"glucose": "EX_glc__D_e"})
    assert model.reactions.get_by_id("EX_glc__D_e").lower_bound == -8.0


def test_zero_uptake_closes_exchange_inflow():
    model = _load_fba_model("textbook", {})
    apply_uptake_bounds(model, {"glucose": 0.0}, {"glucose": "EX_glc__D_e"})
    # lower_bound 0 -> no inflow; with no carbon source growth should be ~0.
    assert model.reactions.get_by_id("EX_glc__D_e").lower_bound == 0.0
    assert model.slim_optimize(error_value=0.0) == pytest.approx(0.0, abs=1e-6)
