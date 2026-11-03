"""Dynamic-FBA composite: Michaelis-Menten uptake -> COBRApy FBA, over time.

Integration test — runs real COBRApy optimizations each scheduled step.
"""
from __future__ import annotations

import logging

import pytest

from viva_cobra import run_dfba
from viva_cobra.models import get_model_spec

CONFIG = {
    "model_file": "textbook",
    "substrate_update_reactions": {"glucose": "EX_glc__D_e", "acetate": "EX_ac_e"},
    "bounds": {
        "EX_o2_e": {"lower": -20.0, "upper": 1000.0},
        "ATPM": {"lower": 1.0, "upper": 1.0},
    },
    "biomass_reaction": "Biomass_Ecoli_core",
    "kinetic_params": {"glucose": [0.5, 10.0], "acetate": [0.5, 2.0]},
}


def test_dfba_batch_grows_and_consumes_glucose():
    history = run_dfba(
        CONFIG,
        initial_substrates={"glucose": 11.1, "acetate": 0.0},
        initial_biomass=0.01,
        steps=40,
        dt=0.05,
    )
    assert history, "emitter produced no trajectory"

    first, last = history[0], history[-1]
    # Biomass grows over the batch.
    assert last["biomass"] > first["biomass"]
    # Glucose is consumed.
    assert last["substrates"]["glucose"] < 11.1
    # Pools never go negative (clamped to available).
    assert all(v >= -1e-9 for step in history for v in step["substrates"].values())


def test_model_registry_ecoli_core_metadata():
    spec = get_model_spec("ecoli_core")
    assert spec["model_file"] == "textbook"
    assert spec["biomass_reaction"] == "Biomass_Ecoli_core"
    assert "glucose" in spec["substrate_update_reactions"]


def test_initial_uptakes_follow_michaelis_menten_not_zero():
    """Regression: the composite used to start with every uptake at 0. Glucose's exchange lower bound is then 0,
    which with the ATPM lower bound of 1 makes the first LP infeasible, so the first step grew nothing."""
    from viva_cobra import dfba_document
    from viva_cobra.processes.michaelis_menten import mm_uptake_rates

    substrates = {"glucose": 11.1, "acetate": 0.0}
    doc = dfba_document(CONFIG, dt=0.05, initial_substrates=substrates, initial_biomass=0.01)
    expected = mm_uptake_rates(substrates, CONFIG["kinetic_params"], ["glucose", "acetate"])
    assert doc["state"]["uptakes"] == pytest.approx(expected)
    assert doc["state"]["uptakes"]["glucose"] > 0.0


def test_first_fba_solve_is_optimal_and_grows_biomass():
    from viva_cobra.core import build_core
    from viva_cobra.processes.fba import FBAStep
    from viva_cobra.processes.michaelis_menten import mm_uptake_rates

    substrates = {"glucose": 11.1, "acetate": 0.0}
    step = FBAStep(config=CONFIG, core=build_core())
    out = step.update(
        {
            "uptakes": mm_uptake_rates(substrates, CONFIG["kinetic_params"], ["glucose", "acetate"]),
            "substrates": substrates,
            "biomass": 0.01,
            "interval": 0.05,
        }
    )
    assert out["biomass"] > 0.0
    assert out["substrates"]["glucose"] < 0.0


def test_batch_runs_past_glucose_exhaustion_and_stops_growing():
    """Regression (0.1.1): running out of glucose is the normal end of a batch culture, but the step after it made the
    LP infeasible (the uptake can no longer cover ATPM >= 1) and FBAStep raised. With this CONFIG glucose runs out at
    step 114 of dt 0.05; 160 steps runs well past it."""
    history = run_dfba(
        CONFIG,
        initial_substrates={"glucose": 11.1, "acetate": 0.0},
        initial_biomass=0.01,
        steps=160,
        dt=0.05,
    )
    assert len(history) == 161
    assert history[-1]["substrates"]["glucose"] == pytest.approx(0.0, abs=1e-3)
    tail = [step["biomass"] for step in history[-30:]]
    assert max(tail) == pytest.approx(min(tail), rel=0, abs=0.0)
    assert history[-1]["biomass"] > 1.0


def _fba_step(config=CONFIG):
    from viva_cobra.core import build_core
    from viva_cobra.processes.fba import FBAStep

    return FBAStep(config=config, core=build_core())


@pytest.mark.parametrize(
    "glucose_uptake",
    [
        0.0,
        # the uptake of the step that raised in 0.1.1: positive, but below what ATPM >= 1 needs
        0.0016270804105548392,
    ],
)
def test_substrate_limited_solve_returns_zero_deltas(glucose_uptake, caplog):
    caplog.set_level(logging.DEBUG, logger="viva_cobra.processes.fba")
    out = _fba_step().update(
        {
            "uptakes": {"glucose": glucose_uptake, "acetate": 0.0},
            "substrates": {"glucose": 8.1e-5, "acetate": 0.0},
            "biomass": 1.02,
            "interval": 0.05,
        }
    )
    assert out == {"substrates": {"glucose": 0.0, "acetate": 0.0}, "biomass": 0.0}
    assert [r.levelno for r in caplog.records if "starved" in r.getMessage()] == [logging.DEBUG]


def test_infeasible_solve_that_more_substrate_cannot_fix_raises():
    """Regression: a contradictory model used to return zero deltas silently. A growth floor no supply can meet is
    infeasible at any uptake, so it is a defect in the model, not starvation."""
    from viva_cobra.processes.fba import _load_fba_model, apply_uptake_bounds

    bounds = {**CONFIG["bounds"], "Biomass_Ecoli_core": {"lower": 500.0, "upper": 1000.0}}
    # the premise: infeasible at a non-zero uptake, and still infeasible with every tracked exchange fully open
    model = _load_fba_model("textbook", bounds)
    for uptake in (10.0, 1000.0):
        apply_uptake_bounds(model, {"glucose": uptake, "acetate": uptake}, CONFIG["substrate_update_reactions"])
        assert model.optimize().status == "infeasible"

    step = _fba_step({**CONFIG, "bounds": bounds})
    with pytest.raises(RuntimeError, match="not optimal"):
        step.update(
            {
                "uptakes": {"glucose": 10.0, "acetate": 0.0},
                "substrates": {"glucose": 11.1, "acetate": 0.0},
                "biomass": 0.01,
                "interval": 0.05,
            }
        )
