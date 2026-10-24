"""Dynamic-FBA composite: Michaelis-Menten uptake -> COBRApy FBA, over time.

Integration test — runs real COBRApy optimizations each scheduled step.
"""
from __future__ import annotations

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
