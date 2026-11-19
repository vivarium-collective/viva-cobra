"""Dynamic-FBA composite: Michaelis-Menten uptake -> COBRApy FBA, over time.

Integration test — runs real COBRApy optimizations each scheduled step.
"""
from __future__ import annotations

import logging

import pytest

from viva_cobra import dfba_document, run_dfba
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


def _emitter():
    from process_bigraph.emitter import emitter_from_wires

    return emitter_from_wires({"global_time": ["global_time"], "biomass": ["biomass"], "substrates": ["substrates"]})


def _process_doc(config, substrates, biomass, dt):
    """``DynamicFBAProcess`` alone on the substrate and biomass stores, with an emitter."""
    return {
        "schema": {"substrates": "map[concentration]", "biomass": "mass"},
        "state": {
            "substrates": dict(substrates),
            "biomass": float(biomass),
            "dfba": {
                "_type": "process",
                "address": "local:DynamicFBAProcess",
                "config": dict(config),
                "interval": float(dt),
                "inputs": {"substrates": ["substrates"], "biomass": ["biomass"]},
                "outputs": {"substrates": ["substrates"], "biomass": ["biomass"]},
            },
            "emitter": _emitter(),
        },
    }


def _run(doc, until):
    from process_bigraph import Composite
    from process_bigraph.emitter import gather_emitter_results

    from viva_cobra.core import build_core

    composite = Composite(doc, core=build_core())
    composite.run(until)
    return gather_emitter_results(composite)[("emitter",)]


def _run_process(config, substrates, biomass, *, steps, dt):
    """Run ``DynamicFBAProcess`` alone; return the emitter trajectory."""
    return _run(_process_doc(config, substrates, biomass, dt), steps * dt)


def _textbook_optimum(glucose_uptake, atpm_lower=None):
    """COBRApy's own growth rate for the textbook model with the glucose exchange at ``-glucose_uptake``."""
    import cobra

    model = cobra.io.load_model("textbook")
    model.reactions.get_by_id("EX_glc__D_e").lower_bound = -glucose_uptake
    if atpm_lower is not None:
        model.reactions.get_by_id("ATPM").lower_bound = atpm_lower
    return model.slim_optimize()


TEXTBOOK = {"model_file": "textbook", "substrate_update_reactions": {"glucose": "EX_glc__D_e"}, "bounds": {}}


def test_process_integrates_as_the_step_composite_does_but_from_the_initial_state():
    """DynamicFBAProcess computes each interval exactly as the MichaelisMentenUptakeProcess -> FBAStep composite does,
    through glucose exhaustion and the starved steps after it. The two differ only at the start. The composite's Step
    fires once when the composite is built, on the seeded uptakes, so it has already grown by one interval at t = 0;
    the Process grows only as time passes, so its first row is the initial state."""
    substrates, biomass = {"glucose": 11.1, "acetate": 0.0}, 0.01
    composite = run_dfba(CONFIG, initial_substrates=substrates, initial_biomass=biomass, steps=140, dt=0.05)
    process = _run_process(CONFIG, substrates, biomass, steps=140, dt=0.05)

    def states(rows):
        return [(row["biomass"], row["substrates"]) for row in rows]

    assert states(process[:1]) == [(biomass, substrates)]
    assert composite[0]["biomass"] > biomass  # the Step's extra interval, applied before any time has passed
    assert states(process[1:]) == states(composite[:-1])


def test_one_interval_follows_michaelis_menten_uptake_biomass_and_interval():
    """One interval of 0.5 from biomass 0.5 and glucose 100, with Km 100 and Vmax 10: the uptake is 10 x 100 / 200 = 5,
    so the growth rate is COBRApy's optimum with the glucose exchange at -5, and glucose falls by 5 x 0.5 x 0.5. Each
    factor differs from 1, so dropping the biomass, the interval or the concentration dependence changes the result."""
    mu = _textbook_optimum(5.0)
    config = {**TEXTBOOK, "kinetic_params": {"glucose": [100.0, 10.0]}}
    rows = _run_process(config, {"glucose": 100.0}, 0.5, steps=1, dt=0.5)
    assert rows[-1]["global_time"] == 0.5
    assert rows[-1]["biomass"] == pytest.approx(0.5 + mu * 0.5 * 0.5, rel=1e-12, abs=0)
    assert rows[-1]["substrates"]["glucose"] == pytest.approx(100.0 - 5.0 * 0.5 * 0.5, rel=1e-12, abs=0)


def test_a_short_pool_caps_the_uptake_so_growth_matches_what_it_supplied():
    """Glucose 0.1 can supply at most 0.1 over an interval of 1.0 at biomass 1.0, far below the kinetic uptake of 10.
    The uptake is capped at pool / (biomass x interval) = 0.1 before the solve, so the growth rate is COBRApy's optimum
    at an exchange of -0.1 and the pool ends at zero, not below it. (ATPM's lower bound is 0 here, so a small uptake is
    feasible rather than starved.)"""
    atpm_free = {**TEXTBOOK, "bounds": {"ATPM": {"lower": 0.0, "upper": 1000.0}}}
    mu = _textbook_optimum(0.1, atpm_lower=0.0)
    assert 0.0 < mu < _textbook_optimum(10.0, atpm_lower=0.0)  # the premise: the cap changes the growth
    config = {**atpm_free, "kinetic_params": {"glucose": [0.0, 10.0]}}
    rows = _run_process(config, {"glucose": 0.1}, 1.0, steps=1, dt=1.0)
    assert rows[-1]["biomass"] == pytest.approx(1.0 + mu, rel=1e-12, abs=0)
    assert rows[-1]["substrates"]["glucose"] == pytest.approx(0.0, abs=1e-12)


def test_a_negative_pool_supplies_nothing_and_is_not_refilled():
    """A pool already below zero supplies no uptake, and the clamp that keeps a delta from overdrawing a pool does not
    turn into a refill: glucose -1.0 stays -1.0 and, with no other carbon source, nothing grows."""
    atpm_free = {**TEXTBOOK, "bounds": {"ATPM": {"lower": 0.0, "upper": 1000.0}}}
    config = {**atpm_free, "kinetic_params": {"glucose": [0.0, 10.0]}}
    rows = _run_process(config, {"glucose": -1.0}, 1.0, steps=1, dt=1.0)
    assert rows[-1]["substrates"]["glucose"] == -1.0
    assert rows[-1]["biomass"] == pytest.approx(1.0, abs=1e-12)


def test_a_saved_and_reloaded_process_does_not_apply_its_deltas_again(tmp_path):
    """Run 0.5, save, reload and run 0.5 more: the Process ends exactly where an uninterrupted run of 1.0 does. A Step
    re-fires when the reloaded composite is built, so the MichaelisMentenUptakeProcess -> FBAStep composite, run the
    same way, applies one interval twice and ends elsewhere."""
    import json

    from process_bigraph import Composite

    from viva_cobra.core import build_core

    substrates, biomass, dt = {"glucose": 11.1, "acetate": 0.0}, 0.01, 0.05

    def resumed(doc, name):
        first = Composite(doc, core=build_core())
        first.run(0.5)
        first.save(filename=f"{name}.pbg", outdir=str(tmp_path))
        saved = json.loads((tmp_path / f"{name}.pbg").read_text())
        state, schema = saved["state"], saved["schema"]
        state["emitter"] = _emitter()  # a fresh emitter: the saved one's results are not part of the model
        for key in ("emitter", "results"):
            schema.pop(key, None)
        state.pop("results", None)
        return _run({"state": state, "schema": schema}, 0.5)[-1]

    def final(row):
        return row["biomass"], row["substrates"]

    process = _process_doc(CONFIG, substrates, biomass, dt)
    assert final(resumed(process, "process")) == final(_run(_process_doc(CONFIG, substrates, biomass, dt), 1.0)[-1])

    composite = dfba_document(CONFIG, dt=dt, initial_substrates=substrates, initial_biomass=biomass)
    uninterrupted = run_dfba(CONFIG, initial_substrates=substrates, initial_biomass=biomass, steps=20, dt=dt)[-1]
    assert final(resumed(composite, "composite")) != final(uninterrupted)


def test_process_requires_kinetic_params_for_every_resource():
    from viva_cobra.core import build_core
    from viva_cobra.processes.fba import DynamicFBAProcess

    config = {**CONFIG, "kinetic_params": {"glucose": [0.5, 10.0]}}
    with pytest.raises(ValueError, match=r"DynamicFBAProcess missing kinetic_params for: \['acetate'\]"):
        DynamicFBAProcess(config=config, core=build_core())


@pytest.mark.parametrize(
    "entry, message",
    [
        ([-0.5, 10.0], "needs finite Km >= 0 and Vmax >= 0"),
        ([0.5, -10.0], "needs finite Km >= 0 and Vmax >= 0"),
        ([float("nan"), 10.0], "needs finite Km >= 0 and Vmax >= 0"),
        ([0.5, float("inf")], "needs finite Km >= 0 and Vmax >= 0"),
        ([0.5], "must be \\[Km, Vmax\\]"),
        ([0.5, 10.0, 1.0], "must be \\[Km, Vmax\\]"),
        ("fast", "must be \\[Km, Vmax\\]"),
        (None, "must be \\[Km, Vmax\\]"),
    ],
)
def test_kinetic_params_must_be_a_finite_non_negative_pair(entry, message):
    """Regression: a negative Km was accepted and gave a near-infinite uptake (Vmax S / (Km + S) with Km close to -S),
    and a malformed entry raised IndexError or KeyError instead of saying what was wrong."""
    from viva_cobra.processes.michaelis_menten import kinetic_table

    with pytest.raises(ValueError, match=message):
        kinetic_table(["glucose"], {"glucose": entry}, "DynamicFBAProcess")
