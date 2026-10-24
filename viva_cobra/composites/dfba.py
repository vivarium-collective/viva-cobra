"""Dynamic-FBA composite: Michaelis-Menten uptake wired to a COBRApy FBA step.

The dFBA loop is two components sharing four stores:

    MichaelisMentenUptakeProcess (temporal)
        reads  substrates, biomass
        writes uptakes, interval
    FBAStep (reactive, non-temporal)
        reads  uptakes, substrates, biomass, interval
        writes substrates (delta), biomass (delta)

Each scheduled step: MM turns the current concentrations into uptake rates -> FBA
sets exchange lower bounds (-uptake), solves, and the realized exchange fluxes +
growth update the shared substrate pools and biomass over ``interval``.

``dfba_spec`` returns just the two process specs (wire the shared stores + an
emitter yourself); ``dfba_document`` returns a complete runnable document; and
``run_dfba`` builds a core, runs it, and returns the emitter trajectory.
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Optional

MM_ADDRESS = "local:!viva_cobra.processes.michaelis_menten.MichaelisMentenUptakeProcess"
FBA_ADDRESS = "local:!viva_cobra.processes.fba.FBAStep"


def dfba_spec(
    config: Mapping,
    dt: float,
    *,
    mm_name: str = "uptake",
    fba_name: str = "fba",
    store_names: Optional[Mapping[str, str]] = None,
) -> Dict[str, dict]:
    """Build the two-process state fragment for a dFBA composite.

    config keys: ``model_file``, ``substrate_update_reactions``,
    ``kinetic_params`` ({resource: [Km, Vmax]}), and optional ``bounds`` /
    ``biomass_reaction``. ``dt`` is the interval the uptake Process is scheduled
    at. ``store_names`` optionally remaps the four shared stores
    (``substrates``, ``biomass``, ``uptakes``, ``interval``).
    """
    names = {
        "substrates": "substrates",
        "biomass": "biomass",
        "uptakes": "uptakes",
        "interval": "interval",
    }
    if store_names:
        names.update(store_names)

    substrate_rxns: Dict[str, str] = dict(config["substrate_update_reactions"])
    resources: List[str] = list(substrate_rxns.keys())

    mm_spec = {
        "_type": "process",
        "address": MM_ADDRESS,
        "config": {
            "resources": resources,
            "kinetic_params": dict(config["kinetic_params"]),
        },
        "interval": float(dt),
        "inputs": {
            "substrates": [names["substrates"]],
            "biomass": [names["biomass"]],
        },
        "outputs": {
            "uptakes": [names["uptakes"]],
            "interval": [names["interval"]],
        },
    }

    fba_spec = {
        "_type": "step",
        "address": FBA_ADDRESS,
        "config": {
            "model_file": config["model_file"],
            "substrate_update_reactions": substrate_rxns,
            "bounds": dict(config.get("bounds") or {}),
            "biomass_reaction": config.get("biomass_reaction"),
        },
        "inputs": {
            "uptakes": [names["uptakes"]],
            "substrates": [names["substrates"]],
            "biomass": [names["biomass"]],
            "interval": [names["interval"]],
        },
        "outputs": {
            "substrates": [names["substrates"]],
            "biomass": [names["biomass"]],
        },
    }

    return {mm_name: mm_spec, fba_name: fba_spec}


def dfba_document(
    config: Mapping,
    dt: float,
    initial_substrates: Mapping[str, float],
    initial_biomass: float,
) -> dict:
    """A complete, runnable dFBA document (two processes + shared stores + emitter)."""
    from process_bigraph.emitter import emitter_from_wires

    state = {
        "substrates": {k: float(v) for k, v in initial_substrates.items()},
        "biomass": float(initial_biomass),
        "uptakes": {r: 0.0 for r in initial_substrates},
        "interval": float(dt),
        "emitter": emitter_from_wires(
            {
                "global_time": ["global_time"],
                "biomass": ["biomass"],
                "substrates": ["substrates"],
            }
        ),
    }
    state.update(dfba_spec(config, dt=dt))
    return {
        "schema": {
            "substrates": "map[concentration]",
            "biomass": "mass",
            "uptakes": "overwrite[map[float]]",
            "interval": "overwrite[float]",
        },
        "state": state,
    }


def run_dfba(
    config: Mapping,
    initial_substrates: Mapping[str, float],
    initial_biomass: float = 0.01,
    *,
    steps: int = 100,
    dt: float = 0.05,
    core=None,
):
    """Build a core, run the dFBA composite ``steps * dt`` time units, return the
    emitter trajectory (a list of ``{global_time, biomass, substrates}`` dicts)."""
    from process_bigraph import Composite
    from process_bigraph.emitter import gather_emitter_results

    from viva_cobra.core import build_core

    if core is None:
        core = build_core()
    doc = dfba_document(config, dt=dt, initial_substrates=initial_substrates, initial_biomass=initial_biomass)
    composite = Composite(doc, core=core)
    composite.run(steps * dt)
    return gather_emitter_results(composite)[("emitter",)]
