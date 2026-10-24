"""FBA as a process-bigraph Step.

``FBAStep`` is non-temporal: it consumes per-resource uptake rates (e.g. from a
Michaelis-Menten uptake Process), plus the current substrate and biomass state,
solves the FBA LP under uptake-derived exchange bounds with COBRApy, and returns
deltas for substrates and biomass. The ``interval`` over which those deltas apply
is read from the input state (emitted by the upstream temporal Process).

Coupling (the dynamic-FBA contract):

    uptake u_a (mmol/gDW/hr, >= 0)  -> exchange lower_bound = -u_a
    realized flux v_a               -> delta_substrate = v_a * biomass * dt
    growth rate mu                  -> delta_biomass    = mu  * biomass * dt

Substrate deltas are clamped to the available mass (no negative pools).

Extracted from CRM_FBA (``crm_dfba/processes/fba.py``); the three module-level
helpers are framework-agnostic and usable without process-bigraph.
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Dict

import cobra
from cobra.io import load_model
from process_bigraph import Step

warnings.filterwarnings("ignore", category=UserWarning, module="cobra.util.solver")
warnings.filterwarnings("ignore", category=FutureWarning, module="cobra.medium.boundary_types")

# Bundled SBML GEMs live next to viva_cobra/models/.
_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def _load_fba_model(model_file: str, bounds: dict):
    """Load a COBRApy model by SBML filename, absolute path, or cobrapy name.

    Resolution order for an ``.xml``/``.sbml`` name: ``$VIVA_COBRA_MODELS_DIR``,
    then the bundled ``viva_cobra/models/`` dir, then the literal path. A
    non-SBML name (e.g. ``"textbook"``) is loaded via ``cobra.io.load_model``.
    """
    search_dirs = [_MODELS_DIR]
    env_dir = os.environ.get("VIVA_COBRA_MODELS_DIR")
    if env_dir:
        search_dirs.insert(0, Path(env_dir))

    if model_file.endswith(".xml") or model_file.endswith(".sbml"):
        for d in search_dirs:
            p = (d / model_file).resolve()
            if p.exists():
                return _apply_bounds(cobra.io.read_sbml_model(str(p)), bounds)
        p = Path(model_file)
        if p.exists():
            return _apply_bounds(cobra.io.read_sbml_model(str(p)), bounds)
        raise FileNotFoundError(f"SBML file not found: {model_file}")

    return _apply_bounds(load_model(model_file), bounds)


def _apply_bounds(model, bounds):
    """Apply a ``{reaction_id: {lower, upper}}`` map of static bound overrides."""
    for rxn_id, limits in (bounds or {}).items():
        rxn = model.reactions.get_by_id(rxn_id)
        lower = limits.get("lower")
        upper = limits.get("upper")
        if lower is not None:
            rxn.lower_bound = float(lower)
        if upper is not None:
            rxn.upper_bound = float(upper)
    return model


def apply_uptake_bounds(model, uptakes: Dict[str, float], substrate_rxns: Dict[str, str]) -> None:
    """Set exchange lower bounds from uptake rates: ``lower_bound = -uptake``."""
    for resource, rxn_id in substrate_rxns.items():
        u = max(0.0, float(uptakes.get(resource, 0.0)))
        rxn = model.reactions.get_by_id(rxn_id)
        lb = -u
        if rxn.upper_bound < lb:
            rxn.upper_bound = lb
        rxn.lower_bound = lb


class FBAStep(Step):
    """Steady-state FBA step for a dynamic-FBA composite.

    config:
        model_file: SBML filename / path / cobrapy name (e.g. "textbook").
        substrate_update_reactions: {resource_name: exchange_reaction_id}.
        bounds: {reaction_id: {lower, upper}} static overrides (e.g. O2, ATPM).
        biomass_reaction: objective reaction id (defaults to the model's own).
    """

    config_schema = {
        "model_file": "string",
        "substrate_update_reactions": "map[string]",
        "bounds": "map[bounds]",
        "biomass_reaction": "maybe[string]",
    }

    def initialize(self, config):
        self.substrate_rxns: Dict[str, str] = dict(config["substrate_update_reactions"])
        self.resources = list(self.substrate_rxns.keys())

        self.model = _load_fba_model(
            model_file=config["model_file"],
            bounds=config.get("bounds", {}) or {},
        )
        biomass_rxn = config.get("biomass_reaction")
        if biomass_rxn:
            self.model.objective = biomass_rxn

    def inputs(self):
        return {
            "uptakes": "map[float]",
            "substrates": "map[concentration]",
            "biomass": "mass",
            "interval": "float",
        }

    def triggers(self):
        return {"uptakes": "map[float]"}

    def outputs(self):
        return {
            "substrates": "map[count]",
            "biomass": "mass",
        }

    def update(self, state):
        uptakes = state.get("uptakes", {}) or {}
        substrates = state.get("substrates", {}) or {}
        biomass = float(state.get("biomass", 0.0))
        interval = float(state.get("interval", 0.0))

        apply_uptake_bounds(self.model, uptakes, self.substrate_rxns)
        solution = self.model.optimize()

        delta_subs = {r: 0.0 for r in self.resources}
        delta_biomass = 0.0

        if solution.status == "optimal" and interval > 0.0:
            mu = float(solution.objective_value)
            delta_biomass = mu * biomass * interval

            for resource, rxn_id in self.substrate_rxns.items():
                flux = float(solution.fluxes[rxn_id]) * biomass * interval
                available = float(substrates.get(resource, 0.0))
                delta_subs[resource] = max(flux, -available)

        return {
            "substrates": delta_subs,
            "biomass": delta_biomass,
        }
