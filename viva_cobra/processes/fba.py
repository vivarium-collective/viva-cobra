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

try:  # contract metadata is optional — never break import if unavailable
    from bigraph_schema.contract import ProcessContract
except Exception:  # pragma: no cover - defensive
    ProcessContract = None

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
    """Steady-state flux-balance analysis as a non-temporal Step.

    Solves the FBA linear program — maximize the biomass objective subject to
    the steady-state mass-balance S·v = 0 and flux bounds lb ≤ v ≤ ub — with
    exchange lower bounds set from the incoming uptake rates, then converts the
    optimal growth rate and exchange fluxes into biomass and substrate deltas
    over the interval Δt supplied by the upstream temporal Process:

        max  cᵀv   s.t.   S·v = 0,   lb ≤ v ≤ ub      (lb_exch,a = -uₐ)
        Δbiomass    = μ · biomass · Δt
        Δsubstrateₐ = vₐ · biomass · Δt   (clamped so the pool stays ≥ 0)

    config:
        model_file: SBML filename / path / cobrapy name (e.g. "textbook").
        substrate_update_reactions: {resource_name: exchange_reaction_id}.
        bounds: {reaction_id: {lower, upper}} static overrides (e.g. O2, ATPM).
        biomass_reaction: objective reaction id (defaults to the model's own).
    """

    description = (
        "Steady-state flux-balance analysis (FBA) LP: maximize the biomass "
        "objective cᵀv subject to the mass-balance constraint S·v = 0 and flux "
        "bounds lb ≤ v ≤ ub, with exchange lower bounds driven by the incoming "
        "uptake rates (lb = -u). The optimal growth rate μ and exchange fluxes "
        "are integrated over the interval Δt into biomass and substrate deltas.\n"
        "   max cᵀv  s.t.  S·v = 0,  lb ≤ v ≤ ub,  lb_exch,a = -uₐ\n"
        "   Δbiomass = μ·biomass·Δt;   Δsubstrateₐ = vₐ·biomass·Δt (clamped ≥ -pool)."
    )

    if ProcessContract is not None:
        contract = ProcessContract(
            summary=(
                "Steady-state FBA linear program (max cᵀv s.t. S·v = 0, "
                "lb ≤ v ≤ ub) with uptake-driven exchange bounds, returning "
                "biomass and substrate deltas over the step interval."
            ),
            symbols={
                "v": "vector of reaction fluxes (mmol / gDW / hr); the LP decision variables",
                "S": "stoichiometric matrix Sᵢⱼ (metabolite i × reaction j)",
                "c": "objective coefficient vector selecting the biomass reaction",
                "cᵀv": "objective value = growth rate μ of the biomass reaction",
                "μ": "optimal growth rate = solution.objective_value (1/hr)",
                "lb": "per-reaction lower flux bounds (uptake direction for exchanges)",
                "ub": "per-reaction upper flux bounds (secretion direction for exchanges)",
                "uₐ": "incoming uptake rate of resource a (≥ 0); sets exchange lb_exch,a = -uₐ",
                "vₐ": "optimal exchange flux of resource a at the solution (signed)",
                "Δt": "interval over which fluxes are integrated (the input `interval`)",
                "Δbiomass": "biomass change this step = μ·biomass·Δt",
                "Δsubstrateₐ": "substrate change = vₐ·biomass·Δt, clamped so the pool stays ≥ 0",
            },
            math=[
                r"\max_{v}\; c^{\mathsf{T}} v \quad \text{s.t.}\quad S\,v = 0,\; \mathrm{lb} \le v \le \mathrm{ub}",
                r"\mathrm{lb}^{\mathrm{exch}}_a = -u_a,\qquad u_a = \max(0,\, u_a)",
                r"\mu = c^{\mathsf{T}} v^{\star}, \qquad \Delta\mathrm{biomass} = \mu\,\cdot\,\mathrm{biomass}\,\cdot\,\Delta t",
                r"\Delta\mathrm{substrate}_a = \max\!\big(v^{\star}_a\,\cdot\,\mathrm{biomass}\,\cdot\,\Delta t,\; -\mathrm{pool}_a\big)",
            ],
            inputs={
                "uptakes": "Per-resource uptake rates uₐ (`map[float]`, ≥ 0); set exchange lower bounds. Also the Step's trigger.",
                "substrates": "Current substrate pools (`map[concentration]`); used to clamp each delta so no pool goes negative.",
                "biomass": "Current biomass (`mass`); scales fluxes and growth into absolute deltas.",
                "interval": "Timestep Δt (`float`) emitted by the upstream temporal Process; deltas are zero when Δt ≤ 0.",
            },
            outputs={
                "substrates": "Per-resource substrate deltas Δsubstrateₐ (`map[count]`), clamped to not exceed the available pool.",
                "biomass": "Biomass delta Δbiomass = μ·biomass·Δt (`mass`).",
            },
            config={
                "model_file": "SBML filename / absolute path / cobrapy model name (e.g. \"textbook\") for the genome-scale model.",
                "substrate_update_reactions": "{resource_name: exchange_reaction_id} mapping each tracked resource to its exchange reaction.",
                "bounds": "{reaction_id: {lower, upper}} static flux-bound overrides applied at load (e.g. O2, ATPM).",
                "biomass_reaction": "Objective reaction id; defaults to the model's own objective when omitted.",
            },
            assumptions=[
                "Metabolism is at quasi-steady state within each step: S·v = 0 (no net internal metabolite accumulation).",
                "The LP maximizes a single linear biomass objective; a non-optimal solver status yields zero deltas.",
                "Uptake rates set only exchange lower bounds (lb = -u); substrate deltas are clamped so pools never go negative.",
                "Fluxes and growth are constant over the interval Δt (explicit forward-Euler integration of dynamic FBA).",
            ],
        )

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
