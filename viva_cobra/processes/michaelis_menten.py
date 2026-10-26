"""Michaelis-Menten (Monod) substrate uptake as a process-bigraph Process.

Reads the current substrate concentrations and biomass, and emits per-resource
uptake rates a downstream ``FBAStep`` turns into exchange-reaction bounds:

    v_a = Vmax_a * S_a / (Km_a + S_a)      (mmol / gDW / hr, >= 0)

``MichaelisMentenUptakeProcess`` is the temporal Process in a dynamic-FBA
composite: its scheduled interval drives the loop and is emitted alongside the
uptakes so the non-temporal FBA Step can convert fluxes into deltas.

Generalized from CRM_FBA's ``MonodCRM`` + ``CRMProcess``; the kinetics match
spatio-flux / multiscale-bats DynamicFBA.
"""
from __future__ import annotations

from typing import Dict, Mapping, Sequence

from process_bigraph import Process

try:  # contract metadata is optional — never break import if unavailable
    from bigraph_schema.contract import ProcessContract
except Exception:  # pragma: no cover - defensive
    ProcessContract = None

# A tiny epsilon keeps the rate finite and zero at S = 0 (matches CRM_FBA).
_EPS = 1e-12


def mm_uptake_rates(
    substrates: Mapping[str, float],
    kinetic_params: Mapping[str, Sequence[float]],
    resources: Sequence[str],
) -> Dict[str, float]:
    """Michaelis-Menten uptake rate per resource: ``Vmax * S / (Km + S)``.

    ``kinetic_params`` maps each resource to ``(Km, Vmax)``. Negative or missing
    concentrations are treated as zero (rate 0).
    """
    out: Dict[str, float] = {}
    for r in resources:
        Km, Vmax = kinetic_params[r][0], kinetic_params[r][1]
        S = max(0.0, float(substrates.get(r, 0.0)))
        out[r] = float(Vmax) * S / (float(Km) + S + _EPS)
    return out


class MichaelisMentenUptakeProcess(Process):
    """Michaelis-Menten (Monod) uptake — Vₐ = Vmaxₐ·Sₐ/(Kmₐ + Sₐ) per resource.

    Temporal Process in a dynamic-FBA composite: each step it reads the current
    substrate concentrations, computes a saturating per-resource uptake rate,
    and emits those rates (plus the scheduling interval) for a downstream
    ``FBAStep`` to turn into exchange-reaction bounds.

        Vₐ = Vmaxₐ · Sₐ / (Kmₐ + Sₐ)      (mmol / gDW / hr, ≥ 0)

    config:
        resources: list of resource names.
        kinetic_params: {resource: [Km, Vmax]} half-saturation + max velocity.
    """

    description = (
        "Michaelis-Menten (Monod) substrate uptake — a saturating per-resource "
        "uptake rate Vₐ = Vmaxₐ·Sₐ/(Kmₐ + Sₐ) that rises linearly at low "
        "substrate and plateaus at Vmaxₐ once Sₐ ≫ Kmₐ.\n"
        "   Vₐ = Vmaxₐ · Sₐ / (Kmₐ + Sₐ)   (mmol / gDW / hr, ≥ 0); "
        "concentrations are floored at zero so the rate is 0 at Sₐ = 0."
    )

    if ProcessContract is not None:
        contract = ProcessContract(
            summary=(
                "Per-resource Michaelis-Menten uptake rate "
                "Vₐ = Vmaxₐ·Sₐ/(Kmₐ+Sₐ), saturating in substrate, emitted with "
                "the scheduling interval to drive a downstream FBA step."
            ),
            symbols={
                "Vₐ": "uptake rate of resource a (mmol / gDW / hr, ≥ 0) — the emitted `uptakes[a]`",
                "Sₐ": "concentration of resource a in the environment (floored at 0)",
                "Vmaxₐ": "maximum uptake velocity of resource a (mmol / gDW / hr)",
                "Kmₐ": "half-saturation constant of resource a: Sₐ at which Vₐ = Vmaxₐ/2",
                "ε": "numerical epsilon (1e-12) keeping the rate finite and exactly 0 at Sₐ = 0",
                "Δt": "scheduling interval of the Process, echoed as `interval`",
            },
            math=[
                r"V_a = \frac{V^{\max}_a\,S_a}{K^m_a + S_a + \varepsilon}",
                r"S_a \leftarrow \max(0,\, S_a)",
                r"\lim_{S_a \to \infty} V_a = V^{\max}_a, \qquad V_a\big|_{S_a = K^m_a} = \tfrac{1}{2}V^{\max}_a",
            ],
            inputs={
                "substrates": "Current environmental concentrations per resource (`map[concentration]`); negative/missing values read as 0.",
                "biomass": "Current biomass (`mass`); read for composite wiring, not used in the rate law.",
            },
            outputs={
                "uptakes": "Per-resource uptake rates Vₐ (`overwrite[map[float]]`), one entry per configured resource, ≥ 0.",
                "interval": "The scheduling interval Δt (`overwrite[float]`), echoed so the non-temporal FBA step can convert fluxes to deltas.",
            },
            config={
                "resources": "List of resource names the Process emits uptakes for.",
                "kinetic_params": "{resource: [Km, Vmax]} — half-saturation constant and maximum velocity per resource; every resource must be present or initialize() raises.",
            },
            assumptions=[
                "Uptake follows single-substrate Michaelis-Menten / Monod kinetics, independent per resource (no competitive coupling).",
                "Substrate concentrations are floored at zero; the rate is non-negative and 0 at Sₐ = 0.",
                "Km and Vmax are constant over the simulation (no enzyme regulation or adaptation).",
            ],
        )

    config_schema = {
        "resources": "list[string]",
        # Carried as `schema` (arbitrary nested value) so the {resource: [Km, Vmax]}
        # map passes through un-typed, as CRM_FBA does for its CRM params.
        "kinetic_params": "schema",
    }

    def initialize(self, config):
        self.resources = list(config["resources"])
        kp = config.get("kinetic_params") or {}
        missing = set(self.resources) - set(kp)
        if missing:
            raise ValueError(
                f"MichaelisMentenUptakeProcess missing kinetic_params for: {sorted(missing)}"
            )
        # Normalize each entry to a (Km, Vmax) float pair (tuples JSON-roundtrip to lists).
        self.kinetic_params: Dict[str, tuple] = {
            r: (float(kp[r][0]), float(kp[r][1])) for r in self.resources
        }

    def inputs(self):
        return {
            "substrates": "map[concentration]",
            "biomass": "mass",
        }

    def outputs(self):
        return {
            "uptakes": "overwrite[map[float]]",
            "interval": "overwrite[float]",
        }

    def update(self, inputs, interval):
        substrates: Dict[str, float] = inputs.get("substrates", {}) or {}
        uptakes = mm_uptake_rates(substrates, self.kinetic_params, self.resources)
        return {
            "uptakes": {r: float(uptakes.get(r, 0.0)) for r in self.resources},
            "interval": float(interval),
        }
