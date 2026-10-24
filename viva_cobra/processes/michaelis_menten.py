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
    """Temporal uptake Process for a dynamic-FBA composite.

    config:
        resources: list of resource names.
        kinetic_params: {resource: [Km, Vmax]} half-saturation + max velocity.
    """

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
