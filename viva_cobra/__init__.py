"""viva-cobra — COBRApy flux-balance analysis as process-bigraph components.

Public API:

    FBAStep                        COBRApy FBA as a reactive Step
    MichaelisMentenUptakeProcess   Michaelis-Menten substrate uptake Process
    mm_uptake_rates                the MM kinetics as a plain function
    dfba_spec / dfba_document / run_dfba   the dynamic-FBA composite
    build_core                     process-bigraph core with viva-cobra types+links
    MODEL_REGISTRY / get_model_spec   bundled GEM metadata
"""
from __future__ import annotations

from viva_cobra.composites.dfba import dfba_document, dfba_spec, run_dfba
from viva_cobra.core import build_core, register_processes, register_types
from viva_cobra.models import MODEL_REGISTRY, get_model_spec
from viva_cobra.processes.fba import (
    FBAStep,
    apply_uptake_bounds,
)
from viva_cobra.processes.michaelis_menten import (
    MichaelisMentenUptakeProcess,
    mm_uptake_rates,
)

__all__ = [
    "FBAStep",
    "MichaelisMentenUptakeProcess",
    "mm_uptake_rates",
    "apply_uptake_bounds",
    "dfba_spec",
    "dfba_document",
    "run_dfba",
    "build_core",
    "register_types",
    "register_processes",
    "MODEL_REGISTRY",
    "get_model_spec",
]

__version__ = "0.1.0"
