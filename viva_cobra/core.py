"""Workspace core factory for the vivarium-workbench / process-bigraph.

``build_core`` returns a process-bigraph core with viva-cobra's domain types and
processes registered, so a workbench (or a downstream repo's ``build_core``) can
introspect the Registry and realize the dFBA composite.

viva-cobra's schemas use a handful of domain type-names (``concentration``,
``mass``, ``count``, ``bounds``) that are not part of the process-bigraph base
types; they are registered here (and should be re-applied as composite
``core_extensions`` so composites realize on whatever core builds them). Process
specs address their classes by dynamic-import address
(``local:!viva_cobra.processes.fba.FBAStep`` etc.); the link registrations below
are for discovery + friendly names.

Extracted from CRM_FBA (``crm_dfba/core.py``).
"""
from __future__ import annotations

from process_bigraph import allocate_core

from viva_cobra.processes.fba import FBAStep
from viva_cobra.processes.michaelis_menten import MichaelisMentenUptakeProcess

# concentration / mass / count are non-negative scalars (floats); bounds is a
# per-reaction {lower, upper} exchange-flux window (map[bounds] in configs).
_TYPE_DEFS = {
    "concentration": {"_type": "float", "_default": 0.0},
    "mass": {"_type": "float", "_default": 0.0},
    "count": {"_type": "float", "_default": 0.0},
    "bounds": {
        "lower": {"_type": "float", "_default": 0.0},
        "upper": {"_type": "float", "_default": 1000.0},
    },
}


def _is_registered(core, name: str) -> bool:
    """True if ``name`` resolves to a real schema (unknown names echo back)."""
    try:
        resolved = core.access(name)
    except Exception:
        return False
    return not (isinstance(resolved, str) and resolved == name)


def register_types(core):
    """Register viva-cobra's domain types on ``core`` (idempotent)."""
    for name, schema in _TYPE_DEFS.items():
        if not _is_registered(core, name):
            core.register_type(name, schema)
    return core


def register_processes(core):
    """Register viva-cobra's process/step links under friendly names."""
    core.register_link("FBAStep", FBAStep)
    core.register_link("MichaelisMentenUptakeProcess", MichaelisMentenUptakeProcess)
    return core


def build_core(core=None):
    """Return a process-bigraph core with viva-cobra's types + processes.

    Pass an existing ``core`` to compose viva-cobra's registrations onto it (so a
    downstream repo can inherit them); omit it to get a fresh ``allocate_core()``.
    """
    if core is None:
        core = allocate_core()
    register_types(core)
    register_processes(core)
    return core
