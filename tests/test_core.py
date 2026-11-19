"""viva-cobra's type table is public as `viva_cobra.core:standard_types`, the name compose-api's runner imports.

compose-api's runner (compose_api/environments/runner.py, `build_core`) imports each `module:attribute` its entry
declares and calls `core.register_type(name, schema)` for every item, on a fresh `allocate_core()`. Without the
table, compose-api's `step-outputs-replace` check cannot resolve `mass`, so it missed that FBAStep's `biomass`
output adds (biosimulations/compose-api#542).
"""
from __future__ import annotations

import importlib

from process_bigraph import allocate_core

from viva_cobra.core import build_core


def test_standard_types_register_as_compose_apis_runner_does():
    module, _, attribute = "viva_cobra.core:standard_types".partition(":")
    table = getattr(importlib.import_module(module), attribute)

    core = allocate_core()
    for name, schema in table.items():
        core.register_type(name, schema)

    assert set(table) == {"concentration", "mass", "count", "bounds"}
    # `mass` is a float, so an update (FBAStep's biomass delta) adds onto the store
    added, _ = core.apply(core.access("mass"), 1.0, 0.5)
    assert added == 1.5


def test_build_core_registers_the_same_table():
    core = build_core()
    for name in ("concentration", "mass", "count", "bounds"):
        assert core.access(name) != name  # an unknown name echoes back unresolved
