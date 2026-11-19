# viva-cobra

COBRApy flux-balance analysis packaged as **process-bigraph** components: a
reactive **FBA step**, a **Michaelis-Menten uptake process**, and a **dynamic-FBA
composite** that wires them together. It bridges the real
[COBRApy](https://opencobra.github.io/cobrapy/) solver — no reimplementation — so
any SBML genome-scale metabolic model (GEM) can be driven dynamically inside a
larger multiscale simulation.

It is the standalone home for the COBRApy dFBA pattern that previously lived,
duplicated, inside `CRM_FBA` (`crm_dfba`) and `multiscale-bats` — both can import
`viva_cobra` instead of carrying their own copy.

## What's in it

| Component | Kind | Role |
|---|---|---|
| `DynamicFBAProcess` | process-bigraph `Process` (temporal) | The whole dFBA step in one Process: Michaelis-Menten uptake from the current concentrations, then the FBA solve, returning substrate + biomass deltas over its own interval. The form compose-api runs (below). |
| `FBAStep` | process-bigraph `Step` (reactive) | Loads a GEM, sets exchange lower bounds from uptake rates, solves the FBA LP, returns substrate + biomass deltas. |
| `MichaelisMentenUptakeProcess` | process-bigraph `Process` (temporal) | Turns substrate concentrations into uptake rates `v = Vmax·S/(Km+S)`, and drives the loop's `interval`. |
| `dfba_spec` / `dfba_document` / `run_dfba` | composite | Wires the two through shared `substrates` / `biomass` / `uptakes` / `interval` stores. |
| `build_core` | core factory | Registers the domain types (`concentration`, `mass`, `count`, `bounds`) + process links. |
| `MODEL_REGISTRY` | metadata | Bundled GEMs (E. coli core, iAF1260, iMM904, iCN900, iJN746, iNF517) with biomass + exchange mappings. |

## The dynamic-FBA contract

Each scheduled step of `dt`:

```
uptake u_a = Vmax_a · S_a / (Km_a + S_a)      (Michaelis-Menten, per resource)
exchange lower_bound  = -u_a                   (FBA constraint)
solve FBA  ->  growth rate mu, realized fluxes v_a
delta_biomass    = mu · biomass · dt
delta_substrate  = v_a · biomass · dt          (clamped to the available pool)
```

## Quickstart

```python
from viva_cobra import run_dfba

config = {
    "model_file": "textbook",                       # cobrapy-bundled E. coli core
    "substrate_update_reactions": {"glucose": "EX_glc__D_e", "acetate": "EX_ac_e"},
    "bounds": {"EX_o2_e": {"lower": -20.0, "upper": 1000.0},
               "ATPM": {"lower": 1.0, "upper": 1.0}},
    "biomass_reaction": "Biomass_Ecoli_core",
    "kinetic_params": {"glucose": [0.5, 10.0], "acetate": [0.5, 2.0]},  # (Km, Vmax)
}

history = run_dfba(config, initial_substrates={"glucose": 11.1, "acetate": 0.0},
                   initial_biomass=0.01, steps=100, dt=0.05)
# history: [{global_time, biomass, substrates}, ...]
```

Build the composite into a larger model with `dfba_spec(config, dt)` (returns the
two process specs — you provide the shared stores + an emitter), or use
`dfba_document(...)` for a complete runnable document.

### One Process instead of the pair

`DynamicFBAProcess` takes the same `config` and is wired to just the
`substrates` and `biomass` stores. It computes every interval exactly as the
pair does, but its deltas apply once per interval because it is a Process. A
Step's deltas apply every time it fires: when it is built on seeded uptakes
(so `run_dfba`'s first row already shows one interval of growth at t = 0), when
a wired input changes, or after a save and reload. compose-api refuses Steps
whose outputs add for that reason, so it runs `DynamicFBAProcess`, not `FBAStep`.

```python
"dfba": {
    "_type": "process",
    "address": "local:viva_cobra.processes.fba.DynamicFBAProcess",
    "config": config,          # as above, kinetic_params included
    "interval": 0.05,
    "inputs":  {"substrates": ["substrates"], "biomass": ["biomass"]},
    "outputs": {"substrates": ["substrates"], "biomass": ["biomass"]},
}
```

## Install

```bash
pip install -e .          # deps: cobra, process-bigraph, bigraph-schema, numpy
pytest
```

## Bundled models

SBML GEMs ship under `viva_cobra/models/` and resolve by filename (e.g.
`"model_file": "iAF1260.xml"`); `"textbook"` loads the cobrapy-bundled core. Point
`VIVA_COBRA_MODELS_DIR` at another directory to add your own.

## Provenance

The FBA step, bundled GEMs, and Monod kinetics were extracted from `CRM_FBA`
(`crm_dfba`); the design mirrors the more polished `multiscale-bats` FBA/MM
processes. The kinetics match spatio-flux / multiscale-bats DynamicFBA.
