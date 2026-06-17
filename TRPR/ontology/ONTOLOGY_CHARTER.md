# Packet Ontology Charter

`packet_ontology.yaml` is **sacred**.

## Rules

1. **Single source of truth** — All packet primitives, derived metrics, regimes, and lifecycle states live here.
2. **No drift** — Do not rename keys, add domain-specific fields, or fork vocabulary in engine code or side YAML files.
3. **Map inward** — Every future project maps its local metrics *to* these primitives; primitives do not bend to the project.
4. **Read-only at runtime** — Engines may load and reference this file; they must not write or mutate it.
5. **Generalize before adding** — New entries require cross-domain justification (market, repo, aviation, institution, ecology, or equivalent).

## Where things live

| Layer | Path | Role |
|-------|------|------|
| Sacred ontology | `TRPR/ontology/packet_ontology.yaml` | Shared vocabulary |
| Read-only loader | `TRPR/ontology/packet_ontology_loader.py` | Optional import for engines |
| Domain mapping | `config/pressure_ontology.yaml` | Finance-crow market sensor → primitive map (references sacred file) |

## Engine integration (optional, not required yet)

```python
from TRPR.ontology.packet_ontology_loader import load_packet_ontology, classify_regime

ontology = load_packet_ontology()  # read-only
regime = classify_regime(0.42, metric="latent_rupture_potential")
```

Do **not** hardwire every engine to depend on the loader until a deliberate integration pass.

## Change process

1. Propose addition in a design note (why it generalizes).
2. Update `packet_ontology.yaml` and bump `schema_version` only if the contract changes.
3. Update domain mapping files — never duplicate primitive definitions elsewhere.
