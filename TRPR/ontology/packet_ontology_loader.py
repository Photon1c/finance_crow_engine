"""Read-only loader for the sacred packet ontology.

Engines may import this module optionally. No engine is required to depend on it yet.
Do not mutate loaded ontology data at runtime.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import yaml

ONTOLOGY_PATH = Path(__file__).resolve().parent / "packet_ontology.yaml"


class OntologyReadError(RuntimeError):
    """Raised when the sacred ontology cannot be loaded."""


@lru_cache(maxsize=1)
def load_packet_ontology(*, path: Optional[Path] = None) -> dict[str, Any]:
    """Load packet_ontology.yaml read-only. Cached after first read."""
    ontology_path = Path(path) if path is not None else ONTOLOGY_PATH
    if not ontology_path.exists():
        raise OntologyReadError(f"Sacred ontology not found: {ontology_path}")

    with ontology_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict):
        raise OntologyReadError(f"Invalid ontology structure in {ontology_path}")

    return data


def schema_version(*, path: Optional[Path] = None) -> str:
    ontology = load_packet_ontology(path=path)
    metadata = ontology.get("metadata", {})
    return str(metadata.get("schema_version", "unknown"))


def get_primitive(name: str, *, path: Optional[Path] = None) -> dict[str, Any]:
    ontology = load_packet_ontology(path=path)
    primitives = ontology.get("core_primitives", {})
    if name not in primitives:
        raise KeyError(f"Unknown core primitive: {name}")
    return dict(primitives[name])


def get_derived_metric(name: str, *, path: Optional[Path] = None) -> dict[str, Any]:
    """Lookup by derived metric block name (e.g. latent_rupture_potential) or key (e.g. LRP)."""
    ontology = load_packet_ontology(path=path)
    derived = ontology.get("derived_metrics", {})
    if name in derived:
        return dict(derived[name])

    for block_name, block in derived.items():
        if isinstance(block, dict) and block.get("key") == name:
            return {"name": block_name, **block}

    raise KeyError(f"Unknown derived metric: {name}")


def get_packet_type(name: str, *, path: Optional[Path] = None) -> dict[str, Any]:
    ontology = load_packet_ontology(path=path)
    types = ontology.get("packet_types", {})
    if name not in types:
        raise KeyError(f"Unknown packet type: {name}")
    return dict(types[name])


def classify_regime(
    value: float,
    *,
    metric: str = "latent_rupture_potential",
    path: Optional[Path] = None,
) -> str:
    """Classify a normalized score using sacred regime ranges."""
    ontology = load_packet_ontology(path=path)
    regimes = ontology.get("regimes", {})
    score = float(value)

    for regime_name, spec in regimes.items():
        if not isinstance(spec, dict):
            continue
        bounds = spec.get("range", [])
        if len(bounds) == 2 and bounds[0] <= score < bounds[1]:
            return regime_name.upper()

    if regimes:
        last = list(regimes.keys())[-1]
        last_range = regimes[last].get("range", [0.0, 1.0])
        if score >= last_range[-1]:
            return last.upper()

    return ""


def list_alert_ids(*, path: Optional[Path] = None) -> list[str]:
    ontology = load_packet_ontology(path=path)
    alerts = ontology.get("alerts", {})
    return list(alerts.keys())


def list_named_regimes(*, path: Optional[Path] = None) -> list[str]:
    """Return sacred named field regime keys (e.g. RESTORED_EQUILIBRIUM)."""
    ontology = load_packet_ontology(path=path)
    named = ontology.get("named_regimes", {})
    keys = []
    for spec in named.values():
        if isinstance(spec, dict) and spec.get("key"):
            keys.append(str(spec["key"]))
    return keys


def get_named_regime(name: str, *, path: Optional[Path] = None) -> dict[str, Any]:
    """Lookup named regime by block name (restored_equilibrium) or key (RESTORED_EQUILIBRIUM)."""
    ontology = load_packet_ontology(path=path)
    named = ontology.get("named_regimes", {})
    if name in named:
        return dict(named[name])
    for block_name, block in named.items():
        if isinstance(block, dict) and block.get("key") == name:
            return {"name": block_name, **block}
    raise KeyError(f"Unknown named regime: {name}")


def ontology_reference() -> str:
    """Return canonical relative path string for documentation and mapping files."""
    return "TRPR/ontology/packet_ontology.yaml"
