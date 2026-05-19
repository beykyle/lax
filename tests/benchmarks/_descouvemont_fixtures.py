from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np

_DATA_DIR = Path(__file__).with_name("data")


@dataclass(frozen=True)
class NpJ1Reference:
    """Published coupled ``n-p`` benchmark outputs for one mesh setup."""

    scale: float
    n_basis: int
    n_intervals: int
    energies: np.ndarray
    phase_11: np.ndarray
    phase_22: np.ndarray
    eta_12: np.ndarray


@dataclass(frozen=True)
class CoupledColumnReference:
    """Published first-column amplitudes and phases for one coupled setup."""

    scale: float
    n_basis: int
    n_intervals: int
    energies: np.ndarray
    amplitudes: tuple[np.ndarray, ...]
    phases: tuple[np.ndarray, ...]


def load_np_j1_references() -> tuple[NpJ1Reference, ...]:
    """Load the Descouvemont coupled ``n-p`` references from JSON."""

    payload = _load_payload("descouvemont_np_j1.json")
    return tuple(_np_j1_reference(entry) for entry in _payload_entries(payload, "references"))


def load_o16_ca44_references() -> tuple[CoupledColumnReference, ...]:
    """Load the ``16O + 44Ca`` benchmark references from JSON."""

    payload = _load_payload("descouvemont_o16_ca44.json")
    return tuple(
        _coupled_column_reference(entry) for entry in _payload_entries(payload, "references")
    )


def load_alpha_c12_references() -> tuple[CoupledColumnReference, ...]:
    """Load the ``α + 12C`` benchmark references from JSON."""

    payload = _load_payload("descouvemont_alpha_c12.json")
    return tuple(
        _coupled_column_reference(entry) for entry in _payload_entries(payload, "references")
    )


def load_alpha_c12_single_interval_demo() -> CoupledColumnReference:
    """Load the full-precision single-interval ``α + 12C`` demo reference."""

    payload = _load_payload("descouvemont_alpha_c12.json")
    return _coupled_column_reference(_payload_value(payload, "single_interval_demo"))


@cache
def _load_payload(filename: str) -> dict[str, object]:
    """Load and cache one JSON benchmark payload."""

    path = _DATA_DIR / filename
    return json.loads(path.read_text(encoding="utf-8"))


def _np_j1_reference(entry: object) -> NpJ1Reference:
    """Convert one JSON object into an ``NpJ1Reference``."""

    values = _as_dict(entry)
    return NpJ1Reference(
        scale=_float_value(values, "scale"),
        n_basis=_int_value(values, "n_basis"),
        n_intervals=_int_value(values, "n_intervals"),
        energies=_float_array(_payload_value(values, "energies")),
        phase_11=_float_array(_payload_value(values, "phase_11")),
        phase_22=_float_array(_payload_value(values, "phase_22")),
        eta_12=_float_array(_payload_value(values, "eta_12")),
    )


def _coupled_column_reference(entry: object) -> CoupledColumnReference:
    """Convert one JSON object into a ``CoupledColumnReference``."""

    values = _as_dict(entry)
    return CoupledColumnReference(
        scale=_float_value(values, "scale"),
        n_basis=_int_value(values, "n_basis"),
        n_intervals=_int_value(values, "n_intervals"),
        energies=_float_array(_payload_value(values, "energies")),
        amplitudes=_tuple_of_float_arrays(_payload_value(values, "amplitudes")),
        phases=_tuple_of_float_arrays(_payload_value(values, "phases")),
    )


def _tuple_of_float_arrays(values: object) -> tuple[np.ndarray, ...]:
    """Convert nested JSON numeric lists into float arrays."""

    rows = _as_list(values)
    return tuple(_float_array(row) for row in rows)


def _float_array(values: object) -> np.ndarray:
    """Convert a JSON numeric list into a ``float64`` array."""

    return np.asarray(_as_list(values), dtype=np.float64)


def _as_dict(value: object) -> dict[str, object]:
    """Narrow a JSON value to a dictionary."""

    if not isinstance(value, dict):
        msg = f"Expected a JSON object, got {type(value)!r}."
        raise TypeError(msg)
    return {str(key): nested for key, nested in value.items()}


def _as_list(value: object) -> list[object]:
    """Narrow a JSON value to a list."""

    if not isinstance(value, list):
        msg = f"Expected a JSON list, got {type(value)!r}."
        raise TypeError(msg)
    return list(value)


def _payload_entries(payload: dict[str, object], key: str) -> list[object]:
    """Return a list-valued payload field."""

    return _as_list(_payload_value(payload, key))


def _payload_value(payload: dict[str, object], key: str) -> object:
    """Return one required payload field."""

    if key not in payload:
        msg = f"Missing required JSON field {key!r}."
        raise KeyError(msg)
    return payload[key]


def _float_value(payload: dict[str, object], key: str) -> float:
    """Return one numeric payload field as ``float``."""

    value = _payload_value(payload, key)
    if not isinstance(value, (int, float)):
        msg = f"Expected numeric JSON field {key!r}, got {type(value)!r}."
        raise TypeError(msg)
    return float(value)


def _int_value(payload: dict[str, object], key: str) -> int:
    """Return one numeric payload field as ``int``."""

    value = _payload_value(payload, key)
    if not isinstance(value, int):
        msg = f"Expected integer JSON field {key!r}, got {type(value)!r}."
        raise TypeError(msg)
    return value


__all__ = [
    "CoupledColumnReference",
    "NpJ1Reference",
    "load_alpha_c12_references",
    "load_alpha_c12_single_interval_demo",
    "load_np_j1_references",
    "load_o16_ca44_references",
]
