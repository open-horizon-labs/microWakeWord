"""Fail-closed checks for declared and realized Kizz batch mixtures.

This is deliberately a small sidecar around the existing stratified sampler.
The sampler remains responsible for drawing examples; this module verifies
that its declared and observed distributions are the distributions the recipe
actually permits.
"""

from __future__ import annotations

import math
from collections.abc import Mapping


SCHEMA_VERSION = 1


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be finite and between zero and one")
    return value


def _targets(guard: Mapping, kind: str) -> Mapping:
    expected = guard.get("expected", {})
    targets = expected.get(kind, {})
    if not isinstance(targets, Mapping) or not targets:
        raise ValueError(f"mixture_guard.expected.{kind} must be a non-empty map")
    return targets


def _tolerance(guard: Mapping, name: str) -> float:
    tolerances = guard.get("tolerances", {})
    if not isinstance(tolerances, Mapping):
        raise ValueError("mixture_guard.tolerances must be a map")
    return _number(tolerances.get(name), f"tolerances.{name}")


def _target_value(target: object, field: str, name: str) -> float:
    if not isinstance(target, Mapping):
        raise ValueError(f"mixture target {name!r} must be a map")
    return _number(target.get(field), f"expected target {name}.{field}")


def _check_distribution(
    actual: Mapping[str, Mapping],
    targets: Mapping,
    *,
    field: str,
    target_field: str,
    tolerance: float,
    require_all: bool,
    label: str,
) -> None:
    actual_names = set(actual)
    target_names = set(targets)
    if require_all and actual_names != target_names:
        missing = sorted(actual_names - target_names)
        extra = sorted(target_names - actual_names)
        raise ValueError(
            f"{label} declaration does not exactly cover active entries; "
            f"undeclared={missing}, unused={extra}"
        )
    for name, target in targets.items():
        if name not in actual:
            raise ValueError(f"{label} target {name!r} is not realized")
        expected = _target_value(target, target_field, f"{label}.{name}")
        observed = _number(actual[name].get(field), f"{label}.{name}.{field}")
        if abs(observed - expected) > tolerance + 1e-12:
            raise ValueError(
                f"{label} {name!r} {field} {observed:.6f} differs from "
                f"declared {expected:.6f} by more than {tolerance:.6f}"
            )


def validate_mixture_guard(guard: Mapping) -> None:
    """Validate the guard's shape before it can be used permissively."""
    if not isinstance(guard, Mapping) or guard.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("mixture_guard requires schema_version 1")
    _tolerance(guard, "sample_share")
    _tolerance(guard, "weighted_pressure_share")
    for kind in ("classes", "groups"):
        targets = _targets(guard, kind)
        sample_total = sum(
            _target_value(item, "sample_share", f"{kind}.{name}")
            for name, item in targets.items()
        )
        pressure_total = sum(
            _target_value(item, "weighted_pressure_share", f"{kind}.{name}")
            for name, item in targets.items()
        )
        if abs(sample_total - 1.0) > 1e-9:
            raise ValueError(
                f"mixture_guard.expected.{kind} sample shares must sum to one"
            )
        if abs(pressure_total - 1.0) > 1e-9:
            raise ValueError(
                f"mixture_guard.expected.{kind} weighted pressure shares "
                "must sum to one"
            )
    if guard.get("require_all_active_groups", True) is not True:
        raise ValueError("mixture_guard.require_all_active_groups must be true")
    minimum_samples = guard.get("minimum_realized_samples", 1)
    if (
        isinstance(minimum_samples, bool)
        or not isinstance(minimum_samples, int)
        or minimum_samples < 1
    ):
        raise ValueError(
            "mixture_guard.minimum_realized_samples must be a positive integer"
        )


def validate_declared_mixture(summary: Mapping, guard: Mapping) -> None:
    """Fail closed if the expanded plan is not the declared mixture."""
    validate_mixture_guard(guard)
    _check_distribution(
        summary.get("classes", {}),
        _targets(guard, "classes"),
        field="sampling_share",
        target_field="sample_share",
        tolerance=_tolerance(guard, "sample_share"),
        require_all=True,
        label="class sampling share",
    )
    _check_distribution(
        summary.get("groups", {}),
        _targets(guard, "groups"),
        field="sampling_share",
        target_field="sample_share",
        tolerance=_tolerance(guard, "sample_share"),
        require_all=True,
        label="group sampling share",
    )
    _check_distribution(
        summary.get("classes", {}),
        _targets(guard, "classes"),
        field="weighted_pressure_share",
        target_field="weighted_pressure_share",
        tolerance=_tolerance(guard, "weighted_pressure_share"),
        require_all=True,
        label="class weighted pressure share",
    )
    _check_distribution(
        summary.get("groups", {}),
        _targets(guard, "groups"),
        field="weighted_pressure_share",
        target_field="weighted_pressure_share",
        tolerance=_tolerance(guard, "weighted_pressure_share"),
        require_all=True,
        label="group weighted pressure share",
    )


def validate_realized_mixture(ledger: Mapping, guard: Mapping) -> None:
    """Fail closed after training if the sampling ledger drifted."""
    validate_mixture_guard(guard)
    if ledger.get("mixture_guard") != guard:
        raise ValueError(
            "sampling ledger mixture_guard does not exactly match the recipe guard"
        )
    minimum = guard["minimum_realized_samples"]
    total = ledger.get("total_samples", 0)
    if isinstance(total, bool) or not isinstance(total, int) or total < minimum:
        raise ValueError(
            f"realized training sample count {total} is below the guarded minimum {minimum}"
        )
    for actual, kind in (
        (ledger.get("realized_classes", {}), "classes"),
        (ledger.get("realized_groups", {}), "groups"),
    ):
        _check_distribution(
            actual,
            _targets(guard, kind),
            field="share",
            target_field="sample_share",
            tolerance=_tolerance(guard, "sample_share"),
            require_all=True,
            label=f"realized {kind[:-1]} sampling share",
        )
        _check_distribution(
            actual,
            _targets(guard, kind),
            field="weighted_pressure_share",
            target_field="weighted_pressure_share",
            tolerance=_tolerance(guard, "weighted_pressure_share"),
            require_all=True,
            label=f"realized {kind[:-1]} weighted pressure share",
        )
