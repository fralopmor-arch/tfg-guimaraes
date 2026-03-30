from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ExecutionMode = Literal["smoke", "fast", "intermediate", "full"]

SMOKE_MIN_SIZE = 5
SMOKE_MAX_SIZE = 10
FAST_MIN_SIZE = 20
FAST_MAX_SIZE = 50

DEFAULT_SMOKE_SIZE = 8
DEFAULT_FAST_SIZE = 30
DEFAULT_INTERMEDIATE_BATCH_SIZE = 40


@dataclass(frozen=True)
class ModeConfig:
    smoke_size: int = DEFAULT_SMOKE_SIZE
    fast_size: int = DEFAULT_FAST_SIZE
    intermediate_batch_size: int = DEFAULT_INTERMEDIATE_BATCH_SIZE
    full_batch_size: int | None = None


def normalize_mode(value: str) -> ExecutionMode:
    mode = value.strip().lower()
    valid_modes = {"smoke", "fast", "intermediate", "full"}
    if mode not in valid_modes:
        raise ValueError(f"Invalid mode '{value}'. Choose one of: smoke, fast, intermediate, full")
    return mode  # type: ignore[return-value]


def _resolve_bounded(
    value: int | None,
    *,
    default: int,
    minimum: int,
    maximum: int,
    label: str,
) -> int:
    resolved = default if value is None else value
    if resolved < minimum or resolved > maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}. Received: {resolved}")
    return resolved


def _resolve_positive(value: int | None, *, default: int, label: str) -> int:
    resolved = default if value is None else value
    if resolved <= 0:
        raise ValueError(f"{label} must be greater than zero. Received: {resolved}")
    return resolved


def _resolve_optional_positive(value: int | None, *, label: str) -> int | None:
    if value is None:
        return None
    if value <= 0:
        raise ValueError(f"{label} must be greater than zero when provided. Received: {value}")
    return value


def resolve_mode_config(
    *,
    smoke_size: int | None = None,
    fast_size: int | None = None,
    intermediate_batch_size: int | None = None,
    full_batch_size: int | None = None,
) -> ModeConfig:
    return ModeConfig(
        smoke_size=_resolve_bounded(
            smoke_size,
            default=DEFAULT_SMOKE_SIZE,
            minimum=SMOKE_MIN_SIZE,
            maximum=SMOKE_MAX_SIZE,
            label="smoke_size",
        ),
        fast_size=_resolve_bounded(
            fast_size,
            default=DEFAULT_FAST_SIZE,
            minimum=FAST_MIN_SIZE,
            maximum=FAST_MAX_SIZE,
            label="fast_size",
        ),
        intermediate_batch_size=_resolve_positive(
            intermediate_batch_size,
            default=DEFAULT_INTERMEDIATE_BATCH_SIZE,
            label="intermediate_batch_size",
        ),
        full_batch_size=_resolve_optional_positive(full_batch_size, label="full_batch_size"),
    )
