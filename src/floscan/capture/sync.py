"""Associate a stream's presentation times with a sensor log's timestamps.

The two clocks share neither origin nor exact rate, and either side may skip
samples, so association never assumes a nominal frame rate or pairs by index.
Each start hypothesis (stream sample ``a`` paired with sensor sample ``b``)
is followed forward: every stream sample is predicted on the sensor clock
from the last matched pair and accepted only if a sensor sample lies within
``tolerance_s``. Re-anchoring on every match follows slow rate differences.
A wrong start can re-lock after its first miss and still match most
samples, so the evidence is the count of samples a hypothesis leaves
unexplained: the runner-up must leave clearly more than the winner, or the
association is reported ambiguous rather than guessed. Uniform sampling, for
example, makes every start offset look equally good.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

AssociationStatus = Literal["verified", "ambiguous", "failed"]
MIN_EVIDENCE_SAMPLES = 5


@dataclass(frozen=True)
class ClockFit:
    """``sensor_s = offset_s + (1 + rate_ppm 1e-6) stream_s`` over matched pairs."""

    offset_s: float
    rate_ppm: float
    max_abs_residual_s: float
    rms_residual_s: float


@dataclass(frozen=True)
class Hypothesis:
    stream_start: int
    sensor_start: int
    matched: int


@dataclass(frozen=True)
class Association:
    """Index maps between the two streams, with the evidence for them.

    ``stream_to_sensor[k]`` is the sensor index of stream sample ``k`` (or
    None), and ``sensor_to_stream`` is the inverse.
    """

    status: AssociationStatus
    reason: str
    tolerance_s: float
    stream_to_sensor: tuple[int | None, ...]
    sensor_to_stream: tuple[int | None, ...]
    chosen: Hypothesis | None
    runner_up: Hypothesis | None
    clock: ClockFit | None

    @property
    def matched(self) -> int:
        return sum(i is not None for i in self.stream_to_sensor)


def _strictly_increasing(values: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) < 2:
        raise ValueError(f"{name} needs at least two timestamps")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite timestamps")
    if (np.diff(array) <= 0).any():
        first = int(np.nonzero(np.diff(array) <= 0)[0][0])
        raise ValueError(
            f"{name} timestamps are not strictly increasing at index {first + 1}"
        )
    return array


def _follow(
    stream: NDArray[np.float64],
    sensor: NDArray[np.float64],
    stream_start: int,
    sensor_start: int,
    tolerance_s: float,
) -> list[int | None]:
    mapping: list[int | None] = [None] * len(stream)
    mapping[stream_start] = sensor_start
    offset = sensor[sensor_start] - stream[stream_start]
    last = sensor_start
    for k in range(stream_start + 1, len(stream)):
        predicted = stream[k] + offset
        j = int(np.searchsorted(sensor, predicted))
        best = None
        for candidate in (j - 1, j):
            if last < candidate < len(sensor):
                error = abs(sensor[candidate] - predicted)
                if error <= tolerance_s and (best is None or error < best[1]):
                    best = (candidate, error)
        if best is not None:
            mapping[k] = best[0]
            last = best[0]
            offset = sensor[best[0]] - stream[k]
    return mapping


def _clock_fit(
    stream: NDArray[np.float64], sensor: NDArray[np.float64], pairs: Sequence[tuple]
) -> ClockFit:
    k = np.array([p[0] for p in pairs])
    j = np.array([p[1] for p in pairs])
    x = stream[k] - stream[k[0]]
    y = sensor[j]
    design = np.column_stack([np.ones_like(x), x])
    (intercept, slope), *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ np.array([intercept, slope])
    return ClockFit(
        offset_s=float(intercept - slope * stream[k[0]]),
        rate_ppm=float((slope - 1.0) * 1e6),
        max_abs_residual_s=float(np.abs(residual).max()),
        rms_residual_s=float(np.sqrt(np.mean(residual**2))),
    )


def associate(
    stream_s: ArrayLike,
    sensor_s: ArrayLike,
    tolerance_s: float,
    max_start_offset: int,
    min_matched_fraction: float,
    min_evidence_gap: float,
) -> Association:
    """Match stream samples to sensor samples by timing alone.

    Args:
        stream_s: presentation times of the stream (for example video PTS).
        sensor_s: timestamps of the sensor log, on its own clock.
        tolerance_s: largest accepted timing error of one match.
        max_start_offset: how many leading samples either side may lack.
        min_matched_fraction: share of stream samples that must match.
        min_evidence_gap: share of stream samples (at least
            ``MIN_EVIDENCE_SAMPLES``) that the runner-up must leave unmatched
            beyond the winner's misses.
    """
    stream = _strictly_increasing(stream_s, "stream")
    sensor = _strictly_increasing(sensor_s, "sensor")
    if not np.isfinite(tolerance_s) or tolerance_s <= 0:
        raise ValueError("tolerance_s must be positive")
    if tolerance_s >= 0.5 * float(np.diff(sensor).min()):
        raise ValueError(
            "tolerance_s must be below half the smallest sensor interval, so a "
            "stream sample can match at most one sensor sample"
        )
    if isinstance(max_start_offset, bool) or not isinstance(max_start_offset, int):
        raise ValueError("max_start_offset must be an integer")
    if max_start_offset < 1:
        raise ValueError(
            "max_start_offset must be at least 1: with no alternative start "
            "tested, the leading alignment would be assumed, not established"
        )
    if not 0 < min_matched_fraction <= 1:
        raise ValueError("min_matched_fraction must be in (0, 1]")
    if not 0 < min_evidence_gap <= 1:
        raise ValueError("min_evidence_gap must be in (0, 1]")
    starts = [(0, b) for b in range(min(max_start_offset, len(sensor) - 1) + 1)]
    starts += [(a, 0) for a in range(1, min(max_start_offset, len(stream) - 1) + 1)]
    results = []
    for a, b in starts:
        mapping = _follow(stream, sensor, a, b, tolerance_s)
        results.append((Hypothesis(a, b, sum(m is not None for m in mapping)), mapping))
    results.sort(key=lambda item: item[0].matched, reverse=True)
    (chosen, mapping), (runner_up, _) = results[0], results[1]

    def outcome(status: AssociationStatus, reason: str, use: bool) -> Association:
        pairs = [(k, j) for k, j in enumerate(mapping) if j is not None]
        inverse: list[int | None] = [None] * len(sensor)
        for k, j in pairs:
            inverse[j] = k
        return Association(
            status=status,
            reason=reason,
            tolerance_s=tolerance_s,
            stream_to_sensor=tuple(mapping) if use else (None,) * len(stream),
            sensor_to_stream=tuple(inverse) if use else (None,) * len(sensor),
            chosen=chosen,
            runner_up=runner_up,
            clock=_clock_fit(stream, sensor, pairs) if len(pairs) >= 2 else None,
        )

    fraction = chosen.matched / len(stream)
    if fraction < min_matched_fraction:
        return outcome(
            "failed",
            f"best start hypothesis matches {chosen.matched}/{len(stream)} stream "
            f"samples ({fraction:.1%}), below {min_matched_fraction:.0%}",
            use=False,
        )
    gap = chosen.matched - runner_up.matched
    if gap < max(MIN_EVIDENCE_SAMPLES, min_evidence_gap * len(stream)):
        return outcome(
            "ambiguous",
            f"start hypotheses (stream {chosen.stream_start} = sensor "
            f"{chosen.sensor_start}) and (stream {runner_up.stream_start} = sensor "
            f"{runner_up.sensor_start}) match {chosen.matched} and "
            f"{runner_up.matched} samples; timing alone cannot decide",
            use=False,
        )
    result = outcome("verified", "", use=True)
    assert result.clock is not None
    if result.clock.max_abs_residual_s > tolerance_s:
        return outcome(
            "failed",
            f"clock fit residual {result.clock.max_abs_residual_s * 1e3:.2f} ms "
            f"exceeds the {tolerance_s * 1e3:.2f} ms tolerance",
            use=False,
        )
    return outcome(
        "verified",
        f"stream {chosen.stream_start} = sensor {chosen.sensor_start}; "
        f"{chosen.matched}/{len(stream)} stream samples matched, runner-up "
        f"{runner_up.matched}",
        use=True,
    )
