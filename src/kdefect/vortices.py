"""Topological charge, sub-plaquette cores, and conservative worldlines."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

ComplexArray = NDArray[np.complexfloating]
IntArray = NDArray[np.integer]
RealArray = NDArray[np.floating]


def wrap_phase(delta: RealArray) -> RealArray:
    """Wrap angle differences into [-pi, pi)."""

    return (delta + np.pi) % (2.0 * np.pi) - np.pi


def plaquette_winding(field: ComplexArray) -> IntArray:
    """Integer U(1) winding around every periodic plaquette.

    A nonzero answer is a topological candidate, not by itself a claim that a
    continuum vortex core is resolved by the lattice.
    """

    phase = np.angle(field)
    p00 = phase
    p10 = np.roll(phase, -1, axis=-2)
    p01 = np.roll(phase, -1, axis=-1)
    p11 = np.roll(p10, -1, axis=-1)
    circulation = (
        wrap_phase(p10 - p00)
        + wrap_phase(p11 - p10)
        + wrap_phase(p01 - p11)
        + wrap_phase(p00 - p01)
    )
    winding = np.rint(circulation / (2.0 * np.pi)).astype(np.int8)
    return np.clip(winding, -1, 1)


@dataclass(frozen=True)
class Vortex:
    """A validated vortex core in physical box coordinates."""

    x: float
    y: float
    charge: int
    residual: float
    core_ratio: float
    plaquette: tuple[int, int]


def _bilinear_value_and_jacobian(
    corners: tuple[complex, complex, complex, complex], x: float, y: float
) -> tuple[complex, NDArray[np.float64]]:
    z00, z10, z01, z11 = corners
    a = z00
    b = z10 - z00
    c = z01 - z00
    d = z11 - z10 - z01 + z00
    value = a + b * x + c * y + d * x * y
    dzdx = b + d * y
    dzdy = c + d * x
    jacobian = np.array(
        [[dzdx.real, dzdy.real], [dzdx.imag, dzdy.imag]], dtype=float
    )
    return value, jacobian


def bilinear_zero(
    corners: tuple[complex, complex, complex, complex],
    *,
    tolerance: float = 1.0e-7,
    max_iterations: int = 30,
) -> tuple[float, float, float] | None:
    """Locate a simultaneous zero of Re(psi) and Im(psi) in one cell.

    Several Newton seeds make this robust to skewed cores.  The returned
    residual is normalized by the largest corner amplitude.
    """

    scale = max(max(abs(z) for z in corners), np.finfo(float).tiny)
    seeds = (
        (0.5, 0.5),
        (0.25, 0.25),
        (0.75, 0.25),
        (0.25, 0.75),
        (0.75, 0.75),
    )
    best: tuple[float, float, float] | None = None
    for x0, y0 in seeds:
        x, y = x0, y0
        for _ in range(max_iterations):
            value, jacobian = _bilinear_value_and_jacobian(corners, x, y)
            residual = abs(value) / scale
            if residual <= tolerance:
                break
            determinant = float(np.linalg.det(jacobian))
            if abs(determinant) < 1.0e-14:
                break
            try:
                update = np.linalg.solve(jacobian, np.array([value.real, value.imag]))
            except np.linalg.LinAlgError:
                break
            x -= float(update[0])
            y -= float(update[1])
            if not (-2.0 <= x <= 3.0 and -2.0 <= y <= 3.0):
                break
        value, _ = _bilinear_value_and_jacobian(corners, x, y)
        residual = abs(value) / scale
        if -tolerance <= x <= 1.0 + tolerance and -tolerance <= y <= 1.0 + tolerance:
            candidate = (
                float(np.clip(x, 0.0, 1.0)),
                float(np.clip(y, 0.0, 1.0)),
                float(residual),
            )
            if best is None or candidate[2] < best[2]:
                best = candidate
    if best is None or best[2] > tolerance:
        return None
    return best


def _core_to_background_ratio(field: ComplexArray, i: int, j: int) -> float:
    nx, ny = field.shape
    core_offsets = {(0, 0), (1, 0), (0, 1), (1, 1)}
    core = [
        abs(field[(i + di) % nx, (j + dj) % ny]) ** 2
        for di, dj in core_offsets
    ]
    ring = [
        abs(field[(i + di) % nx, (j + dj) % ny]) ** 2
        for di in (-1, 0, 1, 2)
        for dj in (-1, 0, 1, 2)
        if (di, dj) not in core_offsets
    ]
    background = float(np.median(ring))
    if background <= np.finfo(float).tiny:
        return float("inf")
    return float(np.mean(core) / background)


def detect_vortices(
    field: ComplexArray,
    *,
    dx: float = 1.0,
    require_resolved: bool = True,
    max_core_ratio: float = 0.85,
    root_tolerance: float = 1.0e-6,
) -> list[Vortex]:
    """Detect winding candidates and validate their interpolated cores.

    Validation requires a bilinear zero of both field components.  The
    default resolved-core gate additionally requires the four cell corners to
    be amplitude-depleted relative to a surrounding twelve-site ring.  This
    rejects many phase-only zeros in under-resolved random fields.
    """

    if field.ndim != 2:
        raise ValueError("detect_vortices expects one two-dimensional field")
    winding = plaquette_winding(field)
    candidates = np.argwhere(winding != 0)
    nx, ny = field.shape
    detections: list[Vortex] = []
    for i_raw, j_raw in candidates:
        i, j = int(i_raw), int(j_raw)
        corners = (
            complex(field[i, j]),
            complex(field[(i + 1) % nx, j]),
            complex(field[i, (j + 1) % ny]),
            complex(field[(i + 1) % nx, (j + 1) % ny]),
        )
        root = bilinear_zero(corners, tolerance=root_tolerance)
        if root is None:
            continue
        x_local, y_local, residual = root
        core_ratio = _core_to_background_ratio(field, i, j)
        if require_resolved and core_ratio > max_core_ratio:
            continue
        detections.append(
            Vortex(
                x=float(((i + x_local) * dx) % (nx * dx)),
                y=float(((j + y_local) * dx) % (ny * dx)),
                charge=int(winding[i, j]),
                residual=residual,
                core_ratio=core_ratio,
                plaquette=(i, j),
            )
        )
    return detections


def periodic_distance_matrix(
    left: RealArray, right: RealArray, box_size: float
) -> RealArray:
    """Pairwise Euclidean distances on a square periodic box."""

    if left.size == 0 or right.size == 0:
        return np.empty((len(left), len(right)), dtype=float)
    delta = np.abs(left[:, None, :] - right[None, :, :])
    delta = np.minimum(delta, box_size - delta)
    return np.sqrt(np.sum(delta**2, axis=-1))


@dataclass(frozen=True)
class TrackPoint:
    frame: int
    time: float
    x: float
    y: float


@dataclass
class VortexTrack:
    """A same-charge sequence linked by bounded periodic displacement."""

    identifier: int
    charge: int
    points: list[TrackPoint] = field(default_factory=list)

    @property
    def start_frame(self) -> int:
        return self.points[0].frame

    @property
    def end_frame(self) -> int:
        return self.points[-1].frame

    @property
    def lifetime(self) -> float:
        return self.points[-1].time - self.points[0].time


def track_vortices(
    frames: Sequence[Sequence[Vortex]],
    times: Sequence[float],
    *,
    box_size: float,
    max_speed: float,
) -> list[VortexTrack]:
    """Link detections with a minimum-cost assignment at each frame.

    Tracks terminate after one missed frame.  This conservative rule avoids
    inventing survival across an unresolved annihilation or pair-creation
    event; callers should report detector cadence and speed cutoff.
    """

    if len(frames) != len(times):
        raise ValueError("frames and times must have equal length")
    if not frames:
        return []
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:  # pragma: no cover - dependency declared by package
        raise RuntimeError("track_vortices requires scipy") from exc

    tracks: list[VortexTrack] = []
    active: dict[int, Vortex] = {}
    next_identifier = 0

    def start(vortex: Vortex, frame_index: int) -> None:
        nonlocal next_identifier
        track = VortexTrack(identifier=next_identifier, charge=vortex.charge)
        track.points.append(
            TrackPoint(frame_index, float(times[frame_index]), vortex.x, vortex.y)
        )
        tracks.append(track)
        active[next_identifier] = vortex
        next_identifier += 1

    for vortex in frames[0]:
        start(vortex, 0)

    for frame_index in range(1, len(frames)):
        dt = float(times[frame_index] - times[frame_index - 1])
        if dt <= 0.0:
            raise ValueError("times must be strictly increasing")
        new_active: dict[int, Vortex] = {}
        assigned_new: set[int] = set()
        current = list(frames[frame_index])
        for charge in (-1, 1):
            old_ids = [identifier for identifier, v in active.items() if v.charge == charge]
            new_ids = [index for index, v in enumerate(current) if v.charge == charge]
            if not old_ids or not new_ids:
                continue
            old_positions = np.array([[active[k].x, active[k].y] for k in old_ids])
            new_positions = np.array([[current[k].x, current[k].y] for k in new_ids])
            costs = periodic_distance_matrix(old_positions, new_positions, box_size)
            rows, columns = linear_sum_assignment(costs)
            for row, column in zip(rows, columns, strict=True):
                distance = float(costs[row, column])
                if distance > max_speed * dt:
                    continue
                identifier = old_ids[int(row)]
                detection_index = new_ids[int(column)]
                vortex = current[detection_index]
                tracks[identifier].points.append(
                    TrackPoint(frame_index, float(times[frame_index]), vortex.x, vortex.y)
                )
                new_active[identifier] = vortex
                assigned_new.add(detection_index)
        active = new_active
        for detection_index, vortex in enumerate(current):
            if detection_index not in assigned_new:
                start(vortex, frame_index)
    return tracks


def initial_survival_fraction(tracks: Sequence[VortexTrack], final_frame: int) -> float:
    """Fraction of frame-zero tracks that remain linked through final_frame."""

    initial = [track for track in tracks if track.start_frame == 0]
    if not initial:
        return float("nan")
    survived = sum(track.end_frame == final_frame for track in initial)
    return float(survived / len(initial))
