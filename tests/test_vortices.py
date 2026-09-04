from __future__ import annotations

import numpy as np
import pytest

from kdefect.vortices import (
    Vortex,
    detect_vortices,
    initial_survival_fraction,
    plaquette_winding,
    track_vortices,
)


def make_vortex(x: float, y: float, charge: int) -> Vortex:
    return Vortex(x, y, charge, 0.0, 0.1, (int(x), int(y)))


def test_plane_wave_has_no_plaquette_vortices() -> None:
    size = 32
    x = np.arange(size)[:, None]
    field = np.exp(2j * np.pi * 3 * x / size) * np.ones((1, size))
    assert np.count_nonzero(plaquette_winding(field)) == 0


def test_sine_field_has_four_balanced_resolved_cores() -> None:
    size = 40
    offset_x, offset_y = 6.35, 8.65
    x = np.arange(size)[:, None]
    y = np.arange(size)[None, :]
    field = np.sin(2.0 * np.pi * (x - offset_x) / size) + 1j * np.sin(
        2.0 * np.pi * (y - offset_y) / size
    )
    raw = plaquette_winding(field)
    detected = detect_vortices(field, max_core_ratio=0.9)
    assert int(np.sum(np.abs(raw))) == 4
    assert len(detected) == 4
    assert sum(vortex.charge for vortex in detected) == 0
    expected_x = np.array([offset_x, offset_x + size / 2.0])
    expected_y = np.array([offset_y, offset_y + size / 2.0])
    for vortex in detected:
        dx = np.min(np.abs(((vortex.x - expected_x + size / 2.0) % size) - size / 2.0))
        dy = np.min(np.abs(((vortex.y - expected_y + size / 2.0) % size) - size / 2.0))
        assert dx < 0.03
        assert dy < 0.03


def test_worldline_assignment_respects_charge_and_periodic_boundary() -> None:
    frames = [
        [make_vortex(9.8, 2.0, 1), make_vortex(4.0, 5.0, -1)],
        [make_vortex(0.1, 2.1, 1), make_vortex(4.2, 5.1, -1)],
        [make_vortex(0.4, 2.2, 1)],
    ]
    tracks = track_vortices(frames, [0.0, 1.0, 2.0], box_size=10.0, max_speed=1.0)
    positive = [track for track in tracks if track.charge == 1 and track.start_frame == 0]
    negative = [track for track in tracks if track.charge == -1 and track.start_frame == 0]
    assert len(positive) == 1 and positive[0].end_frame == 2
    assert len(negative) == 1 and negative[0].end_frame == 1
    assert initial_survival_fraction(tracks, 2) == pytest.approx(0.5)
