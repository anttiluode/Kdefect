"""Linear quenches through a two-dimensional U(1)-breaking transition."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .lattice import energy, euler_maruyama_step, gradient_invariant
from .laws import GradientLaw, get_law
from .vortices import (
    Vortex,
    detect_vortices,
    initial_survival_fraction,
    plaquette_winding,
    track_vortices,
)

ComplexArray = NDArray[np.complexfloating]
RealArray = NDArray[np.floating]


@dataclass(frozen=True)
class QuenchConfig:
    """Dimensionless simulation parameters for one periodic square lattice."""

    size: int = 64
    dx: float = 1.0
    dt: float = 0.04
    tau_q: float = 16.0
    epsilon0: float = 0.5
    kappa: float = 4.0
    quartic: float = 1.0
    regulator: float = 0.03
    temperature: float = 0.006
    equilibration_time: float = 8.0
    post_quench_time: float = 16.0
    sample_interval: float = 0.4
    max_core_ratio: float = 0.85
    initial_scale: float = 0.02

    def validate(self) -> None:
        if self.size < 8:
            raise ValueError("size must be at least 8")
        positive = {
            "dx": self.dx,
            "dt": self.dt,
            "tau_q": self.tau_q,
            "epsilon0": self.epsilon0,
            "kappa": self.kappa,
            "quartic": self.quartic,
            "sample_interval": self.sample_interval,
        }
        for name, value in positive.items():
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.regulator < 0.0 or self.temperature < 0.0:
            raise ValueError("regulator and temperature must be nonnegative")
        if self.equilibration_time < 0.0 or self.post_quench_time < 0.0:
            raise ValueError("run durations must be nonnegative")


@dataclass
class QuenchResult:
    """Sampled observables and validated cores from one realization."""

    arm: str
    seed: int
    config: QuenchConfig
    times: RealArray
    epsilon: RealArray
    mean_power: RealArray
    free_energy: RealArray
    raw_defects: NDArray[np.int64]
    raw_charge: NDArray[np.int64]
    resolved_defects: NDArray[np.int64]
    negative_stiffness_fraction: RealArray
    kappa_gradient_p95: RealArray
    kappa_gradient_max: RealArray
    detections: list[list[Vortex]]
    final_field: ComplexArray
    sampled_fields: list[ComplexArray] | None


@dataclass(frozen=True)
class BirthSurvival:
    """Operational formation time and subsequent worldline survival."""

    birth_frame: int
    survival_frame: int
    birth_time: float
    survival_time: float
    birth_count: int
    later_count: int
    count_ratio: float
    tracked_survival: float
    gap_tolerant_survival: float
    final_reacquired_or_new_fraction: float
    post_birth_track_starts: int


def quench_epsilon(time: float, tau_q: float, epsilon0: float) -> float:
    """Linear ramp from -epsilon0 to +epsilon0, held outside the ramp."""

    return float(epsilon0 * np.clip(time / tau_q, -1.0, 1.0))


def simulate_quench(
    arm: str | GradientLaw,
    config: QuenchConfig,
    *,
    seed: int,
    store_fields: bool = False,
) -> QuenchResult:
    """Run one stochastic quench and sample topology plus thermodynamics."""

    config.validate()
    law = get_law(arm) if isinstance(arm, str) else arm
    rng = np.random.default_rng(seed)
    shape = (config.size, config.size)
    field = config.initial_scale * (
        rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    )

    start = -config.tau_q - config.equilibration_time
    stop = config.tau_q + config.post_quench_time
    number_of_steps = int(np.ceil((stop - start) / config.dt))
    sample_every = max(1, int(round(config.sample_interval / config.dt)))

    times: list[float] = []
    epsilons: list[float] = []
    powers: list[float] = []
    energies: list[float] = []
    raw_counts: list[int] = []
    raw_charges: list[int] = []
    resolved_counts: list[int] = []
    negative_fractions: list[float] = []
    gradient_p95: list[float] = []
    gradient_max: list[float] = []
    detections: list[list[Vortex]] = []
    sampled_fields: list[ComplexArray] | None = [] if store_fields else None

    for step in range(number_of_steps + 1):
        time = start + step * config.dt
        epsilon = quench_epsilon(time, config.tau_q, config.epsilon0)
        in_measurement_window = time >= -config.tau_q - 0.5 * config.dt
        if in_measurement_window and step % sample_every == 0:
            winding = plaquette_winding(field)
            resolved = detect_vortices(
                field,
                dx=config.dx,
                require_resolved=True,
                max_core_ratio=config.max_core_ratio,
            )
            y = gradient_invariant(field, config.dx)
            stiffness = law.longitudinal_stiffness(y, config.kappa)
            times.append(time)
            epsilons.append(epsilon)
            powers.append(float(np.mean(np.abs(field) ** 2)))
            energies.append(
                float(
                    energy(
                        field,
                        law,
                        epsilon=epsilon,
                        kappa=config.kappa,
                        quartic=config.quartic,
                        regulator=config.regulator,
                        dx=config.dx,
                    )
                )
            )
            raw_counts.append(int(np.sum(np.abs(winding))))
            raw_charges.append(int(np.sum(winding)))
            resolved_counts.append(len(resolved))
            negative_fractions.append(float(np.mean(stiffness < 0.0)))
            gradient_p95.append(float(np.quantile(config.kappa * y, 0.95)))
            gradient_max.append(float(np.max(config.kappa * y)))
            detections.append(resolved)
            if sampled_fields is not None:
                sampled_fields.append(field.copy())
        if step == number_of_steps:
            break
        field = euler_maruyama_step(
            field,
            law,
            rng,
            dt=config.dt,
            temperature=config.temperature,
            epsilon=epsilon,
            kappa=config.kappa,
            quartic=config.quartic,
            regulator=config.regulator,
            dx=config.dx,
        )
        if not np.all(np.isfinite(field)):
            raise FloatingPointError(
                f"non-finite field in {law.name} arm at t={time:.6g}; "
                "reduce dt or increase the regulator"
            )

    return QuenchResult(
        arm=law.name,
        seed=seed,
        config=config,
        times=np.asarray(times),
        epsilon=np.asarray(epsilons),
        mean_power=np.asarray(powers),
        free_energy=np.asarray(energies),
        raw_defects=np.asarray(raw_counts, dtype=np.int64),
        raw_charge=np.asarray(raw_charges, dtype=np.int64),
        resolved_defects=np.asarray(resolved_counts, dtype=np.int64),
        negative_stiffness_fraction=np.asarray(negative_fractions),
        kappa_gradient_p95=np.asarray(gradient_p95),
        kappa_gradient_max=np.asarray(gradient_max),
        detections=detections,
        final_field=field,
        sampled_fields=sampled_fields,
    )


def locate_birth_frame(
    result: QuenchResult,
    *,
    minimum_ordered_fraction: float = 0.12,
) -> int | None:
    """Locate the resolved-core formation peak after the critical crossing.

    The search begins only after mean |psi|^2 reaches a declared fraction of
    its noiseless ordered value epsilon0/u.  Among eligible frames, the first
    global maximum of *resolved* core count defines birth.  Returning ``None``
    is preferable to manufacturing a formation time from raw phase winding.
    """

    ordered_power = result.config.epsilon0 / result.config.quartic
    eligible = np.flatnonzero(
        (result.times >= 0.0)
        & (result.mean_power >= minimum_ordered_fraction * ordered_power)
    )
    if len(eligible) == 0:
        return None
    counts = result.resolved_defects[eligible]
    if int(np.max(counts)) == 0:
        return None
    return int(eligible[int(np.argmax(counts))])


def measure_birth_survival(
    result: QuenchResult,
    *,
    survival_lag: float = 8.0,
    max_speed: float = 5.0,
) -> BirthSurvival | None:
    """Measure count decay and same-charge worldline survival after birth."""

    birth_frame = locate_birth_frame(result)
    if birth_frame is None:
        return None
    target_time = float(result.times[birth_frame] + survival_lag)
    later = np.flatnonzero(result.times >= target_time)
    if len(later) == 0:
        return None
    survival_frame = int(later[0])
    birth_count = int(result.resolved_defects[birth_frame])
    later_count = int(result.resolved_defects[survival_frame])
    window_frames = result.detections[birth_frame : survival_frame + 1]
    window_times = result.times[birth_frame : survival_frame + 1]
    tracks = track_vortices(
        window_frames,
        window_times,
        box_size=result.config.size * result.config.dx,
        max_speed=max_speed,
    )
    tracked = initial_survival_fraction(tracks, survival_frame - birth_frame)
    tolerant_tracks = track_vortices(
        window_frames,
        window_times,
        box_size=result.config.size * result.config.dx,
        max_speed=max_speed,
        max_gap_frames=2,
    )
    final_frame = survival_frame - birth_frame
    tolerant = initial_survival_fraction(tolerant_tracks, final_frame)
    final_tracks = [track for track in tolerant_tracks if track.end_frame == final_frame]
    initial_at_final = sum(track.start_frame == 0 for track in final_tracks)
    reacquired_fraction = (
        float(1.0 - initial_at_final / len(final_tracks)) if final_tracks else float("nan")
    )
    later_starts = sum(track.start_frame > 0 for track in tolerant_tracks)
    return BirthSurvival(
        birth_frame=birth_frame,
        survival_frame=survival_frame,
        birth_time=float(result.times[birth_frame]),
        survival_time=float(result.times[survival_frame]),
        birth_count=birth_count,
        later_count=later_count,
        count_ratio=float(later_count / birth_count),
        tracked_survival=tracked,
        gap_tolerant_survival=tolerant,
        final_reacquired_or_new_fraction=reacquired_fraction,
        post_birth_track_starts=later_starts,
    )
