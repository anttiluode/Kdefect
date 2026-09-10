#!/usr/bin/env python3
"""Run one preregistered Gate-2 convergence cell.

The expensive campaign is sharded by GitHub Actions.  This script deliberately
runs only one numerical cell at a time and writes a machine-readable receipt.
See GATE2_CONVERGENCE_PROTOCOL.md for the frozen panel and pass/fail rules.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from kdefect.lattice import gradient_invariant
from kdefect.laws import get_law
from kdefect.quench import QuenchConfig, simulate_quench
from kdefect.statistics import (
    bootstrap_loglog_slope,
    normalized_nn_spacings,
    poisson_nn_ks,
)
from kdefect.vortices import (
    Vortex,
    detect_vortices,
    initial_survival_fraction,
    track_vortices,
)

CORE_ARMS = ("canonical", "sqrt")
LOG_ARMS = ("canonical", "log")
THRESHOLDS = (0.75, 0.85, 0.95)
TAU_VALUES = (8.0, 16.0, 32.0)
REGULATORS = (0.020, 0.030, 0.045, 0.0675, 0.100)
BOX_LENGTH = 64.0
BASE_SEED = 260_909_000
SURVIVAL_LAG = 8.0
BASE_CADENCE = 0.4
COARSE_CADENCE = 0.8
BASE_SPEED = 5.0
SPEEDS = (3.75, 5.0, 6.25)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def mean(values: Sequence[float]) -> float:
    x = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    return float(np.mean(x)) if len(x) else float("nan")


def rel_change(a: float, b: float) -> float:
    scale = max(abs(a), abs(b), np.finfo(float).tiny)
    return float(abs(a - b) / scale)


def _filter_detections(
    detections95: Sequence[Sequence[Vortex]], threshold: float
) -> list[list[Vortex]]:
    return [[v for v in frame if v.core_ratio <= threshold] for frame in detections95]


def _birth_and_survival(
    result,
    detections: Sequence[Sequence[Vortex]],
    *,
    cadence: float,
    max_speed: float,
    survival_lag: float = SURVIVAL_LAG,
) -> dict[str, Any] | None:
    internal = float(result.config.sample_interval)
    stride = max(1, int(round(cadence / internal)))
    source_indices = np.arange(0, len(result.times), stride, dtype=int)
    times = result.times[source_indices]
    powers = result.mean_power[source_indices]
    frames = [detections[int(i)] for i in source_indices]
    counts = np.asarray([len(frame) for frame in frames], dtype=int)

    ordered_power = result.config.epsilon0 / result.config.quartic
    eligible = np.flatnonzero((times >= 0.0) & (powers >= 0.12 * ordered_power))
    if len(eligible) == 0 or int(np.max(counts[eligible])) == 0:
        return None
    birth_local = int(eligible[int(np.argmax(counts[eligible]))])
    target = float(times[birth_local] + survival_lag)
    later = np.flatnonzero(times >= target)
    if len(later) == 0:
        return None
    survival_local = int(later[0])
    window_frames = frames[birth_local : survival_local + 1]
    window_times = times[birth_local : survival_local + 1]
    tracks = track_vortices(
        window_frames,
        window_times,
        box_size=result.config.size * result.config.dx,
        max_speed=max_speed,
        max_gap_frames=2,
    )
    final_frame = survival_local - birth_local
    survival = initial_survival_fraction(tracks, final_frame)
    birth_count = int(counts[birth_local])
    later_count = int(counts[survival_local])
    return {
        "birth_source_index": int(source_indices[birth_local]),
        "survival_source_index": int(source_indices[survival_local]),
        "birth_time": float(times[birth_local]),
        "birth_count": birth_count,
        "birth_density": float(birth_count / (BOX_LENGTH**2)),
        "later_count": later_count,
        "gap_tolerant_survival": float(survival),
        "cadence": float(cadence),
        "max_speed": float(max_speed),
    }


def _all_detections_95(result) -> list[list[Vortex]]:
    assert result.sampled_fields is not None
    return [
        detect_vortices(
            field,
            dx=result.config.dx,
            require_resolved=True,
            max_core_ratio=0.95,
        )
        for field in result.sampled_fields
    ]


def _core_patch_spectrum(
    field: np.ndarray,
    vortices: Sequence[Vortex],
    dx: float,
    *,
    physical_width: float = 16.0,
) -> dict[str, float]:
    if not vortices:
        return {"peak_k": float("nan"), "spectral_centroid_k": float("nan"), "cores": 0}
    y = gradient_invariant(field, dx)
    n = max(8, int(round(physical_width / dx)))
    if n % 2:
        n += 1
    n = min(n, field.shape[0])
    half = n // 2
    window1 = np.hanning(n)
    window = window1[:, None] * window1[None, :]
    wave = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    kx, ky = np.meshgrid(wave, wave, indexing="ij")
    kmag = np.sqrt(kx**2 + ky**2)
    fundamental = 2.0 * np.pi / (n * dx)
    shell = np.rint(kmag / fundamental).astype(int)
    max_shell = int(np.max(shell))
    radial_sum = np.zeros(max_shell + 1, dtype=float)
    radial_count = np.zeros(max_shell + 1, dtype=float)
    total_power = 0.0
    weighted_k = 0.0
    used = 0
    size = field.shape[0]
    for vortex in vortices:
        ci = int(round(vortex.x / dx)) % size
        cj = int(round(vortex.y / dx)) % size
        ii = (np.arange(ci - half, ci - half + n) % size).astype(int)
        jj = (np.arange(cj - half, cj - half + n) % size).astype(int)
        patch = y[np.ix_(ii, jj)].astype(float)
        patch = (patch - np.mean(patch)) * window
        power = np.abs(np.fft.fft2(patch)) ** 2
        radial_sum += np.bincount(shell.ravel(), weights=power.ravel(), minlength=max_shell + 1)
        radial_count += np.bincount(shell.ravel(), minlength=max_shell + 1)
        nonzero = kmag > 0.0
        total_power += float(np.sum(power[nonzero]))
        weighted_k += float(np.sum(power[nonzero] * kmag[nonzero]))
        used += 1
    radial = np.divide(radial_sum, radial_count, out=np.zeros_like(radial_sum), where=radial_count > 0)
    if len(radial):
        radial[0] = 0.0
    peak_shell = int(np.argmax(radial)) if np.any(radial > 0.0) else 0
    return {
        "peak_k": float(peak_shell * fundamental) if peak_shell else float("nan"),
        "spectral_centroid_k": float(weighted_k / total_power) if total_power > 0 else float("nan"),
        "cores": used,
        "patch_sites": n,
        "patch_physical_width": float(n * dx),
    }


def run_core_cell(*, size: int, dt: float, seeds: int, output: Path) -> dict[str, Any]:
    dx = BOX_LENGTH / size
    base = QuenchConfig(
        size=size,
        dx=dx,
        dt=dt,
        kappa=32.0,
        regulator=0.03,
        temperature=0.006,
        sample_interval=0.2,
        max_core_ratio=0.85,
        post_quench_time=20.0,
    )
    records: list[dict[str, Any]] = []
    total = len(CORE_ARMS) * len(TAU_VALUES) * seeds
    done = 0
    for arm in CORE_ARMS:
        for ti, tau_q in enumerate(TAU_VALUES):
            config = replace(base, tau_q=tau_q)
            for si in range(seeds):
                seed = BASE_SEED + 10_000 * ti + si
                result = simulate_quench(arm, config, seed=seed, store_fields=True)
                detections95 = _all_detections_95(result)
                variants: dict[str, Any] = {}
                for threshold in THRESHOLDS:
                    det = _filter_detections(detections95, threshold)
                    variants[f"threshold_{threshold:.2f}"] = _birth_and_survival(
                        result, det, cadence=BASE_CADENCE, max_speed=BASE_SPEED
                    )
                det85 = _filter_detections(detections95, 0.85)
                variants["cadence_0.80"] = _birth_and_survival(
                    result, det85, cadence=COARSE_CADENCE, max_speed=BASE_SPEED
                )
                for speed in (3.75, 6.25):
                    variants[f"speed_{speed:.2f}"] = _birth_and_survival(
                        result, det85, cadence=BASE_CADENCE, max_speed=speed
                    )
                record = {
                    "arm": arm,
                    "tau_q": tau_q,
                    "seed": seed,
                    "variants": variants,
                    "nonzero_raw_charge_frames": int(np.count_nonzero(result.raw_charge)),
                    "sampled_frames": int(len(result.raw_charge)),
                }
                records.append(record)
                done += 1
                print(f"[{done}/{total}] core N={size} dt={dt:g} {arm} tau={tau_q:g} seed={seed}", flush=True)

    summaries: list[dict[str, Any]] = []
    slope_groups: dict[str, list[list[float]]] = {arm: [] for arm in CORE_ARMS}
    for arm in CORE_ARMS:
        for tau_q in TAU_VALUES:
            group = [r for r in records if r["arm"] == arm and r["tau_q"] == tau_q]
            row: dict[str, Any] = {"arm": arm, "tau_q": tau_q, "attempted": len(group)}
            for name in [f"threshold_{x:.2f}" for x in THRESHOLDS] + ["cadence_0.80", "speed_3.75", "speed_6.25"]:
                valid = [r["variants"][name] for r in group if r["variants"].get(name) is not None]
                row[name] = {
                    "resolved": len(valid),
                    "birth_density_mean": mean([v["birth_density"] for v in valid]),
                    "survival_mean": mean([v["gap_tolerant_survival"] for v in valid]),
                }
            base_vals = [
                r["variants"]["threshold_0.85"]["birth_density"]
                for r in group
                if r["variants"].get("threshold_0.85") is not None
            ]
            slope_groups[arm].append(base_vals)
            summaries.append(row)

    slopes: dict[str, Any] = {}
    for ai, arm in enumerate(CORE_ARMS):
        groups = slope_groups[arm]
        slopes[arm] = (
            bootstrap_loglog_slope(TAU_VALUES, groups, draws=2000, seed=88_000 + ai)
            if all(groups)
            else {"median": float("nan"), "low_95": float("nan"), "high_95": float("nan")}
        )

    threshold_changes: list[float] = []
    cadence_changes: list[float] = []
    speed_changes: list[float] = []
    for row in summaries:
        base_birth = row["threshold_0.85"]["birth_density_mean"]
        base_survival = row["threshold_0.85"]["survival_mean"]
        for threshold in (0.75, 0.95):
            threshold_changes.append(rel_change(base_birth, row[f"threshold_{threshold:.2f}"]["birth_density_mean"]))
        cadence_changes.append(abs(base_survival - row["cadence_0.80"]["survival_mean"]))
        for speed in (3.75, 6.25):
            speed_changes.append(abs(base_survival - row[f"speed_{speed:.2f}"]["survival_mean"]))

    receipt = {
        "gate": "GATE2_CORE_CONVERGENCE_CELL",
        "protocol": "GATE2_CONVERGENCE_PROTOCOL.md",
        "size": size,
        "box_length": BOX_LENGTH,
        "dx": dx,
        "dt": dt,
        "seeds_per_arm_tau": seeds,
        "config_base": asdict(base),
        "tau_values": list(TAU_VALUES),
        "records": records,
        "summaries": summaries,
        "birth_slopes": slopes,
        "diagnostics": {
            "max_detector_relative_birth_change": max(threshold_changes, default=float("nan")),
            "max_cadence_absolute_survival_change": max(cadence_changes, default=float("nan")),
            "max_speed_absolute_survival_change": max(speed_changes, default=float("nan")),
            "nonzero_raw_charge_frames": int(sum(r["nonzero_raw_charge_frames"] for r in records)),
            "detector_pass_lt_0p05": bool(threshold_changes and max(threshold_changes) < 0.05),
            "cadence_pass_abs_0p05": bool(cadence_changes and max(cadence_changes) < 0.05),
            "speed_pass_abs_0p05": bool(speed_changes and max(speed_changes) < 0.05),
            "charge_pass": all(r["nonzero_raw_charge_frames"] == 0 for r in records),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(json_safe(receipt), indent=2, allow_nan=False) + "\n")
    return receipt


def run_log_cell(*, size: int, dt: float, seeds: int, output: Path) -> dict[str, Any]:
    dx = BOX_LENGTH / size
    base = QuenchConfig(
        size=size,
        dx=dx,
        dt=dt,
        tau_q=16.0,
        kappa=32.0,
        temperature=0.006,
        sample_interval=0.2,
        max_core_ratio=0.85,
        post_quench_time=20.0,
    )
    records: list[dict[str, Any]] = []
    total = len(LOG_ARMS) * len(REGULATORS) * seeds
    done = 0
    for gi, regulator in enumerate(REGULATORS):
        config = replace(base, regulator=regulator)
        for arm in LOG_ARMS:
            law = get_law(arm)
            for si in range(seeds):
                seed = BASE_SEED + 500_000 + 10_000 * gi + si
                result = simulate_quench(arm, config, seed=seed, store_fields=True)
                detections95 = _all_detections_95(result)
                det85 = _filter_detections(detections95, 0.85)
                metric = _birth_and_survival(result, det85, cadence=BASE_CADENCE, max_speed=BASE_SPEED)
                record: dict[str, Any] = {
                    "arm": arm,
                    "regulator": regulator,
                    "seed": seed,
                    "metric": metric,
                    "nonzero_raw_charge_frames": int(np.count_nonzero(result.raw_charge)),
                }
                if metric is not None:
                    source = int(metric["birth_source_index"])
                    assert result.sampled_fields is not None
                    field = result.sampled_fields[source]
                    vortices = det85[source]
                    spacings = normalized_nn_spacings(vortices, BOX_LENGTH)
                    y = gradient_invariant(field, dx)
                    stiffness = law.longitudinal_stiffness(y, config.kappa)
                    negative = stiffness[stiffness < 0.0]
                    kstar = (
                        float(np.sqrt(np.median(-negative) / (2.0 * regulator)))
                        if len(negative) and regulator > 0
                        else float("nan")
                    )
                    spectrum = _core_patch_spectrum(field, vortices, dx)
                    record.update(
                        {
                            "nn_poisson_ks": poisson_nn_ks(spacings),
                            "negative_stiffness_fraction": float(np.mean(stiffness < 0.0)),
                            "predicted_k_star": kstar,
                            "predicted_wavelength_sites": float(2.0 * np.pi / (kstar * dx)) if np.isfinite(kstar) and kstar > 0 else float("nan"),
                            "core_gradient_spectrum": spectrum,
                        }
                    )
                records.append(record)
                done += 1
                print(f"[{done}/{total}] log N={size} g={regulator:g} {arm} seed={seed}", flush=True)

    summaries: list[dict[str, Any]] = []
    for regulator in REGULATORS:
        by_arm: dict[str, list[dict[str, Any]]] = {
            arm: [r for r in records if r["arm"] == arm and r["regulator"] == regulator and r["metric"] is not None]
            for arm in LOG_ARMS
        }
        row: dict[str, Any] = {"regulator": regulator}
        for arm in LOG_ARMS:
            group = by_arm[arm]
            row[arm] = {
                "resolved": len(group),
                "birth_density_mean": mean([r["metric"]["birth_density"] for r in group]),
                "survival_mean": mean([r["metric"]["gap_tolerant_survival"] for r in group]),
                "nn_poisson_ks_mean": mean([r.get("nn_poisson_ks", float("nan")) for r in group]),
                "predicted_k_star_mean": mean([r.get("predicted_k_star", float("nan")) for r in group]),
                "predicted_wavelength_sites_mean": mean([r.get("predicted_wavelength_sites", float("nan")) for r in group]),
                "core_spectrum_peak_k_mean": mean([r.get("core_gradient_spectrum", {}).get("peak_k", float("nan")) for r in group]),
            }
        ref = {r["seed"]: r for r in by_arm["canonical"]}
        trt = {r["seed"]: r for r in by_arm["log"]}
        common = sorted(ref.keys() & trt.keys())
        ratios = [trt[s]["metric"]["birth_density"] / ref[s]["metric"]["birth_density"] for s in common if ref[s]["metric"]["birth_density"] > 0]
        survival_diff = [trt[s]["metric"]["gap_tolerant_survival"] - ref[s]["metric"]["gap_tolerant_survival"] for s in common]
        nn_diff = [trt[s]["nn_poisson_ks"] - ref[s]["nn_poisson_ks"] for s in common if np.isfinite(trt[s].get("nn_poisson_ks", np.nan)) and np.isfinite(ref[s].get("nn_poisson_ks", np.nan))]
        row["paired"] = {
            "pairs": len(common),
            "log_to_canonical_birth_ratio_mean": mean(ratios),
            "log_minus_canonical_survival_mean": mean(survival_diff),
            "log_minus_canonical_nn_ks_mean": mean(nn_diff),
        }
        summaries.append(row)

    receipt = {
        "gate": "GATE2_LOG_REGULATOR_CELL",
        "protocol": "GATE2_CONVERGENCE_PROTOCOL.md",
        "size": size,
        "box_length": BOX_LENGTH,
        "dx": dx,
        "dt": dt,
        "seeds_per_arm_g": seeds,
        "regulators": list(REGULATORS),
        "config_base": asdict(base),
        "records": records,
        "summaries": summaries,
        "diagnostics": {
            "nonzero_raw_charge_frames": int(sum(r["nonzero_raw_charge_frames"] for r in records)),
            "charge_pass": all(r["nonzero_raw_charge_frames"] == 0 for r in records),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(json_safe(receipt), indent=2, allow_nan=False) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", choices=("core", "log"), required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--dt", type=float, required=True)
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.panel == "core":
        receipt = run_core_cell(size=args.size, dt=args.dt, seeds=args.seeds or 8, output=args.output)
    else:
        receipt = run_log_cell(size=args.size, dt=args.dt, seeds=args.seeds or 6, output=args.output)
    print(json.dumps(json_safe({"gate": receipt["gate"], "size": receipt["size"], "dt": receipt["dt"], "diagnostics": receipt["diagnostics"]}), indent=2))


if __name__ == "__main__":
    main()
