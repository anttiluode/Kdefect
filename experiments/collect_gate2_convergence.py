#!/usr/bin/env python3
"""Collect sharded Gate-2 receipts and apply the frozen convergence rules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def rel_change(a: float, b: float) -> float:
    scale = max(abs(a), abs(b), np.finfo(float).tiny)
    return float(abs(a - b) / scale)


def rank_corr(x: list[float], y: list[float]) -> float:
    if len(x) < 3 or len(x) != len(y):
        return float("nan")
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    rx = np.argsort(np.argsort(xa)).astype(float)
    ry = np.argsort(np.argsort(ya)).astype(float)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def load_receipts(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    core: list[dict[str, Any]] = []
    log: list[dict[str, Any]] = []
    for path in root.rglob("*.json"):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if data.get("gate") == "GATE2_CORE_CONVERGENCE_CELL":
            core.append(data)
        elif data.get("gate") == "GATE2_LOG_REGULATOR_CELL":
            log.append(data)
    return core, log


def core_cell(core: list[dict[str, Any]], size: int, dt: float) -> dict[str, Any]:
    matches = [r for r in core if int(r["size"]) == size and abs(float(r["dt"]) - dt) < 1e-12]
    if len(matches) != 1:
        raise RuntimeError(f"expected one core cell N={size} dt={dt}, found {len(matches)}")
    return matches[0]


def log_cell(log: list[dict[str, Any]], size: int) -> dict[str, Any]:
    matches = [r for r in log if int(r["size"]) == size]
    if len(matches) != 1:
        raise RuntimeError(f"expected one log cell N={size}, found {len(matches)}")
    return matches[0]


def core_summary_row(cell: dict[str, Any], arm: str, tau: float) -> dict[str, Any]:
    rows = [r for r in cell["summaries"] if r["arm"] == arm and abs(float(r["tau_q"]) - tau) < 1e-12]
    if len(rows) != 1:
        raise RuntimeError("missing core summary row")
    return rows[0]


def log_summary_row(cell: dict[str, Any], regulator: float) -> dict[str, Any]:
    rows = [r for r in cell["summaries"] if abs(float(r["regulator"]) - regulator) < 1e-12]
    if len(rows) != 1:
        raise RuntimeError("missing log summary row")
    return rows[0]


def collect(root: Path) -> dict[str, Any]:
    core, log = load_receipts(root)
    required_core = [(64, 0.02), (96, 0.02), (128, 0.02), (96, 0.01), (128, 0.01)]
    for size, dt in required_core:
        core_cell(core, size, dt)
    for size in (64, 96, 128):
        log_cell(log, size)

    within_cells = []
    for size, dt in required_core:
        cell = core_cell(core, size, dt)
        d = cell["diagnostics"]
        passed = bool(d["detector_pass_lt_0p05"] and d["cadence_pass_abs_0p05"] and d["speed_pass_abs_0p05"] and d["charge_pass"])
        within_cells.append({"size": size, "dt": dt, "pass": passed, "diagnostics": d})

    spatial_checks: list[dict[str, Any]] = []
    c96 = core_cell(core, 96, 0.02)
    c128 = core_cell(core, 128, 0.02)
    for arm in ("canonical", "sqrt"):
        for tau in (8.0, 16.0, 32.0):
            a = float(core_summary_row(c96, arm, tau)["threshold_0.85"]["birth_density_mean"])
            b = float(core_summary_row(c128, arm, tau)["threshold_0.85"]["birth_density_mean"])
            change = rel_change(a, b)
            spatial_checks.append({"kind": "birth_density", "arm": arm, "tau_q": tau, "N96": a, "N128": b, "relative_change": change, "pass": change < 0.05})
        slope96 = float(c96["birth_slopes"][arm]["median"])
        slope128 = float(c128["birth_slopes"][arm]["median"])
        diff = abs(slope128 - slope96)
        spatial_checks.append({"kind": "slope", "arm": arm, "N96": slope96, "N128": slope128, "absolute_change": diff, "pass": diff < 0.05})

    time_checks: list[dict[str, Any]] = []
    for size in (96, 128):
        coarse = core_cell(core, size, 0.02)
        fine = core_cell(core, size, 0.01)
        for arm in ("canonical", "sqrt"):
            for tau in (8.0, 16.0, 32.0):
                a = float(core_summary_row(coarse, arm, tau)["threshold_0.85"]["birth_density_mean"])
                b = float(core_summary_row(fine, arm, tau)["threshold_0.85"]["birth_density_mean"])
                change = rel_change(a, b)
                time_checks.append({"kind": "birth_density", "size": size, "arm": arm, "tau_q": tau, "dt0p02": a, "dt0p01": b, "relative_change": change, "pass": change < 0.05})
            s0 = float(coarse["birth_slopes"][arm]["median"])
            s1 = float(fine["birth_slopes"][arm]["median"])
            diff = abs(s1 - s0)
            time_checks.append({"kind": "slope", "size": size, "arm": arm, "dt0p02": s0, "dt0p01": s1, "absolute_change": diff, "pass": diff < 0.05})

    primary_pass = bool(
        all(row["pass"] for row in within_cells)
        and all(row["pass"] for row in spatial_checks)
        and all(row["pass"] for row in time_checks)
    )

    l96 = log_cell(log, 96)
    l128 = log_cell(log, 128)
    regulators = [float(x) for x in l128["regulators"]]
    plateau_checks: list[dict[str, Any]] = []
    spectral_grid_changes: list[float] = []
    for g in regulators:
        r96 = log_summary_row(l96, g)
        r128 = log_summary_row(l128, g)
        ratio96 = float(r96["paired"]["log_to_canonical_birth_ratio_mean"])
        ratio128 = float(r128["paired"]["log_to_canonical_birth_ratio_mean"])
        survival96 = float(r96["paired"]["log_minus_canonical_survival_mean"])
        survival128 = float(r128["paired"]["log_minus_canonical_survival_mean"])
        nn96 = float(r96["paired"]["log_minus_canonical_nn_ks_mean"])
        nn128 = float(r128["paired"]["log_minus_canonical_nn_ks_mean"])
        ratio_change = rel_change(ratio96, ratio128)
        survival_change = abs(survival128 - survival96)
        nn_change = abs(nn128 - nn96)
        peak96 = float(r96["log"]["core_spectrum_peak_k_mean"])
        peak128 = float(r128["log"]["core_spectrum_peak_k_mean"])
        peak_change = rel_change(peak96, peak128) if np.isfinite(peak96) and np.isfinite(peak128) else float("nan")
        if np.isfinite(peak_change):
            spectral_grid_changes.append(peak_change)
        plateau_checks.append({
            "regulator": g,
            "birth_ratio_relative_grid_change": ratio_change,
            "birth_ratio_pass": ratio_change < 0.10,
            "survival_contrast_absolute_grid_change": survival_change,
            "survival_pass": survival_change < 0.05,
            "nn_contrast_absolute_grid_change": nn_change,
            "nn_pass": nn_change < 0.05,
            "core_peak_relative_grid_change": peak_change,
            "core_peak_grid_pass": bool(np.isfinite(peak_change) and peak_change < 0.15),
        })

    predicted: list[float] = []
    measured: list[float] = []
    resolved_g: list[float] = []
    for g in regulators:
        row = log_summary_row(l128, g)["log"]
        sites = float(row["predicted_wavelength_sites_mean"])
        p = float(row["predicted_k_star_mean"])
        m = float(row["core_spectrum_peak_k_mean"])
        if sites >= 8.0 and np.isfinite(p) and np.isfinite(m):
            resolved_g.append(g)
            predicted.append(p)
            measured.append(m)
    rho = rank_corr(predicted, measured)
    enough_resolved = len(resolved_g) >= 3
    spectrum_trend_pass = bool(enough_resolved and np.isfinite(rho) and rho >= 0.6)
    spectrum_grid_pass = bool(len(spectral_grid_changes) >= 3 and float(np.median(spectral_grid_changes)) < 0.15)
    log_long_distance_pass = all(
        row["birth_ratio_pass"] and row["survival_pass"] and row["nn_pass"]
        for row in plateau_checks
    )
    log_pass = bool(log_long_distance_pass and spectrum_trend_pass and spectrum_grid_pass and l96["diagnostics"]["charge_pass"] and l128["diagnostics"]["charge_pass"])

    if log_pass:
        log_classification = "LOG_ARM_GATE2_PLATEAU_CANDIDATE"
    else:
        log_classification = "LOG_ARM_REGULATOR_DEFINED_OR_UNRESOLVED"

    result = {
        "gate": "GATE2_CONVERGENCE_COLLECTOR",
        "protocol": "GATE2_CONVERGENCE_PROTOCOL.md",
        "primary": {
            "classification": "GATE2_PRIMARY_PASS" if primary_pass else "GATE2_PRIMARY_FAIL_OR_UNRESOLVED",
            "pass": primary_pass,
            "within_cell_checks": within_cells,
            "spatial_checks": spatial_checks,
            "time_step_checks": time_checks,
            "gate3_numerically_allowed": primary_pass,
            "gate3_launch_blocked_until_separate_protocol": True,
        },
        "logarithmic": {
            "classification": log_classification,
            "pass": log_pass,
            "plateau_checks": plateau_checks,
            "resolved_g_for_spectrum": resolved_g,
            "predicted_k_star": predicted,
            "measured_core_peak_k": measured,
            "spearman_rank_correlation": rho,
            "spectrum_trend_pass": spectrum_trend_pass,
            "median_core_peak_grid_relative_change": float(np.median(spectral_grid_changes)) if spectral_grid_changes else float("nan"),
            "spectrum_grid_pass": spectrum_grid_pass,
            "long_distance_grid_plateau_pass": log_long_distance_pass,
        },
        "receipt_counts": {"core": len(core), "log": len(log)},
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = collect(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"primary": result["primary"]["classification"], "logarithmic": result["logarithmic"]["classification"]}, indent=2))


if __name__ == "__main__":
    main()
