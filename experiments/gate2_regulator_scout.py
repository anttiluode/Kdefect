#!/usr/bin/env python3
"""Gate 2 scout: expose, rather than hide, fourth-order regulator dependence.

The logarithmic arm is non-elliptic when kappa*Y>1.  Its ``g nabla^4`` term can
therefore select a finite wavelength.  This small paired scan asks whether the
strong-core pilot is visibly regulator-sensitive before any expensive scaling
campaign is attempted.  It is exploratory and cannot establish convergence.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from kdefect.lattice import gradient_invariant
from kdefect.laws import get_law
from kdefect.quench import QuenchConfig, measure_birth_survival, simulate_quench
from kdefect.statistics import density_spectrum_diagnostics

ARMS = ("canonical", "sqrt", "log")


def mean_sem(values: list[float]) -> tuple[float, float]:
    array = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if len(array) == 0:
        return float("nan"), float("nan")
    if len(array) == 1:
        return float(array[0]), float("nan")
    return float(np.mean(array)), float(np.std(array, ddof=1) / np.sqrt(len(array)))


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def run_scout(
    *,
    output: Path,
    figure: Path,
    size: int,
    seeds: int,
    tau_q: float,
    kappa: float,
    regulators: list[float],
    dt: float,
) -> dict[str, Any]:
    base = QuenchConfig(size=size, tau_q=tau_q, kappa=kappa, dt=dt)
    runs: list[dict[str, Any]] = []
    total = len(ARMS) * len(regulators) * seeds
    completed = 0
    for arm in ARMS:
        law = get_law(arm)
        for regulator in regulators:
            config = replace(base, regulator=regulator)
            for seed_index in range(seeds):
                # Identical seed at every g and in every arm: paired scout.
                seed = 86_420_000 + seed_index
                result = simulate_quench(arm, config, seed=seed, store_fields=True)
                metric = measure_birth_survival(result)
                record: dict[str, Any] = {
                    "arm": arm,
                    "regulator": regulator,
                    "seed": seed,
                    "status": "resolved_birth" if metric is not None else "no_resolved_birth",
                }
                if metric is not None:
                    frame = metric.birth_frame
                    assert result.sampled_fields is not None
                    birth_field = result.sampled_fields[frame]
                    y = gradient_invariant(birth_field, config.dx)
                    stiffness = law.longitudinal_stiffness(y, config.kappa)
                    negative = stiffness[stiffness < 0.0]
                    predicted_k = (
                        float(np.sqrt(np.median(-negative) / (2.0 * regulator)))
                        if len(negative) > 0 and regulator > 0.0
                        else float("nan")
                    )
                    record.update(asdict(metric))
                    record.update(
                        {
                            "raw_charge_at_birth": int(result.raw_charge[frame]),
                            "negative_stiffness_fraction_at_birth": float(
                                result.negative_stiffness_fraction[frame]
                            ),
                            "median_local_predicted_k_star": predicted_k,
                            "kappa_gradient_p95_at_birth": float(
                                result.kappa_gradient_p95[frame]
                            ),
                            "birth_density_spectrum": density_spectrum_diagnostics(
                                birth_field, config.dx
                            ),
                        }
                    )
                runs.append(record)
                completed += 1
                print(
                    f"[{completed:03d}/{total:03d}] arm={arm:9s} g={regulator:g} "
                    f"seed={seed} status={record['status']}",
                    flush=True,
                )

    summaries: list[dict[str, Any]] = []
    for arm in ARMS:
        for regulator in regulators:
            group = [
                record
                for record in runs
                if record["arm"] == arm
                and record["regulator"] == regulator
                and record["status"] == "resolved_birth"
            ]
            fields = {
                "birth_count": [float(record["birth_count"]) for record in group],
                "gap_tolerant_survival": [
                    float(record["gap_tolerant_survival"]) for record in group
                ],
                "final_reacquired_or_new_fraction": [
                    float(record["final_reacquired_or_new_fraction"]) for record in group
                ],
                "negative_stiffness_fraction_at_birth": [
                    float(record["negative_stiffness_fraction_at_birth"])
                    for record in group
                ],
                "radial_peak_k": [
                    float(record["birth_density_spectrum"]["radial_peak_k"])
                    for record in group
                ],
                "median_local_predicted_k_star": [
                    float(record["median_local_predicted_k_star"]) for record in group
                ],
            }
            summary: dict[str, Any] = {
                "arm": arm,
                "regulator": regulator,
                "successful_realizations": len(group),
            }
            for name, values in fields.items():
                average, sem = mean_sem(values)
                summary[f"{name}_mean"] = average
                summary[f"{name}_sem"] = sem
            summaries.append(summary)

    paired_birth_ratios: list[dict[str, Any]] = []
    for regulator in regulators:
        reference = {
            record["seed"]: record
            for record in runs
            if record["arm"] == "canonical"
            and record["regulator"] == regulator
            and record["status"] == "resolved_birth"
        }
        for treatment in ("sqrt", "log"):
            treated = {
                record["seed"]: record
                for record in runs
                if record["arm"] == treatment
                and record["regulator"] == regulator
                and record["status"] == "resolved_birth"
            }
            common = sorted(reference.keys() & treated.keys())
            ratios = [
                float(treated[seed]["birth_count"] / reference[seed]["birth_count"])
                for seed in common
                if reference[seed]["birth_count"] > 0
            ]
            average, sem = mean_sem(ratios)
            paired_birth_ratios.append(
                {
                    "treatment": treatment,
                    "reference": "canonical",
                    "regulator": regulator,
                    "finite_pairs": len(ratios),
                    "birth_count_ratio_mean": average,
                    "birth_count_ratio_sem": sem,
                }
            )

    log_birth = np.asarray(
        [
            row["birth_count_mean"]
            for row in summaries
            if row["arm"] == "log" and np.isfinite(row["birth_count_mean"])
        ]
    )
    relative_span = (
        float((np.max(log_birth) - np.min(log_birth)) / np.mean(log_birth))
        if len(log_birth) > 1
        else float("nan")
    )
    canonical_birth = np.asarray(
        [
            row["birth_count_mean"]
            for row in summaries
            if row["arm"] == "canonical" and np.isfinite(row["birth_count_mean"])
        ]
    )
    canonical_relative_span = (
        float((np.max(canonical_birth) - np.min(canonical_birth)) / np.mean(canonical_birth))
        if len(canonical_birth) > 1
        else float("nan")
    )
    log_ratios = np.asarray(
        [
            row["birth_count_ratio_mean"]
            for row in paired_birth_ratios
            if row["treatment"] == "log"
            and np.isfinite(row["birth_count_ratio_mean"])
        ]
    )
    log_ratio_relative_span = (
        float((np.max(log_ratios) - np.min(log_ratios)) / np.mean(log_ratios))
        if len(log_ratios) > 1
        else float("nan")
    )
    valid_runs = [record for record in runs if record["status"] == "resolved_birth"]
    receipt: dict[str, Any] = {
        "gate": "GATE_2_REGULATOR_SCOUT",
        "classification": "EXPLORATORY_REGULATOR_SCOUT_NOT_A_CONVERGENCE_RESULT",
        "analysis_role": "exploratory_one_grid_regulator_scout",
        "config_base": asdict(base),
        "regulator_values": regulators,
        "seeds_per_cell": seeds,
        "common_random_numbers_across_arms_and_regulators": True,
        "runs": runs,
        "summaries": summaries,
        "paired_birth_count_ratios": paired_birth_ratios,
        "diagnostic_outcomes": {
            "all_realizations_have_resolved_birth": len(valid_runs) == len(runs),
            "all_raw_birth_charges_zero": all(
                record["raw_charge_at_birth"] == 0 for record in valid_runs
            ),
            "log_birth_count_relative_span_over_g": relative_span,
            "canonical_birth_count_relative_span_over_g": canonical_relative_span,
            "log_to_canonical_birth_ratio_relative_span_over_g": log_ratio_relative_span,
            "visible_absolute_regulator_sensitivity_at_10_percent_scale": (
                relative_span > 0.10
            ),
            "visible_control_normalized_sensitivity_at_10_percent_scale": (
                log_ratio_relative_span > 0.10
            ),
            "regulator_independence_demonstrated": False,
        },
        "claim_boundary": (
            "One small grid and a few seeds can reveal obvious g-dependence but cannot "
            "demonstrate a continuum plateau or regulator-independent physics."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(json_safe(receipt), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    make_figure(summaries, regulators, figure)
    return receipt


def make_figure(
    summaries: list[dict[str, Any]], regulators: list[float], output: Path
) -> None:
    colors = {"canonical": "#005f73", "sqrt": "#ca6702", "log": "#9b2226"}
    markers = {"canonical": "o", "sqrt": "s", "log": "^"}
    figure, axes = plt.subplots(2, 2, figsize=(10.2, 7.6), sharex=True)
    for arm in ARMS:
        rows = [row for row in summaries if row["arm"] == arm]
        x = np.asarray(regulators)
        for axis, field, error in (
            (axes[0, 0], "birth_count_mean", "birth_count_sem"),
            (
                axes[0, 1],
                "gap_tolerant_survival_mean",
                "gap_tolerant_survival_sem",
            ),
            (
                axes[1, 0],
                "negative_stiffness_fraction_at_birth_mean",
                "negative_stiffness_fraction_at_birth_sem",
            ),
            (axes[1, 1], "radial_peak_k_mean", "radial_peak_k_sem"),
        ):
            axis.errorbar(
                x,
                [row[field] for row in rows],
                yerr=[row[error] for row in rows],
                color=colors[arm],
                marker=markers[arm],
                label=arm,
                capsize=2,
            )
        if arm == "log":
            axes[1, 1].plot(
                x,
                [row["median_local_predicted_k_star_mean"] for row in rows],
                color=colors[arm],
                linestyle="--",
                label="log local prediction",
            )
    axes[0, 0].set_ylabel("resolved birth count")
    axes[0, 1].set_ylabel("gap-tolerant survival")
    axes[1, 0].set_ylabel("negative-stiffness area")
    axes[1, 1].set_ylabel("density-spectrum peak k")
    for axis in axes.flat:
        axis.set_xscale("log")
        axis.set_xlabel("biharmonic regulator g")
        axis.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False)
    axes[1, 1].legend(frameon=False, fontsize=8)
    figure.suptitle("Gate 2 regulator scout — one grid, exploratory only")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/gate2_regulator_scout.json"))
    parser.add_argument("--figure", type=Path, default=Path("figures/gate2_regulator_scout.png"))
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--tau-q", type=float, default=16.0)
    parser.add_argument("--kappa", type=float, default=32.0)
    parser.add_argument("--regulator", type=float, nargs="+", default=[0.015, 0.03, 0.06, 0.12])
    parser.add_argument("--dt", type=float, default=0.04)
    args = parser.parse_args()
    receipt = run_scout(
        output=args.output,
        figure=args.figure,
        size=args.size,
        seeds=args.seeds,
        tau_q=args.tau_q,
        kappa=args.kappa,
        regulators=args.regulator,
        dt=args.dt,
    )
    print(json.dumps(receipt["diagnostic_outcomes"], indent=2))


if __name__ == "__main__":
    main()
