#!/usr/bin/env python3
"""Gate 1 pilot: separate resolved-core birth from worldline survival.

This script is intentionally a pilot.  Its default seed count checks that the
measurement pipeline works and emits effect-size estimates; it cannot certify
Kibble-Zurek exponents or a new universality class.  The preregistered large
campaign is specified in PREREGISTRATION.md.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from kdefect.quench import QuenchConfig, measure_birth_survival, simulate_quench
from kdefect.statistics import (
    bootstrap_loglog_slope,
    bootstrap_loglog_slope_difference,
    charge_form_factor,
    counting_cumulants,
    normalized_nn_spacings,
    paired_bootstrap_mean_difference,
    poisson_nn_ks,
)

ARMS = ("canonical", "sqrt", "log")


def json_safe(value: Any) -> Any:
    """Recursively replace non-finite floats with JSON null."""

    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def mean_sem(values: list[float]) -> tuple[float, float]:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if len(finite) == 0:
        return float("nan"), float("nan")
    if len(finite) == 1:
        return float(finite[0]), float("nan")
    return float(np.mean(finite)), float(np.std(finite, ddof=1) / np.sqrt(len(finite)))


def run_pilot(
    *,
    output: Path,
    figure: Path,
    size: int,
    seeds: int,
    tau_values: list[float],
    dt: float,
    temperature: float,
    regulator: float,
    kappa: float,
    survival_lag: float,
    analysis_role: str,
) -> dict[str, Any]:
    base = QuenchConfig(
        size=size,
        dt=dt,
        temperature=temperature,
        regulator=regulator,
        kappa=kappa,
        post_quench_time=max(16.0, survival_lag + 4.0),
    )
    runs: list[dict[str, Any]] = []
    groups: dict[tuple[str, float], list[dict[str, Any]]] = {}
    total = len(ARMS) * len(tau_values) * seeds
    completed = 0

    for arm in ARMS:
        for tau_index, tau_q in enumerate(tau_values):
            config = replace(base, tau_q=tau_q)
            group: list[dict[str, Any]] = []
            for seed_index in range(seeds):
                # The same initial field and noise stream are replayed in every arm.
                # This common-random-number design makes arm contrasts paired.
                seed = 17_290_000 + 1000 * tau_index + seed_index
                result = simulate_quench(arm, config, seed=seed)
                metric = measure_birth_survival(result, survival_lag=survival_lag)
                record: dict[str, Any] = {
                    "arm": arm,
                    "tau_q": tau_q,
                    "seed": seed,
                    "status": "resolved_birth" if metric is not None else "no_resolved_birth",
                }
                if metric is not None:
                    frame = metric.birth_frame
                    birth_vortices = result.detections[frame]
                    spacings = normalized_nn_spacings(
                        birth_vortices, config.size * config.dx
                    )
                    record.update(asdict(metric))
                    record.update(
                        {
                            "raw_count_at_birth": int(result.raw_defects[frame]),
                            "raw_charge_at_birth": int(result.raw_charge[frame]),
                            "charge_at_birth": int(sum(v.charge for v in birth_vortices)),
                            "negative_stiffness_fraction_at_birth": float(
                                result.negative_stiffness_fraction[frame]
                            ),
                            "nn_poisson_ks_at_birth": poisson_nn_ks(spacings),
                            "kappa_gradient_p95_at_birth": float(
                                result.kappa_gradient_p95[frame]
                            ),
                            "kappa_gradient_max_at_birth": float(
                                result.kappa_gradient_max[frame]
                            ),
                            "low_k_charge_form_factor": charge_form_factor(
                                birth_vortices, config.size * config.dx
                            ),
                        }
                    )
                group.append(record)
                runs.append(record)
                completed += 1
                print(
                    f"[{completed:03d}/{total:03d}] arm={arm:9s} "
                    f"tau_Q={tau_q:g} seed={seed} status={record['status']}",
                    flush=True,
                )
            groups[(arm, tau_q)] = group

    summaries: list[dict[str, Any]] = []
    slope_receipts: dict[str, Any] = {}
    for arm in ARMS:
        count_groups: list[list[float]] = []
        all_groups_valid = True
        for tau_q in tau_values:
            group = groups[(arm, tau_q)]
            valid = [record for record in group if record["status"] == "resolved_birth"]
            birth_counts = [float(record["birth_count"]) for record in valid]
            tracked = [float(record["tracked_survival"]) for record in valid]
            tolerant = [float(record["gap_tolerant_survival"]) for record in valid]
            reacquired = [
                float(record["final_reacquired_or_new_fraction"]) for record in valid
            ]
            ratios = [float(record["count_ratio"]) for record in valid]
            negative = [
                float(record["negative_stiffness_fraction_at_birth"])
                for record in valid
            ]
            gradient_p95 = [float(record["kappa_gradient_p95_at_birth"]) for record in valid]
            nn_ks = [float(record["nn_poisson_ks_at_birth"]) for record in valid]
            birth_mean, birth_sem = mean_sem(birth_counts)
            tracked_mean, tracked_sem = mean_sem(tracked)
            tolerant_mean, tolerant_sem = mean_sem(tolerant)
            ratio_mean, ratio_sem = mean_sem(ratios)
            summaries.append(
                {
                    "arm": arm,
                    "tau_q": tau_q,
                    "successful_realizations": len(valid),
                    "attempted_realizations": len(group),
                    "birth_count_mean": birth_mean,
                    "birth_count_sem": birth_sem,
                    "birth_count_cumulants": counting_cumulants(birth_counts),
                    "tracked_survival_mean": tracked_mean,
                    "tracked_survival_sem": tracked_sem,
                    "gap_tolerant_survival_mean": tolerant_mean,
                    "gap_tolerant_survival_sem": tolerant_sem,
                    "final_reacquired_or_new_fraction_mean": mean_sem(reacquired)[0],
                    "count_ratio_mean": ratio_mean,
                    "count_ratio_sem": ratio_sem,
                    "negative_stiffness_fraction_mean": mean_sem(negative)[0],
                    "kappa_gradient_p95_mean": mean_sem(gradient_p95)[0],
                    "nn_poisson_ks_mean": mean_sem(nn_ks)[0],
                }
            )
            count_groups.append(birth_counts)
            all_groups_valid &= len(birth_counts) > 0 and all(value > 0 for value in birth_counts)
        slope_receipts[arm] = (
            bootstrap_loglog_slope(
                tau_values,
                count_groups,
                draws=2000,
                seed=2026 + ARMS.index(arm),
            )
            if all_groups_valid
            else {"median": float("nan"), "low_95": float("nan"), "high_95": float("nan")}
        )

    paired_contrasts: list[dict[str, Any]] = []
    slope_contrasts: dict[str, Any] = {}
    for treatment in ("sqrt", "log"):
        reference_count_groups: list[list[float]] = []
        treatment_count_groups: list[list[float]] = []
        for tau_index, tau_q in enumerate(tau_values):
            reference_by_seed = {
                record["seed"]: record
                for record in groups[("canonical", tau_q)]
                if record["status"] == "resolved_birth"
            }
            treatment_by_seed = {
                record["seed"]: record
                for record in groups[(treatment, tau_q)]
                if record["status"] == "resolved_birth"
            }
            common = sorted(reference_by_seed.keys() & treatment_by_seed.keys())
            reference_counts = [float(reference_by_seed[seed]["birth_count"]) for seed in common]
            treatment_counts = [float(treatment_by_seed[seed]["birth_count"]) for seed in common]
            reference_count_groups.append(reference_counts)
            treatment_count_groups.append(treatment_counts)
            contrast: dict[str, Any] = {
                "treatment": treatment,
                "reference": "canonical",
                "tau_q": tau_q,
                "paired_realizations": len(common),
            }
            for observable in (
                "birth_count",
                "tracked_survival",
                "gap_tolerant_survival",
                "count_ratio",
                "final_reacquired_or_new_fraction",
            ):
                reference_values = [
                    float(reference_by_seed[seed][observable]) for seed in common
                ]
                treatment_values = [
                    float(treatment_by_seed[seed][observable]) for seed in common
                ]
                contrast[f"{observable}_difference"] = paired_bootstrap_mean_difference(
                    reference_values,
                    treatment_values,
                    draws=5000,
                    seed=90210 + 100 * tau_index + ARMS.index(treatment),
                )
            paired_contrasts.append(contrast)
        if all(reference_count_groups) and all(treatment_count_groups):
            slope_contrasts[f"{treatment}_minus_canonical"] = (
                bootstrap_loglog_slope_difference(
                    tau_values,
                    reference_count_groups,
                    treatment_count_groups,
                    draws=5000,
                    seed=777 + ARMS.index(treatment),
                )
            )

    valid_runs = [record for record in runs if record["status"] == "resolved_birth"]
    log_runs = [record for record in valid_runs if record["arm"] == "log"]
    receipt: dict[str, Any] = {
        "gate": "GATE_1_BIRTH_VERSUS_SURVIVAL_PILOT",
        "classification": "UNDERPOWERED_PAIRED_PIPELINE_PILOT",
        "analysis_role": analysis_role,
        "config_base": asdict(base),
        "tau_q_values": tau_values,
        "arms": list(ARMS),
        "seeds_per_cell": seeds,
        "common_random_numbers_across_arms": True,
        "survival_lag": survival_lag,
        "worldline_max_speed": 5.0,
        "gap_tolerant_worldline_allowance_frames": 2,
        "operational_birth_definition": (
            "First global maximum of resolved-core count after t>=0 and mean |psi|^2 "
            "reaches 12% of epsilon0/u."
        ),
        "runs": runs,
        "summaries": summaries,
        "bootstrap_birth_count_slopes": slope_receipts,
        "paired_arm_contrasts": paired_contrasts,
        "paired_birth_slope_contrasts": slope_contrasts,
        "diagnostic_outcomes": {
            "all_realizations_have_resolved_birth": len(valid_runs) == len(runs),
            "all_raw_birth_charges_zero": all(
                record["raw_charge_at_birth"] == 0 for record in valid_runs
            ),
            "log_negative_stiffness_activated": any(
                record["negative_stiffness_fraction_at_birth"] > 0.0
                for record in log_runs
            ),
            "maximum_log_negative_stiffness_fraction": max(
                (
                    record["negative_stiffness_fraction_at_birth"]
                    for record in log_runs
                ),
                default=float("nan"),
            ),
            "maximum_log_kappa_gradient_at_birth": max(
                (record["kappa_gradient_max_at_birth"] for record in log_runs),
                default=float("nan"),
            ),
            "lineage_ambiguity_present": any(
                record["final_reacquired_or_new_fraction"] > 0.25
                for record in valid_runs
                if np.isfinite(record["final_reacquired_or_new_fraction"])
            ),
            "confirmatory_inference_allowed": False,
        },
        "claim_boundary": (
            "Default statistics are a CI-sized instrument check. Exponent comparison, "
            "regulator extrapolation, and spatial universality require the preregistered "
            "large campaign."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(json_safe(receipt), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    make_figure(summaries, tau_values, figure)
    return receipt


def make_figure(summaries: list[dict[str, Any]], tau_values: list[float], output: Path) -> None:
    colors = {"canonical": "#005f73", "sqrt": "#ca6702", "log": "#9b2226"}
    markers = {"canonical": "o", "sqrt": "s", "log": "^"}
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1))
    for arm in ARMS:
        rows = [row for row in summaries if row["arm"] == arm]
        x = np.asarray(tau_values)
        birth = np.asarray([row["birth_count_mean"] for row in rows])
        birth_error = np.asarray([row["birth_count_sem"] for row in rows])
        survival = np.asarray([row["gap_tolerant_survival_mean"] for row in rows])
        survival_error = np.asarray([row["gap_tolerant_survival_sem"] for row in rows])
        negative = np.asarray([row["negative_stiffness_fraction_mean"] for row in rows])
        axes[0].errorbar(
            x,
            birth,
            yerr=birth_error,
            color=colors[arm],
            marker=markers[arm],
            label=arm,
            capsize=2,
        )
        axes[1].errorbar(
            x,
            survival,
            yerr=survival_error,
            color=colors[arm],
            marker=markers[arm],
            capsize=2,
        )
        axes[2].plot(x, negative, color=colors[arm], marker=markers[arm])
    axes[0].set(
        xscale="log",
        yscale="log",
        xlabel=r"quench time $\tau_Q$",
        ylabel="resolved cores at birth",
    )
    axes[1].set(
        xscale="log",
        xlabel=r"quench time $\tau_Q$",
        ylabel="gap-tolerant cohort survival",
        ylim=(-0.03, 1.03),
    )
    axes[2].set(
        xscale="log",
        xlabel=r"quench time $\tau_Q$",
        ylabel="negative-stiffness area at birth",
        ylim=(-0.03, 1.03),
    )
    axes[0].legend(frameon=False)
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.suptitle("Gate 1 pilot — instrument receipt, not an exponent claim", y=1.02)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/gate1_pilot.json"))
    parser.add_argument("--figure", type=Path, default=Path("figures/gate1_pilot.png"))
    parser.add_argument("--size", type=int, default=48)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--tau", type=float, nargs="+", default=[8.0, 16.0, 32.0])
    parser.add_argument("--dt", type=float, default=0.04)
    parser.add_argument("--temperature", type=float, default=0.006)
    parser.add_argument("--regulator", type=float, default=0.03)
    parser.add_argument("--kappa", type=float, default=4.0)
    parser.add_argument("--survival-lag", type=float, default=8.0)
    parser.add_argument(
        "--analysis-role",
        default="preregistered_weak_core_pipeline_pilot",
        help="Receipt label; use an explicit exploratory label for parameter scouting.",
    )
    args = parser.parse_args()
    receipt = run_pilot(
        output=args.output,
        figure=args.figure,
        size=args.size,
        seeds=args.seeds,
        tau_values=args.tau,
        dt=args.dt,
        temperature=args.temperature,
        regulator=args.regulator,
        kappa=args.kappa,
        survival_lag=args.survival_lag,
        analysis_role=args.analysis_role,
    )
    summary_keys = ("gate", "classification", "bootstrap_birth_count_slopes")
    print(json.dumps(json_safe({key: receipt[key] for key in summary_keys}), indent=2))


if __name__ == "__main__":
    main()
