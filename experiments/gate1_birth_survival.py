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
    charge_form_factor,
    counting_cumulants,
    normalized_nn_spacings,
    poisson_nn_ks,
)

ARMS = ("canonical", "sqrt", "log")


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
    survival_lag: float,
) -> dict[str, Any]:
    base = QuenchConfig(
        size=size,
        dt=dt,
        temperature=temperature,
        regulator=regulator,
        post_quench_time=max(16.0, survival_lag + 4.0),
    )
    runs: list[dict[str, Any]] = []
    groups: dict[tuple[str, float], list[dict[str, Any]]] = {}
    total = len(ARMS) * len(tau_values) * seeds
    completed = 0

    for arm in ARMS:
        for tau_q in tau_values:
            config = replace(base, tau_q=tau_q)
            group: list[dict[str, Any]] = []
            for seed_index in range(seeds):
                seed = 10_000 * ARMS.index(arm) + 100 * int(round(tau_q)) + seed_index
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
                            "charge_at_birth": int(sum(v.charge for v in birth_vortices)),
                            "negative_stiffness_fraction_at_birth": float(
                                result.negative_stiffness_fraction[frame]
                            ),
                            "nn_poisson_ks_at_birth": poisson_nn_ks(spacings),
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
            ratios = [float(record["count_ratio"]) for record in valid]
            negative = [
                float(record["negative_stiffness_fraction_at_birth"])
                for record in valid
            ]
            nn_ks = [float(record["nn_poisson_ks_at_birth"]) for record in valid]
            birth_mean, birth_sem = mean_sem(birth_counts)
            tracked_mean, tracked_sem = mean_sem(tracked)
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
                    "count_ratio_mean": ratio_mean,
                    "count_ratio_sem": ratio_sem,
                    "negative_stiffness_fraction_mean": mean_sem(negative)[0],
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

    receipt: dict[str, Any] = {
        "gate": "GATE_1_BIRTH_VERSUS_SURVIVAL_PILOT",
        "classification": "PIPELINE_PILOT_ONLY_NOT_A_UNIVERSALITY_CLAIM",
        "config_base": asdict(base),
        "tau_q_values": tau_values,
        "arms": list(ARMS),
        "seeds_per_cell": seeds,
        "survival_lag": survival_lag,
        "worldline_max_speed": 5.0,
        "operational_birth_definition": (
            "First global maximum of resolved-core count after t>=0 and mean |psi|^2 "
            "reaches 12% of epsilon0/u."
        ),
        "runs": runs,
        "summaries": summaries,
        "bootstrap_birth_count_slopes": slope_receipts,
        "claim_boundary": (
            "Default statistics are a CI-sized instrument check. Exponent comparison, "
            "regulator extrapolation, and spatial universality require the preregistered "
            "large campaign."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, allow_nan=True) + "\n", encoding="utf-8")
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
        survival = np.asarray([row["tracked_survival_mean"] for row in rows])
        survival_error = np.asarray([row["tracked_survival_sem"] for row in rows])
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
        ylabel="tracked survival fraction",
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
    parser.add_argument("--survival-lag", type=float, default=8.0)
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
        survival_lag=args.survival_lag,
    )
    summary_keys = ("gate", "classification", "bootstrap_birth_count_slopes")
    print(json.dumps({key: receipt[key] for key in summary_keys}, indent=2))


if __name__ == "__main__":
    main()
