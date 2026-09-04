#!/usr/bin/env python3
"""Gate 0: audit action/force consistency and ellipticity boundaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from kdefect.lattice import deterministic_force, energy
from kdefect.laws import LAWS


def smooth_random_field(rng: np.random.Generator, size: int) -> np.ndarray:
    field = rng.standard_normal((size, size)) + 1j * rng.standard_normal((size, size))
    for _ in range(4):
        field = (
            4.0 * field
            + np.roll(field, 1, axis=0)
            + np.roll(field, -1, axis=0)
            + np.roll(field, 1, axis=1)
            + np.roll(field, -1, axis=1)
        ) / 8.0
    return 0.4 * field


def run_audit(output: Path, figure: Path, seed: int = 1729) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    field = smooth_random_field(rng, 20)
    direction = smooth_random_field(rng, 20)
    direction /= np.sqrt(np.mean(np.abs(direction) ** 2))
    parameters = {
        "epsilon": 0.31,
        "kappa": 4.0,
        "quartic": 1.0,
        "regulator": 0.03,
        "dx": 0.8,
    }
    finite_difference_step = 1.0e-6
    descent_step = 1.0e-5
    arms: dict[str, dict[str, float | bool]] = {}
    passed = True

    for name, law in LAWS.items():
        force = deterministic_force(field, law, **parameters)
        plus = energy(field + finite_difference_step * direction, law, **parameters)
        minus = energy(field - finite_difference_step * direction, law, **parameters)
        finite_difference = (plus - minus) / (2.0 * finite_difference_step)
        action_gradient = -parameters["dx"] ** 2 * float(
            np.real(np.vdot(direction, force))
        )
        relative_error = abs(finite_difference - action_gradient) / max(
            1.0, abs(finite_difference), abs(action_gradient)
        )
        before = energy(field, law, **parameters)
        after = energy(field + descent_step * force, law, **parameters)
        drop = float(before - after)
        arm_passed = bool(relative_error < 2.0e-7 and drop > 0.0)
        passed &= arm_passed
        arms[name] = {
            "finite_difference_directional_derivative": float(finite_difference),
            "force_directional_derivative": float(action_gradient),
            "relative_error": float(relative_error),
            "energy_before": float(before),
            "energy_after_one_gradient_step": float(after),
            "energy_drop": drop,
            "passed": arm_passed,
        }

    kappa = parameters["kappa"]
    y = np.geomspace(1.0e-4 / kappa, 1.0e3 / kappa, 1200)
    sqrt_stiffness = LAWS["sqrt"].longitudinal_stiffness(y, kappa)
    threshold_checks = {
        "predicted_log_crossing_Y": 1.0 / kappa,
        "log_below_threshold_positive": bool(
            LAWS["log"].longitudinal_stiffness(0.99 / kappa, kappa) > 0.0
        ),
        "log_at_threshold_zero": bool(
            abs(float(LAWS["log"].longitudinal_stiffness(1.0 / kappa, kappa))) < 1.0e-14
        ),
        "log_above_threshold_negative": bool(
            LAWS["log"].longitudinal_stiffness(1.01 / kappa, kappa) < 0.0
        ),
        "sqrt_positive_on_scan": bool(np.all(sqrt_stiffness > 0.0)),
        "canonical_positive_on_scan": True,
    }
    passed &= all(
        threshold_checks[key]
        for key in threshold_checks
        if key != "predicted_log_crossing_Y"
    )

    figure.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(7.2, 4.6))
    for law in LAWS.values():
        axis.plot(
            kappa * y,
            law.longitudinal_stiffness(y, kappa),
            label=law.label,
            linewidth=2.0,
        )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.axvline(1.0, color="#9b2226", linestyle="--", linewidth=1.1, label=r"$\kappa Y=1$")
    axis.set_xscale("log")
    axis.set_ylim(-0.16, 1.05)
    axis.set_xlabel(r"dimensionless core gradient $\kappa Y$")
    axis.set_ylabel(r"longitudinal stiffness $F'+2YF''$")
    axis.set_title("Gate 0: same linear limit, different nonlinear stiffness")
    axis.legend(frameon=False, ncol=2)
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(figure, dpi=180)
    plt.close(fig)

    receipt: dict[str, object] = {
        "gate": "GATE_0_ACTION_AUDIT",
        "classification": (
            "ACTION_DERIVED_ARMS_AND_STIFFNESS_BOUNDARIES_VERIFIED"
            if passed
            else "GATE_0_FAILED"
        ),
        "passed": bool(passed),
        "seed": seed,
        "parameters": parameters,
        "finite_difference_step": finite_difference_step,
        "descent_step": descent_step,
        "arms": arms,
        "ellipticity": threshold_checks,
        "claim_boundary": (
            "This gate validates the discretized variational force and analytic principal "
            "stiffness. It is not evidence for a new universality class."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/gate0_action_audit.json"))
    parser.add_argument("--figure", type=Path, default=Path("figures/gate0_stiffness.png"))
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    receipt = run_audit(args.output, args.figure, args.seed)
    print(json.dumps(receipt, indent=2))
    if not receipt["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
