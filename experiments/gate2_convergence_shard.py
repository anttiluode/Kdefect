#!/usr/bin/env python3
"""Parallel execution wrapper for the frozen Gate-2 campaign.

One core shard = one (N, dt, tau_Q) cell containing both primary arms.
One log shard  = one (N, g) cell containing canonical + logarithmic arms.
The scientific configuration, including the monolithic campaign's seed schedule,
is unchanged from GATE2_CONVERGENCE_PROTOCOL.md and gate2_convergence_campaign.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import gate2_convergence_campaign as c

# Frozen monolithic schedules. Sharding must not reset local enumerate() indices
# and thereby silently reuse seed groups across tau/g cells.
FULL_TAUS = (8.0, 16.0, 32.0)
FULL_REGULATORS = (0.020, 0.030, 0.045, 0.0675, 0.100)
ORIGINAL_BASE_SEED = int(c.BASE_SEED)

# Measurement-only optimization: retain the loosest preregistered 0.95 core
# detector once, then derive 0.85/0.75 by filtering core_ratio. The detector
# threshold does not enter field evolution.
_OriginalConfig = c.QuenchConfig


def _superset_config(*args, **kwargs):
    kwargs["max_core_ratio"] = 0.95
    return _OriginalConfig(*args, **kwargs)


def _already_measured_95(result):
    if abs(float(result.config.max_core_ratio) - 0.95) > 1e-12:
        raise RuntimeError("shard runner requires the 0.95 measurement superset")
    return [list(frame) for frame in result.detections]


c.QuenchConfig = _superset_config
c._all_detections_95 = _already_measured_95


def exact_index(value: float, schedule: tuple[float, ...], label: str) -> int:
    matches = [i for i, expected in enumerate(schedule) if abs(float(value) - expected) < 1e-12]
    if len(matches) != 1:
        raise ValueError(f"{label}={value} is not in the frozen schedule {schedule}")
    return matches[0]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel", choices=("core", "log"), required=True)
    p.add_argument("--size", type=int, required=True)
    p.add_argument("--dt", type=float, required=True)
    p.add_argument("--tau", type=float)
    p.add_argument("--regulator", type=float)
    p.add_argument("--seeds", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()

    if a.panel == "core":
        if a.tau is None:
            p.error("--tau is required for core shards")
        tau = float(a.tau)
        tau_index = exact_index(tau, FULL_TAUS, "tau")
        # run_core_cell locally enumerates its one TAU as ti=0. Shift BASE_SEED
        # so its formula BASE + 10000*ti + si exactly equals the monolithic
        # formula ORIGINAL_BASE + 10000*tau_index + si.
        c.BASE_SEED = ORIGINAL_BASE_SEED + 10_000 * tau_index
        c.TAU_VALUES = (tau,)
        result = c.run_core_cell(size=a.size, dt=a.dt, seeds=a.seeds, output=a.output)
        result["shard"] = {
            "panel": "core",
            "tau_q": tau,
            "frozen_tau_index": tau_index,
            "effective_base_seed": int(c.BASE_SEED),
        }
    else:
        if a.regulator is None:
            p.error("--regulator is required for log shards")
        regulator = float(a.regulator)
        g_index = exact_index(regulator, FULL_REGULATORS, "regulator")
        # run_log_cell adds 500000 + 10000*gi + si. With one local g its gi=0,
        # so shift BASE_SEED by the frozen monolithic g index.
        c.BASE_SEED = ORIGINAL_BASE_SEED + 10_000 * g_index
        c.REGULATORS = (regulator,)
        result = c.run_log_cell(size=a.size, dt=a.dt, seeds=a.seeds, output=a.output)
        result["shard"] = {
            "panel": "log",
            "regulator": regulator,
            "frozen_regulator_index": g_index,
            "effective_base_seed": int(c.BASE_SEED),
        }

    # run_* already wrote the receipt; rewrite once with explicit shard metadata.
    a.output.write_text(json.dumps(c.json_safe(result), indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
