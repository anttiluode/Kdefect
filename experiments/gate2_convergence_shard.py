#!/usr/bin/env python3
"""Parallel execution wrapper for the frozen Gate-2 campaign.

One core shard = one (N, dt, tau_Q) cell containing both primary arms.
One log shard  = one (N, g) cell containing canonical + logarithmic arms.
The scientific configuration is unchanged from GATE2_CONVERGENCE_PROTOCOL.md.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import gate2_convergence_campaign as c

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
        c.TAU_VALUES = (float(a.tau),)
        result = c.run_core_cell(size=a.size, dt=a.dt, seeds=a.seeds, output=a.output)
        # Make shard identity explicit because a one-point slope is intentionally undefined.
        result["shard"] = {"panel": "core", "tau_q": float(a.tau)}
    else:
        if a.regulator is None:
            p.error("--regulator is required for log shards")
        c.REGULATORS = (float(a.regulator),)
        result = c.run_log_cell(size=a.size, dt=a.dt, seeds=a.seeds, output=a.output)
        result["shard"] = {"panel": "log", "regulator": float(a.regulator)}
    # run_* already wrote the receipt; rewrite once with explicit shard metadata.
    import json
    a.output.write_text(json.dumps(c.json_safe(result), indent=2, allow_nan=False) + "\n")

if __name__ == "__main__":
    main()
