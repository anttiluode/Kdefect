#!/usr/bin/env python3
"""Execution-only optimization for the frozen Gate-2 campaign.

`max_core_ratio` affects measurement only, not field evolution.  The first
campaign implementation evolved with the 0.85 detector and then re-ran the
expensive root finder on every stored frame to obtain the 0.95 superset.
This wrapper instead asks `simulate_quench` to retain the preregistered loosest
0.95 detector during evolution and derives 0.85/0.75 by filtering `core_ratio`.
No seed, dynamics, observable, threshold, or pass/fail criterion changes.
"""

from __future__ import annotations

import sys
from dataclasses import replace

import gate2_convergence_campaign as campaign

_QuenchConfig = campaign.QuenchConfig


def _measurement_superset_config(*args, **kwargs):
    kwargs["max_core_ratio"] = 0.95
    return _QuenchConfig(*args, **kwargs)


def _already_measured_95(result):
    if abs(float(result.config.max_core_ratio) - 0.95) > 1e-12:
        raise RuntimeError("fast Gate-2 runner requires max_core_ratio=0.95")
    return [list(frame) for frame in result.detections]


campaign.QuenchConfig = _measurement_superset_config
campaign._all_detections_95 = _already_measured_95

if __name__ == "__main__":
    campaign.main()
