#!/usr/bin/env python3
"""Collect the 30 parallel shards of the frozen Gate-2 campaign."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from kdefect.statistics import bootstrap_loglog_slope

TAUS = (8.0, 16.0, 32.0)
GS = (0.020, 0.030, 0.045, 0.0675, 0.100)


def rel_change(a: float, b: float) -> float:
    scale = max(abs(a), abs(b), np.finfo(float).tiny)
    return float(abs(a-b)/scale)


def finite_float(value: Any) -> float:
    """Convert JSON numeric/null to a finite float or NaN."""
    if value is None:
        return float("nan")
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def average_ranks(values: list[float]) -> np.ndarray:
    """Return 0-based average ranks, including exact ties.

    The previous double-argsort shortcut assigned arbitrary different ranks to
    tied FFT-shell peaks. Gate 2 compares discretized spectral peaks, so tie
    handling must be explicit before any result is inspected.
    """
    x = np.asarray(values, dtype=float)
    if x.ndim != 1:
        raise ValueError("rank input must be one-dimensional")
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and x[order[end]] == x[order[start]]:
            end += 1
        average = 0.5 * (start + end - 1)
        ranks[order[start:end]] = average
        start = end
    return ranks


def rank_corr(x: list[float], y: list[float]) -> float:
    """Tie-aware Spearman rank correlation without pooled-vortex inference."""
    if len(x) < 3 or len(x) != len(y):
        return float("nan")
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    valid = np.isfinite(xa) & np.isfinite(ya)
    if np.count_nonzero(valid) < 3:
        return float("nan")
    rx = average_ranks(xa[valid].tolist())
    ry = average_ranks(ya[valid].tolist())
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def load(root: Path) -> tuple[list[dict[str,Any]], list[dict[str,Any]]]:
    core=[]; log=[]
    for p in root.rglob("*.json"):
        try: d=json.loads(p.read_text())
        except Exception: continue
        shard=d.get("shard",{})
        if shard.get("panel")=="core": core.append(d)
        elif shard.get("panel")=="log": log.append(d)
    return core,log


def core_shard(core,size,dt,tau):
    m=[d for d in core if int(d["size"])==size and abs(float(d["dt"])-dt)<1e-12 and abs(float(d["shard"]["tau_q"])-tau)<1e-12]
    if len(m)!=1: raise RuntimeError(f"need one core shard N={size} dt={dt} tau={tau}; got {len(m)}")
    return m[0]

def log_shard(log,size,g):
    m=[d for d in log if int(d["size"])==size and abs(float(d["shard"]["regulator"])-g)<1e-12]
    if len(m)!=1: raise RuntimeError(f"need one log shard N={size} g={g}; got {len(m)}")
    return m[0]
def core_row(d,arm):
    m=[r for r in d["summaries"] if r["arm"]==arm]
    if len(m)!=1: raise RuntimeError("core summary arm missing")
    return m[0]
def log_row(d):
    if len(d["summaries"])!=1: raise RuntimeError("log shard must have one g summary")
    return d["summaries"][0]


def collect(root: Path) -> dict[str,Any]:
    core,log=load(root)
    cells=((64,.02),(96,.02),(128,.02),(96,.01),(128,.01))
    for n,dt in cells:
        for tau in TAUS: core_shard(core,n,dt,tau)
    for n in (64,96,128):
        for g in GS: log_shard(log,n,g)

    within=[]
    for n,dt in cells:
        shards=[core_shard(core,n,dt,t) for t in TAUS]
        max_detector=max(float(s["diagnostics"]["max_detector_relative_birth_change"]) for s in shards)
        max_cadence=max(float(s["diagnostics"]["max_cadence_absolute_survival_change"]) for s in shards)
        max_speed=max(float(s["diagnostics"]["max_speed_absolute_survival_change"]) for s in shards)
        charge=sum(int(s["diagnostics"]["nonzero_raw_charge_frames"]) for s in shards)
        passed=max_detector<.05 and max_cadence<.05 and max_speed<.05 and charge==0
        within.append({"size":n,"dt":dt,"pass":passed,"max_detector_relative_birth_change":max_detector,"max_cadence_absolute_survival_change":max_cadence,"max_speed_absolute_survival_change":max_speed,"nonzero_raw_charge_frames":charge})

    # Reconstruct slopes from individual-realization birth densities at threshold .85.
    slopes={}
    for n,dt in cells:
        slopes[f"{n}_{dt}"]={}
        for ai,arm in enumerate(("canonical","sqrt")):
            groups=[]
            for tau in TAUS:
                s=core_shard(core,n,dt,tau)
                vals=[r["variants"]["threshold_0.85"]["birth_density"] for r in s["records"] if r["arm"]==arm and r["variants"].get("threshold_0.85") is not None]
                groups.append(vals)
            slopes[f"{n}_{dt}"][arm]=bootstrap_loglog_slope(TAUS,groups,draws=5000,seed=99100+n+ai)

    spatial=[]
    for arm in ("canonical","sqrt"):
        for tau in TAUS:
            a=float(core_row(core_shard(core,96,.02,tau),arm)["threshold_0.85"]["birth_density_mean"])
            b=float(core_row(core_shard(core,128,.02,tau),arm)["threshold_0.85"]["birth_density_mean"])
            ch=rel_change(a,b)
            spatial.append({"kind":"birth_density","arm":arm,"tau_q":tau,"N96":a,"N128":b,"relative_change":ch,"pass":ch<.05})
        a=float(slopes["96_0.02"][arm]["median"]); b=float(slopes["128_0.02"][arm]["median"]); ch=abs(b-a)
        spatial.append({"kind":"slope","arm":arm,"N96":a,"N128":b,"absolute_change":ch,"pass":ch<.05})

    timestep=[]
    for n in (96,128):
        for arm in ("canonical","sqrt"):
            for tau in TAUS:
                a=float(core_row(core_shard(core,n,.02,tau),arm)["threshold_0.85"]["birth_density_mean"])
                b=float(core_row(core_shard(core,n,.01,tau),arm)["threshold_0.85"]["birth_density_mean"])
                ch=rel_change(a,b)
                timestep.append({"kind":"birth_density","size":n,"arm":arm,"tau_q":tau,"dt0p02":a,"dt0p01":b,"relative_change":ch,"pass":ch<.05})
            a=float(slopes[f"{n}_0.02"][arm]["median"]); b=float(slopes[f"{n}_0.01"][arm]["median"]); ch=abs(b-a)
            timestep.append({"kind":"slope","size":n,"arm":arm,"dt0p02":a,"dt0p01":b,"absolute_change":ch,"pass":ch<.05})

    primary_pass=all(x["pass"] for x in within+spatial+timestep)

    plateau=[]; grid_peak_changes=[]
    for g in GS:
        a=log_row(log_shard(log,96,g)); b=log_row(log_shard(log,128,g))
        ratio96=float(a["paired"]["log_to_canonical_birth_ratio_mean"]); ratio128=float(b["paired"]["log_to_canonical_birth_ratio_mean"])
        surv96=float(a["paired"]["log_minus_canonical_survival_mean"]); surv128=float(b["paired"]["log_minus_canonical_survival_mean"])
        nn96=float(a["paired"]["log_minus_canonical_nn_ks_mean"]); nn128=float(b["paired"]["log_minus_canonical_nn_ks_mean"])
        rc=rel_change(ratio96,ratio128); sc=abs(surv128-surv96); nc=abs(nn128-nn96)
        p96=finite_float(a["log"].get("core_spectrum_peak_k_mean")); p128=finite_float(b["log"].get("core_spectrum_peak_k_mean"))
        pc=rel_change(p96,p128) if np.isfinite(p96) and np.isfinite(p128) else float("nan")
        if np.isfinite(pc): grid_peak_changes.append(pc)
        plateau.append({"regulator":g,"birth_ratio_N96":ratio96,"birth_ratio_N128":ratio128,"birth_ratio_relative_grid_change":rc,"birth_ratio_pass":rc<.10,"survival_contrast_N96":surv96,"survival_contrast_N128":surv128,"survival_contrast_absolute_grid_change":sc,"survival_pass":sc<.05,"nn_contrast_N96":nn96,"nn_contrast_N128":nn128,"nn_contrast_absolute_grid_change":nc,"nn_pass":nc<.05,"core_peak_N96":p96 if np.isfinite(p96) else None,"core_peak_N128":p128 if np.isfinite(p128) else None,"core_peak_relative_grid_change":pc if np.isfinite(pc) else None,"core_peak_grid_pass":bool(np.isfinite(pc) and pc<.15)})

    predicted=[]; measured=[]; resolved=[]
    for g in GS:
        row=log_row(log_shard(log,128,g))["log"]
        sites=finite_float(row.get("predicted_wavelength_sites_mean")); pk=finite_float(row.get("predicted_k_star_mean")); mk=finite_float(row.get("core_spectrum_peak_k_mean"))
        if sites>=8 and np.isfinite(pk) and np.isfinite(mk): resolved.append(g); predicted.append(pk); measured.append(mk)
    rho=rank_corr(predicted,measured)
    spectrum_trend=bool(len(resolved)>=3 and np.isfinite(rho) and rho>=.6)
    spectrum_grid=bool(len(grid_peak_changes)>=3 and float(np.median(grid_peak_changes))<.15)
    long_plateau=all(x["birth_ratio_pass"] and x["survival_pass"] and x["nn_pass"] for x in plateau)
    log_charge=all(log_shard(log,n,g)["diagnostics"]["charge_pass"] for n in (64,96,128) for g in GS)
    log_pass=bool(long_plateau and spectrum_trend and spectrum_grid and log_charge)

    return {
      "gate":"GATE2_CONVERGENCE_SHARD_COLLECTOR",
      "protocol":"GATE2_CONVERGENCE_PROTOCOL.md",
      "collector_note":"Tie-aware Spearman ranks were frozen before any campaign artifact was inspected; unresolved spectral values remain null rather than being coerced.",
      "receipt_counts":{"core_shards":len(core),"log_shards":len(log)},
      "primary":{"classification":"GATE2_PRIMARY_PASS" if primary_pass else "GATE2_PRIMARY_FAIL_OR_UNRESOLVED","pass":primary_pass,"gate3_numerically_allowed":primary_pass,"gate3_launch_blocked_until_separate_protocol":True,"within_cell_checks":within,"spatial_checks":spatial,"time_step_checks":timestep,"birth_slopes":slopes},
      "logarithmic":{"classification":"LOG_ARM_GATE2_PLATEAU_CANDIDATE" if log_pass else "LOG_ARM_REGULATOR_DEFINED_OR_UNRESOLVED","pass":log_pass,"plateau_checks":plateau,"resolved_g_for_spectrum":resolved,"predicted_k_star":predicted,"measured_core_peak_k":measured,"spearman_rank_correlation":rho if np.isfinite(rho) else None,"spectrum_trend_pass":spectrum_trend,"median_core_peak_grid_relative_change":float(np.median(grid_peak_changes)) if grid_peak_changes else None,"spectrum_grid_pass":spectrum_grid,"long_distance_grid_plateau_pass":long_plateau,"raw_charge_pass":log_charge}
    }


def main():
    p=argparse.ArgumentParser(); p.add_argument("--root",type=Path,required=True); p.add_argument("--output",type=Path,required=True); a=p.parse_args()
    out=collect(a.root); a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"primary":out["primary"]["classification"],"logarithmic":out["logarithmic"]["classification"]},indent=2))
if __name__=="__main__": main()
