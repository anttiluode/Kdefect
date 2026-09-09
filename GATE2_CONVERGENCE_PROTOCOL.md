# Gate 2 convergence campaign — executable preregistration

Status: **frozen before the first run of this campaign**.

This document operationalizes Gate 2 of `PREREGISTRATION.md`. It does not change H1–H4. It fixes the numerical panel, seeds, and pass/fail summaries needed before the expensive Gate 3 scaling campaign is allowed to start.

## Why Gate 2 comes first

`PREREGISTRATION.md` requires numerical convergence before confirmatory scaling. `RESULTS.md` additionally called for `L=64,96,128`, a smaller time step, at least five regulator values, better cadence/lineage checks, and a core-conditioned spectrum. Gate 3 is therefore blocked until this campaign is classified.

## Common physical box and random numbers

All resolution comparisons use a periodic physical box of side

```text
L_box = 64
```

with

```text
dx = L_box / N
N in {64, 96, 128}.
```

The stochastic initial condition/noise stream is paired across arms and across numerical variants whenever a comparison permits it. New campaign seeds begin at `260909000`; no Gate-0/1/1b/2 pilot seed is reused.

The nonlinear strength is fixed at `kappa=32`, selected by the already-labelled exploratory strong-core scout **before** this convergence campaign. This is a parameter-selection step, not a confirmatory result. If Gate 3 is reached, its inference uses new seeds again.

## Primary canonical / square-root convergence panel

For each numerical cell:

```text
arms       = canonical, sqrt
tau_Q      = 8, 16, 32
seeds      = 8 per arm/tau cell
regulator  = 0.03
temperature= 0.006
```

Spatial panel:

```text
N=64,  96, 128 at dt=0.02
```

Time-step panel on the two finer grids:

```text
N=96, 128 at dt=0.01
```

Every simulation is sampled internally at `0.2` time units so that the declared `0.4` cadence and a halved-cadence `0.8` attacker can be derived from the same stochastic path.

### Detector attack

The exact same stored field trajectory is re-read with

```text
max_core_ratio = 0.75, 0.85, 0.95.
```

The Gate-2 preregistration requires `<5%` core-acceptance sensitivity. We score the maximum relative change in mean birth density from the `0.85` analysis. A cell passes this item only if it is `<0.05`.

### Cadence and worldline-speed attack

The baseline worldline analysis uses

```text
cadence = 0.4
max_speed = 5.0
survival_lag = 8.0.
```

Attackers are

```text
cadence = 0.8
max_speed = 3.75 and 6.25.
```

`PREREGISTRATION.md` says survival must be stable but did not assign a numeric word to “stable.” Before seeing this campaign, we operationalize it as a maximum **absolute** change of `0.05` in cohort-survival fraction. The raw differences are always retained even if this convention later proves too strict or too loose.

### Charge check

Every sampled periodic frame must have raw plaquette net charge exactly zero. A nonzero frame is retained and counted; it fails the Gate-2 charge item.

### Spatial/time convergence classification

The final collector compares independently run cells.

- Between the two finest spatial grids (`N=96` and `N=128`, both `dt=0.02`), canonical and square-root mean birth densities must change by `<5%` at each `tau_Q`.
- Their fitted log-log birth exponents must change by `<0.05`.
- Between `dt=0.02` and `0.01` on `N=96` and `N=128`, the fitted exponents must change by `<0.05`; mean birth density must change by `<5%` at each `tau_Q`.

These are direct executable versions of the numeric boundaries already stated in Gate 2.

## Logarithmic regulator panel

The log arm remains exploratory and can be **killed independently** of the canonical/square-root program.

Use

```text
N          = 64, 96, 128
L_box      = 64
dt         = 0.01
tau_Q      = 16
seeds      = 6
kappa      = 32
g          = 0.020, 0.030, 0.045, 0.0675, 0.100
arms       = canonical, log
```

The five `g` values were fixed before this run. The finest-grid core-spectrum diagnostic reports the number of sites per predicted wavelength. A point with fewer than eight sites per predicted wavelength is marked unresolved and cannot support H3.

### Core-conditioned gradient spectrum

At the resolved birth frame, each detected core supplies a periodic square patch of physical width approximately `16`. For each patch:

1. compute `Y=|grad psi|^2`;
2. subtract the patch mean;
3. apply a separable Hann window;
4. compute the 2-D power spectrum;
5. radially bin physical wave number `k`;
6. ignore the DC bin and average the radial spectra over cores.

The resulting peak is compared with

```math
k_star = sqrt(median(-lambda_parallel)/(2 g))
```

on negative-stiffness sites at birth. This diagnostic is specifically intended to avoid the low-k vortex-spacing dominance seen in the earlier full-field density spectrum.

### Regulator/continuum kill rule

The logarithmic arm is **not allowed forward** merely because it has more defects. On the two finest grids (`N=96,128`):

- for each `g`, the paired log/canonical birth-count ratio must change by `<10%`;
- the log-minus-canonical gap-tolerant survival difference must change by `<0.05` absolute;
- the log-minus-canonical nearest-neighbor Poisson-KS difference must change by `<0.05` absolute;
- at least three resolved `g` points on the finest grid must have a core-spectrum peak; across those points, Spearman rank correlation between measured peak and local `k_star` must be positive and at least `0.6`;
- to reject a grid/Nyquist-locked spectral artifact, the median relative change in the **physical** core-spectrum peak between `N=96` and `N=128` must be `<15%` across at least three finite `g` comparisons.

If the long-distance contrasts fail the grid plateau, or the spectral peak remains tied to grid/Nyquist resolution rather than moving with the local prediction, the classification is

```text
LOG_ARM_REGULATOR_DEFINED_OR_UNRESOLVED
```

and H3 is sharply downgraded per the original kill condition.

A pass is only permission to do the larger log-arm study; it is not evidence for a new universality class.

## Gate-3 launch condition

The expensive Gate-3 scaling campaign is allowed only if:

1. the canonical/square-root Gate-2 spatial, time-step, detector, cadence, speed and charge checks pass; and
2. a separate Gate-3 protocol freezes either the finite-temperature/BKT calibration or the already-permitted zero-temperature noise-seeded Landau-instability protocol before its data are generated.

The log arm is optional for Gate 3 and may be killed while the primary canonical/square-root question survives.
