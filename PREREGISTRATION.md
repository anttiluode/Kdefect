# Preregistration: birth versus survival of noncanonical U(1) defects

Version 0.1 — frozen before the first Kdefect pilot.

## Research question

For gradient laws with an identical quadratic small-field limit, does a finite-rate U(1)
transition generate the same Kibble–Zurek birth scaling while nonlinear core physics
changes subsequent survival and spatial organization?

The canonical and square-root arms are the clean primary comparison. The logarithmic
arm is an exploratory, regulator-dependent pattern-forming arm until it passes its own
continuum/scaling tests.

## Model and fixed hypotheses

The free energy and dynamics are those printed in the README and implemented in
`src/kdefect/lattice.py`. In all arms, `F'(0)=1`; with mean-field `nu=1/2` and Model-A
`z=2`, the reference prediction for point-defect density in two dimensions is

```math
\rho_{\rm birth}\propto \tau_Q^{-D\nu/(1+z\nu)}=\tau_Q^{-1/2}.
```

The hypotheses are:

- **H1 — shared birth law:** after finite-size, time-step, detector, and core-size
  convergence, canonical and square-root birth exponents differ by less than `0.10`.
- **H2 — separated survival:** square-root screening lowers core flux and therefore
  produces at least a `10%` relative change in worldline survival at one or more fixed
  scaled lags, with a paired 95% bootstrap interval excluding zero.
- **H3 — logarithmic finite-k scale:** where its longitudinal stiffness is negative,
  local linearization predicts a fastest regularized mode
  `k_star = sqrt(|lambda_parallel|/(2g))`. A logarithmic-arm interpretation requires
  the measured spectral peak to follow this local prediction where applicable and all
  claimed long-distance observables to approach a grid-independent plateau.
- **H4 — stochastic geometry:** at resolved birth, the canonical arm approaches a 2D
  Poisson point process after conditioning on density. Noncanonical departure is only
  accepted from run-level bootstrap intervals across several system sizes—not from a
  pooled-vortex p-value.

H2 is a directional mechanistic prediction, not a guaranteed discovery. Failure is a
valid and informative result.

## Gates

### Gate 0 — variational identity

- Central directional derivative of discrete energy agrees with minus the force inner
  product to relative error `< 1e-6` for every arm.
- A sufficiently small deterministic Euler step lowers the energy for every arm.
- Analytic stiffness signs and the logarithmic crossing at `kappa Y=1` are reproduced.

Failure stops all physics interpretation.

### Gate 1 — measurement pilot

- Synthetic periodic fields recover all known vortices with net charge zero and
  sub-grid position error `< 0.1 dx`.
- Plane waves produce no plaquette vortices.
- Same-charge worldlines link correctly across a periodic boundary and stop on a
  missing detection.
- A small stochastic run must produce machine-readable records for raw winding,
  resolved cores, negative-stiffness fraction, birth time, count decay, worldline
  survival, nearest-neighbor statistic, and low-`k` charge form factor.

The default Gate 1 run is explicitly underpowered and cannot confirm H1–H4.

### Gate 2 — numerical convergence

Run at a minimum of three spatial resolutions and two smaller time steps while holding
physical box size and dimensionless physical parameters fixed. Required plateaus:

- canonical/square-root birth density changes `< 5%` between the two finest grids;
- their fitted birth exponents change `< 0.05`;
- core acceptance changes `< 5%` under each preregistered detector threshold
  (`max_core_ratio = 0.75, 0.85, 0.95`);
- survival estimates are stable under halving sample cadence and varying the speed
  cutoff by `±25%`;
- total topological charge is zero to lattice precision in every periodic frame.

For the logarithmic arm, scan at least five `g` values and resolve the predicted
finite-wavelength peak by at least eight sites. No `g -> 0` limit is presumed to exist.

### Gate 3 — scaling campaign

Primary target:

- square boxes `L >= 256 xi_0` and one doubled-size check;
- at least eight quench times spanning the empirically identified slow-quench window;
- `200` independent noise realizations per arm/time, increased to `400` near ambiguous
  interval boundaries;
- common random-number seeds across arms for paired contrasts;
- fast-quench points excluded only by the growth-curve collapse criterion, fixed before
  fitting defect exponents.

The equilibration time is extracted from the condensate-norm crossover between early
exponential and later adiabatic growth. The first maximum of resolved visible cores is
an independent formation marker. Both are reported; neither may be silently replaced by
a hand-selected observation time.

## Primary observables

1. Resolved birth density and its log-log exponent versus `tau_Q`.
2. Full counting cumulants `kappa_1` through `kappa_4` across realizations.
3. Charge-preserving worldline survival at lags `0.5, 1, 2, 4` times the measured
   equilibration time.
4. Vortex/antivortex pair correlation, nearest-neighbor spacing, Voronoi cell-area
   statistics, and density/charge spatial form factors.
5. Core radius, mobility, annihilation time, negative-stiffness volume fraction, and
   energy decomposition.

Inference resamples whole realizations within each quench-time cell. Vortices inside one
field are spatially correlated and are never treated as independent samples.

## Fixed exclusions

- numerical blow-up or non-finite energy;
- a failed Gate 1 core validation;
- a run whose periodic net charge is nonzero after resolved detection is retained but
  flagged; it is not silently deleted;
- points outside the predeclared scaling window remain plotted and are marked excluded;
- no seed is dropped because its count looks anomalous.

## Kill conditions

The core hypothesis is rejected or sharply downgraded if any of the following occurs:

1. After rescaling length by measured core radius and time by isolated-pair annihilation
   time, canonical and square-root survival curves *and* spatial statistics collapse.
   Conclusion: screening renormalizes microscopic units but does not create a distinct
   dynamical class.
2. Logarithmic-arm observables follow `g`, `dx`, or the Nyquist scale without a
   long-distance plateau. Conclusion: regulator-defined Swift–Hohenberg-like pattern
   dynamics, not regulator-independent k-defect physics.
3. Birth-exponent contrasts change materially with core threshold, sample cadence, or
   the equilibration-time estimator. Conclusion: measurement artifact.
4. No common finite-size-safe Kibble–Zurek window exists across the primary arms.
   Conclusion: H1 is untestable with this model/parameter range.
5. H2's paired effect is below `10%` with intervals narrow enough to exclude that
   threshold. Conclusion: no practically meaningful survival separation.

## Claim discipline

Passing Gate 3 would establish a result for this stochastic dissipative field model.
It would not by itself establish a relativistic k-essence result, an early-universe
prediction, a quantum-gravity mechanism, or experimental universality. Those require a
separate mapping and new tests.
