# Gate 2 convergence result

Date: 2026-09-10

Status: **frozen negative result**.

This document records the outcome of the preregistered Gate-2 convergence/regulator campaign in `GATE2_CONVERGENCE_PROTOCOL.md`. The run used the frozen scientific configuration on GitHub Actions. The execution-only optimization retained the loosest preregistered `max_core_ratio=0.95` detections once and derived the stricter `0.85/0.75` analyses by filtering `core_ratio`; it did not change field evolution, seeds, observables, or pass/fail criteria.

The completed Actions run was `34381161350` at commit `87bcf89920ceb67a3bd152b29b450bcc02c9637c`. Its summary artifact was `gate2-fast-convergence-summary`, artifact id `10127410897`, digest:

```text
sha256:d15d3baa6c7717637695ac2e880f5afb1a9fbad343e14e7c933762e52fbb4218
```

## Primary canonical/square-root classification

```text
GATE2_PRIMARY_FAIL_OR_UNRESOLVED
```

Gate 3 is **not numerically allowed**.

The failure is not marginal. At fixed physical box size, changing from `N=96` to `N=128` changes mean resolved birth density by:

| arm | tau_Q=8 | tau_Q=16 | tau_Q=32 |
|---|---:|---:|---:|
| canonical | 27.97% | 38.06% | 23.41% |
| square-root | 38.69% | 41.20% | 38.61% |

The preregistered spatial tolerance was `<5%`.

The fitted birth-density slopes themselves were more stable under spatial refinement: canonical changed by `0.0448` and square-root by `0.00064`, both inside the `<0.05` slope boundary. That does not rescue the gate because the underlying birth-density observable is strongly grid dependent.

Detector robustness also fails in every numerical cell. The maximum relative change in birth density when the resolved-core threshold is moved from the baseline `0.85` to `0.75/0.95` is:

```text
N64  dt=.02 :  9.87%
N96  dt=.02 : 12.97%
N128 dt=.02 : 14.15%
N96  dt=.01 : 10.74%
N128 dt=.01 : 11.49%
```

The frozen detector criterion was `<5%`.

Cadence sensitivity also exceeds the frozen absolute `0.05` survival boundary in all five cells (the `N128, dt=.01` value is `0.05020`, just over the line). By contrast the worldline-speed attack passes everywhere, with maximum absolute survival changes between `0.0175` and `0.0413`, and every sampled periodic frame has exactly zero raw net plaquette charge.

Time-step convergence is mixed rather than clean. Examples that fail the `<5%` birth-density rule include canonical `N96, tau=16` (`6.25%`), canonical `N96, tau=32` (`12.31%`), square-root `N96, tau=8` (`11.10%`), canonical `N128, tau=16` (`9.03%`), and square-root `N128, tau=16` (`6.57%`). Canonical slope changes also miss the `<0.05` boundary at both `N=96` (`0.0653`) and `N=128` (`0.0523`).

Therefore the large Gate-3 birth-law campaign is blocked by the preregistration. No exponent/universality claim is allowed from these data.

## Logarithmic-arm regulator classification

```text
LOG_ARM_REGULATOR_DEFINED_OR_UNRESOLVED
```

The logarithmic arm fails the regulator/grid plateau test.

Between `N=96` and `N=128`, the paired log/canonical birth-ratio changes are:

| g | relative grid change | pass `<10%`? |
|---:|---:|---|
| 0.020 | 18.41% | no |
| 0.030 | 13.03% | no |
| 0.045 | 29.78% | no |
| 0.0675 | 20.79% | no |
| 0.100 | 2.98% | yes |

The nearest-neighbor Poisson-KS contrast also fails its `<0.05` grid-stability rule at `g=.020` (`0.0705`) and `g=.030` (`0.0782`). The survival contrast itself is grid-stable at all five regulator values, but the preregistration required the full long-distance plateau, not one favorable observable.

The core-conditioned spectrum gives an especially direct negative result. On the finest grid, the locally predicted fastest scale moves substantially:

```text
g=.030   predicted k_star = 1.3593
g=.045   predicted k_star = 1.1115
g=.0675  predicted k_star = 0.8987
g=.100   predicted k_star = 0.7145
```

while the measured core-spectrum peak is the same discrete Fourier bin in every resolved case:

```text
measured peak k = 0.5235987756
```

Thus the measured core modulation does not track the local prediction. The first collector printed a Spearman value of `-1.0`; a later pre-result audit found that its double-`argsort` implementation mishandled tied ranks. With proper tied ranks, a constant measured series has undefined Spearman correlation. This correction does not change the preregistered decision: an invariant measured peak cannot satisfy the required positive `rho >= 0.6` tracking criterion.

## Claim boundary

Gate 2 therefore kills the result we hoped to test at Gate 3 in the present discretization/detector:

> We do **not** have evidence that the birth law has reached a regulator- and grid-independent continuum regime while nonlinear core physics changes only survival/spatial organization.

What remains supported is narrower:

- the action/force implementation passed Gate 0;
- exploratory strong-core runs produced large noncanonical birth-count differences;
- those differences are not numerically converged under the current resolved-core observable;
- the logarithmic branch is regulator-defined or unresolved by the declared tests;
- Gate 3 must not be launched without a new, separately justified numerical formulation or observable and a new preregistration.

This is a useful falsification result, not a failed software run.
