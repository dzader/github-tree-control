# Causal Loom v0.25 — pre-holdout development rationale

## Status

The scientific protocol and executable were frozen before B0007 or B0018 payload access. Development used only B0005 and B0006 from the immutable nested NASA archive `BatteryAgingARC-FY08Q4.zip`.

## Architectural change forced by v0.24

The v0.24 one-dimensional charge-coordinate candidate failed its development gates. Inspection of the already-authorized B0005/B0006 inputs showed that a NASA `discharge` record contains two causally distinct regimes:

1. active current-driven discharge; and
2. current-off electrothermal relaxation, during which charge is nearly constant while voltage rebounds and temperature cools.

A single cumulative-charge coordinate is therefore non-injective with respect to the full observed response. v0.25 replaces it with an input-defined hybrid atlas whose event boundary is the final sample with `abs(Current_measured) > 0.2 A`.

## Frozen representation

The stitched coordinate is

```text
z = q/q_off                                         during active discharge
z = 1 + (t-t_off)/(t_end-t_off)                    during relaxation
```

Training cycles are chord-residualized, centered, RMS-normalized, and sign-aligned separately for voltage and temperature. The base mode `h0` is the pointwise median. The tangent mode `h1` is the first orthogonal residual SVD mode and is oriented by discharge-cycle order. A new cycle uses exactly five output-blind probes to fit output-specific affine/write coefficients and one chart coordinate `eta` shared across voltage and temperature.

An exact strong-local fallback is retained. A convex blend is selected only by leave-one-probe-out evidence, with ties toward the fallback.

## Predeclared development evidence

On leave-one-cell-out development across 60 target cycles from B0005/B0006:

- 60/60 valid cycles;
- zero catastrophic failures;
- safe-atlas/local geometric-mean ratio 0.448156;
- safe-atlas/local median ratio 0.441968;
- safe-atlas/local win fraction 0.883333;
- tail geometric-mean ratio 0.496079;
- tail win fraction 0.883333;
- relaxation geometric-mean ratio 0.456498;
- relaxation win fraction 0.836364;
- raw atlas/generic-hybrid-cubic geometric-mean ratio 0.175812;
- raw atlas/generic-hybrid-cubic win fraction 1.0;
- cross-cell projective correlations: voltage h0 0.994843, voltage h1 0.977026, temperature h0 0.996029, temperature h1 0.965779;
- shared `eta` versus discharged capacity Pearson correlation -0.832357.

These values passed every frozen development gate. They are development evidence only and do not authorize a scientific claim until the separate B0007/B0018 holdout job reproduces the candidate lock and clears the frozen primary gates.

## Evidence rule

If development reproduction changes or misses any gate, B0007/B0018 remain unopened. If the holdout runs and fails, the first result is permanent; no tuning or scientific rerun on those cells is allowed.
