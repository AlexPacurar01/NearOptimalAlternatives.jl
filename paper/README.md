# Paper case study: Norse with Max-Distance

Reproducible script and data behind Section 3.3 of the NearOptimalAlternatives.jl
(NOA) SoftwareX manuscript: four Max-Distance alternatives on the Norse example
system of TulipaEnergyModel.jl, and the technology trade-offs they show.

## Contents

| Path | What it is |
| :--- | :--- |
| `norse_max_distance.jl` | The experiment. One NOA call, then tables and a figure. |
| `data/Norse/` | Norse inputs from TulipaEnergyModel.jl v0.22.2 (`test/inputs/Norse`), with investment limits added (see below). |
| `results/norse_max_distance/` | Output of the run reported in the paper: `alternatives.csv`, `distances.csv`, `summary.csv`, `table.tex`, `bandwidth.png`. |
| `Project.toml` / `Manifest.toml` | The exact Julia environment used. |

## Data change

The Norse case shipped with TulipaEnergyModel.jl has no `investment_limit` on
most investable assets. Max-Distance then spends the whole cost budget on the
one technology whose capacity is unbounded (hydrogen storage, which has a
near-zero investment cost), which says nothing about the system. In
`data/Norse/asset-commission.csv` and `data/Norse/flow-commission.csv` every
investable asset and transport flow therefore gets an `investment_limit` of
ten times its cost-optimal investment. Nothing else is changed; `diff -r`
against the Tulipa copy shows exactly these cells.

## Running it

From the repository root:

```sh
julia --project=paper -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=paper paper/norse_max_distance.jl
```

Max-Distance maximises a convex quadratic (the squared distance to the previous
solutions) over the near-optimal polytope. That is a nonconvex QP, so it needs a
solver with global nonconvex support. The script uses the open-source SCIP by
default; set `NOA_SOLVER=gurobi` to use Gurobi (NonConvex=2) instead. HiGHS
rejects the problem ("Cannot solve non-convex QP problems with HiGHS") and
Ipopt, a local NLP solver, does not converge on it at this size. All the other
NOA methods (HSJ, SPORES, Min/Max, Random Vector, DWV) are linear and run with
HiGHS.

Other knobs (environment variables): `NOA_N_ALT` (default 4), `NOA_EPS`
(default 0.1), `NOA_NORSE` (input folder), `NOA_OUT` (output folder),
`NOA_TIME` (per-solve time limit in seconds, default 1800).
