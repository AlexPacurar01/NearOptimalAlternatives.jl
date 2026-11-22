# Paper case study: near-optimal alternatives for Norse

Reproducible script and data behind Section 3.3 of the NearOptimalAlternatives.jl
(NOA) SoftwareX manuscript: three SPORES alternatives for the Norse example
system of TulipaEnergyModel.jl within a 20% cost slack, and how they differ
from the cost-optimal design and from each other.

## Contents

| Path | What it is |
| :--- | :--- |
| `norse_mga.jl` | The experiment. One NOA call, then tables and a figure. |
| `FINDINGS.md` | What the runs show, problems found on the way, and how they were fixed. |
| `results/norse_hsj/` | The same run with HSJ, used for the comparison in the paper. |
| `data/Norse/` | Norse inputs from TulipaEnergyModel.jl v0.22.2 (`test/inputs/Norse`), with investment limits added (see below). |
| `results/norse_spores/` | Tables and data of the run reported in the paper (see below). |
| `figures/` | Figures for the manuscript: `norse_spores.png`/`.pdf` (the paper's figure) and `norse_hsj.png`/`.pdf`. |
| `Project.toml` / `Manifest.toml` | The exact Julia environment used. |

The results folder holds:

| File | Content |
| :--- | :--- |
| `alternatives.csv` | Installed capacity (GW) per technology for the optimum and each alternative, plus costs. |
| `change_vs_optimum.csv` | Each alternative minus the optimum, per technology (GW). |
| `distances.csv` | Euclidean distances (GW) between the optimum and the alternatives, all pairs. |
| `summary.csv` | Method, settings, run times, package versions. |
| `table.tex`, `distances.tex` | The same as LaTeX tables. |

## Data change

SPORES scores each decision by how close it is to its upper bound, so every
decision needs a finite bound. Several investable assets and transport lines in
the Norse case shipped with TulipaEnergyModel.jl have no `investment_limit`. In
`data/Norse/asset-commission.csv` and `data/Norse/flow-commission.csv` each of
them gets one of ten times its cost-optimal investment. The two CCGTs get
412.5 GW and the electrolyser 100 GW, since the CCGT in Midgard and the
electrolyser are not built at the optimum. Wind, solar, battery, pumped hydro
and the gas turbine keep Tulipa's original limits. Nothing else is changed;
`diff -r` against the Tulipa copy shows exactly these cells.

## Units: why the script works in GW

Tulipa's investment variables count units, and unit sizes in Norse range from
3 MW (a wind turbine) to 1000 MW (a transmission line). Methods that weight the
decision variables, such as SPORES and HSJ, depend on that choice. On unit
counts, SPORES penalises a GW of wind about 90 times harder than a GW of nuclear,
and wind makes up over 80% of its first objective, so every alternative simply
drops wind and the alternatives end up within a few GW of each other. The script
therefore adds one bounded variable per decision holding its capacity in GW and
hands those to NOA:

```julia
@variable(model, 0 <= cap_gw[i = 1:n] <= upper_bound(units[i]) * gw[i])
@constraint(model, [i = 1:n], cap_gw[i] == gw[i] * units[i])
```

Set `NOA_SPACE=units` to reproduce the unit-count behaviour.

## Running it

From the repository root:

```sh
julia --project=paper -e 'using Pkg; Pkg.instantiate()'
julia --project=paper paper/norse_mga.jl
```

The base solve takes about 30 seconds and the three alternatives under 10
seconds with HiGHS on a laptop.

Other knobs (environment variables): `NOA_METHOD` (default `Spores`; also `HSJ`,
`Min_Max_Variables`, `Random_Vector`, `Directionally_Weighted_Variables`,
`Max_Distance`), `NOA_N_ALT` (default 3), `NOA_EPS` (default 0.2), `NOA_SPACE`
(`gw` or `units`), `NOA_NORSE` (input folder), `NOA_OUT` (output folder),
`NOA_TIME` (per-solve time limit in seconds).

`Max_Distance` maximises a convex quadratic, the squared distance to earlier
solutions, which makes it a nonconvex QP. HiGHS rejects it and Ipopt does not
converge on it at this size. Run it with `NOA_SOLVER=scip` or
`NOA_SOLVER=gurobi` after adding that solver package to the environment. All
other methods are linear and run with HiGHS.
