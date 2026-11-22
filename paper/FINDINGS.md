# Findings from the Norse case study

Notes from preparing Section 3.3 of the NearOptimalAlternatives.jl (NOA)
SoftwareX manuscript, November 2026. They record what the Norse runs show, what
went wrong along the way, and what was fixed, so that co-authors can check the
reasoning behind the published numbers.

## Summary

- SPORES on Norse, with a 20% cost slack and three alternatives, gives three
  clearly different designs once the decisions are measured in GW.
- Run on Tulipa's unit counts instead, SPORES returned near-copies. The cause is
  the units of the variables, not the method.
- NOA reported alternative costs without the objective's constant term. This is
  fixed in the package and covered by a test.
- Max-Distance cannot be run with HiGHS or Ipopt on Norse, because it is a
  nonconvex problem.

## Setup

| Item | Value |
| :--- | :--- |
| Model | Norse, TulipaEnergyModel.jl v0.22.2, LP relaxation |
| Size | 7,180 variables, 11,563 constraints, 16 investment decisions |
| Method | SPORES, `generate_alternatives_optimization!` |
| Cost slack | 20% |
| Alternatives | 3 |
| Solver | HiGHS 1.27 (pinned in `paper/Manifest.toml`) |
| Run time | about 40 s base solve, about 11 s for the three alternatives |

Script: `paper/norse_mga.jl`. Outputs: `paper/results/norse_spores/`.

## Results

All three alternatives cost exactly 1.20 times the optimum. Installed capacity
in GW, with the change against the optimum in brackets:

| Technology | Optimum | Alt. 1 | Alt. 2 | Alt. 3 |
| :--- | ---: | ---: | ---: | ---: |
| Asgard solar | 25.3 | 0.6 (−24.7) | 2.3 (−23.0) | 2.6 (−22.7) |
| Valhalla H2 storage | 27.3 | 9.7 (−17.5) | 33.5 (+6.3) | 22.4 (−4.9) |
| Valhalla H2 generator | 15.9 | 0.0 (−15.9) | 7.0 (−8.9) | 0.0 (−15.9) |
| Midgard nuclear SMR | 15.4 | 30.1 (+14.7) | 23.3 (+7.8) | 25.8 (+10.4) |
| Midgard wind | 14.7 | 12.8 (−1.9) | 0.5 (−14.2) | 6.1 (−8.6) |
| Line Asgard–Midgard | 8.2 | 20.1 (+11.9) | 8.3 (+0.1) | 11.5 (+3.4) |
| Valhalla fuel cell | 13.3 | 4.5 (−8.8) | 16.4 (+3.1) | 10.8 (−2.4) |
| Asgard battery | 6.6 | 0.0 (−6.6) | 1.5 (−5.1) | 1.6 (−5.0) |
| Valhalla electrolyser | 0.0 | 1.2 (+1.2) | 6.4 (+6.4) | 6.0 (+6.0) |
| Line Asgard–Valhalla | 8.5 | 2.9 (−5.6) | 13.7 (+5.2) | 10.2 (+1.7) |
| Asgard CCGT | 41.2 | 41.2 | 41.2 | 41.2 |

Three rows with changes under 3 GW are omitted here; the full table is in
`paper/results/norse_spores/table.tex`. Midgard CCGT and the Valhalla gas
turbine are not built in any design.

Euclidean distance between designs, in GW over all 16 decisions:

| | Optimum | Alt. 1 | Alt. 2 | Alt. 3 |
| :--- | ---: | ---: | ---: | ---: |
| Optimum | 0 | 41.1 | 31.8 | 32.6 |
| Alt. 1 | | 0 | 35.3 | 20.5 |
| Alt. 2 | | | 0 | 16.2 |

**What all alternatives share.** Each alternative moves away from the optimum
in the same four ways:

- Solar falls from 25 GW to under 3 GW.
- Nuclear grows by 8 to 15 GW.
- An electrolyser is built, although the optimum builds none.
- The Asgard CCGT stays at 41 GW, so it looks like a must-have.

**How they differ.** The alternatives disagree on how to replace the solar:

- Alternative 1 keeps most of the wind and replaces the hydrogen chain with
  nuclear and a much stronger Asgard–Midgard line.
- Alternative 2 drops the wind and relies on hydrogen, with the most H2 storage
  and fuel-cell capacity and more transmission into Valhalla.
- Alternative 3 lies between the two.

The alternatives are about as far from each other as from the optimum, which is
what SPORES's accumulating weights are meant to achieve.

**HSJ for comparison.** The same call with `modeling_method = :HSJ` finds one
design 44 GW from the optimum and then stays within 1.7 GW of it. HSJ's weights
only mark which technologies were built in the previous alternative, and that
set barely changes. Outputs: `paper/results/norse_hsj/`.

## Problems found and how they were handled

### Variable units change what SPORES and HSJ find

Tulipa's investment variables count units, and unit sizes in Norse range from
3 MW for a wind turbine to 1000 MW for a transmission line. SPORES scores each
decision by its share of its upper bound, which is unit-free, but then
multiplies that score by the variable itself, which is not. On unit counts this
made a GW of wind about 90 times more expensive in the SPORES objective than a
GW of nuclear, and wind made up 81.5% of the first SPORES objective. Every
alternative removed all wind, and the four alternatives of the first draft were
only 2 to 6 GW apart.

Fix in the case study: add one bounded variable per decision holding its
capacity in GW and pass those to NOA. This adds 16 variables and 16 equality
constraints to the solved Tulipa model. The paper's Limitations section now
states that weighted methods are not invariant to variable units.

Open question for the package: SPORES could divide each term by the variable's
upper bound, which would make it unit-invariant without user action. That
changes the method's behaviour and existing tests, so it is left for discussion.

### Reported costs missed the objective's constant term

The Norse objective has a constant of 250,000. When NOA adds the budget
constraint, JuMP moves that constant into the constraint's right-hand side. NOA
then read each alternative's cost from the constraint's left-hand side, so
costs were reported 250,000 too low and appeared as 1.199 times the optimum
instead of 1.200. The budget itself was always enforced correctly.

Fix in the package: NOA now stores the original objective when it builds the
budget constraint and evaluates it for each alternative. This is applied in
`generate_alternatives_optimization!`, the sweep and the arclength generators. A
regression test with a constant term is in `test/test-results.jl`. The full
test suite passes on the `upstream-clean` branch: 231 tests.

### Max-Distance needs a nonconvex solver

Max-Distance maximises a convex quadratic, the squared distance to earlier
solutions, which is a nonconvex quadratic program. On Norse:

| Solver | Outcome |
| :--- | :--- |
| HiGHS | Refuses it: "Cannot solve non-convex QP problems". |
| Ipopt | Diverges and hits the 3,000-iteration limit, both from its default start and warm-started at the optimum. |
| SCIP | Solves it, but took over ten minutes per alternative with bounded data. |
| Gurobi | Supported with `NonConvex=2`; not tested because no licence was available. |

The paper therefore uses SPORES. The script accepts `NOA_SOLVER=scip` or
`NOA_SOLVER=gurobi` for Max-Distance, and the Limitations section mentions the
solver requirement.

### Investment limits had to be added

SPORES needs a finite upper bound on every decision. Several Norse assets and
lines have none, so each was given ten times its cost-optimal investment, with
412.5 GW for the two CCGTs and 100 GW for the electrolyser. The limits on wind,
solar, battery, pumped hydro and the gas turbine are Tulipa's own. See
`paper/README.md`.

### Results depend slightly on the HiGHS version

The SPORES problems have many equally good solutions. HiGHS 1.25 and 1.27 pick
slightly different ones, and capacities differ by up to 0.1 GW. The conclusions
do not change. The published numbers come from the environment pinned in
`paper/Manifest.toml`.

### Small examples in Sections 3.1 and 3.2

- The original two-variable HSJ example in Section 3.1 returned the optimum
  five times, because its near-optimal region has no vertex HSJ can move to. It
  was replaced by a three-variable model, where HSJ alternates between two
  vertices; the text explains why.
- In Section 3.2, the two SPORES directions on the three-variable model return
  identical points. The manuscript does not claim otherwise, but the example
  could be revisited.

## Reproducing

From the repository root:

```sh
julia --project=paper -e 'using Pkg; Pkg.instantiate()'
julia --project=paper paper/norse_mga.jl                     # SPORES, the paper's run
NOA_METHOD=HSJ julia --project=paper paper/norse_mga.jl      # HSJ comparison
NOA_SPACE=units julia --project=paper paper/norse_mga.jl     # unit-count behaviour
```
