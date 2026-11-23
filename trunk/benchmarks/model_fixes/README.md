# Corrected multilinear EMT benchmarks

> **Periodicity correction (23 November 2026):** The original closure diagnostic missed
> the multilinear `theta_generator_*` states and counted their normal 2π advance as
> drift. Corrected original closure: IEEE39 0.001435, IEEE118 0.041858. A further
> off-nominal transformer tap scaling bug was found and corrected. Earlier IEEE118
> modes are superseded. See [periodicity validation](../periodicity_validation/README.md) for current checks.


See [RESULTS.md](../RESULTS.md#corrected-multilinear-models-and-completed-ieee118-ssa)
for the generator/transformer root causes, sparse consistency reduction, current
full-analysis timings, and the upstream startup port. `summary.json` is compact.

Both corrected grids return six modes. IEEE118 total time is about 147–150 s;
3.78 s refers only to its consistency reduction. The periodic-orbit check is
not satisfied, so these modes are not a validated stability conclusion.
`ieee118` is an interrupted diagnostic using the original dense reduction.
`ieee118_sparse` and `ieee118_arnoldi` are completed analyses. The upstream
snapshot run was interrupted during its native SSA, after a 0.838 s build.
