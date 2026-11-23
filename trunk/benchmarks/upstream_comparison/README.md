# Initialization optimization and upstream comparison

See [RESULTS.md](../RESULTS.md#initialization-acceleration-and-pinned-upstream-trial)
for timings, methodology, and API differences. `summary.json` is compact.
`downstream_optimized` contains the successful IEEE39 trial and exact saved-output
agreement check. `downstream_optimized_ieee118` records fast construction followed
by capture failure. `upstream_native_compatible` records the actual 180-second
upstream construction timeout. `upstream` and `upstream_native` are preliminary
API-compatibility failures, excluded from timing comparisons.

The saved `.veragrid` file holds the identical network/model input used for the
upstream native and optimized IEEE39 trials. The `.prof` file is a profiled old
run, not an ordinary wall-time baseline; `profile_summary.txt` is readable.

The upstream snapshot has subsequently been patched and rerun successfully.
`upstream_initialization_speed.patch` contains the portable source changes and
regression tests; `upstream_optimized` contains the completed 39.78-second native
upstream trial. Both generated-Python options are required to reproduce that
short-study result. See the final upstream-patch section in `RESULTS.md`.
