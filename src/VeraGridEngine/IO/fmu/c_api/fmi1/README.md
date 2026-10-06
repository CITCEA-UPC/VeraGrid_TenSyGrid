# Official FMI 1.0.1 C headers

These files are unmodified copies from the Modelica Association Project FMI
release archives. They are vendored so VeraGrid can compile deterministic,
host-native FMI 1 Model Exchange and Stand-Alone Co-Simulation binaries without
network access during export.

Co-Simulation package:
`https://fmi-standard.org/assets/releases/FMI_for_CoSimulation_v1.0.1.zip`

- Package SHA-256: `81c87527febece29b5bb00bac89f9a58c5aed976aeb0afbfd357cf5aeebabd21`
- `co_simulation/fmiFunctions.h`: `847a34b9cc6b338751020ff8fc6f686b8c85ac0006da612a7b18ed30ff1a6a6b`
- `co_simulation/fmiPlatformTypes.h`: `d0239ce64251059500b8fb3f1bb55690576bf4b26e51bf2db15c8a713aece5b8`

Model Exchange package:
`https://fmi-standard.org/assets/releases/FMI_for_ModelExchange_v1.0.1.zip`

- Package SHA-256: `966c948707531c5a764b70fa7bbe23030da962d8a939e01dc8bc9af9d8264cd1`
- `model_exchange/fmiModelFunctions.h`: `8263da2171bca0485e500d506ea2179eb767b9541a8d0adae034362e2a4f0028`
- `model_exchange/fmiModelTypes.h`: `1b7382ab15a1ce3c9bd057c456ff887aaadd9c0325e4d9e910f43165bd6c4c2a`

The original BSD license text remains embedded in each header.
