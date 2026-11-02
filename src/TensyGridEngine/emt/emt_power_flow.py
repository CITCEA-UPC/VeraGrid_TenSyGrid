"""Power-flow operating point for EMT studies with optional three-phase data."""

from VeraGridEngine.Simulations.PowerFlow.power_flow_driver import PowerFlowDriver
from VeraGridEngine.Simulations.PowerFlow3ph.power_flow_driver_3ph import PowerFlowDriver3Ph


def run_emt_power_flows(grid, options):
    """Return converged balanced and, when available, three-phase PF results.

    Older saved EMT grids may omit usable generator sequence impedances. They
    can still initialize from a balanced operating point, while unrelated
    three-phase errors must remain visible to the caller.
    """
    balanced = PowerFlowDriver(grid=grid, options=options)
    balanced.run()
    if not bool(balanced.results.converged):
        raise RuntimeError("Balanced power flow failed before EMT analysis")

    three_phase_results = None
    try:
        three_phase = PowerFlowDriver3Ph(grid=grid, options=options)
        three_phase.run()
    except ValueError as exc:
        message = str(exc)
        if not (message.startswith("U1 power-flow Z0 must be finite and nonzero")
                or message.startswith("U1 power-flow Z2 must be finite and nonzero")):
            raise
        print(f"Three-phase power flow unavailable ({exc}); using balanced power flow for EMT initialization.")
    else:
        if not bool(three_phase.results.converged):
            raise RuntimeError("Three-phase power flow failed before EMT analysis")
        three_phase_results = three_phase.results
    return balanced.results, three_phase_results
