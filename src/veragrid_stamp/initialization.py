"""Power-flow-consistent initialization of the polar STAMP dynamic network."""
import numpy as np
from scipy.optimize import root


def initialize_dynamic_network(problem):
    """Seed dynamic network states and solve algebraic balance at fixed states."""
    grid = problem.grid
    voltage = {bus.idtag: value for bus, value in zip(grid.buses, problem.power_flow_results.voltage)}
    state_indices = {str(var): problem.uid2idx_vars[var.uid] for var in problem.state_vars}
    x = problem.get_x0()

    for line in grid.lines:
        name = line.rms_model.name
        if not name.startswith('NET.') or f'{name}.iq' not in state_indices:
            continue
        # The series current excludes line charging, which is modeled by
        # separate bus capacitors in this network.
        current = (voltage[line.bus_from.idtag] - voltage[line.bus_to.idtag]) / complex(line.R, line.X)
        x[state_indices[f'{name}.iq']] = current.real
        x[state_indices[f'{name}.id']] = -current.imag

    for load in grid.loads:
        name = load.rms_model.name
        v = voltage[load.bus.idtag]
        if name.startswith('STAMP bus capacitor '):
            x[state_indices[f'{name}.vc_q']] = v.real
            x[state_indices[f'{name}.vc_d']] = -v.imag
        elif f'{name}.ilq' in state_indices:
            current = -1j * (load.Q / grid.Sbase) * v / abs(v)**2
            x[state_indices[f'{name}.ilq']] = current.real
            x[state_indices[f'{name}.ild']] = -current.imag

    n_states = len(problem.state_vars)
    dx = np.zeros(problem.get_diff_var_number())

    def residual(algebraic):
        return problem.rhs_algebraic(np.r_[x[:n_states], algebraic], dx)

    solution = root(residual, x[n_states:], tol=1e-10)
    x[n_states:] = solution.x
    error = max(np.max(abs(residual(solution.x))), np.max(abs(problem.rhs_state(x, dx))))
    if not np.isfinite(error) or error > 1e-6:
        raise RuntimeError(f'STAMP dynamic-network initialization failed. Residual {error:.3g}')
    for uid, index in problem.uid2idx_vars.items():
        problem.init_guess[uid] = float(x[index])
