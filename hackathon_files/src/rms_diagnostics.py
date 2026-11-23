"""Read-only operating-point diagnostics for RMS tuning participants."""
from __future__ import annotations
from typing import TYPE_CHECKING
import math
import numpy as np
from scipy.linalg import eig
from VeraGridEngine.Devices.multi_circuit import MultiCircuit

from VeraGridEngine.Simulations.PowerFlow.power_flow_driver import PowerFlowDriver
from VeraGridEngine.Simulations.SmallSignalStabilityRms.small_signal_driver import SmallSignalStabilityRmsDriver, compute_state_matrix, compute_participation_factors, compute_participation_factors_generalized
from VeraGridEngine.Simulations.SmallSignalStabilityRms.small_signal_options import RmsSmallSignalStabilityOptions
if TYPE_CHECKING:
    from .rms_tuning import TunableGrid, TunableParameter

def get_modes_from_unstable_eigenvalues(analysis, *, real_tolerance: float=1e-08, top_n: int | None=5, include_conjugates: bool=False) -> tuple:
    """Return growing modes, worst real part first; omit conjugate duplicates.

        top_n limits participants per mode, not the number of unstable modes.
        Set it to None for all variables. Near-zero/neutral modes are excluded
        by real_tolerance; stable grids return an empty tuple.
        """
    if not math.isfinite(real_tolerance) or real_tolerance < 0:
        raise ValueError('real_tolerance must be finite and nonnegative')
    if top_n is not None and (isinstance(top_n, bool) or not isinstance(top_n, int) or top_n <= 0):
        raise ValueError('top_n must be a positive integer or None')
    indices = [i for i, value in enumerate(analysis['eigenvalues']) if np.isfinite(value) and value.real > real_tolerance]
    indices.sort(key=lambda i: (-analysis['eigenvalues'][i].real, -analysis['eigenvalues'][i].imag, i))
    modes = []
    for i in indices:
        value = complex(analysis['eigenvalues'][i])
        if not include_conjugates and value.imag != 0 and any((np.isclose(value, mode['eigenvalue'].conjugate(), rtol=1e-07, atol=1e-10) for mode in modes if mode['eigenvalue'].imag != 0)):
            continue
        factors = analysis['participation_factors'][:, i]
        order = np.argsort(-factors, kind='stable')[:top_n]
        modes.append(dict(index=i, eigenvalue=value, frequency_hz=abs(value.imag) / (2 * np.pi), damping_ratio=-value.real / abs(value), participants=tuple((dict(variable=analysis['variables'][j], participation=float(factors[j])) for j in order))))
    return tuple(modes)

def _readonly(array):
    result = np.array(array.toarray() if hasattr(array, 'toarray') else array, copy=True)
    result.flags.writeable = False
    return result

def analyze_grid(source: MultiCircuit | TunableGrid | dict) -> dict:
    """Analyze a grid copy or the initialized problem supplied to tune().

    No tuning proposal or transient simulation is run and no test budget is
    consumed. Pass a returned snapshot to reuse an existing eigensolve.
    """
    from .rms_tuning import TunableGrid
    if isinstance(source, dict):
        return source
    if isinstance(source, MultiCircuit):
        grid = source.copy()
        pf = PowerFlowDriver(grid)
        pf.run()
        if not bool(pf.results.converged):
            raise RuntimeError('Power flow did not converge')
        problem = SmallSignalStabilityRmsDriver(grid=grid, pf_results=pf.results, sss_options=RmsSmallSignalStabilityOptions(k=0, ss_assessment_time=0)).problem
        tunables = ()
    elif isinstance(source, TunableGrid):
        problem = source.problem
        grid = problem.grid
        tunables = source.tunable_parameters
    else:
        raise TypeError('Expected MultiCircuit, TunableGrid, or an analysis dictionary')
    x = problem.get_x0().copy()
    dx = np.zeros_like(x)
    variables = list(problem.state_vars)
    mass = None
    if problem.get_diff_var_number():
        variables += list(problem.algebraic_vars)
        mass = _readonly(-problem.get_E_matrix(x, dx))
        matrix = _readonly(problem.get_static_state_matrix(x, dx))
    else:
        _, matrix = compute_state_matrix(problem, x, dx)
    eigenvalues, left, right = eig(matrix, b=mass, left=True, right=True)
    left = left.conj().astype(complex)
    right = right.astype(complex)
    if mass is None:
        participation = compute_participation_factors(right, left)
    else:
        participation = compute_participation_factors_generalized(right, left, mass)
    if matrix.shape != (len(variables), len(variables)) or participation.shape != (len(variables), len(eigenvalues)):
        raise RuntimeError('Native SSA matrix/participation dimensions do not match variable ordering')
    if not np.isfinite(matrix).all() or not np.isfinite(participation).all() or (not np.any(np.isfinite(eigenvalues))):
        raise RuntimeError('Small-signal analysis returned invalid matrices or no finite eigenvalues')
    owners = {}
    for device, device_vars in problem.get_device_vars_dict().items():
        for var in device_vars:
            owners[var.uid] = device
    blocks = {}
    for device in grid.get_all_elements_iter():
        root = getattr(device, 'rms_model', None)
        if root is None:
            continue
        pending = [(root, root.name or 'root')]
        while pending:
            block, path = pending.pop()
            handles = tuple((p for p in tunables if p.device_idtag == device.idtag and p.name in {v.name for v in block.event_dict}))
            for var in list(block.state_vars) + list(block.algebraic_vars):
                blocks[var.uid] = (device, path, handles)
            pending.extend(((child, f'{path}/{child.name or f'block[{j}]'}') for j, child in enumerate(block.children)))
    labels = []
    for i, var in enumerate(variables):
        owner, path, handles = blocks.get(var.uid, (owners.get(var.uid), None, ()))
        labels.append(dict(index=i, uid=var.uid, name=var.name, device_idtag=getattr(owner, 'idtag', None), device_name=getattr(owner, 'name', None), block_path=path, tunable_parameters=handles))
    return dict(state_matrix=_readonly(matrix), mass_matrix=mass, eigenvalues=_readonly(eigenvalues), participation_factors=_readonly(participation), variables=tuple(labels))
