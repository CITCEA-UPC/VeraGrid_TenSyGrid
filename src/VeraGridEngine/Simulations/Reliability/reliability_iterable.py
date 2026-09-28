# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
from typing import Tuple, Union
from VeraGridEngine.Simulations.PowerFlow.power_flow_worker import PowerFlowOptions, multi_island_pf_nc, PowerFlowResults
from VeraGridEngine.Devices.multi_circuit import MultiCircuit
from VeraGridEngine.Compilers.circuit_to_data import compile_numerical_circuit_at
from VeraGridEngine.Simulations.Reliability.reliability2 import compute_transition_probabilities
from VeraGridEngine.basic_structures import IntVec, Logger


class ReliabilityIterable:
    """
    RealTimeStateEnumeration iterable for reliability analysis.
    """
    __slots__ = (
        "grid",
        "nt",
        "t_idx",
        "logger",
        "pf_options",
        "p_up_branches",
        "p_dwn_branches",
        "p_up_gen",
        "p_dwn_gen",
    )

    def __init__(
        self,
        grid: MultiCircuit,
        forced_mttf: Union[None, float] = None,
        forced_mttr: Union[None, float] = None,
        logger: Logger = Logger(),
    ) -> None:
        """
        ReliabilityIterable constructor.

        :param grid: MultiCircuit instance
        :param forced_mttf: Override the branches MTTF with this value
        :param forced_mttr: Override the branches MTTR with this value
        :param logger: Logger instance
        """
        self.grid: MultiCircuit = grid

        # number of time steps
        self.nt: int = grid.get_time_number()

        # time index
        self.t_idx: int = 0

        self.logger: Logger = logger

        # declare the power flow options
        self.pf_options: PowerFlowOptions = PowerFlowOptions()

        # compile the time step
        nc = compile_numerical_circuit_at(self.grid, t_idx=None, logger=logger)

        # compute the transition probabilities
        self.p_up_branches, self.p_dwn_branches = compute_transition_probabilities(
            mttf=nc.passive_branch_data.mttf,
            mttr=nc.passive_branch_data.mttr,
            forced_mttf=forced_mttf,
            forced_mttr=forced_mttr,
        )

        self.p_up_gen, self.p_dwn_gen = compute_transition_probabilities(
            mttf=nc.generator_data.mttf,
            mttr=nc.generator_data.mttr,
            forced_mttf=forced_mttf,
            forced_mttr=forced_mttr,
        )

    def __iter__(self) -> "ReliabilityIterable":
        """
        Return the iterator itself.

        :return: Self
        """
        return self

    def __next__(self) -> Tuple[IntVec, PowerFlowResults]:
        """
        Compute the next reliability power flow state.

        :return: Tuple of active branches and power flow results
        """
        if self.nt == 0:  # no time steps, no fun
            print('No time steps :/')
            raise StopIteration
        else:
            pass

        # compile the time step
        nc = compile_numerical_circuit_at(self.grid, t_idx=self.t_idx, logger=self.logger)

        # determine the Markov states
        p = np.random.random(nc.nbr)
        br_active = (p > self.p_dwn_branches).astype(int)

        # apply the transitioning states
        nc.passive_branch_data.active = br_active

        pf_res = multi_island_pf_nc(nc=nc, options=self.pf_options)

        # determine the next state
        if self.t_idx < (self.nt - 1):
            # advance to the next step
            self.t_idx += 1
        else:
            # raise StopIteration
            self.t_idx = 0  # restart

        return br_active, pf_res
