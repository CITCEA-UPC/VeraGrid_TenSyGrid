# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations
import numpy as np
from typing import Union, TYPE_CHECKING
from VeraGridEngine.Simulations.PowerFlow.power_flow_worker import PowerFlowOptions, multi_island_pf_nc, PowerFlowResults
from VeraGridEngine.Devices.multi_circuit import MultiCircuit
from VeraGridEngine.Compilers.circuit_to_data import compile_numerical_circuit_at
from VeraGridEngine.Simulations.Reliability.reliability2 import compute_transition_probabilities
from VeraGridEngine.Simulations.Stochastic.stochastic_power_flow_input import StochasticPowerFlowInput
from VeraGridEngine.basic_structures import Logger, CxVec

if TYPE_CHECKING:  # Only imports the below statements during type checking
    from VeraGridEngine.Simulations.OPF.opf_results import OptimalPowerFlowResults


class AiIterable:
    """
    AI-ready power flow stochastic iterable
    """
    __slots__ = (
        "grid",
        "logger",
        "pf_options",
        "modify_injections",
        "modify_branches_state",
        "p_up_branches",
        "p_dwn_branches",
        "p_up_gen",
        "p_dwn_gen",
        "mc_input",
        "nc",
        "base_branch_active",
    )

    def __init__(
        self,
        grid: MultiCircuit,
        forced_mttf: Union[None, float] = None,
        forced_mttr: Union[None, float] = None,
        pf_options: PowerFlowOptions = PowerFlowOptions(),
        modify_injections: bool = True,
        modify_branches_state: bool = True,
        opf_results: Union[OptimalPowerFlowResults, None] = None,
        t_idx: int | None = None,
        logger: Logger = Logger(),
    ) -> None:
        """
        AiIterable constructor.

        :param grid: MultiCircuit instance
        :param forced_mttf: Override the branches MTTF with this value
        :param forced_mttr: Override the branches MTTR with this value
        :param pf_options: Power flow options
        :param modify_injections: Flag to randomize injections
        :param modify_branches_state: Flag to randomize branch states
        :param opf_results: Optional OPF results for initialization
        :param t_idx: Optional time step index
        :param logger: Logger instance
        """
        self.grid: MultiCircuit = grid
        self.logger: Logger = logger
        self.pf_options: PowerFlowOptions = pf_options
        self.modify_injections: bool = modify_injections
        self.modify_branches_state: bool = modify_branches_state

        # compile the time step
        nc = compile_numerical_circuit_at(self.grid,
                                          t_idx=t_idx,
                                          logger=logger,
                                          opf_results=opf_results)

        # compute the transition probabilities
        self.p_up_branches, self.p_dwn_branches = compute_transition_probabilities(mttf=nc.passive_branch_data.mttf,
                                                                                   mttr=nc.passive_branch_data.mttr,
                                                                                   forced_mttf=forced_mttf,
                                                                                   forced_mttr=forced_mttr)

        self.p_up_gen, self.p_dwn_gen = compute_transition_probabilities(mttf=nc.generator_data.mttf,
                                                                         mttr=nc.generator_data.mttr,
                                                                         forced_mttf=forced_mttf,
                                                                         forced_mttr=forced_mttr)

        if not grid.has_time_series:
            raise ValueError("The grid must have time series declared!")
        else:
            pass

        self.mc_input = StochasticPowerFlowInput(self.grid)

        # compile the time step
        self.nc = compile_numerical_circuit_at(self.grid, t_idx=None, logger=self.logger)
        self.base_branch_active = self.nc.passive_branch_data.active.copy()

    def __iter__(self) -> "AiIterable":
        return self

    def __next__(self) -> PowerFlowResults:

        if self.modify_branches_state:
            # determine the Markov states
            p = np.random.random(self.nc.nbr)
            br_active = (p > self.p_dwn_branches).astype(int)

            # apply the transitioning states
            self.nc.passive_branch_data.active = br_active
        else:
            pass

        if self.modify_injections:
            # sample monte-carlo injections
            x = np.random.random(self.nc.nbus)
            Sbus: CxVec = self.mc_input.get_at(x=x) / self.nc.Sbase

            pf_res = multi_island_pf_nc(nc=self.nc, options=self.pf_options, Sbus_input=Sbus)

        else:
            # just run without injections variation, and pick the ones from the numerical circuit
            pf_res = multi_island_pf_nc(nc=self.nc, options=self.pf_options)

        return pf_res

    def reset(self) -> None:
        """
        Reset the iterable
        """
        self.nc = compile_numerical_circuit_at(self.grid, t_idx=None, logger=self.logger)
        self.base_branch_active = self.nc.passive_branch_data.active.copy()

