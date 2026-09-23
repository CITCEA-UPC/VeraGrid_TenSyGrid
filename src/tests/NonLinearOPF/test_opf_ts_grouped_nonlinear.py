# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
import numpy as np
import pandas as pd

import VeraGridEngine.api as vg
import VeraGridEngine.Simulations.OPF.opf_ts_driver as opf_ts_driver_module
from VeraGridEngine.enumerations import SolverType, TimeGrouping


class ProgressRecorder:
    """
    Store progress values emitted by a driver.
    """

    __slots__ = ("values",)

    def __init__(self) -> None:
        """
        Constructor.

        :return: None.
        """
        self.values: list[float] = list()

    def emit(self, value: float) -> None:
        """
        Store one progress value.

        :param value: Emitted progress value.
        :return: None.
        """
        self.values.append(float(value))


class FakeNonlinearOpfResult:
    """
    Minimal nonlinear OPF result surface consumed by the time-series driver.
    """

    __slots__ = (
        "V",
        "S",
        "lam_p",
        "Pg",
        "Qg",
        "Pcost",
        "Qsh",
        "Sf",
        "St",
        "sl_sf",
        "sl_st",
        "loading",
        "tap_phase",
        "tap_module",
        "hvdc_Pf",
        "hvdc_loading",
        "converged",
    )

    def __init__(self, value: float) -> None:
        """
        Constructor.

        :param value: Time-index marker encoded in the fake result.
        :return: None.
        """
        self.V: np.ndarray = np.array([value + 0.0j], dtype=np.complex128)
        self.S: np.ndarray = np.array([value + 0.0j], dtype=np.complex128)
        self.lam_p: np.ndarray = np.array([value], dtype=float)
        self.Pg: np.ndarray = np.array([value], dtype=float)
        self.Qg: np.ndarray = np.array([value], dtype=float)
        self.Pcost: np.ndarray = np.array([value], dtype=float)
        self.Qsh: np.ndarray = np.zeros(0, dtype=float)
        self.Sf: np.ndarray = np.zeros(0, dtype=np.complex128)
        self.St: np.ndarray = np.zeros(0, dtype=np.complex128)
        self.sl_sf: np.ndarray = np.zeros(0, dtype=float)
        self.sl_st: np.ndarray = np.zeros(0, dtype=float)
        self.loading: np.ndarray = np.zeros(0, dtype=float)
        self.tap_phase: np.ndarray = np.zeros(0, dtype=float)
        self.tap_module: np.ndarray = np.zeros(0, dtype=float)
        self.hvdc_Pf: np.ndarray = np.zeros(0, dtype=float)
        self.hvdc_loading: np.ndarray = np.zeros(0, dtype=float)
        self.converged: bool = True


def build_grouped_opf_grid(periods: int = 48) -> vg.MultiCircuit:
    """
    Build a tiny time-series grid with two calendar days.

    :param periods: Number of hourly time points.
    :return: Test grid.
    """
    grid: vg.MultiCircuit = vg.MultiCircuit()
    bus: vg.Bus = grid.add_bus(vg.Bus(name="B1", Vnom=10.0, is_slack=True))
    grid.add_generator(bus=bus, api_obj=vg.Generator(name="G1", P=1.0, Pmax=10.0))
    grid.time_profile = pd.date_range(start="2026-01-01 00:00:00", periods=periods, freq="h")
    return grid


def fake_run_nonlinear_opf(grid: vg.MultiCircuit,
                           opf_options: vg.OptimalPowerFlowOptions,
                           t_idx: int,
                           logger: vg.Logger) -> FakeNonlinearOpfResult:
    """
    Return a fake result whose values identify the absolute solved time index.

    :param grid: Grid.
    :param opf_options: OPF options.
    :param t_idx: Absolute time index requested by the driver.
    :param logger: Logger.
    :return: Fake nonlinear result.
    """
    return FakeNonlinearOpfResult(value=float(t_idx + 1))


def test_grouped_nonlinear_opf_writes_each_day_to_distinct_rows(monkeypatch) -> None:
    """
    Daily nonlinear OPF grouping must not overwrite every chunk into the first result rows.
    """
    grid: vg.MultiCircuit = build_grouped_opf_grid()
    options: vg.OptimalPowerFlowOptions = vg.OptimalPowerFlowOptions(
        solver=SolverType.NONLINEAR_OPF,
        time_grouping=TimeGrouping.Daily,
    )
    driver: vg.OptimalPowerFlowTimeSeriesDriver = vg.OptimalPowerFlowTimeSeriesDriver(
        grid=grid,
        options=options,
        time_indices=grid.get_all_time_indices(),
    )
    progress: ProgressRecorder = ProgressRecorder()
    driver.progress_signal = progress

    monkeypatch.setattr(opf_ts_driver_module, "run_nonlinear_opf", fake_run_nonlinear_opf)

    driver.run()

    expected: np.ndarray = np.arange(1, 49, dtype=float)
    assert np.allclose(driver.results.generator_power[:, 0], expected * grid.Sbase)
    assert progress.values == [0.0, 50.0, 100.0]


def test_grouped_nonlinear_opf_uses_absolute_time_and_local_result_rows(monkeypatch) -> None:
    """
    Grouped nonlinear OPF must solve absolute selected times and store them in local result rows.
    """
    grid: vg.MultiCircuit = build_grouped_opf_grid(periods=72)
    options: vg.OptimalPowerFlowOptions = vg.OptimalPowerFlowOptions(
        solver=SolverType.NONLINEAR_OPF,
        time_grouping=TimeGrouping.Daily,
    )
    selected_time_indices: np.ndarray = np.arange(24, 72, dtype=int)
    driver: vg.OptimalPowerFlowTimeSeriesDriver = vg.OptimalPowerFlowTimeSeriesDriver(
        grid=grid,
        options=options,
        time_indices=selected_time_indices,
    )

    monkeypatch.setattr(opf_ts_driver_module, "run_nonlinear_opf", fake_run_nonlinear_opf)

    driver.run()

    expected: np.ndarray = np.arange(25, 73, dtype=float)
    assert driver.results.generator_power.shape == (48, 1)
    assert np.allclose(driver.results.generator_power[:, 0], expected * grid.Sbase)
