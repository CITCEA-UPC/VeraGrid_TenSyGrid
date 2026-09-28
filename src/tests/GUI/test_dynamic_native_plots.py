# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""Native dynamic-results plotting lifecycle tests."""

import numpy as np
from PySide6 import QtCore, QtWidgets

from VeraGrid.Gui.DynamicModelEditor.Plots.dynamic_plots_handler import DynamicsResultsHandler
from VeraGrid.Gui.PlotDialogue.qt_chart_widget import GraphsWidget
from VeraGridEngine.Simulations.Rms.rms_results import RmsResults
from VeraGridEngine.Utils.Symbolic.symbolic import Var
from VeraGridEngine.enumerations import DeviceType, DynamicPlotEntryRole, DynamicPlotMode


class DynamicPlotTestDevice:
    """Small hashable device record sufficient for the dynamic results tree."""

    __slots__ = ('name', 'idtag', 'device_type')

    def __init__(self, name: str, idtag: str) -> None:
        """Create one deterministic device identity.

        :param name: Visible device name.
        :param idtag: Stable device identifier.
        :return: None.
        """
        self.name: str = name
        self.idtag: str = idtag
        self.device_type: DeviceType = DeviceType.NoDevice

    def __hash__(self) -> int:
        """Hash the device by its stable identifier.

        :return: Stable hash value.
        """
        return hash(self.idtag)

    def __str__(self) -> str:
        """Return the visible tree label.

        :return: Device name.
        """
        return self.name


def build_dynamic_plot_results() -> RmsResults:
    """Create two finite RMS traces sharing one event group.

    :return: Dynamic results populated with two variables.
    """
    first_variable: Var = Var(name='omega', uid=1)
    second_variable: Var = Var(name='efd', uid=2)
    device: DynamicPlotTestDevice = DynamicPlotTestDevice(name='Generator', idtag='generator-1')
    results: RmsResults = RmsResults(
        time_array=np.array((0.0, 0.5, 1.0, 1.5), dtype=float),
        rms_events_group_names=np.array(('Base',), dtype=str),
        rms_events_group_idtags=np.array(('base-group',), dtype=str),
        variables=[first_variable, second_variable],
        uid2idx={1: 0, 2: 1},
        vars_glob_name2uid={'generator-1:omega:1': 1, 'generator-1:efd:2': 2},
        devices_vars_info={device: [first_variable, second_variable]},
        initial_parameter_value_maps=[dict()],
        has_event_group_results=np.array((True,), dtype=bool),
    )
    results.values[:, :, 0] = np.array(
        ((1.0, 2.0), (1.1, 2.4), (1.2, 2.8), (1.3, 3.2)),
        dtype=float,
    )
    return results


def test_dynamic_native_dialogs_survive_repeated_plot_and_close_cycles(
        qt_app: QtWidgets.QApplication) -> None:
    """Exercise dynamic time, XY, and direct-series dialogs through teardown.

    :param qt_app: Shared Qt application fixture.
    :return: None.
    """
    application: QtWidgets.QApplication = qt_app
    cycle_index: int
    for cycle_index in range(16):
        results: RmsResults = build_dynamic_plot_results()
        handler: DynamicsResultsHandler = DynamicsResultsHandler(results=results)
        assert handler.create_plot_group(name=f'Time {cycle_index}')
        assert handler.add_var_to_group(group_name=f'Time {cycle_index}', var_uid=1)
        assert handler.add_var_to_group(group_name=f'Time {cycle_index}', var_uid=2)
        assert handler.plot_group(plot_group_name=f'Time {cycle_index}')
        time_chart: GraphsWidget = handler._open_plot_dialogues[-1].chart
        time_x_minimum: float
        time_x_maximum: float
        time_x_minimum, time_x_maximum = time_chart.axis_x.get_range()
        time_y_minimum: float
        time_y_maximum: float
        time_y_minimum, time_y_maximum = time_chart.axis_y.get_range()
        assert np.isclose(time_x_minimum, -0.075)
        assert np.isclose(time_x_maximum, 1.575)
        assert np.isclose(time_y_minimum, 0.89)
        assert np.isclose(time_y_maximum, 3.31)

        first_series = handler.series_by_var_uid[1][0]
        second_series = handler.series_by_var_uid[2][0]
        assert handler.create_plot_group(name=f'XY {cycle_index}', mode=DynamicPlotMode.XY)
        assert handler.add_series_to_group_with_role(
            group_name=f'XY {cycle_index}',
            series_key=first_series.get_key(),
            role=DynamicPlotEntryRole.X_AXIS,
        )
        assert handler.add_series_to_group_with_role(
            group_name=f'XY {cycle_index}',
            series_key=second_series.get_key(),
            role=DynamicPlotEntryRole.Y_AXIS,
        )
        assert handler.plot_group(plot_group_name=f'XY {cycle_index}')
        xy_chart: GraphsWidget = handler._open_plot_dialogues[-1].chart
        x_minimum: float
        x_maximum: float
        x_minimum, x_maximum = xy_chart.axis_x.get_range()
        y_minimum: float
        y_maximum: float
        y_minimum, y_maximum = xy_chart.axis_y.get_range()
        assert np.isclose(x_minimum, 0.985)
        assert np.isclose(x_maximum, 1.315)
        assert np.isclose(y_minimum, 1.94)
        assert np.isclose(y_maximum, 3.26)
        handler.plot_series(series=first_series)
        application.processEvents()

        assert len(handler._open_plot_dialogues) == 3
        plot_dialogue = handler._open_plot_dialogues[0]
        assert len(plot_dialogue.chart._series) == 2
        handler.close_plot_dialogs()
        assert len(handler._open_plot_dialogues) == 0
        assert plot_dialogue.chart._disposed
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
        application.processEvents()
