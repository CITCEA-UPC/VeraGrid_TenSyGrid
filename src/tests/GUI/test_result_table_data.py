# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""Tests for copying result table data into native Qt chart buffers."""

import numpy as np

from VeraGrid.Gui.PlotDialogue.result_table_data import get_result_table_series
from VeraGridEngine.enumerations import DeviceType
from VeraGridEngine.Simulations.results_table import ResultsTable


def make_result_table() -> ResultsTable:
    """Create a small dated table used to verify native plot extraction.

    :return: Results table with numeric columns and a NumPy time index.
    """
    return ResultsTable(
        data=np.array(((0.0, 2.0), (1.0, 3.0), (2.0, 4.0)), dtype=float),
        columns=np.array(('Voltage', 'Loading'), dtype=str),
        index=np.array(('2026-01-01T00:00', '2026-01-01T01:00', '2026-01-01T02:00'), dtype='datetime64[m]'),
        title='Voltage',
        cols_device_type=DeviceType.NoDevice,
        idx_device_type=DeviceType.NoDevice,
    )


def test_result_table_data_keeps_time_coordinates_and_isolates_values() -> None:
    """Verify native series receive copied time coordinates and finite value buffers.

    :return: None.
    """
    table: ResultsTable = make_result_table()
    plot_data: tuple[np.ndarray, list[str], list[np.ndarray]] | None = get_result_table_series(
        table=table,
        selected_col_idx=np.array((0,), dtype=np.int64),
        selected_rows=np.array((0, 2), dtype=np.int64),
        hide_zero_values=True,
    )

    assert plot_data is not None
    x_values: np.ndarray = plot_data[0]
    series_names: list[str] = plot_data[1]
    series_values: list[np.ndarray] = plot_data[2]
    assert np.issubdtype(x_values.dtype, np.datetime64)
    assert series_names == ['Voltage']
    assert np.isnan(series_values[0][0])
    assert series_values[0][1] == 2.0
    table.data_c[2, 0] = 99.0
    assert series_values[0][1] == 2.0


def test_result_table_data_rejects_invalid_empty_and_out_of_range_selection() -> None:
    """Verify invalid table selections fail before mutating any native chart state.

    :return: None.
    """
    table: ResultsTable = make_result_table()
    assert get_result_table_series(
        table=table,
        selected_col_idx=np.array((), dtype=np.int64),
    ) is None
    assert get_result_table_series(
        table=table,
        selected_col_idx=np.array((3,), dtype=np.int64),
    ) is None
    assert get_result_table_series(
        table=table,
        selected_rows=np.array((-1,), dtype=np.int64),
    ) is None
