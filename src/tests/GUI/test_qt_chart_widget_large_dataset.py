# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""Large real-world time-series coverage for the native QWidget chart."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PySide6 import QtCore, QtGui, QtWidgets

from VeraGrid.Gui.PlotDialogue.plot_dialogue import PlotDialogue
from VeraGrid.Gui.PlotDialogue.qt_chart_widget import GraphsWidget


def load_large_chart_data(data_path: Path) -> tuple[np.ndarray, tuple[str, ...], tuple[np.ndarray, ...]]:
    """Load the supplied wide time-series CSV without adding a plotting dependency.

    :param data_path: CSV fixture with one time column and numeric series columns.
    :return: Time values, series names, and one NumPy view for each source series.
    """
    data_frame: pd.DataFrame = pd.read_csv(data_path, index_col=0, parse_dates=True)
    time_values: np.ndarray = np.asarray(data_frame.index.to_numpy(), dtype='datetime64[ns]')
    value_matrix: np.ndarray = data_frame.to_numpy(dtype=float, copy=False)
    series_names: tuple[str, ...] = tuple(str(column_name) for column_name in data_frame.columns)
    series_values: tuple[np.ndarray, ...] = tuple(
        value_matrix[:, series_index] for series_index in range(value_matrix.shape[1])
    )
    return time_values, series_names, series_values


def render_chart(chart: QtWidgets.QWidget) -> QtGui.QImage:
    """Render one QWidget chart into a standalone image paint device.

    :param chart: Visible native chart widget to exercise.
    :return: Non-null raster image containing its paint output.
    """
    image: QtGui.QImage = QtGui.QImage(
        chart.size(),
        QtGui.QImage.Format.Format_ARGB32_Premultiplied,
    )
    image.fill(QtGui.QColor('#ffffff'))
    painter: QtGui.QPainter = QtGui.QPainter(image)
    chart.render(painter, QtCore.QPoint(0, 0))
    painter.end()
    return image


@pytest.mark.skip("File too large")
def test_wide_time_series_uses_one_x_buffer_and_bounded_paint_work(
        qt_app: QtWidgets.QApplication) -> None:
    """Render all supplied lines through the owned-dialog lifecycle without a crash.

    :param qt_app: Shared Qt application fixture.
    :return: None.
    """
    app: QtWidgets.QApplication = qt_app
    data_path: Path = Path(__file__).parents[1] / 'data' / 'large_dataset_for_graphics.csv'
    time_values: np.ndarray
    series_names: tuple[str, ...]
    series_values: tuple[np.ndarray, ...]
    time_values, series_names, series_values = load_large_chart_data(data_path=data_path)
    assert len(time_values) == 8791
    assert len(series_names) == 2224

    dialog: PlotDialogue = PlotDialogue(title='Wide time-series')
    dialog.resize(1280, 760)
    dialog.show()
    app.processEvents()
    assert dialog.set_time_series(
        time_values=time_values,
        series_names=series_names,
        series_values=series_values,
        title='Wide time-series',
        y_axis_title='p.u.',
    )
    app.processEvents()

    chart: GraphsWidget = dialog.chart
    assert chart.get_series_count() == len(series_names)
    assert dialog.ui.seriesSelectorFrame.isVisible()
    assert dialog.ui.seriesListWidget.count() == len(series_names)
    shared_x_data: np.ndarray = chart._series[0].get_x_data()
    series_index: int
    for series_index in range(chart.get_series_count()):
        assert chart._series[series_index].get_x_data() is shared_x_data

    plot_rect: QtCore.QRectF = chart._get_plot_rect()
    paint_point_count: int = 0
    for series_index in range(chart.get_series_count()):
        paint_point_count += len(chart._sample_xy_indices(
            x_data=chart._series[series_index].get_x_data(),
            y_data=chart._series[series_index].get_y_data(),
            x_is_monotonic=chart._series[series_index].get_x_is_monotonic(),
            plot_rect=plot_rect,
            visible_line_count=chart.get_series_count(),
        ))
    assert paint_point_count <= 500000
    assert paint_point_count > chart.get_series_count() * 2
    rendered_image: QtGui.QImage = render_chart(chart=chart)
    assert not rendered_image.isNull()

    full_x_minimum: float
    full_x_maximum: float
    full_x_minimum, full_x_maximum = chart.axis_x.get_range()
    chart.axis_x.set_zoom(4.0)
    chart.axis_x.set_pan((full_x_maximum - full_x_minimum) * 0.25)
    zoomed_indices: np.ndarray = chart._sample_xy_indices(
        x_data=shared_x_data,
        y_data=chart._series[0].get_y_data(),
        x_is_monotonic=chart._series[0].get_x_is_monotonic(),
        plot_rect=plot_rect,
        visible_line_count=chart.get_series_count(),
    )
    assert int(zoomed_indices[0]) > 0
    zoomed_image: QtGui.QImage = render_chart(chart=chart)
    assert not zoomed_image.isNull()

    dialog.reject()
    assert chart._disposed
    dialog.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    app.processEvents()
