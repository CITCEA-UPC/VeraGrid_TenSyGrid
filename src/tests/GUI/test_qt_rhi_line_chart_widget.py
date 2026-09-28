# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""Basic data and resource-lifecycle checks for the experimental QRhi plot."""

from collections.abc import Sequence

import numpy as np
from PySide6 import QtWidgets
from PySide6.QtGui import QColor

from VeraGrid.Gui.PlotDialogue.qt_chart_widget import GraphsWidget
from VeraGrid.Gui.PlotDialogue.qt_rhi_line_chart_widget import RhiLineChartWidget


def test_rhi_line_chart_packs_series_colors_and_clears_invalid_replacement(
        qt_app: QtWidgets.QApplication) -> None:
    """Check data packing and invalid-input reset without requiring a GPU backend.

    :param qt_app: Shared Qt application fixture.
    :return: None.
    """
    _ = qt_app
    chart: RhiLineChartWidget = RhiLineChartWidget()
    vertices: np.ndarray = np.array(
        ((-1.0, -1.0), (0.0, 0.0), (1.0, 1.0), (-1.0, 1.0), (0.0, 0.0)),
        dtype=np.float32,
    )
    colors: Sequence[QColor] = (QColor("#ff0000"), QColor("#00ff00"))
    assert chart.set_vertices(vertices, (3, 2), colors)
    assert chart._vertices.shape == (5, 5)
    assert chart._vertices.dtype == np.float32
    assert chart._series_count == 2
    assert np.array_equal(chart._series_first_vertices, np.array((0, 3), dtype=np.int32))
    assert np.array_equal(chart._vertices[:3, 2:], np.tile((1.0, 0.0, 0.0), (3, 1)))
    assert np.array_equal(chart._vertices[3:, 2:], np.tile((0.0, 1.0, 0.0), (2, 1)))

    assert not chart.set_vertices(vertices, (4,), (QColor("#0000ff"),))
    assert chart._vertices.size == 0
    assert chart._series_count == 0

    chart.releaseResources()
    chart.releaseResources()
    chart.deleteLater()
    QtWidgets.QApplication.sendPostedEvents(None)


def test_graphs_widget_keeps_qpainter_as_the_default_renderer(
        qt_app: QtWidgets.QApplication) -> None:
    """Verify opt-in creates the child before painting and remains optional.

    :param qt_app: Shared Qt application fixture.
    :return: None.
    """
    _ = qt_app
    chart: GraphsWidget = GraphsWidget()
    assert not chart._rhi_line_enabled
    assert chart._rhi_line_widget is None

    chart.set_rhi_line_rendering(enabled=True, series_threshold=4)
    assert chart._rhi_line_enabled
    assert isinstance(chart._rhi_line_widget, RhiLineChartWidget)
    chart.set_rhi_line_rendering(enabled=False)
    assert not chart._rhi_line_enabled

    chart.dispose()
    chart.deleteLater()
    QtWidgets.QApplication.sendPostedEvents(None)


def test_graphs_widget_routes_dense_lines_to_optional_rhi_layer(
        qt_app: QtWidgets.QApplication) -> None:
    """Check the threshold selects QRhi only after explicit opt-in.

    :param qt_app: Shared Qt application fixture.
    :return: None.
    """
    _ = qt_app
    chart: GraphsWidget = GraphsWidget()
    chart.resize(720, 480)
    line_index: int
    for line_index in range(4):
        chart.add_line_series(
            name=f'Line {line_index}',
            x_values=np.array((0.0, 1.0, 2.0), dtype=float),
            y_values=np.array((float(line_index), float(line_index + 1), float(line_index)), dtype=float),
        )
    assert chart.get_series_count() == 4

    chart.set_rhi_line_rendering(enabled=True, series_threshold=4)
    assert chart._prepare_rhi_line_layer(plot_rect=chart._get_plot_rect())
    assert isinstance(chart._rhi_line_widget, RhiLineChartWidget)
    assert chart._rhi_line_widget._series_count == 16
    assert chart._rhi_line_widget._background_color == chart._plot_color
    assert np.array_equal(
        chart._rhi_line_widget._vertices[:2, :2],
        np.array(((-1.0, -1.0), (-1.0, 1.0)), dtype=np.float32),
    )
    assert np.array_equal(
        chart._rhi_line_widget._vertices[:2, 2:],
        np.tile(
            np.array((chart._grid_color.redF(), chart._grid_color.greenF(),
                      chart._grid_color.blueF()), dtype=np.float32),
            (2, 1),
        ),
    )

    chart.set_rhi_line_rendering(enabled=False)
    assert not chart._prepare_rhi_line_layer(plot_rect=chart._get_plot_rect())
    chart.dispose()
    chart.deleteLater()
    QtWidgets.QApplication.sendPostedEvents(None)


def test_graphs_widget_keeps_cumulative_areas_on_qpainter(
        qt_app: QtWidgets.QApplication) -> None:
    """Keep stacked area bands on QPainter until QRhi matches its rendering.

    :param qt_app: Shared Qt application fixture.
    :return: None.
    """
    _ = qt_app
    chart: GraphsWidget = GraphsWidget()
    chart.resize(720, 480)
    x_values: np.ndarray = np.linspace(0.0, 10.0, 20000, dtype=float)
    first_values: np.ndarray = np.sin(x_values)
    second_values: np.ndarray = np.cos(x_values)
    assert chart.set_cumulative_area_series(
        x_values=x_values,
        series_names=("First", "Second"),
        series_values=(first_values, second_values),
        colors=("#2563eb", "#f97316"),
    )
    positive_stack_maximum: float = float(
        np.max(np.maximum(first_values, 0.0) + np.maximum(second_values, 0.0))
    )
    negative_stack_minimum: float = float(
        np.min(np.minimum(first_values, 0.0) + np.minimum(second_values, 0.0))
    )
    assert np.isclose(chart._y_min, negative_stack_minimum)
    assert np.isclose(chart._y_max, positive_stack_maximum)
    chart.set_rhi_line_rendering(enabled=True)
    assert not chart._prepare_rhi_line_layer(plot_rect=chart._get_plot_rect())
    assert chart._rhi_line_widget is not None
    assert not chart._rhi_line_widget.isVisible()
    assert chart._content_type.name == "CUMULATIVE_AREA"

    chart.dispose()
    chart.deleteLater()
    QtWidgets.QApplication.sendPostedEvents(None)


def test_rhi_line_chart_packs_large_line_batch(qt_app: QtWidgets.QApplication) -> None:
    """Exercise CPU packing at the approximate supplied graphics-data scale.

    :param qt_app: Shared Qt application fixture.
    :return: None.
    """
    _ = qt_app
    series_count: int = 2224
    vertices_per_series: int = 224
    point_count: int = series_count * vertices_per_series
    x_values: np.ndarray = np.linspace(-1.0, 1.0, point_count, dtype=np.float32)
    vertices: np.ndarray = np.empty((point_count, 2), dtype=np.float32)
    vertices[:, 0] = x_values
    vertices[:, 1] = np.sin(x_values * np.float32(10.0))
    series_lengths: tuple[int, ...] = (vertices_per_series,) * series_count
    series_colors: tuple[QColor, ...] = tuple(
        QColor.fromHsv((series_index * 37) % 360, 220, 210) for series_index in range(series_count)
    )
    chart: RhiLineChartWidget = RhiLineChartWidget()
    assert chart.set_vertices(vertices, series_lengths, series_colors)
    assert chart._vertices.shape == (point_count, 5)
    assert chart._vertices.nbytes == point_count * 5 * np.dtype(np.float32).itemsize
    chart.releaseResources()
    chart.deleteLater()
    QtWidgets.QApplication.sendPostedEvents(None)
