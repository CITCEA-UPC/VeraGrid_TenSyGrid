from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtWidgets

from VeraGrid.Gui.Main.SubClasses.Results.results import ResultsMain
from VeraGrid.Gui.PlotDialogue.plot_dialogue import PlotDialogue
from VeraGrid.Gui.PlotDialogue.qt_chart_widget import ChartContentType, PolarAngleUnit
from VeraGrid.Gui.results_model import ResultsModel
from VeraGridEngine.enumerations import DeviceType
from VeraGridEngine.Simulations.results_table import ResultsTable


class ResultsPlotOwner(QtWidgets.QWidget):
    """Minimal Qt owner for exercising the Results plotting method."""

    __slots__ = ("_open_plot_dialogs",)

    def __init__(self) -> None:
        """Create the dialog owner and its retained plot list.

        :return: None.
        """
        super().__init__()
        self._open_plot_dialogs: list[PlotDialogue] = list()

    def register_open_plot_dialog(self, dialog: QtWidgets.QDialog) -> None:
        """
        Track one open plot dialog in the owner.

        :param dialog: Plot dialog to track.
        :return: None.
        """
        if dialog in self._open_plot_dialogs:
            pass
        else:
            self._open_plot_dialogs.append(dialog)



def close_plot_dialogue(dialog: PlotDialogue, app: QtWidgets.QApplication) -> None:
    """Close one plot dialog through its real Qt ownership lifecycle.

    :param dialog: Dialog whose chart buffers must be released before deletion.
    :param app: Application processing the deferred native deletion event.
    :return: None.
    """
    dialog.reject()
    assert dialog.chart._disposed
    dialog.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    app.processEvents()


def test_plot_dialogue_time_area_polar_export_and_lifecycle(
        qt_app: QtWidgets.QApplication,
        tmp_path: Path) -> None:
    """Exercise every native plot mode through repeated open and close cycles.

    :param qt_app: Shared Qt application fixture.
    :param tmp_path: Temporary output folder provided by pytest.
    :return: None.
    """
    app: QtWidgets.QApplication = qt_app
    cycle_index: int
    for cycle_index in range(24):
        dialog: PlotDialogue = PlotDialogue(title=f'Chart cycle {cycle_index}')
        dialog.resize(900, 620)
        dialog.show()
        app.processEvents()

        time_values: np.ndarray = np.array(
            ('2026-01-01T00:00', '2026-01-01T01:00', '2026-01-01T02:00', '2026-01-01T03:00'),
            dtype='datetime64[m]',
        )
        assert dialog.set_time_series(
            time_values=time_values,
            series_names=('Active power', 'Reactive power'),
            series_values=(np.array((1.0, 2.0, 3.0, 2.0)), np.array((0.2, 0.4, 0.6, 0.4))),
            colors=('#2563eb', '#f97316'),
            title='Time plot',
            y_axis_title='MW',
        )
        assert dialog.chart._x_is_datetime
        dialog.chart.axis_x.set_zoom(4.0)
        dialog.chart.axis_y.set_zoom(4.0)
        dialog.ui.actionResetView.trigger()
        assert dialog.chart.axis_x.get_zoom() == 1.0
        assert dialog.chart.axis_y.get_zoom() == 1.0

        assert dialog.set_cumulative_area_series(
            x_values=time_values,
            series_names=('Wind', 'Solar'),
            series_values=(np.array((1.0, 1.5, 1.1, 0.9)), np.array((0.5, 0.7, 1.2, 0.6))),
            colors=('#2563eb', '#f97316'),
            title='Cumulative generation',
            x_axis_title='Time',
            y_axis_title='MW',
        )
        assert dialog.chart._content_type == ChartContentType.CUMULATIVE_AREA
        dialog.repaint()
        app.processEvents()

        assert dialog.set_polar_series(
            series_names=('Impedance',),
            angle_values=(np.array((0.0, 90.0, 180.0, 270.0, 360.0)),),
            radius_values=(np.array((1.0, 0.8, 1.2, 0.9, 1.0)),),
            colors=('#7c3aed',),
            title='Polar plot',
            radius_title='Per unit',
            angle_unit=PolarAngleUnit.DEGREES,
        )
        assert dialog.chart._content_type == ChartContentType.POLAR
        dialog.repaint()
        app.processEvents()

        assert dialog.set_polar_series(
            series_names=('Voltage phasors',),
            angle_values=(np.array((0.0, 120.0, 240.0)),),
            radius_values=(np.array((1.0, 0.98, 1.02)),),
            title='Voltage phasors',
            radius_title='p.u.',
            angle_unit=PolarAngleUnit.DEGREES,
            connect_points=False,
        )
        dialog.repaint()
        app.processEvents()

        png_path: Path = tmp_path / f'plot-{cycle_index}.png'
        svg_path: Path = tmp_path / f'plot-{cycle_index}.svg'
        assert dialog.save_image_to_file(file_name=str(png_path))
        assert dialog.save_image_to_file(file_name=str(svg_path))
        assert png_path.stat().st_size > 0
        assert svg_path.stat().st_size > 0
        assert '<svg' in svg_path.read_text(encoding='utf-8')

        close_plot_dialogue(dialog=dialog, app=app)


def test_plot_dialogue_histogram_tabs_release_every_chart(
        qt_app: QtWidgets.QApplication,
        tmp_path: Path) -> None:
    """Exercise repeated histogram tabs, tab switching, export, and shutdown.

    :param qt_app: Shared Qt application fixture.
    :param tmp_path: Temporary output folder provided by pytest.
    :return: None.
    """
    app: QtWidgets.QApplication = qt_app
    cycle_index: int
    for cycle_index in range(24):
        dialog: PlotDialogue = PlotDialogue(title=f'Histogram cycle {cycle_index}')
        dialog.resize(900, 620)
        dialog.show()
        app.processEvents()

        first_values: np.ndarray = np.array((0.0, 0.1, 0.5, 0.7, 1.3, np.nan, np.inf), dtype=float)
        assert dialog.set_histogram(
            values=first_values,
            bin_count=4,
            title='Resistance distribution',
            x_axis_title='R',
            y_axis_title='Count',
        )
        dialog.set_current_tab_title('Resistance')
        first_chart = dialog.chart
        assert first_chart._content_type == ChartContentType.HISTOGRAM
        assert len(first_chart._series) == 1
        assert not first_chart.set_histogram(values=first_values, bin_count=0)
        assert len(first_chart._series) == 1

        second_values: np.ndarray = np.array((-3.0, -1.0, -1.0, 0.0, 2.0, 4.0), dtype=float)
        assert dialog.add_histogram_tab(
            tab_title='Reactance',
            values=second_values,
            bin_count=4,
            title='Reactance distribution',
            x_axis_title='X',
            y_axis_title='Count',
        )
        second_chart = dialog.get_current_chart()
        assert dialog.ui.plotTabs.count() == 2
        assert second_chart is not first_chart
        assert second_chart._content_type == ChartContentType.HISTOGRAM
        second_chart.axis_x.set_zoom(3.0)
        second_chart.axis_y.set_zoom(3.0)
        dialog.ui.actionResetView.trigger()
        assert second_chart.axis_x.get_zoom() == 1.0
        assert second_chart.axis_y.get_zoom() == 1.0
        dialog.repaint()
        app.processEvents()

        histogram_path: Path = tmp_path / f'histogram-{cycle_index}.png'
        assert dialog.save_image_to_file(file_name=str(histogram_path))
        assert histogram_path.stat().st_size > 0
        dialog.ui.plotTabs.setCurrentIndex(0)
        assert dialog.get_current_chart() is first_chart
        dialog.ui.plotTabs.setCurrentIndex(1)
        assert dialog.get_current_chart() is second_chart

        dialog.reject()
        assert first_chart._disposed
        assert second_chart._disposed
        dialog.deleteLater()
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
        app.processEvents()


def test_plot_dialogue_series_selector_filters_and_hides_lines(
        qt_app: QtWidgets.QApplication) -> None:
    """Use the large-series legend without retaining chart-owned Qt objects.

    :param qt_app: Shared Qt application fixture.
    :return: None.
    """
    app: QtWidgets.QApplication = qt_app
    dialog: PlotDialogue = PlotDialogue(title='Many lines')
    dialog.resize(900, 620)
    dialog.show()
    app.processEvents()
    x_values: np.ndarray = np.linspace(0.0, 1.0, 32, dtype=float)
    series_names: tuple[str, ...] = tuple(f'Series {series_index:02d}' for series_index in range(21))
    series_values: tuple[np.ndarray, ...] = tuple(
        x_values + float(series_index) for series_index in range(len(series_names))
    )
    assert dialog.set_line_series(
        x_values=x_values,
        series_names=series_names,
        series_values=series_values,
    )
    assert dialog.ui.seriesSelectorFrame.isVisible()
    assert dialog.ui.seriesTreeWidget.topLevelItemCount() == 1
    assert dialog.ui.seriesTreeWidget.topLevelItem(0).childCount() == len(series_names)
    assert not dialog.chart._legend_visible
    first_item: QtWidgets.QTreeWidgetItem = dialog.ui.seriesTreeWidget.topLevelItem(0).child(0)
    assert first_item.foreground(0).color() == dialog.chart.get_series_color(series_index=0)

    dialog.ui.seriesSearchLineEdit.setText('series 20')
    app.processEvents()
    series_group: QtWidgets.QTreeWidgetItem = dialog.ui.seriesTreeWidget.topLevelItem(0)
    assert not series_group.child(20).isHidden()
    assert series_group.child(0).isHidden()

    first_item.setCheckState(0, QtCore.Qt.CheckState.Unchecked)
    app.processEvents()
    assert not dialog.chart.get_series_visible(series_index=0)
    close_plot_dialogue(dialog=dialog, app=app)


def test_plot_tabs_can_be_added_and_closed_without_disposing_survivors(
        qt_app: QtWidgets.QApplication) -> None:
    """Dispose only the requested tab chart and keep another tab usable.

    :param qt_app: Shared Qt application fixture.
    :return: None.
    """
    app: QtWidgets.QApplication = qt_app
    dialog: PlotDialogue = PlotDialogue(title='Tabs')
    first_chart = dialog.chart
    dialog.ui.actionAddPlot.trigger()
    assert dialog.ui.plotTabs.count() == 2
    second_chart = dialog.get_current_chart()
    assert second_chart is not first_chart
    dialog.ui.plotTabs.tabCloseRequested.emit(1)
    app.processEvents()
    assert dialog.ui.plotTabs.count() == 1
    assert first_chart in dialog._tab_charts
    assert second_chart not in dialog._tab_charts
    assert dialog.get_current_chart() is first_chart
    close_plot_dialogue(dialog=dialog, app=app)


def test_plot_dialogue_save_picker_is_child_owned_and_asynchronous(
        qt_app: QtWidgets.QApplication) -> None:
    """Release the asynchronous image picker when its plot dialog closes.

    :param qt_app: Shared Qt application fixture.
    :return: None.
    """
    app: QtWidgets.QApplication = qt_app
    dialog: PlotDialogue = PlotDialogue(title='Save lifecycle')
    dialog.show()
    app.processEvents()
    dialog.save_image()
    save_dialog: QtWidgets.QFileDialog | None = dialog._active_save_dialog
    assert save_dialog is not None
    assert save_dialog.parent() is dialog
    assert save_dialog.isVisible()
    save_dialog.close()
    app.processEvents()
    assert dialog._active_save_dialog is None
    close_plot_dialogue(dialog=dialog, app=app)


def test_plot_dialogue_dense_rhi_is_opt_in_and_applies_to_new_tabs(
        qt_app: QtWidgets.QApplication) -> None:
    """Keep ordinary dialogs on QPainter while carrying explicit RHI opt-in to tabs.

    :param qt_app: Shared Qt application fixture.
    :return: None.
    """
    app: QtWidgets.QApplication = qt_app
    painter_dialog: PlotDialogue = PlotDialogue(title='Default renderer')
    assert not painter_dialog.chart._rhi_line_enabled
    painter_dialog.reject()

    rhi_dialog: PlotDialogue = PlotDialogue(
        title='Dense results',
        use_rhi_for_many_series=True,
    )
    assert rhi_dialog.chart._rhi_line_enabled
    new_chart = rhi_dialog.add_tab(title='Additional results')
    assert new_chart._rhi_line_enabled
    assert new_chart._rhi_line_widget is not None
    rhi_dialog.reject()

    dialogue: PlotDialogue
    for dialogue in (painter_dialog, rhi_dialog):
        dialogue.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    app.processEvents()


def test_results_wide_time_series_stays_on_qpainter(
        qt_app: QtWidgets.QApplication) -> None:
    """Keep wide Results plots off the QRhi path that distorted their lines.

    :param qt_app: Shared Qt application fixture.
    :return: None.
    """
    app: QtWidgets.QApplication = qt_app
    series_count: int = 80
    time_values: np.ndarray = np.array(
        ('2026-01-01T00:00', '2026-01-01T01:00', '2026-01-01T02:00'),
        dtype='datetime64[m]',
    )
    branch_values: np.ndarray = (
        np.array((1.0, 2.0, 3.0), dtype=float)[:, np.newaxis]
        + np.arange(series_count, dtype=float)[np.newaxis, :]
    )
    table: ResultsTable = ResultsTable(
        data=branch_values,
        columns=np.array(tuple(f'branch {i}' for i in range(series_count)), dtype=str),
        index=time_values,
        title='Branch Loading',
        cols_device_type=DeviceType.NoDevice,
        idx_device_type=DeviceType.TimeDevice,
        xlabel='Time',
        ylabel='Loading (%)',
    )
    model: ResultsModel = ResultsModel(table=table)
    owner: ResultsPlotOwner = ResultsPlotOwner()

    ResultsMain.open_native_results_series_plot(
        owner,
        mdl=model,
        selected_columns=None,
        selected_rows=None,
        stacked=False,
    )

    assert len(owner._open_plot_dialogs) == 1
    plot_dialogue: PlotDialogue = owner._open_plot_dialogs[0]
    assert plot_dialogue.chart.get_series_count() == series_count
    assert not plot_dialogue.chart._rhi_line_enabled
    assert plot_dialogue.chart._rhi_line_widget is None

    close_plot_dialogue(dialog=plot_dialogue, app=app)
    owner.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    app.processEvents()


def test_plot_dialogue_native_close_releases_chart_once(
        qt_app: QtWidgets.QApplication) -> None:
    """Release chart resources when the native window close button is used.

    :param qt_app: Shared Qt application fixture.
    :return: None.
    """
    dialog: PlotDialogue = PlotDialogue(title='Native close')
    chart: GraphsWidget = dialog.chart
    dialog.show()
    qt_app.processEvents()
    dialog.close()
    qt_app.processEvents()
    assert dialog._closed
    assert chart._disposed
