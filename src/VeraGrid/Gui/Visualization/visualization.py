# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""Native Qt colour helpers used by diagram widgets."""

from __future__ import annotations

from collections.abc import Sequence

from PySide6 import QtGui


class NativeColorMap:
    """Interpolate fixed Qt colour stops for native Qt rendering."""

    __slots__ = ("_positions", "_colors")

    def __init__(self, stops: Sequence[tuple[float, str]]) -> None:
        """Copy normalized colour stops into Qt value types.

        :param stops: Ordered `(position, colour)` definitions.
        :return: None.
        """
        self._positions: list[float] = list()
        self._colors: list[QtGui.QColor] = list()
        position: float
        color_name: str
        for position, color_name in stops:
            self._positions.append(float(position))
            self._colors.append(QtGui.QColor(color_name))

    def __call__(self, value: float) -> tuple[float, float, float, float]:
        """Return a normalized RGBA tuple for a scalar value.

        :param value: Normalized scalar to map to a colour.
        :return: Red, green, blue, and alpha components in `[0, 1]`.
        """
        bounded_value: float = min(max(float(value), 0.0), 1.0)
        stop_index: int
        for stop_index in range(1, len(self._positions)):
            if bounded_value <= self._positions[stop_index]:
                start_position: float = self._positions[stop_index - 1]
                end_position: float = self._positions[stop_index]
                if end_position > start_position:
                    fraction: float = (bounded_value - start_position) / (end_position - start_position)
                else:
                    fraction = 0.0
                start_color: QtGui.QColor = self._colors[stop_index - 1]
                end_color: QtGui.QColor = self._colors[stop_index]
                return (
                    start_color.redF() + (end_color.redF() - start_color.redF()) * fraction,
                    start_color.greenF() + (end_color.greenF() - start_color.greenF()) * fraction,
                    start_color.blueF() + (end_color.blueF() - start_color.blueF()) * fraction,
                    start_color.alphaF() + (end_color.alphaF() - start_color.alphaF()) * fraction,
                )
            else:
                pass

        final_color: QtGui.QColor = self._colors[len(self._colors) - 1]
        return final_color.redF(), final_color.greenF(), final_color.blueF(), final_color.alphaF()


def get_voltage_color_map() -> NativeColorMap:
    """Return the diagram voltage colour scale.

    :return: Native callable voltage colour map.
    """
    return NativeColorMap(stops=(
        (0.0, "black"),
        (0.8 / 1.2, "blue"),
        (1.0 / 1.2, "green"),
        (1.05 / 1.2, "orange"),
        (1.0, "red"),
    ))


def get_loading_color_map() -> NativeColorMap:
    """Return the diagram loading colour scale.

    :return: Native callable loading colour map.
    """
    return NativeColorMap(stops=(
        (0.0, "gray"),
        (0.8 / 1.5, "green"),
        (1.2 / 1.5, "orange"),
        (1.0, "red"),
    ))


def has_null_coordinates(coordinates: Sequence[tuple[float, float]]) -> bool:
    """Return whether any coordinate lies at the null origin.

    :param coordinates: Coordinates to inspect.
    :return: Whether one coordinate has both components set to zero.
    """
    x_value: float
    y_value: float
    for x_value, y_value in coordinates:
        if x_value == 0.0 and y_value == 0.0:
            return True
        else:
            pass
    return False


def get_n_colours(n: int, colormap: str = "") -> list[tuple[float, float, float, float]]:
    """Return visually separated HSV colours without an external renderer.

    :param n: Number of colours to create.
    :param colormap: Retained compatibility argument ignored by Qt colouring.
    :return: Normalized RGBA tuples.
    """
    _ = colormap
    colors: list[tuple[float, float, float, float]] = list()
    color_index: int
    if n > 0:
        for color_index in range(n):
            color: QtGui.QColor = QtGui.QColor.fromHsvF(float(color_index) / float(n), 0.78, 0.92)
            colors.append((color.redF(), color.greenF(), color.blueF(), color.alphaF()))
    else:
        pass
    return colors
