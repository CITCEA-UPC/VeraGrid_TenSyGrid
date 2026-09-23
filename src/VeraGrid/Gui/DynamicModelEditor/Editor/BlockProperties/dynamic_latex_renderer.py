# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""
LaTeX equation rendering for the Dynamic Block Editor equations table.

Architecture::

    LaTeX string
        ↓
    LatexRenderer  (Qt text → QPixmap + QSize, cached)
        ↓
    LatexEquationDelegate  (sizeHint + paint, no scaling)
        ↓
    QTableView  (ResizeToContents on vertical header)
        ↓
    Row height adapts to the real equation size.

This module lives entirely on the GUI side and never touches the
symbolic expression tree.
"""

from __future__ import annotations

from html import escape

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap


class RenderedEquation:
    """Rendered equation image and its preferred Qt cell size."""

    __slots__ = ("_pixmap", "_size", "_uses_mathtext")

    def __init__(self,
                 pixmap: QPixmap,
                 size: QtCore.QSize,
                 uses_mathtext: bool) -> None:
        """Store one immutable rendering result.

        :param pixmap: Equation image.
        :param size: Preferred table-cell size.
        :param uses_mathtext: Whether a mathematical text parser was used.
        :return: None.
        """
        self._pixmap: QPixmap = pixmap
        self._size: QtCore.QSize = size
        self._uses_mathtext: bool = uses_mathtext

    def get_pixmap(self) -> QPixmap:
        """
        :return: The rendered equation image.
        """
        return self._pixmap

    def get_size(self) -> QtCore.QSize:
        """
        :return: The preferred table-cell size.
        """
        return self._size

    def get_uses_mathtext(self) -> bool:
        """
        :return: Whether a mathematical text parser was used.
        """
        return self._uses_mathtext


class RenderedSvgEquation:
    """Self-contained vector equation and its intrinsic document size."""

    __slots__ = ("_data", "_size", "_uses_mathtext")

    def __init__(self,
                 data: QtCore.QByteArray,
                 size: QtCore.QSize,
                 uses_mathtext: bool) -> None:
        """Store one immutable SVG rendering result.

        :param data: Complete SVG document payload.
        :param size: Intrinsic logical display size.
        :param uses_mathtext: Whether a mathematical text parser was used.
        :return: None.
        """
        self._data: QtCore.QByteArray = data
        self._size: QtCore.QSize = size
        self._uses_mathtext: bool = uses_mathtext

    def get_data(self) -> QtCore.QByteArray:
        """
        :return: The self-contained SVG document payload.
        """
        return self._data

    def get_size(self) -> QtCore.QSize:
        """
        :return: The intrinsic SVG display size.
        """
        return self._size

    def get_uses_mathtext(self) -> bool:
        """
        :return: Whether a mathematical text parser was used.
        """
        return self._uses_mathtext


def normalize_mathtext_latex(latex: str) -> str:
    """Return LaTeX source unchanged for compatibility with existing callers.

    :param latex: LaTeX emitted by the symbolic printer.
    :return: Original LaTeX source.
    """
    return latex


# ---------------------------------------------------------------------------
# LatexRenderer  –  Qt text → QPixmap with cache
# ---------------------------------------------------------------------------


class LatexRenderer:
    """
    Render equation source into a :class:`QPixmap` using Qt text APIs.

    The pixmap has a fully transparent background so Qt's native
    selection / hover / alternating-row painting shows through.

    The source is displayed literally because the GUI no longer embeds a
    plotting renderer. A cache keyed by the raw source avoids re-rendering
    the same equation on every repaint / sizeHint query.
    """

    __slots__ = (
        "_font_size",
        "_dpi",
        "_minimum_row_height",
        "_maximum_mathtext_length",
        "_maximum_fallback_width",
        "_cache",
        "_svg_cache",
    )

    def __init__(self, font_size: int = 12, dpi: int = 96) -> None:
        """Create a renderer with an isolated image cache.

        :param font_size: Equation font size in points.
        :param dpi: Raster rendering resolution.
        :return: None.
        """
        self._font_size: int = font_size
        self._dpi: int = dpi
        self._minimum_row_height: int = 24
        self._maximum_mathtext_length: int = 2000
        self._maximum_fallback_width: int = 960
        self._cache: dict[str, RenderedEquation] = dict()
        self._svg_cache: dict[str, RenderedSvgEquation] = dict()

    # -- public API --------------------------------------------------------

    def render(self, latex: str) -> RenderedEquation:
        """
        Return a cached :class:`RenderedEquation` for *latex*.

        If not yet in the cache it is rendered with Qt and stored.

        :param latex: LaTeX source handled by the operation.
        :return: A cached :class:`RenderedEquation` for *latex*.
        """
        cached: RenderedEquation | None = self._cache.get(latex, None)
        if cached is not None:
            return cached
        else:
            if len(latex) > self._maximum_mathtext_length:
                preview_message: str = (
                    f"Equation is too large for graphical preview ({len(latex)} characters). "
                    "See Python code."
                )
                rendered: RenderedEquation = self._render_to_pixmap(
                    preview_message,
                    False,
                    self._maximum_fallback_width,
                )
            else:
                rendered = self._render_to_pixmap(latex, False)
            self._cache[latex] = rendered
            return rendered

    def render_plain_text(self, text: str) -> RenderedEquation:
        """Render literal text with Qt's native font renderer.

        This is used by PDF metadata and source previews.

        :param text: Literal text to rasterize.
        :return: Rendered text image and preferred size.
        """
        cache_key: str = f"__plain__:{text}"
        cached: RenderedEquation | None = self._cache.get(cache_key, None)
        if cached is not None:
            return cached
        else:
            rendered: RenderedEquation = self._render_to_pixmap(text, False)
            self._cache[cache_key] = rendered
            return rendered

    def render_svg(self, latex: str) -> RenderedSvgEquation:
        """Return a cached SVG containing the literal equation source.

        :param latex: LaTeX source.
        :return: Self-contained SVG text and its intrinsic logical size.
        """
        cached: RenderedSvgEquation | None = self._svg_cache.get(latex, None)
        if cached is not None:
            return cached
        else:
            if len(latex) > self._maximum_mathtext_length:
                preview_message: str = (
                    f"Equation is too large for graphical preview ({len(latex)} characters). "
                    "See Python code."
                )
                rendered: RenderedSvgEquation = self._render_to_svg(
                    preview_message,
                    False,
                    self._maximum_fallback_width,
                )
            else:
                rendered = self._render_to_svg(latex, False)
            self._svg_cache[latex] = rendered
            return rendered

    def render_plain_text_svg(self, text: str) -> RenderedSvgEquation:
        """Return literal text as SVG text.

        :param text: Literal label text.
        :return: Self-contained SVG label and its logical size.
        """
        cache_key: str = f"__plain_svg__:{text}"
        cached: RenderedSvgEquation | None = self._svg_cache.get(cache_key, None)
        if cached is not None:
            return cached
        else:
            rendered: RenderedSvgEquation = self._render_to_svg(text, False)
            self._svg_cache[cache_key] = rendered
            return rendered

    def invalidate(self, latex: str) -> None:
        """
        Remove one entry from the cache (called after an edit).

        :param latex: LaTeX source handled by the operation.
        :return: None.
        """
        self._cache.pop(latex, None)
        self._svg_cache.pop(latex, None)

    def clear(self) -> None:
        """
        Drop the entire cache.

        :return: None.
        """
        self._cache.clear()
        self._svg_cache.clear()

    def get_maximum_mathtext_length(self) -> int:
        """
        Return the maximum equation source length shown in the preview.

        :return: Maximum equation source length shown in the preview.
        """
        return self._maximum_mathtext_length

    # -- internals ---------------------------------------------------------

    def _render_to_pixmap(
            self,
            latex: str,
            use_mathtext: bool,
            maximum_width: int | None = None,
    ) -> RenderedEquation:
        """Render equation source with a Qt painter.

        :param latex: Equation source.
        :param use_mathtext: Compatibility flag; Qt renders source literally.
        :param maximum_width: Optional logical-pixel cap for fallback messages.
        :return: Rendered Qt image and preferred size.
        """
        _ = use_mathtext
        font: QtGui.QFont = QtWidgets.QApplication.font()
        font.setPixelSize(max(1, int(round(float(self._font_size * self._dpi) / 72.0))))
        metrics: QtGui.QFontMetrics = QtGui.QFontMetrics(font)
        padding: int = 2
        draw_text: str = latex
        if maximum_width is not None:
            available_width: int = max(1, maximum_width - padding * 2)
            if metrics.horizontalAdvance(draw_text) > available_width:
                draw_text = metrics.elidedText(
                    draw_text,
                    Qt.TextElideMode.ElideRight,
                    available_width,
                )
            else:
                pass
        else:
            pass
        measured_width: int = metrics.horizontalAdvance(draw_text) + padding * 2
        if maximum_width is None:
            image_width: int = max(1, measured_width)
        else:
            image_width = max(1, min(maximum_width, measured_width))
        image_height: int = max(self._minimum_row_height, metrics.height() + padding * 2)
        image: QtGui.QImage = QtGui.QImage(
            image_width,
            image_height,
            QtGui.QImage.Format.Format_ARGB32_Premultiplied,
        )
        image.fill(QtGui.QColor(0, 0, 0, 0))
        painter: QtGui.QPainter = QtGui.QPainter(image)
        painter.setFont(font)
        painter.setPen(QtWidgets.QApplication.palette().color(QtGui.QPalette.ColorRole.Text))
        painter.drawText(padding, padding + metrics.ascent(), draw_text)
        painter.end()
        pixmap: QPixmap = QPixmap.fromImage(image)
        size: QtCore.QSize = QtCore.QSize(image_width, image_height)
        return RenderedEquation(
            pixmap=pixmap,
            size=size,
            uses_mathtext=False,
        )

    def _render_to_svg(
            self,
            latex: str,
            use_mathtext: bool,
            maximum_width: int | None = None,
    ) -> RenderedSvgEquation:
        """Render equation source as self-contained SVG text.

        :param latex: Equation source.
        :param use_mathtext: Compatibility flag; Qt renders source literally.
        :param maximum_width: Optional logical-pixel cap for fallback messages.
        :return: SVG text payload and intrinsic logical size.
        """
        _ = use_mathtext
        font: QtGui.QFont = QtWidgets.QApplication.font()
        font.setPixelSize(max(1, int(round(float(self._font_size * self._dpi) / 72.0))))
        metrics: QtGui.QFontMetrics = QtGui.QFontMetrics(font)
        padding: int = 2
        draw_text: str = latex
        if maximum_width is not None:
            available_width: int = max(1, maximum_width - padding * 2)
            if metrics.horizontalAdvance(draw_text) > available_width:
                draw_text = metrics.elidedText(
                    draw_text,
                    Qt.TextElideMode.ElideRight,
                    available_width,
                )
            else:
                pass
        else:
            pass
        measured_width: int = metrics.horizontalAdvance(draw_text) + padding * 2
        if maximum_width is None:
            image_width: int = max(1, measured_width)
        else:
            image_width = max(1, min(maximum_width, measured_width))
        image_height: int = max(1, metrics.height() + padding * 2)
        baseline: int = padding + metrics.ascent()
        font_size: int = max(1, font.pixelSize())
        svg_source: str = (
            '<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{image_width}px" height="{image_height}px" '
            f'viewBox="0 0 {image_width} {image_height}">'
            f'<text x="{padding}" y="{baseline}" '
            f'font-family="{escape(font.family(), quote=True)}" '
            f'font-size="{font_size}px" fill="#000000">{escape(draw_text)}</text></svg>'
        )
        svg_data: QtCore.QByteArray = QtCore.QByteArray(svg_source.encode("utf-8"))
        return RenderedSvgEquation(
            data=svg_data,
            size=QtCore.QSize(image_width, image_height),
            uses_mathtext=False,
        )


# ---------------------------------------------------------------------------
# LatexEquationDelegate  –  sizeHint + paint, no scaling
# ---------------------------------------------------------------------------

class LatexEquationDelegate(QtWidgets.QStyledItemDelegate):
    """
    Delegate that renders LaTeX equations in the equations table.

    Implements ``sizeHint()`` using the same cached render that
    ``paint()`` uses, so Qt can set the row height to the real
    equation size via ``ResizeToContents``.

    ``paint()`` never scales the pixmap — it centres the original-
    size image inside the cell rectangle.
    """

    __slots__ = ("_renderer",)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        """Create a delegate with an explicitly owned renderer cache.

        :param parent: Owning table widget.
        :return: None.
        """
        super().__init__(parent)
        self._renderer: LatexRenderer = LatexRenderer()

    # -- sizeHint ----------------------------------------------------------

    def sizeHint(self,
                 option: QtWidgets.QStyleOptionViewItem,
                 index: QtCore.QModelIndex) -> QtCore.QSize:
        """Return the real cached image dimensions for a table cell.

        :param option: Qt style information.
        :param index: Equation model index.
        :return: Preferred equation-cell size.
        """
        text: object = index.data(Qt.ItemDataRole.DisplayRole)
        if not text:
            return super().sizeHint(option, index)
        else:
            rendered: RenderedEquation = self._renderer.render(str(text))
            return rendered.get_size()

    # -- paint -------------------------------------------------------------

    def paint(
        self,
        painter: QtGui.QPainter,
        option: QtWidgets.QStyleOptionViewItem,
        index: QtCore.QModelIndex,
    ) -> None:
        """Paint the cached equation image over Qt's native cell background.

        :param painter: Active table-view painter.
        :param option: Qt style and geometry information.
        :param index: Equation model index.
        :return: None.
        """
        self.initStyleOption(option, index)
        style: QtWidgets.QStyle = (
            option.widget.style() if option.widget else QtWidgets.QApplication.style()
        )
        style.drawPrimitive(
            QtWidgets.QStyle.PrimitiveElement.PE_PanelItemViewItem,
            option, painter, option.widget,
        )

        text: object = index.data(Qt.ItemDataRole.DisplayRole)
        if not text:
            return
        else:
            rendered: RenderedEquation = self._renderer.render(str(text))
            pixmap: QPixmap = rendered.get_pixmap()

            rect: QtCore.QRect = option.rect
            x_pos: int = rect.x() + (rect.width() - pixmap.width()) // 2
            y_pos: int = rect.y() + (rect.height() - pixmap.height()) // 2
            painter.drawPixmap(x_pos, y_pos, pixmap)

    # -- cache management --------------------------------------------------

    def invalidate(self, latex: str) -> None:
        """
        Drop one cached entry after the underlying data changes.

        :param latex: LaTeX source handled by the operation.
        :return: None.
        """
        self._renderer.invalidate(latex)

    def clear_cache(self) -> None:
        """
        Drop the entire rendering cache.

        :return: None.
        """
        self._renderer.clear()
