# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""Experimental GPU-backed line plots using Qt's QWidget RHI surface."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.support import VoidPtr


class RhiLineChartWidget(QtWidgets.QRhiWidget):
    """Render line series from one contiguous GPU vertex buffer."""

    __slots__ = ("_vertices", "_series_lengths", "_series_first_vertices", "_series_count",
                 "_buffer", "_buffer_size", "_pipeline", "_rhi",
                 "_vertex_count", "_data_revision", "_uploaded_revision", "_bindings",
                 "_background_color", "_disposed")

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        """Create an empty chart without allocating native graphics resources.

        :param parent: Optional Qt widget owner.
        :return: None.
        """
        super().__init__(parent)
        self._vertices: np.ndarray = np.zeros((0, 5), dtype=np.float32)
        self._series_lengths: np.ndarray = np.zeros(0, dtype=np.int32)
        self._series_first_vertices: np.ndarray = np.zeros(0, dtype=np.int32)
        self._series_count: int = 0
        self._buffer: QtGui.QRhiBuffer | None = None
        self._buffer_size: int = 0
        self._pipeline: QtGui.QRhiGraphicsPipeline | None = None
        self._rhi: QtGui.QRhi | None = None
        self._vertex_count: int = 0
        self._data_revision: int = 0
        self._uploaded_revision: int = -1
        self._bindings: QtGui.QRhiShaderResourceBindings | None = None
        self._background_color: QtGui.QColor = QtGui.QColor("#f7fbff")
        self._disposed: bool = False

    def set_background_color(self, color: QtGui.QColor) -> None:
        """Set the plot fill used when QRhi clears its opaque child surface.

        :param color: Current themed chart plot color.
        :return: None.
        """
        if self._background_color != color:
            self._background_color = QtGui.QColor(color)
            if not self._disposed:
                self.update()
            else:
                pass
        else:
            pass

    def set_vertices(self,
                     vertices: np.ndarray,
                     series_lengths: Sequence[int],
                     series_colors: Sequence[QtGui.QColor]) -> bool:
        """Replace the plot with finite clip-space line geometry.

        :param vertices: XY vertex coordinates shaped as ``(points, 2)``.
        :param series_lengths: Vertex count for each contiguous line.
        :param series_colors: Color for each line.
        :return: Whether the buffers contain valid line strips.
        """
        if self._disposed:
            return False
        else:
            pass
        if (vertices.ndim == 2 and vertices.shape[1] == 2
                and len(series_lengths) == len(series_colors)
                and np.all(np.isfinite(vertices))):
            lengths: np.ndarray = np.asarray(series_lengths, dtype=np.int32).copy()
            if (lengths.size > 0
                    and np.all(lengths >= 2)
                    and int(np.sum(lengths)) == vertices.shape[0]):
                packed_vertices: np.ndarray = np.empty((vertices.shape[0], 5), dtype=np.float32)
                packed_vertices[:, :2] = vertices
                first_vertices: np.ndarray = np.empty(lengths.size, dtype=np.int32)
                vertex_offset: int = 0
                series_index: int
                for series_index in range(lengths.size):
                    line_color: QtGui.QColor = series_colors[series_index]
                    line_length: int = int(lengths[series_index])
                    first_vertices[series_index] = vertex_offset
                    packed_vertices[vertex_offset:vertex_offset + line_length, 2] = line_color.redF()
                    packed_vertices[vertex_offset:vertex_offset + line_length, 3] = line_color.greenF()
                    packed_vertices[vertex_offset:vertex_offset + line_length, 4] = line_color.blueF()
                    vertex_offset += line_length
                self._vertices = np.ascontiguousarray(packed_vertices)
                self._series_lengths = lengths
                self._series_first_vertices = first_vertices
                self._series_count = int(lengths.size)
                self._vertex_count = int(vertices.shape[0])
                self._data_revision += 1
                self.update()
                return True
            else:
                self._clear_data()
        else:
            self._clear_data()
        return False

    def _clear_data(self) -> None:
        """Drop stale CPU chart data and schedule a blank frame.

        :return: None.
        """
        self._vertices = np.zeros((0, 5), dtype=np.float32)
        self._series_lengths = np.zeros(0, dtype=np.int32)
        self._series_first_vertices = np.zeros(0, dtype=np.int32)
        self._series_count = 0
        self._vertex_count = 0
        self._data_revision += 1
        if not self._disposed:
            self.update()
        else:
            pass

    def clear_data(self) -> None:
        """Release retained CPU vertices while keeping reusable GPU resources.

        :return: None.
        """
        if not self._disposed:
            self._clear_data()
        else:
            pass

    def dispose(self) -> None:
        """Release CPU and GPU resources before the chart owner is destroyed.

        :return: None.
        """
        if not self._disposed:
            self.hide()
            self._disposed = True
            self._clear_data()
            self._destroy_graphics_resources()
            self._rhi = None
        else:
            pass

    def initialize(self, command_buffer: QtGui.QRhiCommandBuffer) -> None:
        """Create a graphics pipeline for the current QRhi instance.

        :param command_buffer: Command buffer provided by QRhiWidget.
        :return: None.
        """
        _ = command_buffer
        current_rhi: QtGui.QRhi | None = self.rhi()
        if not self._disposed and current_rhi is not self._rhi:
            self._destroy_graphics_resources()
            self._rhi = current_rhi
            if current_rhi is not None:
                self._create_pipeline(current_rhi)
            else:
                pass
        else:
            pass

    def _create_pipeline(self, rhi: QtGui.QRhi) -> None:
        """Build a line-strip pipeline from packaged shader packs.

        :param rhi: Active renderer instance owned by Qt.
        :return: None.
        """
        shader_directory: Path = Path(__file__).with_name("shaders")
        vertex_shader: QtGui.QShader | None = self._load_shader(shader_directory / "line.vert.qsb")
        fragment_shader: QtGui.QShader | None = self._load_shader(shader_directory / "line.frag.qsb")
        if vertex_shader is not None and fragment_shader is not None:
            bindings: QtGui.QRhiShaderResourceBindings = rhi.newShaderResourceBindings()
            bindings.setBindings([])
            pipeline: QtGui.QRhiGraphicsPipeline = rhi.newGraphicsPipeline()
            stages: list[QtGui.QRhiShaderStage] = [
                QtGui.QRhiShaderStage(QtGui.QRhiShaderStage.Type.Vertex, vertex_shader),
                QtGui.QRhiShaderStage(QtGui.QRhiShaderStage.Type.Fragment, fragment_shader),
            ]
            input_layout: QtGui.QRhiVertexInputLayout = QtGui.QRhiVertexInputLayout()
            input_layout.setBindings([QtGui.QRhiVertexInputBinding(20)])
            input_layout.setAttributes([
                QtGui.QRhiVertexInputAttribute(
                    0, 0, QtGui.QRhiVertexInputAttribute.Format.Float2, 0
                ),
                QtGui.QRhiVertexInputAttribute(
                    0, 1, QtGui.QRhiVertexInputAttribute.Format.Float3, 8
                ),
            ])
            pipeline.setShaderStages(stages)
            pipeline.setShaderResourceBindings(bindings)
            pipeline.setVertexInputLayout(input_layout)
            pipeline.setTopology(QtGui.QRhiGraphicsPipeline.Topology.LineStrip)
            pipeline.setRenderPassDescriptor(self.renderTarget().renderPassDescriptor())
            if bindings.create() and pipeline.create():
                self._pipeline = pipeline
                self._bindings = bindings
            else:
                pipeline.destroy()
                bindings.destroy()
                self.renderFailed.emit()
        else:
            self.renderFailed.emit()

    @staticmethod
    def _load_shader(shader_path: Path) -> QtGui.QShader | None:
        """Load a compiled Qt shader pack from this widget's package directory.

        :param shader_path: Compiled shader asset path.
        :return: Valid shader, or ``None`` when the asset is unavailable.
        """
        shader_file: QtCore.QFile = QtCore.QFile(str(shader_path))
        if shader_file.open(QtCore.QIODevice.OpenModeFlag.ReadOnly):
            shader: QtGui.QShader = QtGui.QShader.fromSerialized(shader_file.readAll())
            shader_file.close()
            if shader.isValid():
                return shader
            else:
                return None
        else:
            return None

    def render(self, command_buffer: QtGui.QRhiCommandBuffer) -> None:
        """Upload changed CPU data once, then draw each series as a line strip.

        :param command_buffer: Command buffer provided by QRhiWidget.
        :return: None.
        """
        render_target: QtGui.QRhiRenderTarget | None = self.renderTarget()
        if not self._disposed and render_target is not None:
            update_batch: QtGui.QRhiResourceUpdateBatch | None = None
            if self._pipeline is not None and self._uploaded_revision != self._data_revision:
                if self._vertex_count > 0:
                    rhi: QtGui.QRhi | None = self.rhi()
                    if rhi is not None:
                        if self._buffer is None or self._buffer_size != self._vertices.nbytes:
                            if self._buffer is not None:
                                self._buffer.destroy()
                                self._buffer = None
                                self._buffer_size = 0
                            else:
                                pass
                            buffer: QtGui.QRhiBuffer = rhi.newBuffer(
                                QtGui.QRhiBuffer.Type.Dynamic,
                                QtGui.QRhiBuffer.UsageFlag.VertexBuffer,
                                int(self._vertices.nbytes),
                            )
                            if buffer.create():
                                self._buffer = buffer
                                self._buffer_size = int(self._vertices.nbytes)
                            else:
                                buffer.destroy()
                                self.renderFailed.emit()
                        else:
                            pass
                        if self._buffer is not None:
                            update_batch = rhi.nextResourceUpdateBatch()
                            update_batch.updateDynamicBuffer(
                                self._buffer,
                                0,
                                int(self._vertices.nbytes),
                                VoidPtr(self._vertices.ctypes.data, int(self._vertices.nbytes)),
                            )
                            self._uploaded_revision = self._data_revision
                        else:
                            pass
                    else:
                        pass
                else:
                    self._uploaded_revision = self._data_revision
            else:
                pass
            clear_value: QtGui.QRhiDepthStencilClearValue = QtGui.QRhiDepthStencilClearValue(1.0, 0)
            command_buffer.beginPass(render_target, self._background_color, clear_value, update_batch)
            if self._pipeline is not None and self._buffer is not None:
                output_size: QtCore.QSize = render_target.pixelSize()
                viewport: QtGui.QRhiViewport = QtGui.QRhiViewport(
                    0.0, 0.0, float(output_size.width()), float(output_size.height())
                )
                command_buffer.setViewport(viewport)
                vertex_buffer: tuple[QtGui.QRhiBuffer, int] = (self._buffer, 0)
                command_buffer.setVertexInput(0, [vertex_buffer])
                series_index: int = 0
                while series_index < self._series_count:
                    command_buffer.setGraphicsPipeline(self._pipeline)
                    command_buffer.draw(int(self._series_lengths[series_index]), 1,
                                        int(self._series_first_vertices[series_index]), 0)
                    series_index += 1
            else:
                pass
            command_buffer.endPass()
        else:
            pass

    def releaseResources(self) -> None:
        """Destroy GPU resources while Qt's QRhi context is still available.

        :return: None.
        """
        self._destroy_graphics_resources()
        self._rhi = None

    def _destroy_graphics_resources(self) -> None:
        """Release owned buffer and pipeline handles idempotently.

        :return: None.
        """
        if self._buffer is not None:
            self._buffer.destroy()
            self._buffer = None
        else:
            pass
        self._buffer_size = 0
        if self._pipeline is not None:
            self._pipeline.destroy()
            self._pipeline = None
        else:
            pass
        if self._bindings is not None:
            self._bindings.destroy()
            self._bindings = None
        else:
            pass
        self._uploaded_revision = -1
