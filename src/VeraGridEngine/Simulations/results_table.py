# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from typing import Union, List
import numpy as np
import pandas as pd
from VeraGridEngine.enumerations import ResultTypes, DeviceType, ResultTablePlotType
from VeraGridEngine.basic_structures import StrVec, Mat, Vec
from VeraGridEngine.Devices.types import ALL_DEV_TYPES




class ResultsTable:
    """
    Class to populate a Qt table view with data from the results
    """
    __slots__ = (
        "data_c",
        "cols_c",
        "index_c",
        "editable",
        "editable_min_idx",
        "palette",
        "title",
        "x_label",
        "y_label",
        "units",
        "r",
        "c",
        "isDate",
        "decimals",
        "format_string",
        "formatter",
        "cols_device_type",
        "idx_device_type",
        "_col_devices",
        "_idx_devices",
        "plot_type",
        "damping_ratio_boundary",
        "plot_title",
        "complex_plot_x_column",
        "complex_plot_y_columns",
        "complex_plot_y_scales",
    )

    def __init__(self,
                 data: Union[Mat, Vec],
                 columns: StrVec,
                 index: StrVec | pd.DatetimeIndex,
                 title: str,
                 cols_device_type: DeviceType,
                 idx_device_type: DeviceType,
                 units: str = "",
                 xlabel: str = "",
                 ylabel: str = "",
                 editable: bool = False,
                 palette: object | None = None,
                 editable_min_idx: int = -1,
                 decimals: int = 6,
                 plot_type: ResultTablePlotType = ResultTablePlotType.SERIES,
                 damping_ratio_boundary: float | None = None,
                 plot_title: str | None = None,
                 complex_plot_x_column: str | None = None,
                 complex_plot_y_columns: StrVec | None = None,
                 complex_plot_y_scales: Vec | None = None) -> None:
        """
        ResultsTable constructor
        :param data:
        :param columns:
        :param index:
        :param palette:
        :param title:
        :param xlabel:
        :param ylabel:
        :param editable:
        :param editable_min_idx:
        :param decimals:
        :param plot_type: Graphical representation used when plotting this table.
        :param damping_ratio_boundary: Optional damping-ratio guide for complex-point plots.
        :param plot_title: Optional title used only by the plot representation.
        :param complex_plot_x_column: Column name used as the real coordinate.
        :param complex_plot_y_columns: Allowed imaginary-coordinate column names.
        :param complex_plot_y_scales: Overlay scales corresponding to the allowed imaginary columns.
        :return: None.
        """
        if data.ndim == 1:
            # assert compatible dimensions
            assert len(data) == len(index)

            self.data_c = data.reshape(-1, 1)

        elif data.ndim == 2:
            # assert compatible dimensions
            assert data.shape[0] == len(index)
            assert data.shape[1] == len(columns)

            self.data_c = data
        else:
            raise Exception("Unsupported number of dimensions {}".format(data.ndim))

        self.cols_c = columns
        self.index_c = index

        self.editable = editable
        self.editable_min_idx = editable_min_idx
        self.palette = palette
        self.title = title
        self.x_label = xlabel
        self.y_label = ylabel
        self.units = units
        self.r, self.c = self.data_c.shape
        self.isDate = False
        if self.r > 0 and self.c > 0:
            if isinstance(self.index_c[0], np.datetime64):
                self.index_c = pd.to_datetime(self.index_c)
                self.isDate = True

        self.decimals: int = decimals
        self.format_string = '.' + str(decimals) + 'f'
        self.formatter = lambda x: self.format_string % x

        self.cols_device_type: DeviceType = cols_device_type
        self.idx_device_type: DeviceType = idx_device_type

        # list of devices that match the columns or rows for filtering
        self._col_devices = list()
        self._idx_devices = list()

        # Plot semantics travel with the data so filtering and slicing retain
        # the same graphical interpretation as the source result.
        self.plot_type: ResultTablePlotType = plot_type
        self.damping_ratio_boundary: float | None = damping_ratio_boundary
        self.plot_title: str | None = plot_title
        self.complex_plot_x_column: str | None = complex_plot_x_column
        if complex_plot_y_columns is None:
            self.complex_plot_y_columns: StrVec = np.empty(0, dtype=np.str_)
        else:
            self.complex_plot_y_columns = np.asarray(complex_plot_y_columns, dtype=np.str_)
        if complex_plot_y_scales is None:
            self.complex_plot_y_scales: Vec = np.empty(0, dtype=float)
        else:
            self.complex_plot_y_scales = np.asarray(complex_plot_y_scales, dtype=float)

        if len(self.complex_plot_y_columns) == len(self.complex_plot_y_scales):
            pass
        else:
            raise ValueError("Complex plot Y columns and scales must have the same length.")

    @property
    def data(self):
        """
        Backward-compatible alias for the table numeric payload.

        :return: Table data array.
        """
        return self.data_c

    @property
    def col_devices(self):
        """

        :return:
        """
        return self._col_devices

    @property
    def idx_devices(self):
        """

        :return:
        """
        return self._idx_devices

    def set_col_devices(self, devices_list: List[ALL_DEV_TYPES]):
        """
        Set the list of devices that matches the results for filtering
        :param devices_list:
        """
        self._col_devices = devices_list

    def set_idx_devices(self, devices_list: List[ALL_DEV_TYPES]):
        """
        Set the list of devices that matches the results for filtering
        :param devices_list:
        """
        self._idx_devices = devices_list

    def transpose(self):
        """
        Transpose the results in-place
        """
        self.data_c = self.data_c.copy().transpose()
        self.r, self.c = self.data_c.shape
        self.x_label, self.y_label = self.y_label, self.x_label
        self.cols_c, self.index_c = self.index_c, self.cols_c
        self._col_devices, self._idx_devices = self._idx_devices, self._col_devices

        # Structured complex plots assign semantic roles to rows and columns.
        # Transposition changes those roles, so the transformed table reverts
        # explicitly to the ordinary series representation.
        self.plot_type = ResultTablePlotType.SERIES
        self.damping_ratio_boundary = None
        self.plot_title = None
        self.complex_plot_x_column = None
        self.complex_plot_y_columns = np.empty(0, dtype=np.str_)
        self.complex_plot_y_scales = np.empty(0, dtype=float)

    def sort_column(self, c: int, max_to_min: bool = True):
        """

        :param c:
        :param max_to_min:
        :return:
        """
        try:
            sorting_arr = self.data_c[:, c].astype(float)
        except ValueError:
            print("Not a float column...")
            sorting_arr = self.data_c[:, c]

        if max_to_min:
            idx = sorting_arr.argsort()[::-1]
        else:
            idx = sorting_arr.argsort()

        self.data_c = self.data_c[idx]
        self.index_c = self.index_c[idx]

    def slice_cols(self, col_idx) -> "ResultsTable":
        """
        Make column slicing
        :param col_idx: indices of the columns
        :return: Nothing
        """
        sliced_model = ResultsTable(data=self.data_c[:, col_idx],
                                    columns=np.array([self.cols_c[i] for i in col_idx]),
                                    index=np.array(self.index_c),
                                    palette=None,
                                    title=self.title,
                                    xlabel=self.x_label,
                                    ylabel=self.y_label,
                                    units=self.units,
                                    editable=self.editable,
                                    editable_min_idx=self.editable_min_idx,
                                    decimals=self.decimals,
                                    cols_device_type=self.cols_device_type,
                                    idx_device_type=self.idx_device_type,
                                    plot_type=self.plot_type,
                                    damping_ratio_boundary=self.damping_ratio_boundary,
                                    plot_title=self.plot_title,
                                    complex_plot_x_column=self.complex_plot_x_column,
                                    complex_plot_y_columns=self.complex_plot_y_columns,
                                    complex_plot_y_scales=self.complex_plot_y_scales)

        return sliced_model

    def slice_rows(self, idx) -> "ResultsTable":
        """
        Make rows slicing
        :param idx: indices of the columns
        :return: Nothing
        """
        sliced_model = ResultsTable(data=self.data_c[idx, :],
                                    columns=self.cols_c,
                                    index=np.array([self.index_c[i] for i in idx]),
                                    palette=None,
                                    title=self.title,
                                    xlabel=self.x_label,
                                    ylabel=self.y_label,
                                    units=self.units,
                                    editable=self.editable,
                                    editable_min_idx=self.editable_min_idx,
                                    decimals=self.decimals,
                                    cols_device_type=self.cols_device_type,
                                    idx_device_type=self.idx_device_type,
                                    plot_type=self.plot_type,
                                    damping_ratio_boundary=self.damping_ratio_boundary,
                                    plot_title=self.plot_title,
                                    complex_plot_x_column=self.complex_plot_x_column,
                                    complex_plot_y_columns=self.complex_plot_y_columns,
                                    complex_plot_y_scales=self.complex_plot_y_scales)

        return sliced_model

    def slice_all(self, row_idx, col_idx) -> "ResultsTable":
        """
        Make rows slicing
        :param row_idx: indices of the rows
        :param col_idx: indices of the columns
        :return: ResultsTable
        """
        sliced_model = ResultsTable(data=self.data_c[row_idx, :][:, col_idx],
                                    columns=np.array([self.cols_c[i] for i in col_idx]),
                                    index=np.array([self.index_c[i] for i in row_idx]),
                                    palette=None,
                                    title=self.title,
                                    xlabel=self.x_label,
                                    ylabel=self.y_label,
                                    units=self.units,
                                    editable=self.editable,
                                    editable_min_idx=self.editable_min_idx,
                                    decimals=self.decimals,
                                    cols_device_type=self.cols_device_type,
                                    idx_device_type=self.idx_device_type,
                                    plot_type=self.plot_type,
                                    damping_ratio_boundary=self.damping_ratio_boundary,
                                    plot_title=self.plot_title,
                                    complex_plot_x_column=self.complex_plot_x_column,
                                    complex_plot_y_columns=self.complex_plot_y_columns,
                                    complex_plot_y_scales=self.complex_plot_y_scales)
        return sliced_model

    def search_in_columns(self, txt):
        """
        Search stuff
        :param txt:
        :return:
        """
        idx = list()
        txt2 = str(txt).lower()
        for i, val in enumerate(self.cols_c):
            if txt2 in val.lower():
                idx.append(i)
        idx = np.array(idx, dtype=int)
        if len(idx) > 0:
            return self.slice_cols(idx)
        else:
            return None

    def search_in_rows(self, txt):
        """
        Search stuff
        :param txt:
        :return:
        """
        idx = list()
        txt2 = str(txt).lower()
        for i, val in enumerate(self.index_c):
            if txt2 in str(val).lower():
                idx.append(i)
        idx = np.array(idx, dtype=int)
        if len(idx) > 0:
            return self.slice_rows(idx)
        else:
            return None

    def copy_to_column(self, row: int, col: int):
        """
        Copies one value to all the column
        @param row: Row of the value
        @param col: Column of the value
        @return: Nothing
        """
        self.data_c[:, col] = self.data_c[row, col]

    def is_complex(self) -> bool:
        """
        Is the data complex?
        :return:
        """
        return self.data_c.dtype == complex

    def get_data(self):
        """
        Returns: index, columns, data
        """
        n = len(self.cols_c)

        if n > 0:
            # gather values
            if isinstance(self.cols_c, pd.Index):
                names = self.cols_c.values

                if len(names) > 0:
                    if isinstance(names[0], ResultTypes):
                        names = [str(val) for val in names]
            else:
                names = [str(val) for val in self.cols_c]

            values = self.data_c

            return self.index_c, names, values
        else:
            # there are no elements
            return self.index_c, list(), self.data_c

    def convert_to_cdf(self):
        """
        Convert the data in-place to CDF based
        :return:
        """

        # calculate the proportional values of samples
        n = self.data_c.shape[0]
        if n > 1:
            self.index_c = np.arange(n, dtype=float) / (n - 1)
        else:
            self.index_c = np.arange(n, dtype=float)

        for i in range(self.data_c.shape[1]):
            self.data_c[:, i] = np.sort(self.data_c[:, i].copy(), axis=0)

        self.x_label = 'Probability of value<=x'

        # Independent column sorting destroys complex coordinate pairs and
        # eigenvector component relationships.
        self.plot_type = ResultTablePlotType.SERIES
        self.damping_ratio_boundary = None
        self.plot_title = None
        self.complex_plot_x_column = None
        self.complex_plot_y_columns = np.empty(0, dtype=np.str_)
        self.complex_plot_y_scales = np.empty(0, dtype=float)

    def convert_to_abs(self):
        """
        Convert the data to abs
        :return:
        """
        try:
            self.data_c = np.abs(self.data_c)

            # Magnitudes no longer contain a real-imaginary plane, so their
            # natural representation is the existing series plot.
            self.plot_type = ResultTablePlotType.SERIES
            self.damping_ratio_boundary = None
            self.plot_title = None
            self.complex_plot_x_column = None
            self.complex_plot_y_columns = np.empty(0, dtype=np.str_)
            self.complex_plot_y_scales = np.empty(0, dtype=float)
        except TypeError:
            print('Could not convert to abs :/')

    def to_df(self) -> pd.DataFrame:
        """
        get DataFrame
        """
        index, columns, data = self.get_data()

        return pd.DataFrame(data=data, index=index, columns=columns)

    def save_to_excel(self, file_name):
        """
        save data to excel
        :param file_name:
        """
        self.to_df().to_excel(file_name)

    def save_to_csv(self, file_name):
        """
        Save data to csv
        :param file_name:
        """
        self.to_df().to_csv(file_name)

    def get_data_frame(self):
        """
        Save data to csv
        """
        index, columns, data = self.get_data()
        return pd.DataFrame(data=data, index=index, columns=columns)
