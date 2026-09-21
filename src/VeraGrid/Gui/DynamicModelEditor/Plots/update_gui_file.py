# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
"""Regenerate the dynamic plots page Python UI module."""

import os

from VeraGrid.Gui.update_gui_common import convert_ui_file


if __name__ == '__main__':
    uic_cmd: str = 'pyside6-uic'
    if os.name == 'nt':
        uic_cmd += '.exe'
    else:
        pass
    convert_ui_file(source='dynamic_plots_page_ui.ui', uic_cmd=uic_cmd)
