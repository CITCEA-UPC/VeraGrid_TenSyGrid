# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
import os
import numpy as np
from VeraGridEngine.basic_structures import Logger
import VeraGridEngine.api as gce
from VeraGridEngine.Devices.Events.dynamic_plot_entry import compare_dynamic_plots
from VeraGridEngine.Utils.Symbolic import Block
from VeraGridEngine.Utils.Symbolic.symbolic_io import compare_blocks


def test_load_save_load_dynamic_versions() -> None:
    """
    This test checks if the saving and load process is correct for old dynamic versions with legacy.

    The test consists in:
    - Loading grids created from gui and from scripting with dynamic models.
    - The grids were created in old VeraGrid versions in which some symbolism might have changed compared to the current version.

    """
    folder = os.path.join('data', 'dynamic_versions')

    if not os.path.exists(os.path.join("data", "output")):
        os.makedirs(os.path.join("data", "output"))


    for name in [
        'kundur_from_gui_6_0_22.veragrid',
        'kundur_from_gui_6_1_3.veragrid',
        'kundur_from_gui_6_1_4.veragrid',
        'kundur_from_gui_6_2_0.veragrid',
        'kundur_from_gui_6_2_11.veragrid',
        'kundur_from_gui_6_2_3.veragrid',
        'kundur_from_gui_6_2_4.veragrid',
        'kundur_from_gui_6_3_2.veragrid',
        'kundur_from_gui_6_3_4.veragrid',
        'kundur_from_gui_6_3_8.veragrid',
        'kundur_from_gui_6_4_0.veragrid',
        'kundur_from_gui_6_4_2.veragrid',
        'kundur_from_gui_6_5_0.veragrid',
        'kundur_from_gui_6_5_17.veragrid',
        'kundur_from_gui_6_5_22.veragrid',
        'kundur_from_gui_6_5_29.veragrid',
        'kundur_rms_6_0_22.veragrid',
        'kundur_rms_6_1_3.veragrid',
        'kundur_rms_6_1_4.veragrid',
        'kundur_rms_6_2_0.veragrid',
        'kundur_rms_6_2_11.veragrid',
        'kundur_rms_6_2_3.veragrid',
        'kundur_rms_6_2_4.veragrid',
        'kundur_rms_6_3_2.veragrid',
        'kundur_rms_6_3_4.veragrid',
        'kundur_rms_6_3_8.veragrid',
        'kundur_rms_6_4_0.veragrid',
        'kundur_rms_6_4_2.veragrid',
        'kundur_rms_6_5_0.veragrid',
        'kundur_rms_6_5_17.veragrid',
        'kundur_rms_6_5_22.veragrid',
        'kundur_rms_6_5_29.veragrid',
        'kundur_static_6_0_22.veragrid',
        'kundur_static_6_1_3.veragrid',
        'kundur_static_6_1_4.veragrid',
        'kundur_static_6_2_0.veragrid',
        'kundur_static_6_2_11.veragrid',
        'kundur_static_6_2_3.veragrid',
        'kundur_static_6_2_4.veragrid',
        'kundur_static_6_3_2.veragrid',
        'kundur_static_6_3_4.veragrid',
        'kundur_static_6_3_8.veragrid',
        'kundur_static_6_4_0.veragrid',
        'kundur_static_6_4_2.veragrid',
        'kundur_static_6_5_0.veragrid',
        'kundur_static_6_5_17.veragrid',
        'kundur_static_6_5_22.veragrid',
        'kundur_static_6_5_29.veragrid',
        'emt_syst_from_gui_6_0_22.veragrid', #NOT WORKING
        'emt_syst_from_gui_6_1_3.veragrid',
        'emt_syst_from_gui_6_1_4.veragrid',
        'emt_syst_from_gui_6_2_0.veragrid',
        'emt_syst_from_gui_6_2_3.veragrid',
        'emt_syst_from_gui_6_2_4.veragrid',
        'emt_syst_from_gui_6_2_11.veragrid',
        'emt_syst_from_gui_6_3_2.veragrid',
        'emt_syst_from_gui_6_3_4.veragrid',
        'emt_syst_from_gui_6_3_8.veragrid',
        # 'emt_syst_from_gui_6_4_0.veragrid',
        'emt_syst_from_gui_6_4_2.veragrid',
        'emt_syst_from_gui_6_5_0.veragrid',
        'emt_syst_from_gui_6_5_17.veragrid',
        'emt_syst_from_gui_6_5_22.veragrid',
        'emt_syst_from_gui_6_5_29.veragrid',
        # 'emt_syst_from_gui_6_5_37.veragrid',
        'emt_syst_rms_6_1_3.veragrid',
        'emt_syst_rms_6_1_4.veragrid',
        'emt_syst_rms_6_2_0.veragrid',
        'emt_syst_rms_6_2_3.veragrid',
        'emt_syst_rms_6_2_4.veragrid',
        'emt_syst_rms_6_2_11.veragrid',
        'emt_syst_rms_6_3_2.veragrid',
        'emt_syst_rms_6_3_4.veragrid',
        'emt_syst_rms_6_3_8.veragrid',
        # 'emt_syst_rms_6_4_0.veragrid',
        'emt_syst_rms_6_4_2.veragrid',
        'emt_syst_rms_6_5_0.veragrid',
        'emt_syst_rms_6_5_17.veragrid',
        'emt_syst_rms_6_5_22.veragrid',
        # 'emt_syst_rms_6_5_29.veragrid',
        'emt_syst_rms_6_5_37.veragrid',
        'emt_syst_static_6_1_3.veragrid',
        'emt_syst_static_6_1_4.veragrid',
        'emt_syst_static_6_2_0.veragrid',
        'emt_syst_static_6_2_3.veragrid',
        'emt_syst_static_6_2_4.veragrid',
        'emt_syst_static_6_2_11.veragrid',
        'emt_syst_static_6_3_2.veragrid',
        'emt_syst_static_6_3_4.veragrid',
        'emt_syst_static_6_3_8.veragrid',
        # 'emt_syst_static_6_4_0.veragrid',
        'emt_syst_static_6_4_2.veragrid',
        'emt_syst_static_6_5_0.veragrid',
        'emt_syst_static_6_5_17.veragrid',
        'emt_syst_static_6_5_22.veragrid',
        # 'emt_syst_static_6_5_29.veragrid',
        'emt_syst_static_6_5_37.veragrid',

    ]:

        print(f"Testing file {name}")
        fname = os.path.join(folder, name)

        # open the main grid
        grid1 = gce.open_file(fname)

        name, ext = os.path.splitext(os.path.basename(fname))

        fname2 = os.path.join("data", "output", name + '_to_save.veragrid')

        gce.save_file(grid=grid1, filename=fname2)

        # open the main grid again
        grid2 = gce.open_file(fname2)

        # compare the original grid with the saved one to check that they are equal
        equal, logger = grid1.compare_circuits(grid2, detailed_profile_comparison=True)

        if not equal:
            logger.print()

        # asset for failing
        assert equal

        # if all ok, we can delete the test file
        os.remove(fname2)

