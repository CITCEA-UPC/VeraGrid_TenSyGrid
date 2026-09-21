# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
"""Compiled, schema, ABI, and loopback evidence for FMI 1 exports."""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as element_tree
import zipfile

import numpy as np
from fmpy import simulate_fmu
from fmpy.validation import validate_fmu

from VeraGridEngine.enumerations import FmiVersion
from VeraGridEngine.IO.fmu.export_platform import (
    binary_directory,
    detect_target_platform,
    library_suffix,
)
from VeraGridEngine.IO.fmu.exporter.api import export_fmu
from VeraGridEngine.IO.fmu.exporter.compat import Block, Const, Var
from VeraGridEngine.IO.fmu.exporter.config import ExportConfig as CsExportConfig
from VeraGridEngine.IO.fmu.exporter_me.api import export_fmu_me
from VeraGridEngine.IO.fmu.exporter_me.config import ExportConfig as MeExportConfig


def test_fmi_one_co_simulation_export_is_schema_valid_and_loopback_ready(
    tmp_path: Path,
) -> None:
    """Compile, validate, inspect, and execute one FMI 1 Stand-Alone FMU.

    :param tmp_path: Isolated export and simulation directory from pytest.
    :return: None.
    """

    state: Var = Var("x")
    derivative: Var = Var("dx", base_var=state)
    source_block: Block = Block(
        state_vars=list((state,)),
        state_eqs=list((Const(1.0),)),
        diff_vars=list((derivative,)),
        init_values=dict(((state, Const(0.0)),)),
        out_vars=list((state,)),
        name="FmiOneCs",
    )
    model_identifier: str = "FmiOneCertifiedCs"
    archive_path: Path = export_fmu(
        source_block,
        CsExportConfig(
            model_name=model_identifier,
            output_path=tmp_path / (model_identifier + ".fmu"),
            fixed_step=1.0e-3,
            fmi_version=FmiVersion.FMI_1_0,
        ),
    )

    assert validate_fmu(str(archive_path)) == list()
    expected_binary: str = (
        "binaries/"
        + binary_directory(detect_target_platform(), FmiVersion.FMI_1_0)
        + "/"
        + model_identifier
        + library_suffix(detect_target_platform())
    )
    with zipfile.ZipFile(archive_path) as archive:
        archive_names: set[str] = set(archive.namelist())
        root = element_tree.fromstring(archive.read("modelDescription.xml"))
    assert expected_binary in archive_names
    assert root.attrib["fmiVersion"] == "1.0"
    assert root.attrib["modelIdentifier"] == model_identifier
    assert root.attrib["numberOfContinuousStates"] == "0"
    assert root.find("CoSimulation") is None
    assert root.find("ModelStructure") is None
    standalone = root.find("./Implementation/CoSimulation_StandAlone")
    assert standalone is not None
    capabilities = standalone.find("Capabilities")
    assert capabilities is not None
    assert capabilities.attrib["canHandleVariableCommunicationStepSize"] == "true"
    assert capabilities.attrib["canRunAsynchronuously"] == "false"
    assert "initial=" not in element_tree.tostring(root, encoding="unicode")

    result: np.ndarray = simulate_fmu(
        filename=str(archive_path),
        stop_time=3.0e-3,
        step_size=1.0e-3,
        output=("x",),
    )
    np.testing.assert_allclose(
        result["x"],
        result["time"],
        rtol=0.0,
        atol=1.0e-12,
    )


def test_fmi_one_model_exchange_export_is_schema_valid_and_loopback_ready(
    tmp_path: Path,
) -> None:
    """Compile, validate, inspect, and execute one FMI 1 Model Exchange FMU.

    :param tmp_path: Isolated export and simulation directory from pytest.
    :return: None.
    """

    state: Var = Var("x")
    derivative: Var = Var("dx", base_var=state)
    source_block: Block = Block(
        state_vars=list((state,)),
        state_eqs=list((Const(1.0),)),
        diff_vars=list((derivative,)),
        init_values=dict(((state, Const(0.0)),)),
        out_vars=list((state,)),
        name="FmiOneMe",
    )
    model_identifier: str = "FmiOneCertifiedMe"
    archive_path: Path = export_fmu_me(
        source_block,
        MeExportConfig(
            model_name=model_identifier,
            output_path=tmp_path / (model_identifier + ".fmu"),
            fixed_step=1.0e-3,
            fmi_version=FmiVersion.FMI_1_0,
        ),
    )

    assert validate_fmu(str(archive_path)) == list()
    expected_binary: str = (
        "binaries/"
        + binary_directory(detect_target_platform(), FmiVersion.FMI_1_0)
        + "/"
        + model_identifier
        + library_suffix(detect_target_platform())
    )
    with zipfile.ZipFile(archive_path) as archive:
        archive_names: set[str] = set(archive.namelist())
        root = element_tree.fromstring(archive.read("modelDescription.xml"))
    assert expected_binary in archive_names
    assert root.attrib["fmiVersion"] == "1.0"
    assert root.attrib["modelIdentifier"] == model_identifier
    assert root.attrib["numberOfContinuousStates"] == "1"
    assert root.find("Implementation") is None
    assert root.find("ModelExchange") is None
    assert root.find("ModelStructure") is None
    assert "initial=" not in element_tree.tostring(root, encoding="unicode")

    result: np.ndarray = simulate_fmu(
        filename=str(archive_path),
        stop_time=3.0e-3,
        step_size=1.0e-3,
        output=("x",),
    )
    np.testing.assert_allclose(
        result["x"],
        result["time"],
        rtol=0.0,
        atol=1.0e-8,
    )
