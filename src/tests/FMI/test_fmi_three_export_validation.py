# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
"""Static conformance checks for FMI 3 export assets and platform layout."""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path

from VeraGridEngine.enumerations import FmiVersion
from VeraGridEngine.IO.fmu.export_platform import (
    TargetPlatform,
    binary_directory,
    select_target_platform,
)


def test_fmi_three_binary_directories_follow_platform_tuples() -> None:
    """Map supported host identities to exact FMI 3 platform tuples.

    :return: None.
    """
    assert select_target_platform("Windows", "AMD64") is TargetPlatform.WIN64
    assert select_target_platform("Linux", "x86_64") is TargetPlatform.LINUX64
    assert select_target_platform("Darwin", "x86_64") is TargetPlatform.DARWIN64
    assert select_target_platform("Darwin", "arm64") is TargetPlatform.DARWIN_ARM64
    assert binary_directory(TargetPlatform.WIN64, FmiVersion.FMI_3_0) == "x86_64-windows"
    assert binary_directory(TargetPlatform.LINUX64, FmiVersion.FMI_3_0) == "x86_64-linux"
    assert binary_directory(TargetPlatform.DARWIN64, FmiVersion.FMI_3_0) == "x86_64-darwin"
    assert binary_directory(TargetPlatform.DARWIN_ARM64, FmiVersion.FMI_3_0) == "aarch64-darwin"


def test_fmi_three_headers_match_official_release() -> None:
    """Keep vendored FMI 3 headers byte-identical to the reviewed 3.0.2 files.

    :return: None.
    """
    fmi_three_root: Path = (
        Path(__file__).resolve().parents[2] / "VeraGridEngine" / "IO" / "fmu" / "c_api" / "fmi3"
    )
    expected_headers: tuple[tuple[str, str], ...] = (
        ("fmi3Functions.h", "038358fb0209c3547fe8c6f03b80582960c93ce0a7d342b0d9348c8ce6137784"),
        ("fmi3FunctionTypes.h", "d247d1c5fbf58e32379f1d38e7f73990709e0501fc4a9b3901ce3f9638503a0f"),
        ("fmi3PlatformTypes.h", "1aa81efddeb84b0d9425f9226969525bd034cb63848c086b6696655179500616"),
    )
    for header_name, expected_sha256 in expected_headers:
        header_path: Path = fmi_three_root / header_name
        actual_sha256: str = hashlib.sha256(header_path.read_bytes()).hexdigest()
        assert actual_sha256 == expected_sha256


def test_legacy_fmi_two_template_and_me_renderers_are_absent() -> None:
    """Enforce the single static owner selected by the migration contract.

    :return: None.
    """
    fmu_root: Path = Path(__file__).resolve().parents[2] / "VeraGridEngine" / "IO" / "fmu"
    assert not (fmu_root / "exporter" / "c_runtime" / "fmi2_template").exists()
    me_build_source: str = (fmu_root / "exporter_me" / "build.py").read_text(encoding="utf-8")
    deleted_renderers: tuple[str, ...] = (
        "def render_model_instance_h(",
        "def render_model_instance_c(",
        "def render_solver_c(",
        "def render_runtime_fmi2_c(",
    )
    for deleted_renderer in deleted_renderers:
        assert deleted_renderer not in me_build_source


def test_product_fmi_tests_do_not_depend_on_comparison_examples() -> None:
    """Keep product tests independent from the non-production comparison tree.

    :return: None.
    """
    test_root: Path = Path(__file__).resolve().parent
    forbidden_module: str = "tru" + "nk"
    forbidden_forward_path: str = forbidden_module + "/"
    forbidden_back_path: str = forbidden_module + "\\"
    for test_path in test_root.rglob("*.py"):
        source_text: str = test_path.read_text(encoding="utf-8")
        syntax_tree: ast.Module = ast.parse(source_text, filename=str(test_path))
        for syntax_node in ast.walk(syntax_tree):
            if isinstance(syntax_node, ast.Import):
                imported_modules: tuple[str, ...] = tuple(alias.name for alias in syntax_node.names)
                assert all(not name.startswith(forbidden_module) for name in imported_modules)
            elif isinstance(syntax_node, ast.ImportFrom):
                imported_from: str = syntax_node.module if syntax_node.module is not None else ""
                assert not imported_from.startswith(forbidden_module)
            elif isinstance(syntax_node, ast.Constant) and isinstance(syntax_node.value, str):
                normalized_value: str = syntax_node.value.lower()
                assert forbidden_forward_path not in normalized_value
                assert forbidden_back_path not in normalized_value
            else:
                pass


def test_fmi_package_initializers_retain_mpl_license() -> None:
    """Reject any FMI package initializer that omits the MPL notice.

    :return: None.
    """
    fmu_root: Path = Path(__file__).resolve().parents[2] / "VeraGridEngine" / "IO" / "fmu"
    initializer_paths: list[Path] = list(fmu_root.rglob("__init__.py"))
    assert len(initializer_paths) > 0
    for initializer_path in initializer_paths:
        source_text: str = initializer_path.read_text(encoding="utf-8")
        assert source_text.startswith(
            "# This Source Code Form is subject to the terms of the Mozilla Public\n"
            "# License, v. 2.0. If a copy of the MPL was not distributed with this\n"
            "# file, You can obtain one at https://mozilla.org/MPL/2.0/.\n"
            "# SPDX-License-Identifier: MPL-2.0\n"
        )

