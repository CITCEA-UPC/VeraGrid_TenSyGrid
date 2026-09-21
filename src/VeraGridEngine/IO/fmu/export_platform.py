# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from enum import Enum
import platform

from VeraGridEngine.enumerations import FmiVersion


class TargetPlatform(str, Enum):
    """Identify one supported native FMU host platform."""

    WIN64 = "win64"
    LINUX64 = "linux64"
    DARWIN64 = "darwin64"
    DARWIN_ARM64 = "darwin64-arm64"


def select_target_platform(system_name: str, machine_name: str) -> TargetPlatform:
    """Select a supported FMU platform from normalized host identifiers.

    :param system_name: Lowercase operating-system name.
    :param machine_name: Lowercase machine architecture name.
    :return: Matching native FMU platform.
    """

    normalized_system_name: str = system_name.strip().lower()
    normalized_machine_name: str = machine_name.strip().lower()
    if normalized_system_name == "windows":
        if normalized_machine_name == "amd64" or normalized_machine_name == "x86_64":
            result: TargetPlatform = TargetPlatform.WIN64
        else:
            raise RuntimeError("FMI export supports only AMD64 Windows hosts")
    else:
        if normalized_system_name == "linux":
            if normalized_machine_name == "amd64" or normalized_machine_name == "x86_64":
                result = TargetPlatform.LINUX64
            else:
                raise RuntimeError("FMI export supports only x86_64 Linux hosts")
        else:
            if normalized_system_name == "darwin":
                if normalized_machine_name == "amd64" or normalized_machine_name == "x86_64":
                    result = TargetPlatform.DARWIN64
                else:
                    if normalized_machine_name == "arm64" or normalized_machine_name == "aarch64":
                        result = TargetPlatform.DARWIN_ARM64
                    else:
                        raise RuntimeError("FMI export supports only x86_64 or arm64 macOS hosts")
            else:
                raise RuntimeError("FMI export does not support this operating system")
    return result


def detect_target_platform() -> TargetPlatform:
    """Detect the current supported native FMU host platform.

    :return: Host platform selected from the operating system and architecture.
    """

    system_name: str = platform.system().lower()
    machine_name: str = platform.machine().lower()
    return select_target_platform(system_name, machine_name)


def library_suffix(target_platform: TargetPlatform) -> str:
    """Return the native shared-library suffix for a target platform.

    :param target_platform: Native FMU platform.
    :return: Platform shared-library suffix including its leading dot.
    """

    if target_platform == TargetPlatform.WIN64:
        suffix: str = ".dll"
    else:
        if target_platform == TargetPlatform.LINUX64:
            suffix = ".so"
        else:
            suffix = ".dylib"
    return suffix


def binary_directory(target_platform: TargetPlatform, fmi_version: FmiVersion) -> str:
    """Return the standard FMU binary directory for one FMI generation.

    :param target_platform: Native FMU platform.
    :param fmi_version: FMI generation used by the archive.
    :return: Standard relative binary-directory name.
    """

    if fmi_version in (FmiVersion.FMI_1_0, FmiVersion.FMI_2_0):
        if target_platform == TargetPlatform.WIN64:
            directory_name: str = "win64"
        else:
            if target_platform == TargetPlatform.LINUX64:
                directory_name = "linux64"
            else:
                directory_name = "darwin64"
    else:
        if fmi_version == FmiVersion.FMI_3_0:
            if target_platform == TargetPlatform.WIN64:
                directory_name = "x86_64-windows"
            else:
                if target_platform == TargetPlatform.LINUX64:
                    directory_name = "x86_64-linux"
                else:
                    if target_platform == TargetPlatform.DARWIN64:
                        directory_name = "x86_64-darwin"
                    else:
                        directory_name = "aarch64-darwin"
        else:
            raise NotImplementedError(
                f"FMI {fmi_version.value} binary-directory naming is not implemented"
            )
    return directory_name
