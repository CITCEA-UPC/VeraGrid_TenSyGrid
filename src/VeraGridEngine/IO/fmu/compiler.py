# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from enum import Enum
from pathlib import Path
import hashlib
import os
import platform
import shutil
import subprocess
import tempfile
import threading


class FmuBinaryInterface(Enum):
    """Identify the FMI binary interface whose entry points must be exported."""

    FMI_ONE_CO_SIMULATION = "fmi_one_co_simulation"
    FMI_ONE_MODEL_EXCHANGE = "fmi_one_model_exchange"
    FMI_TWO_CO_SIMULATION = "fmi_two_co_simulation"
    FMI_TWO_MODEL_EXCHANGE = "fmi_two_model_exchange"
    FMI_THREE_CO_SIMULATION = "fmi_three_co_simulation"
    FMI_THREE_MODEL_EXCHANGE = "fmi_three_model_exchange"


def _toolchain_tool_path(tool_name: str) -> Path:
    """Return one executable exposed by the approved Clang distribution.

    :param tool_name: Logical tool name used by the FMI compiler.
    :return: Absolute executable path.
    :raises FileNotFoundError: If the approved command is not installed.
    """

    if tool_name == "clang":
        binary_name: str = "clang"
        command_name: str = "clang-tool-chain-clang"
    else:
        if tool_name == "ld.lld":
            binary_name = "ld.lld"
            command_name = "clang-tool-chain-ld"
        else:
            if tool_name == "llvm-nm":
                binary_name = "llvm-nm"
                command_name = "clang-tool-chain-nm"
            else:
                if tool_name == "llvm-objdump":
                    binary_name = "llvm-objdump"
                    command_name = "clang-tool-chain-objdump"
                else:
                    if tool_name == "ld64.lld":
                        binary_name = "ld64.lld"
                        command_name = "clang-tool-chain-ld"
                    else:
                        raise ValueError("Unsupported Clang tool: " + tool_name)
    package_platform: str
    package_arch: str
    selection: tuple[str, str, str, str, str] = _host_toolchain_selection(
        platform.system().lower(), platform.machine().lower()
    )
    package_platform = selection[0]
    package_arch = selection[1]
    if platform.system().lower() == "windows":
        binary_path: Path = (
            Path.home()
            / ".clang-tool-chain"
            / "clang"
            / "win"
            / package_arch
            / "bin"
            / (binary_name + ".exe")
        )
    else:
        binary_path = (
            Path.home()
            / ".clang-tool-chain"
            / "clang"
            / package_platform
            / package_arch
            / "bin"
            / binary_name
        )
    if binary_path.is_file() and not binary_path.is_symlink():
        return binary_path.resolve()
    else:
        pass
    executable: str | None = shutil.which(command_name)
    if executable is not None:
        return Path(executable).resolve()
    else:
        raise FileNotFoundError(
            "The approved clang-tool-chain command is unavailable: " + command_name
        )


def _host_toolchain_selection(system_name: str, machine_name: str) -> tuple[str, str, str, str, str]:
    """Select the host-native compiler ABI and executable names.

    :param system_name: Lowercase operating-system name.
    :param machine_name: Lowercase machine architecture name.
    :return: Package platform, package architecture, target triple, library
        suffix, and linker tool name.
    """

    if system_name == "windows":
        if machine_name == "amd64" or machine_name == "x86_64":
            return (
                "win", "mingw-x86_64", "x86_64-w64-windows-gnu", ".dll", "ld.lld",
            )
        else:
            raise RuntimeError("FMI export compilation supports only AMD64 Windows hosts")
    else:
        if system_name == "linux":
            if machine_name == "x86_64" or machine_name == "amd64":
                return (
                    "linux", "x86_64", "x86_64-unknown-linux-gnu", ".so", "ld.lld",
                )
            else:
                raise RuntimeError("FMI export compilation supports only x86_64 Linux hosts")
        else:
            if system_name == "darwin":
                if machine_name == "x86_64" or machine_name == "amd64":
                    return (
                        "darwin", "x86_64", "x86_64-apple-macos11.0", ".dylib", "ld64.lld",
                    )
                else:
                    if machine_name == "arm64" or machine_name == "aarch64":
                        return (
                            "darwin", "arm64", "arm64-apple-macos11.0", ".dylib", "ld64.lld",
                        )
                    else:
                        raise RuntimeError("FMI export compilation supports only x86_64 or arm64 macOS hosts")
            else:
                raise RuntimeError("FMI export compilation does not support this operating system")


def fmu_compiler_available() -> bool:
    """Return whether the approved Clang command set is available.

    This probe is side-effect free because test collection and UI capability
    checks must not download or execute a compiler.

    :return: ``True`` when the host row and all required wrapper commands exist.
    """

    try:
        selection: tuple[str, str, str, str, str] = _host_toolchain_selection(
            platform.system().lower(), platform.machine().lower()
        )
        compiler_path: Path = _toolchain_tool_path("clang")
        linker_path: Path = _toolchain_tool_path(selection[4])
        nm_path: Path = _toolchain_tool_path("llvm-nm")
        objdump_path: Path = _toolchain_tool_path("llvm-objdump")
    except (FileNotFoundError, OSError, RuntimeError, ValueError):
        return False
    else:
        return all(
            (
                compiler_path.is_file(),
                linker_path.is_file(),
                nm_path.is_file(),
                objdump_path.is_file(),
            )
        )


def _resolve_toolchain_tools() -> tuple[Path, Path, Path, Path, str, str, str, str]:
    """Resolve the compiler, linker, and inspectors from the approved package.

    :return: Absolute tools followed by platform, architecture, target, suffix.
    """

    system_name: str = platform.system().lower()
    machine_name: str = platform.machine().lower()
    package_platform: str
    package_arch: str
    target_triple: str
    library_suffix: str
    linker: str
    (
        package_platform,
        package_arch,
        target_triple,
        library_suffix,
        linker,
    ) = _host_toolchain_selection(system_name, machine_name)
    compiler_path: Path = _toolchain_tool_path("clang")
    linker_path: Path = _toolchain_tool_path(linker)
    nm_path: Path = _toolchain_tool_path("llvm-nm")
    objdump_path: Path = _toolchain_tool_path("llvm-objdump")
    return (
        compiler_path,
        linker_path,
        nm_path,
        objdump_path,
        package_platform,
        package_arch,
        target_triple,
        library_suffix,
    )


def _create_compiler_session_tools(
    workspace_parent: Path,
) -> tuple[Path, Path, Path, Path, Path, str, str, str, str, tuple[str, ...]]:
    """Create one private workspace and authenticate the selected tool paths.

    :param workspace_parent: Existing parent for the session-owned temporary root.
    :return: Workspace, four tools, host metadata, and ordered tool digests.
    """

    workspace_root: Path = Path(tempfile.mkdtemp(prefix="veragrid_fmu_compiler_session_", dir=workspace_parent))
    try:
        resolved: tuple[Path, Path, Path, Path, str, str, str, str] = _resolve_toolchain_tools()
        tool_paths: tuple[Path, ...] = resolved[:4]
        tool_hashes: list[str] = [""] * len(tool_paths)
        tool_index: int
        tool_path: Path
        for tool_index, tool_path in enumerate(tool_paths):
            with tool_path.open("rb") as tool_stream:
                tool_hashes[tool_index] = hashlib.file_digest(tool_stream, "sha256").hexdigest()
    except BaseException as initialization_error:
        try:
            shutil.rmtree(workspace_root)
        except BaseException as cleanup_error:
            raise initialization_error from cleanup_error
        raise
    return (
        workspace_root,
        resolved[0],
        resolved[1],
        resolved[2],
        resolved[3],
        resolved[4],
        resolved[5],
        resolved[6],
        resolved[7],
        tuple(tool_hashes),
    )


def _compile_command(compiler_path: Path, linker_path: Path, source_files: tuple[Path, ...], include_directories: tuple[Path, ...], output_path: Path, package_platform: str, target_triple: str) -> tuple[str, ...]:
    """Build the single explicit Clang compile-and-link command.

    :param compiler_path: Absolute build-scoped Clang executable.
    :param linker_path: Absolute build-scoped LLD executable.
    :param source_files: Ordered generated C translation units.
    :param include_directories: Ordered generated and runtime include roots.
    :param output_path: Expected host-native shared library.
    :param package_platform: Reviewed package platform name.
    :param target_triple: Reviewed host-native LLVM target triple.
    :return: Immutable compiler argument sequence.
    """

    command: list[str] = list()
    command.append(str(compiler_path))
    command.append("--target=" + target_triple)
    command.append("--ld-path=" + str(linker_path))
    if package_platform == "darwin":
        sdk_result: subprocess.CompletedProcess[str] = subprocess.run(("/usr/bin/xcrun", "--sdk", "macosx", "--show-sdk-path"), check=False, capture_output=True, text=True)
        if sdk_result.returncode == 0 and len(sdk_result.stdout.strip()) > 0:
            sdk_path: Path = Path(sdk_result.stdout.strip())
            if sdk_path.is_dir():
                command.extend(("-dynamiclib", "-fPIC", "-mmacosx-version-min=11.0", "-isysroot", str(sdk_path)))
            else:
                raise FileNotFoundError("The macOS SDK reported by xcrun does not exist")
        else:
            raise RuntimeError("A native macOS SDK is required for FMI export")
    else:
        command.append("-shared")
        if package_platform == "linux":
            command.append("-fPIC")
        else:
            if package_platform == "win":
                pass
            else:
                raise RuntimeError("The compiler platform is not part of the reviewed matrix")
    command.extend(("-std=c99", "-O2"))
    include_directory: Path
    for include_directory in include_directories:
        command.extend(("-I", str(include_directory.resolve())))
    command.extend(("-o", str(output_path.resolve())))
    source_file: Path
    for source_file in source_files:
        command.append(str(source_file.resolve()))
    if package_platform == "win":
        pass
    else:
        command.append("-lm")
    return tuple(command)


def _binary_header_matches(header_text: str, package_platform: str, package_arch: str) -> bool:
    """Compare parsed binary header text with one reviewed host row.

    :param header_text: ``llvm-objdump -f`` output.
    :param package_platform: Reviewed package platform name.
    :param package_arch: Reviewed package architecture name.
    :return: Whether format and machine match.
    """

    normalized_text: str = header_text.lower()
    if package_platform == "win":
        return "coff-x86-64" in normalized_text and "architecture: x86_64" in normalized_text
    else:
        if package_platform == "linux":
            return "elf64-x86-64" in normalized_text and "architecture: x86_64" in normalized_text
        else:
            if package_platform == "darwin":
                if package_arch == "arm64":
                    return "mach-o" in normalized_text and ("arm64" in normalized_text or "aarch64" in normalized_text)
                else:
                    return "mach-o" in normalized_text and "x86-64" in normalized_text
            else:
                return False


def _validate_binary_header(output_path: Path, objdump_path: Path, package_platform: str, package_arch: str) -> None:
    """Reject a shared library whose format or machine differs from its host row.

    :param output_path: Compiled shared library.
    :param objdump_path: Absolute build-scoped ``llvm-objdump`` executable.
    :param package_platform: Reviewed package platform name.
    :param package_arch: Reviewed package architecture name.
    :return: None.
    """

    inspection: subprocess.CompletedProcess[str] = subprocess.run((str(objdump_path), "-f", str(output_path)), check=False, capture_output=True, text=True)
    if inspection.returncode == 0:
        pass
    else:
        diagnostic: str = (inspection.stdout + "\n" + inspection.stderr)[-4000:]
        raise RuntimeError("Unable to inspect the FMU binary header:\n" + diagnostic)
    if _binary_header_matches(inspection.stdout, package_platform, package_arch):
        pass
    else:
        raise RuntimeError("The FMU shared library has an unexpected format or machine architecture")


def _required_fmi_symbols(
    interface: FmuBinaryInterface,
    model_identifier: str | None = None,
) -> tuple[str, ...]:
    """Return the complete required symbol inventory for one FMI interface.

    :param interface: FMI binary interface being packaged.
    :param model_identifier: Explicit FMI 1 symbol prefix, when required.
    :return: Immutable required symbol inventory.
    """

    fmi_one_co_simulation_suffixes: tuple[str, ...] = (
        "fmiGetTypesPlatform", "fmiGetVersion", "fmiSetDebugLogging",
        "fmiInstantiateSlave", "fmiInitializeSlave", "fmiTerminateSlave",
        "fmiResetSlave", "fmiFreeSlaveInstance", "fmiSetReal",
        "fmiSetInteger", "fmiSetBoolean", "fmiSetString", "fmiGetReal",
        "fmiGetInteger", "fmiGetBoolean", "fmiGetString",
        "fmiSetRealInputDerivatives", "fmiGetRealOutputDerivatives",
        "fmiCancelStep", "fmiDoStep", "fmiGetStatus", "fmiGetRealStatus",
        "fmiGetIntegerStatus", "fmiGetBooleanStatus", "fmiGetStringStatus",
    )
    fmi_one_model_exchange_suffixes: tuple[str, ...] = (
        "fmiGetModelTypesPlatform", "fmiGetVersion", "fmiSetDebugLogging",
        "fmiInstantiateModel", "fmiFreeModelInstance", "fmiSetTime",
        "fmiSetContinuousStates", "fmiCompletedIntegratorStep", "fmiSetReal",
        "fmiSetInteger", "fmiSetBoolean", "fmiSetString", "fmiGetReal",
        "fmiGetInteger", "fmiGetBoolean", "fmiGetString", "fmiInitialize",
        "fmiGetDerivatives", "fmiGetEventIndicators", "fmiEventUpdate",
        "fmiGetContinuousStates", "fmiGetNominalContinuousStates",
        "fmiGetStateValueReferences", "fmiTerminate",
    )
    if interface in (
        FmuBinaryInterface.FMI_ONE_CO_SIMULATION,
        FmuBinaryInterface.FMI_ONE_MODEL_EXCHANGE,
    ):
        if model_identifier is not None and len(model_identifier) > 0:
            pass
        else:
            raise ValueError(
                "FMI 1 symbol validation requires an explicit model identifier"
            )
        if interface == FmuBinaryInterface.FMI_ONE_CO_SIMULATION:
            fmi_one_suffixes: tuple[str, ...] = fmi_one_co_simulation_suffixes
        else:
            fmi_one_suffixes = fmi_one_model_exchange_suffixes
        prefixed_symbols: list[str] = [""] * len(fmi_one_suffixes)
        suffix_index: int
        for suffix_index in range(len(fmi_one_suffixes)):
            prefixed_symbols[suffix_index] = (
                model_identifier + "_" + fmi_one_suffixes[suffix_index]
            )
        required_symbols: tuple[str, ...] = tuple(prefixed_symbols)
    else:
        required_symbols = tuple()
    fmi_three_symbols: tuple[str, ...] = (
        "fmi3GetVersion",
        "fmi3SetDebugLogging",
        "fmi3InstantiateModelExchange",
        "fmi3InstantiateCoSimulation",
        "fmi3InstantiateScheduledExecution",
        "fmi3FreeInstance",
        "fmi3EnterInitializationMode",
        "fmi3ExitInitializationMode",
        "fmi3EnterEventMode",
        "fmi3Terminate",
        "fmi3Reset",
        "fmi3GetFloat32",
        "fmi3GetFloat64",
        "fmi3GetInt8",
        "fmi3GetUInt8",
        "fmi3GetInt16",
        "fmi3GetUInt16",
        "fmi3GetInt32",
        "fmi3GetUInt32",
        "fmi3GetInt64",
        "fmi3GetUInt64",
        "fmi3GetBoolean",
        "fmi3GetString",
        "fmi3GetBinary",
        "fmi3GetClock",
        "fmi3SetFloat32",
        "fmi3SetFloat64",
        "fmi3SetInt8",
        "fmi3SetUInt8",
        "fmi3SetInt16",
        "fmi3SetUInt16",
        "fmi3SetInt32",
        "fmi3SetUInt32",
        "fmi3SetInt64",
        "fmi3SetUInt64",
        "fmi3SetBoolean",
        "fmi3SetString",
        "fmi3SetBinary",
        "fmi3SetClock",
        "fmi3GetNumberOfVariableDependencies",
        "fmi3GetVariableDependencies",
        "fmi3GetFMUState",
        "fmi3SetFMUState",
        "fmi3FreeFMUState",
        "fmi3SerializedFMUStateSize",
        "fmi3SerializeFMUState",
        "fmi3DeserializeFMUState",
        "fmi3GetDirectionalDerivative",
        "fmi3GetAdjointDerivative",
        "fmi3EnterConfigurationMode",
        "fmi3ExitConfigurationMode",
        "fmi3GetIntervalDecimal",
        "fmi3GetIntervalFraction",
        "fmi3GetShiftDecimal",
        "fmi3GetShiftFraction",
        "fmi3SetIntervalDecimal",
        "fmi3SetIntervalFraction",
        "fmi3SetShiftDecimal",
        "fmi3SetShiftFraction",
        "fmi3EvaluateDiscreteStates",
        "fmi3UpdateDiscreteStates",
        "fmi3EnterContinuousTimeMode",
        "fmi3CompletedIntegratorStep",
        "fmi3SetTime",
        "fmi3SetContinuousStates",
        "fmi3GetContinuousStateDerivatives",
        "fmi3GetEventIndicators",
        "fmi3GetContinuousStates",
        "fmi3GetNominalsOfContinuousStates",
        "fmi3GetNumberOfEventIndicators",
        "fmi3GetNumberOfContinuousStates",
        "fmi3EnterStepMode",
        "fmi3GetOutputDerivatives",
        "fmi3DoStep",
        "fmi3ActivateModelPartition",
    )
    fmi_two_common_symbols: tuple[str, ...] = (
        "fmi2GetTypesPlatform", "fmi2GetVersion", "fmi2SetDebugLogging", "fmi2Instantiate", "fmi2FreeInstance",
        "fmi2SetupExperiment", "fmi2EnterInitializationMode", "fmi2ExitInitializationMode", "fmi2Terminate", "fmi2Reset",
        "fmi2GetReal", "fmi2GetInteger", "fmi2GetBoolean", "fmi2GetString", "fmi2SetReal", "fmi2SetInteger",
        "fmi2SetBoolean", "fmi2SetString", "fmi2GetFMUstate", "fmi2SetFMUstate", "fmi2FreeFMUstate",
        "fmi2SerializedFMUstateSize", "fmi2SerializeFMUstate", "fmi2DeSerializeFMUstate", "fmi2GetDirectionalDerivative",
    )
    if len(required_symbols) > 0:
        pass
    else:
        if interface == FmuBinaryInterface.FMI_TWO_CO_SIMULATION:
            interface_symbols: tuple[str, ...] = (
                "fmi2SetRealInputDerivatives", "fmi2GetRealOutputDerivatives", "fmi2DoStep", "fmi2CancelStep", "fmi2GetStatus",
                "fmi2GetRealStatus", "fmi2GetIntegerStatus", "fmi2GetBooleanStatus", "fmi2GetStringStatus",
            )
            required_symbols = fmi_two_common_symbols + interface_symbols
        else:
            if interface == FmuBinaryInterface.FMI_TWO_MODEL_EXCHANGE:
                interface_symbols = (
                    "fmi2EnterEventMode", "fmi2NewDiscreteStates", "fmi2EnterContinuousTimeMode", "fmi2CompletedIntegratorStep",
                    "fmi2SetTime", "fmi2SetContinuousStates", "fmi2GetDerivatives", "fmi2GetEventIndicators",
                    "fmi2GetContinuousStates", "fmi2GetNominalsOfContinuousStates",
                )
                required_symbols = fmi_two_common_symbols + interface_symbols
            else:
                if (
                    interface == FmuBinaryInterface.FMI_THREE_CO_SIMULATION
                    or interface == FmuBinaryInterface.FMI_THREE_MODEL_EXCHANGE
                ):
                    required_symbols = fmi_three_symbols
                else:
                    raise RuntimeError("The FMI binary interface is unsupported")
    if len(required_symbols) == len(set(required_symbols)):
        return required_symbols
    else:
        raise RuntimeError("The required FMI symbol inventory contains duplicates")

def _missing_exported_symbols(
    symbol_text: str,
    interface: FmuBinaryInterface,
    package_platform: str,
    model_identifier: str | None = None,
) -> tuple[str, ...]:
    """Compare an ``llvm-nm`` inventory with the required FMI interface.

    :param symbol_text: ``llvm-nm --format=posix`` output.
    :param interface: FMI binary interface being packaged.
    :param package_platform: Reviewed package platform name.
    :param model_identifier: Explicit FMI 1 symbol prefix, when required.
    :return: Required symbols absent from the binary.
    """

    exported_symbols: set[str] = set()
    output_line: str
    for output_line in symbol_text.splitlines():
        fields: list[str] = output_line.split()
        if len(fields) > 0:
            symbol_name: str = fields[0]
            if package_platform == "darwin" and symbol_name.startswith("_"):
                symbol_name = symbol_name[1:]
            else:
                pass
            exported_symbols.add(symbol_name)
        else:
            pass
    missing_symbols: list[str] = list()
    required_symbol: str
    for required_symbol in _required_fmi_symbols(interface, model_identifier):
        if required_symbol in exported_symbols:
            pass
        else:
            missing_symbols.append(required_symbol)
    return tuple(missing_symbols)


def _validate_exported_symbols(
    output_path: Path,
    nm_path: Path,
    interface: FmuBinaryInterface,
    package_platform: str,
    model_identifier: str | None = None,
) -> None:
    """Require the complete FMI symbol inventory for one interface.

    :param output_path: Compiled shared library.
    :param nm_path: Absolute build-scoped ``llvm-nm`` executable.
    :param interface: FMI binary interface being packaged.
    :param package_platform: Reviewed package platform name.
    :param model_identifier: Explicit FMI 1 symbol prefix, when required.
    :return: None.
    """

    inspection: subprocess.CompletedProcess[str] = subprocess.run((str(nm_path), "--defined-only", "--extern-only", "--format=posix", str(output_path)), check=False, capture_output=True, text=True)
    if inspection.returncode == 0:
        missing_symbols: tuple[str, ...] = _missing_exported_symbols(
            inspection.stdout,
            interface,
            package_platform,
            model_identifier,
        )
    else:
        diagnostic: str = (inspection.stdout + "\n" + inspection.stderr)[-4000:]
        raise RuntimeError("Unable to inspect the FMU exported symbols:\n" + diagnostic)
    if len(missing_symbols) == 0:
        pass
    else:
        raise RuntimeError("The FMU binary is missing required symbols: " + ", ".join(missing_symbols))


def _runtime_dependency_violations(dependency_text: str, package_platform: str) -> tuple[str, ...]:
    """Parse dependencies and reject malformed or nonportable inventories.

    :param dependency_text: ``llvm-objdump -p`` output.
    :param package_platform: Reviewed package platform name.
    :return: Structural, runtime-path, and dependency violations.
    """

    violations: list[str] = list()
    upper_text: str = dependency_text.upper()
    if "RPATH" in upper_text or "RUNPATH" in upper_text:
        violations.append("forbidden runtime search path")
    else:
        pass
    dependencies: list[str] = list()
    output_lines: list[str] = dependency_text.splitlines()
    if package_platform == "win":
        imports_seen: bool = False
        exports_seen: bool = False
        inside_imports: bool = False
        output_line: str
        for output_line in output_lines:
            stripped_line: str = output_line.strip()
            if stripped_line == "The Import Tables:":
                imports_seen = True
                inside_imports = True
            else:
                if stripped_line == "Export Table:":
                    exports_seen = True
                    inside_imports = False
                else:
                    pass
            if inside_imports and stripped_line.lower().startswith("dll name:"):
                dependencies.append(stripped_line.split(":", 1)[1].strip())
            else:
                pass
        structure_valid: bool = imports_seen and exports_seen
    else:
        if package_platform == "linux":
            structure_valid = "Dynamic Section:" in dependency_text
            output_line = ""
            for output_line in output_lines:
                stripped_line = output_line.strip()
                if stripped_line.startswith("NEEDED"):
                    fields: list[str] = stripped_line.split()
                    if len(fields) == 2:
                        dependencies.append(fields[1])
                    else:
                        structure_valid = False
                else:
                    pass
        else:
            if package_platform == "darwin":
                structure_valid = "Mach header" in dependency_text and "Load command" in dependency_text
                pending_name: bool = False
                for output_line in output_lines:
                    stripped_line = output_line.strip()
                    dependency_commands: tuple[str, ...] = (
                        "cmd LC_LOAD_DYLIB",
                        "cmd LC_LOAD_WEAK_DYLIB",
                        "cmd LC_REEXPORT_DYLIB",
                        "cmd LC_LAZY_LOAD_DYLIB",
                        "cmd LC_LOAD_UPWARD_DYLIB",
                    )
                    if stripped_line in dependency_commands:
                        if pending_name:
                            structure_valid = False
                        else:
                            pending_name = True
                    else:
                        unknown_dylib_command: bool = (
                            stripped_line.startswith("cmd LC_")
                            and stripped_line.endswith("DYLIB")
                            and stripped_line != "cmd LC_ID_DYLIB"
                        )
                        if unknown_dylib_command:
                            structure_valid = False
                        else:
                            pass
                        if pending_name and stripped_line.startswith("name "):
                            name_fields: list[str] = stripped_line.split()
                            if len(name_fields) >= 2:
                                dependencies.append(name_fields[1])
                                pending_name = False
                            else:
                                structure_valid = False
                        else:
                            pass
                if pending_name:
                    structure_valid = False
                else:
                    pass
            else:
                structure_valid = False
    if structure_valid:
        pass
    else:
        violations.append("unrecognized " + package_platform + " dependency table")

    dependency_name: str
    for dependency_name in dependencies:
        if package_platform == "win":
            normalized_name: str = dependency_name.lower()
            allowed: bool = (
                normalized_name == "kernel32.dll" or normalized_name == "msvcrt.dll" or normalized_name == "ucrtbase.dll"
                or normalized_name.startswith("api-ms-win-crt-") and normalized_name.endswith(".dll")
                or normalized_name.startswith("api-ms-win-core-") and normalized_name.endswith(".dll")
            )
        else:
            if package_platform == "linux":
                allowed = dependency_name in ("libc.so.6", "libm.so.6", "libgcc_s.so.1")
            else:
                if package_platform == "darwin":
                    allowed = dependency_name == "/usr/lib/libSystem.B.dylib"
                else:
                    allowed = False
        if allowed:
            pass
        else:
            violations.append("undeclared runtime dependency: " + dependency_name)
    return tuple(violations)


def _validate_runtime_dependencies(output_path: Path, objdump_path: Path, package_platform: str) -> None:
    """Reject build-only, malformed, or undeclared shared-library dependencies.

    :param output_path: Compiled shared library.
    :param objdump_path: Absolute build-scoped ``llvm-objdump`` executable.
    :param package_platform: Reviewed package platform name.
    :return: None.
    """

    inspection: subprocess.CompletedProcess[str] = subprocess.run((str(objdump_path), "-p", str(output_path)), check=False, capture_output=True, text=True)
    if inspection.returncode == 0:
        violations: tuple[str, ...] = _runtime_dependency_violations(inspection.stdout, package_platform)
    else:
        diagnostic: str = (inspection.stdout + "\n" + inspection.stderr)[-4000:]
        raise RuntimeError("Unable to inspect the FMU runtime dependencies:\n" + diagnostic)
    if len(violations) == 0:
        pass
    else:
        raise RuntimeError("The FMU binary failed dependency validation: " + "; ".join(violations))


class FmuCompilerSession:
    """Own one authenticated compiler command session for a sequential FMI batch."""

    __slots__ = (
        "_workspace_root", "_compiler_path", "_linker_path", "_nm_path",
        "_objdump_path", "_package_platform", "_package_arch", "_target_triple",
        "_library_suffix", "_tool_hashes", "_creator_pid", "_creator_thread",
        "_compile_active", "_closed",
    )

    def __init__(self) -> None:
        """Create and authenticate one private compiler session.

        :return: None.
        """

        self._creator_pid: int = os.getpid()
        self._creator_thread: threading.Thread = threading.current_thread()
        self._compile_active: bool = False
        self._closed: bool = False
        (
            self._workspace_root,
            self._compiler_path,
            self._linker_path,
            self._nm_path,
            self._objdump_path,
            self._package_platform,
            self._package_arch,
            self._target_triple,
            self._library_suffix,
            self._tool_hashes,
        ) = _create_compiler_session_tools(Path(tempfile.gettempdir()))

    def __reduce_ex__(self, protocol: int) -> object:
        """Reject serialization because tool ownership is process-local.

        :param protocol: Pickle protocol requested by the caller.
        :return: This method never returns.
        """

        if protocol >= 0:
            raise TypeError("FMI compiler sessions cannot be serialized")
        else:
            raise TypeError("Invalid serialization protocol for FMI compiler session")

    def _assert_owner(self) -> None:
        """Require the creating process and exact creating thread object.

        :return: None.
        """

        if os.getpid() == self._creator_pid and threading.current_thread() is self._creator_thread:
            pass
        else:
            raise RuntimeError("FMI compiler session use is restricted to its creating process and thread")

    def _assert_tools_unchanged(self) -> None:
        """Reauthenticate all executable tools before each compilation.

        :return: None.
        """

        tool_paths: tuple[Path, ...] = (
            self._compiler_path,
            self._linker_path,
            self._nm_path,
            self._objdump_path,
        )
        tool_index: int
        tool_path: Path
        for tool_index, tool_path in enumerate(tool_paths):
            canonical_tool: Path = tool_path.resolve()
            valid_path: bool = bool(
                not tool_path.is_symlink()
                and tool_path.is_file()
                and canonical_tool == tool_path
            )
            if valid_path:
                with tool_path.open("rb") as tool_stream:
                    actual_hash: str = hashlib.file_digest(tool_stream, "sha256").hexdigest()
                if actual_hash == self._tool_hashes[tool_index]:
                    pass
                else:
                    raise RuntimeError("An FMI compiler session tool changed after authentication")
            else:
                raise RuntimeError("An FMI compiler session tool path is no longer trusted")

    def close(self) -> None:
        """Remove the exact session-owned workspace and mark the session closed.

        :return: None.
        """

        self._assert_owner()
        if self._compile_active:
            raise RuntimeError("Cannot close an FMI compiler session during compilation")
        else:
            pass
        if self._closed:
            return
        else:
            if self._workspace_root.exists():
                shutil.rmtree(self._workspace_root)
            else:
                pass
        if self._workspace_root.exists():
            raise RuntimeError("FMI compiler session cleanup did not remove its workspace")
        else:
            self._closed = True

    def compile_shared_library(
        self,
        source_files: tuple[Path, ...],
        include_directories: tuple[Path, ...],
        output_path: Path,
        interface: FmuBinaryInterface,
        model_identifier: str | None = None,
    ) -> Path:
        """Compile and validate one library through this authenticated session.

        :param source_files: Ordered generated C translation units.
        :param include_directories: Ordered generated and runtime include roots.
        :param output_path: Expected host-native shared-library path.
        :param interface: FMI interface whose complete symbols must be present.
        :param model_identifier: Explicit FMI 1 symbol prefix, when required.
        :return: Validated shared-library path.
        """

        self._assert_owner()
        if self._closed:
            raise RuntimeError("Cannot use a closed FMI compiler session")
        else:
            pass
        if self._compile_active:
            raise RuntimeError("FMI compiler session compilation cannot be reentered")
        else:
            pass
        if len(source_files) > 0 and len(include_directories) > 0:
            pass
        else:
            raise ValueError("FMI compilation requires source files and include directories")
        source_file: Path
        for source_file in source_files:
            if source_file.is_file():
                pass
            else:
                raise FileNotFoundError("FMI compilation source is missing: " + str(source_file))
        include_directory: Path
        for include_directory in include_directories:
            if include_directory.is_dir():
                pass
            else:
                raise FileNotFoundError("FMI compilation include directory is missing: " + str(include_directory))
        self._assert_tools_unchanged()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix.lower() == self._library_suffix:
            pass
        else:
            raise ValueError("The requested FMU binary suffix does not match the native host")
        self._compile_active = True
        try:
            command: tuple[str, ...] = _compile_command(
                self._compiler_path,
                self._linker_path,
                source_files,
                include_directories,
                output_path,
                self._package_platform,
                self._target_triple,
            )
            compilation: subprocess.CompletedProcess[str] = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                cwd=output_path.parent,
            )
            if compilation.returncode == 0:
                pass
            else:
                diagnostic: str = (compilation.stdout + "\n" + compilation.stderr)[-4000:]
                raise RuntimeError("Clang failed to compile the FMI shared library:\n" + diagnostic)
            if output_path.is_file():
                pass
            else:
                raise FileNotFoundError("Clang completed without producing the FMI shared library")
            _validate_binary_header(output_path, self._objdump_path, self._package_platform, self._package_arch)
            _validate_exported_symbols(output_path, self._nm_path, interface, self._package_platform, model_identifier)
            _validate_runtime_dependencies(output_path, self._objdump_path, self._package_platform)
            return output_path
        finally:
            self._compile_active = False


def _close_owned_compiler_session(
    compiler_session: FmuCompilerSession,
    primary_error: BaseException | None,
) -> None:
    """Close an owned session without replacing an earlier work failure.

    :param compiler_session: Session owned by the current wrapper.
    :param primary_error: Earlier compilation failure, when present.
    :return: None.
    """

    try:
        compiler_session.close()
    except BaseException as cleanup_error:
        if primary_error is None:
            raise
        else:
            raise primary_error from cleanup_error
    if primary_error is None:
        pass
    else:
        raise primary_error


def compile_fmu_shared_library(
    source_files: tuple[Path, ...],
    include_directories: tuple[Path, ...],
    output_path: Path,
    interface: FmuBinaryInterface,
    model_identifier: str | None = None,
    compiler_session: FmuCompilerSession | None = None,
) -> Path:
    """Compile and validate one host-native FMI shared library.

    :param source_files: Ordered generated C translation units.
    :param include_directories: Ordered generated and runtime include roots.
    :param output_path: Expected host-native shared-library path.
    :param interface: FMI interface whose complete symbols must be present.
    :param model_identifier: Explicit FMI 1 symbol prefix, when required.
    :param compiler_session: Optional authenticated session reused by an explicit batch.
    :return: Validated shared-library path.
    """

    if len(source_files) > 0 and len(include_directories) > 0:
        pass
    else:
        raise ValueError("FMI compilation requires source files and include directories")
    source_file: Path
    for source_file in source_files:
        if source_file.is_file():
            pass
        else:
            raise FileNotFoundError("FMI compilation source is missing: " + str(source_file))
    include_directory: Path
    for include_directory in include_directories:
        if include_directory.is_dir():
            pass
        else:
            raise FileNotFoundError(
                "FMI compilation include directory is missing: " + str(include_directory)
            )
    if compiler_session is None:
        owned_session: FmuCompilerSession = FmuCompilerSession()
        try:
            result_path: Path = owned_session.compile_shared_library(
                source_files, include_directories, output_path, interface, model_identifier,
            )
        except BaseException as primary_error:
            _close_owned_compiler_session(owned_session, primary_error)
            raise RuntimeError("Unreachable compiler-session failure path")
        else:
            _close_owned_compiler_session(owned_session, None)
            return result_path
    else:
        return compiler_session.compile_shared_library(
            source_files, include_directories, output_path, interface, model_identifier,
        )
