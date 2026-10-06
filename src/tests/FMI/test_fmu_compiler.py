# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
from pathlib import Path
import hashlib
import inspect
import os
import pickle
import platform
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from VeraGridEngine.IO.fmu.compiler import (
    FmuBinaryInterface,
    FmuCompilerSession,
    _binary_header_matches,
    _compile_command,
    _close_owned_compiler_session,
    _create_compiler_session_tools,
    _host_toolchain_selection,
    _missing_exported_symbols,
    _required_fmi_symbols,
    _runtime_dependency_violations,
    compile_fmu_shared_library,
    fmu_compiler_available,
)


def _write_complete_fmi_two_cs_stub(source_path: Path) -> None:
    """Write a minimal C translation unit exporting every FMI 2 CS symbol.

    :param source_path: Destination C source path.
    :return: None.
    """

    symbols: tuple[str, ...] = _required_fmi_symbols(FmuBinaryInterface.FMI_TWO_CO_SIMULATION)
    lines: list[str] = [
        "#if defined(_WIN32)",
        "#define VG_EXPORT __declspec(dllexport)",
        "#else",
        "#define VG_EXPORT __attribute__((visibility(\"default\")))",
        "#endif",
    ]
    symbol: str
    for symbol in symbols:
        lines.append("VG_EXPORT int " + symbol + "(void) { return 0; }")
    source_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _build_synthetic_session_for_cleanup_failure(tmp_path: Path) -> FmuCompilerSession:
    """Build a non-executing session state whose workspace removal fails.

    :param tmp_path: Isolated test directory.
    :return: Session object used only for lifecycle guard tests.
    """

    workspace_path: Path = tmp_path / "workspace-is-a-file"
    workspace_path.write_bytes(b"not a directory")
    compiler_session: FmuCompilerSession = FmuCompilerSession.__new__(FmuCompilerSession)
    compiler_session._workspace_root = workspace_path
    compiler_session._compiler_path = workspace_path
    compiler_session._linker_path = workspace_path
    compiler_session._nm_path = workspace_path
    compiler_session._objdump_path = workspace_path
    compiler_session._package_platform = "win"
    compiler_session._package_arch = "mingw-x86_64"
    compiler_session._target_triple = "x86_64-w64-windows-gnu"
    compiler_session._library_suffix = ".dll"
    compiler_session._tool_hashes = ("", "", "", "")
    compiler_session._creator_pid = os.getpid()
    compiler_session._creator_thread = threading.current_thread()
    compiler_session._compile_active = False
    compiler_session._closed = False
    return compiler_session


@pytest.mark.parametrize(
    (
        "system_name", "machine_name", "package_platform", "package_arch",
        "target_triple", "suffix", "linker",
    ),
    (
        (
            "windows", "amd64", "win", "mingw-x86_64",
            "x86_64-w64-windows-gnu", ".dll", "ld.lld",
        ),
        (
            "linux", "x86_64", "linux", "x86_64",
            "x86_64-unknown-linux-gnu", ".so", "ld.lld",
        ),
        (
            "darwin", "x86_64", "darwin", "x86_64",
            "x86_64-apple-macos11.0", ".dylib", "ld64.lld",
        ),
        (
            "darwin", "arm64", "darwin", "arm64",
            "arm64-apple-macos11.0", ".dylib", "ld64.lld",
        ),
    ),
)
def test_host_selection_is_explicit(
    system_name: str,
    machine_name: str,
    package_platform: str,
    package_arch: str,
    target_triple: str,
    suffix: str,
    linker: str,
) -> None:
    """Map every supported native host to its sealed compiler row.

    :param system_name: Explicit operating-system test value.
    :param machine_name: Explicit machine test value.
    :param package_platform: Expected package platform.
    :param package_arch: Expected package architecture.
    :param target_triple: Expected host-native LLVM target triple.
    :param suffix: Expected shared-library suffix.
    :param linker: Expected linker.
    :return: None.
    """

    selection: tuple[str, str, str, str, str] = _host_toolchain_selection(system_name, machine_name)
    assert selection[0] == package_platform
    assert selection[1] == package_arch
    assert selection[2] == target_triple
    assert selection[3] == suffix
    assert selection[4] == linker


@pytest.mark.parametrize(
    ("system_name", "machine_name"),
    (("windows", "arm64"), ("linux", "aarch64"), ("darwin", "mips"), ("freebsd", "x86_64")),
)
def test_host_selection_rejects_unreviewed_pairs(system_name: str, machine_name: str) -> None:
    """Reject operating-system and machine pairs outside the sealed matrix.

    :param system_name: Explicit operating-system test value.
    :param machine_name: Explicit machine test value.
    :return: None.
    """

    with pytest.raises(RuntimeError, match="supports only|does not support"):
        _host_toolchain_selection(system_name, machine_name)


def test_compile_command_binds_absolute_linker(tmp_path: Path) -> None:
    """Bind compilation to the selected build-scoped linker path.

    :param tmp_path: Isolated path owner supplied by pytest.
    :return: None.
    """

    compiler_path: Path = (tmp_path / "tools" / "clang").resolve()
    linker_path: Path = (tmp_path / "tools" / "ld.lld").resolve()
    source_path: Path = tmp_path / "model.c"
    output_path: Path = tmp_path / "model.dll"
    command: tuple[str, ...] = _compile_command(
        compiler_path,
        linker_path,
        (source_path,),
        (tmp_path,),
        output_path,
        "win",
        "x86_64-w64-windows-gnu",
    )
    assert command[0] == str(compiler_path)
    assert command[2] == "--ld-path=" + str(linker_path)
    assert "-shared" in command


@pytest.mark.parametrize(
    ("header_text", "package_platform", "package_arch", "expected"),
    (
        ("file format coff-x86-64\narchitecture: x86_64", "win", "mingw-x86_64", True),
        ("file format elf64-x86-64\narchitecture: x86_64", "linux", "x86_64", True),
        ("file format mach-o arm64\narchitecture: arm64", "darwin", "arm64", True),
        ("file format mach-o 64-bit x86-64\narchitecture: x86-64", "darwin", "x86_64", True),
        ("file format elf64-x86-64\narchitecture: x86_64", "win", "mingw-x86_64", False),
    ),
)
def test_binary_header_matching(
    header_text: str,
    package_platform: str,
    package_arch: str,
    expected: bool,
) -> None:
    """Accept only the binary format and machine of the selected host row.

    :param header_text: Representative llvm-objdump header output.
    :param package_platform: Reviewed package platform.
    :param package_arch: Reviewed package architecture.
    :param expected: Expected comparison result.
    :return: None.
    """

    assert _binary_header_matches(header_text, package_platform, package_arch) is expected


def test_fmi_symbol_inventories_are_complete_and_distinct() -> None:
    """Own complete, nonempty, interface-specific FMI symbol sets.

    :return: None.
    """

    fmi_one_cs_symbols: tuple[str, ...] = _required_fmi_symbols(
        FmuBinaryInterface.FMI_ONE_CO_SIMULATION,
        "CertifiedModel",
    )
    fmi_one_me_symbols: tuple[str, ...] = _required_fmi_symbols(
        FmuBinaryInterface.FMI_ONE_MODEL_EXCHANGE,
        "CertifiedModel",
    )
    assert len(fmi_one_cs_symbols) == 25
    assert len(fmi_one_me_symbols) == 24
    assert "CertifiedModel_fmiDoStep" in fmi_one_cs_symbols
    assert "CertifiedModel_fmiDoStep" not in fmi_one_me_symbols
    assert "CertifiedModel_fmiGetDerivatives" in fmi_one_me_symbols
    assert "CertifiedModel_fmiGetDerivatives" not in fmi_one_cs_symbols
    with pytest.raises(ValueError, match="explicit model identifier"):
        _required_fmi_symbols(FmuBinaryInterface.FMI_ONE_CO_SIMULATION)

    cs_symbols: tuple[str, ...] = _required_fmi_symbols(FmuBinaryInterface.FMI_TWO_CO_SIMULATION)
    me_symbols: tuple[str, ...] = _required_fmi_symbols(FmuBinaryInterface.FMI_TWO_MODEL_EXCHANGE)
    assert len(cs_symbols) == 34
    assert len(me_symbols) == 35
    assert "fmi2DoStep" in cs_symbols
    assert "fmi2DoStep" not in me_symbols
    assert "fmi2GetDerivatives" in me_symbols
    assert "fmi2GetDerivatives" not in cs_symbols

    darwin_inventory: str = "\n".join("_" + symbol + " T 0 0" for symbol in cs_symbols)
    assert _missing_exported_symbols(darwin_inventory, FmuBinaryInterface.FMI_TWO_CO_SIMULATION, "darwin") == tuple()

    fmi_one_inventory: str = "\n".join(
        "_" + symbol + " T 0 0" for symbol in fmi_one_cs_symbols
    )
    assert _missing_exported_symbols(
        fmi_one_inventory,
        FmuBinaryInterface.FMI_ONE_CO_SIMULATION,
        "darwin",
        "CertifiedModel",
    ) == tuple()


@pytest.mark.parametrize(
    ("dependency_text", "package_platform"),
    (
        ("The Import Tables:\n  DLL Name: KERNEL32.dll\nExport Table:\n", "win"),
        ("Dynamic Section:\n  NEEDED libc.so.6\n  NEEDED libm.so.6\n", "linux"),
        ("Mach header\nLoad command 0\n cmd LC_LOAD_DYLIB\n name /usr/lib/libSystem.B.dylib (offset 24)\n", "darwin"),
    ),
)
def test_runtime_dependency_parser_accepts_native_allowlists(dependency_text: str, package_platform: str) -> None:
    """Accept only recognized native structures with portable dependencies.

    :param dependency_text: Representative llvm-objdump dependency output.
    :param package_platform: Platform whose grammar and allowlist apply.
    :return: None.
    """

    assert _runtime_dependency_violations(dependency_text, package_platform) == tuple()


@pytest.mark.parametrize("package_platform", ("win", "linux", "darwin"))
def test_runtime_dependency_parser_fails_closed(package_platform: str) -> None:
    """Reject successful inspection output whose structure is unrecognized.

    :param package_platform: Platform grammar selected for the malformed text.
    :return: None.
    """

    violations: tuple[str, ...] = _runtime_dependency_violations("unrecognized output", package_platform)
    assert len(violations) == 1
    assert "unrecognized" in violations[0]


def test_runtime_dependency_parser_rejects_rpath_and_foreign_library() -> None:
    """Reject runtime search paths and libraries outside the Linux closure.

    :return: None.
    """

    dependency_text: str = "Dynamic Section:\n  RUNPATH /tmp/build\n  NEEDED libpython3.11.so\n"
    violations: tuple[str, ...] = _runtime_dependency_violations(dependency_text, "linux")
    assert "forbidden runtime search path" in violations
    assert "undeclared runtime dependency: libpython3.11.so" in violations


def test_compiler_rejects_empty_inputs(tmp_path: Path) -> None:
    """Reject a compilation request without source or include ownership.

    :param tmp_path: Isolated output directory supplied by pytest.
    :return: None.
    """

    output_path: Path = tmp_path / "empty.dll"
    with pytest.raises(ValueError, match="requires source files and include directories"):
        compile_fmu_shared_library(tuple(), tuple(), output_path, FmuBinaryInterface.FMI_TWO_CO_SIMULATION)


def test_compiler_rejects_missing_source(tmp_path: Path) -> None:
    """Reject a declared C translation unit that does not exist.

    :param tmp_path: Isolated source and output directory supplied by pytest.
    :return: None.
    """

    missing_source: Path = tmp_path / "missing.c"
    with pytest.raises(FileNotFoundError, match="compilation source is missing"):
        compile_fmu_shared_library((missing_source,), (tmp_path,), tmp_path / "missing.dll", FmuBinaryInterface.FMI_TWO_MODEL_EXCHANGE)


@pytest.mark.skipif(not fmu_compiler_available(), reason="The approved compiler command set is unavailable on this host")
def test_compiler_rejects_binary_without_fmi_symbols(tmp_path: Path) -> None:
    """Compile a valid shared library and reject its absent FMI 1 contract.

    :param tmp_path: Isolated source, build, and output directory from pytest.
    :return: None.
    """

    source_path: Path = tmp_path / "not_an_fmu.c"
    source_path.write_text("int ordinary_function(void) { return 0; }\n", encoding="utf-8")
    system_name: str = platform.system().lower()
    if system_name == "windows":
        library_suffix: str = ".dll"
    else:
        if system_name == "linux":
            library_suffix = ".so"
        else:
            if system_name == "darwin":
                library_suffix = ".dylib"
            else:
                pytest.fail("The provisioned compiler reports an unsupported host")
                library_suffix = ""
    with pytest.raises(RuntimeError, match="missing required symbols"):
        compile_fmu_shared_library(
            (source_path,),
            (tmp_path,),
            tmp_path / ("not_an_fmu" + library_suffix),
            FmuBinaryInterface.FMI_ONE_CO_SIMULATION,
            "MissingSymbols",
        )


def test_macho_dependency_parser_rejects_weak_external_library() -> None:
    """Apply the same allowlist to weak and reexported Mach-O dependencies.

    :return: None.
    """

    dependency_text: str = (
        "Mach header\n"
        "Load command 0\n"
        " cmd LC_LOAD_WEAK_DYLIB\n"
        " name /opt/vendor/libexternal.dylib (offset 24)\n"
    )
    violations: tuple[str, ...] = _runtime_dependency_violations(
        dependency_text,
        "darwin",
    )
    assert "undeclared runtime dependency: /opt/vendor/libexternal.dylib" in violations


@pytest.mark.skipif(
    not fmu_compiler_available(),
    reason="The approved compiler command set is unavailable on this host",
)
def test_compiler_rejects_foreign_library_suffix(tmp_path: Path) -> None:
    """Reject a foreign output suffix before compiling generated source.

    :param tmp_path: Isolated source and output root supplied by pytest.
    :return: None.
    """

    source_path: Path = tmp_path / "model.c"
    source_path.write_text("int model(void) { return 0; }\n", encoding="utf-8")
    system_name: str = platform.system().lower()
    if system_name == "windows":
        foreign_suffix: str = ".so"
    else:
        foreign_suffix = ".dll"
    with pytest.raises(ValueError, match="suffix does not match"):
        compile_fmu_shared_library(
            (source_path,),
            (tmp_path,),
            tmp_path / ("model" + foreign_suffix),
            FmuBinaryInterface.FMI_TWO_CO_SIMULATION,
        )


@pytest.mark.skipif(not fmu_compiler_available(), reason="The approved compiler command set is unavailable on this host")
def test_compiler_session_reuses_one_workspace_and_closes(tmp_path: Path) -> None:
    """Compile twice through one session and remove its workspace exactly once.

    :param tmp_path: Isolated source and binary directory.
    :return: None.
    """

    source_path: Path = tmp_path / "complete.c"
    _write_complete_fmi_two_cs_stub(source_path)
    library_suffix: str = _host_toolchain_selection(platform.system().lower(), platform.machine().lower())[3]
    compiler_session: FmuCompilerSession = FmuCompilerSession()
    workspace_root: Path = compiler_session._workspace_root
    try:
        first_path: Path = compiler_session.compile_shared_library(
            (source_path,), (tmp_path,), tmp_path / ("first" + library_suffix), FmuBinaryInterface.FMI_TWO_CO_SIMULATION,
        )
        second_path: Path = compiler_session.compile_shared_library(
            (source_path,), (tmp_path,), tmp_path / ("second" + library_suffix), FmuBinaryInterface.FMI_TWO_CO_SIMULATION,
        )
        assert first_path.is_file()
        assert second_path.is_file()
        assert workspace_root.is_dir()
    finally:
        compiler_session.close()
    assert not workspace_root.exists()
    compiler_session.close()
    assert compiler_session._closed


def test_compiler_session_rejects_use_after_close(tmp_path: Path) -> None:
    """Reject a compile before touching inputs when the session is closed.

    :param tmp_path: Isolated synthetic state directory.
    :return: None.
    """

    compiler_session: FmuCompilerSession = _build_synthetic_session_for_cleanup_failure(tmp_path)
    compiler_session._closed = True
    with pytest.raises(RuntimeError, match="closed"):
        compiler_session.compile_shared_library(tuple(), tuple(), tmp_path / "unused.dll", FmuBinaryInterface.FMI_TWO_CO_SIMULATION)


def test_compiler_session_rejects_cross_thread_use_and_close(tmp_path: Path) -> None:
    """Reject compile and close from a different Thread object.

    :param tmp_path: Isolated synthetic state directory.
    :return: None.
    """

    compiler_session: FmuCompilerSession = _build_synthetic_session_for_cleanup_failure(tmp_path)
    with ThreadPoolExecutor(max_workers=1) as executor:
        compile_future = executor.submit(
            compiler_session.compile_shared_library,
            tuple(), tuple(), tmp_path / "unused.dll", FmuBinaryInterface.FMI_TWO_CO_SIMULATION,
        )
        close_future = executor.submit(compiler_session.close)
        with pytest.raises(RuntimeError, match="creating process and thread"):
            compile_future.result()
        with pytest.raises(RuntimeError, match="creating process and thread"):
            close_future.result()


def test_compiler_session_rejects_pid_mismatch(tmp_path: Path) -> None:
    """Reject an inherited session whose creator process differs.

    :param tmp_path: Isolated synthetic state directory.
    :return: None.
    """

    compiler_session: FmuCompilerSession = _build_synthetic_session_for_cleanup_failure(tmp_path)
    compiler_session._creator_pid = os.getpid() + 1
    with pytest.raises(RuntimeError, match="creating process and thread"):
        compiler_session.close()


def test_compiler_session_rejects_serialization(tmp_path: Path) -> None:
    """Reject pickle transport of process-local executable ownership.

    :param tmp_path: Isolated synthetic state directory.
    :return: None.
    """

    compiler_session: FmuCompilerSession = _build_synthetic_session_for_cleanup_failure(tmp_path)
    with pytest.raises(TypeError, match="cannot be serialized"):
        pickle.dumps(compiler_session)


def test_compiler_session_rejects_tool_identity_drift(tmp_path: Path) -> None:
    """Reject a tool whose bytes changed after initial authentication.

    :param tmp_path: Isolated synthetic tool directory.
    :return: None.
    """

    compiler_session: FmuCompilerSession = _build_synthetic_session_for_cleanup_failure(tmp_path)
    compiler_session._workspace_root.unlink()
    compiler_session._workspace_root.mkdir()
    tool_paths: list[Path] = [compiler_session._workspace_root / ("tool" + str(index)) for index in range(4)]
    tool_path: Path
    for tool_path in tool_paths:
        tool_path.write_bytes(b"reviewed")
    compiler_session._compiler_path, compiler_session._linker_path, compiler_session._nm_path, compiler_session._objdump_path = tool_paths
    tool_hash: str = hashlib.sha256(b"reviewed").hexdigest()
    compiler_session._tool_hashes = (tool_hash, tool_hash, tool_hash, tool_hash)
    compiler_session._assert_tools_unchanged()
    tool_paths[0].write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="changed"):
        compiler_session._assert_tools_unchanged()


def test_compiler_session_rejects_symlinked_tool(tmp_path: Path) -> None:
    """Reject a tool path replaced by a symbolic link.

    :param tmp_path: Isolated synthetic tool directory.
    :return: None.
    """

    compiler_session: FmuCompilerSession = _build_synthetic_session_for_cleanup_failure(tmp_path)
    compiler_session._workspace_root.unlink()
    compiler_session._workspace_root.mkdir()
    target_path: Path = compiler_session._workspace_root / "target"
    link_path: Path = compiler_session._workspace_root / "link"
    target_path.write_bytes(b"reviewed")
    try:
        link_path.symlink_to(target_path)
    except OSError:
        pytest.skip("This host does not permit test symbolic links")
    compiler_session._compiler_path = link_path
    compiler_session._linker_path = target_path
    compiler_session._nm_path = target_path
    compiler_session._objdump_path = target_path
    tool_hash: str = hashlib.sha256(b"reviewed").hexdigest()
    compiler_session._tool_hashes = (tool_hash, tool_hash, tool_hash, tool_hash)
    with pytest.raises(RuntimeError, match="no longer trusted"):
        compiler_session._assert_tools_unchanged()


def test_compiler_session_rejects_reentry_and_close_during_compile(tmp_path: Path) -> None:
    """Reject reentry and cleanup while the session is marked active.

    :param tmp_path: Isolated synthetic state directory.
    :return: None.
    """

    compiler_session: FmuCompilerSession = _build_synthetic_session_for_cleanup_failure(tmp_path)
    compiler_session._compile_active = True
    with pytest.raises(RuntimeError, match="reentered"):
        compiler_session.compile_shared_library(tuple(), tuple(), tmp_path / "unused.dll", FmuBinaryInterface.FMI_TWO_CO_SIMULATION)
    with pytest.raises(RuntimeError, match="during compilation"):
        compiler_session.close()


def test_compiler_session_cleanup_failure_is_retriable(tmp_path: Path) -> None:
    """Keep cleanup retryable until the exact workspace is absent.

    :param tmp_path: Isolated synthetic state directory.
    :return: None.
    """

    compiler_session: FmuCompilerSession = _build_synthetic_session_for_cleanup_failure(tmp_path)
    with pytest.raises(OSError):
        compiler_session.close()
    assert not compiler_session._closed
    compiler_session._workspace_root.unlink()
    compiler_session._workspace_root.mkdir()
    compiler_session.close()
    assert compiler_session._closed


def test_owned_session_cleanup_failure_preserves_primary_error(tmp_path: Path) -> None:
    """Raise the work error from cleanup when both operations fail.

    :param tmp_path: Isolated synthetic state directory.
    :return: None.
    """

    compiler_session: FmuCompilerSession = _build_synthetic_session_for_cleanup_failure(tmp_path)
    primary_error: ValueError = ValueError("primary compilation failure")
    with pytest.raises(ValueError, match="primary compilation failure") as captured:
        _close_owned_compiler_session(compiler_session, primary_error)
    assert isinstance(captured.value.__cause__, OSError)


@pytest.mark.skipif(not fmu_compiler_available(), reason="The approved compiler command set is unavailable on this host")
def test_compiler_session_remains_reusable_after_compile_failure(tmp_path: Path) -> None:
    """Clear active state after failure so a valid compile can follow.

    :param tmp_path: Isolated source and binary directory.
    :return: None.
    """

    source_path: Path = tmp_path / "model.c"
    source_path.write_text("this is invalid C\n", encoding="utf-8")
    library_suffix: str = _host_toolchain_selection(platform.system().lower(), platform.machine().lower())[3]
    compiler_session: FmuCompilerSession = FmuCompilerSession()
    try:
        with pytest.raises(RuntimeError, match="Clang failed"):
            compiler_session.compile_shared_library(
                (source_path,), (tmp_path,), tmp_path / ("invalid" + library_suffix), FmuBinaryInterface.FMI_TWO_CO_SIMULATION,
            )
        assert not compiler_session._compile_active
        _write_complete_fmi_two_cs_stub(source_path)
        valid_path: Path = compiler_session.compile_shared_library(
            (source_path,), (tmp_path,), tmp_path / ("valid" + library_suffix), FmuBinaryInterface.FMI_TWO_CO_SIMULATION,
        )
        assert valid_path.is_file()
    finally:
        compiler_session.close()


def test_compiler_session_initialization_and_cleanup_failure_preserves_primary_error() -> None:
    """Require Python 3.8-compatible primary-from-cleanup constructor logic.

    :return: None.
    """

    source_text: str = inspect.getsource(_create_compiler_session_tools)
    assert "raise initialization_error from cleanup_error" in source_text
    assert "add_note" not in source_text
    assert "ExceptionGroup" not in source_text
