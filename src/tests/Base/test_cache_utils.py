# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from pathlib import Path

from VeraGridEngine.Utils.cache import clean_pycache_folders


def test_clean_pycache_folders(tmp_path: Path) -> None:
    """
    Verify that Python bytecode cache folders are removed below a root path.

    :param tmp_path: Temporary pytest folder.
    :return: None.
    """
    keep_folder: Path = tmp_path / "module"
    pycache_folder: Path = keep_folder / "__pycache__"
    nested_folder: Path = keep_folder / "nested" / "__pycache__"

    # Build a small directory tree with two cache folders and one normal folder.
    pycache_folder.mkdir(parents=True)
    nested_folder.mkdir(parents=True)

    # Clean only the cache folders below the selected root.
    removed_count: int = clean_pycache_folders(tmp_path)

    assert removed_count == 2
    assert keep_folder.is_dir()
    assert not pycache_folder.exists()
    assert not nested_folder.exists()
