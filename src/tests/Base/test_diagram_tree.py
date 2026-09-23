# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from pathlib import Path
import VeraGridEngine.api as vge
from VeraGridEngine.Devices.Diagrams.diagram_tree import DiagramFolder, DiagramTree
from VeraGridEngine.Devices.Diagrams.schematic_diagram import SchematicDiagram
from VeraGridEngine.Devices.Diagrams.map_diagram import MapDiagram


def test_diagram_folder_nesting_and_membership():
    """
    Verify folders upon folders can be nested arbitrarily, holding diagrams and subfolders.
    """
    root_tree = DiagramTree()
    assert len(root_tree) == 0
    assert not root_tree

    # Add unclassified diagram to root
    d_root = SchematicDiagram(name="Root Diagram")
    root_tree.append(d_root)
    assert len(root_tree) == 1
    assert d_root in root_tree
    assert root_tree[0] is d_root

    # Add top-level folder
    f_tx = root_tree.add_folder("Transmission")
    assert f_tx.parent is None
    assert f_tx in root_tree.folders

    # Add subfolder
    f_sub = f_tx.add_folder("Substations")
    assert f_sub.parent is f_tx
    assert f_sub in f_tx.folders

    # Add diagrams into subfolder
    d_sub1 = SchematicDiagram(name="Substation Alpha")
    d_sub2 = MapDiagram(name="Substation Beta Map")
    f_sub.add_diagram(d_sub1)
    f_sub.add_diagram(d_sub2)

    assert d_sub1.group is f_sub
    assert d_sub2.group is f_sub

    # DiagramTree sequence protocol should include all diagrams recursively
    all_diagrams = list(root_tree)
    assert len(all_diagrams) == 3
    assert len(root_tree) == 3
    assert d_root in root_tree
    assert d_sub1 in root_tree
    assert d_sub2 in root_tree

    # Test lookups
    assert root_tree.find_folder(f_sub.idtag) is f_sub
    assert root_tree.find_folder(f_tx.idtag) is f_tx
    assert root_tree.find_diagram_parent(d_sub1) is f_sub
    assert root_tree.find_diagram_parent(d_root) is None

    # Test moving diagram to root
    root_tree.move_diagram(d_sub1, target_folder=None)
    assert d_sub1.group is None
    assert d_sub1 in root_tree.diagrams
    assert d_sub1 not in f_sub.diagrams
    assert len(root_tree) == 3

    # Test removing diagram
    root_tree.remove(d_sub2)
    assert len(root_tree) == 2
    assert d_sub2 not in root_tree


def test_diagram_tree_serialization_roundtrip():
    """
    Verify get_data_dict and parse_data preserve nested folder structure.
    """
    tree = DiagramTree()
    d1 = SchematicDiagram(name="D1")
    d2 = SchematicDiagram(name="D2")
    d3 = MapDiagram(name="D3")

    f1 = tree.add_folder("Region A")
    f1_1 = f1.add_folder("Sub-Area 1")
    f1_1.add_diagram(d1)
    f1.add_diagram(d2)
    tree.add_diagram(d3)

    data = tree.get_data_dict()
    assert "folders" in data
    assert "diagram_ids" in data
    assert d3.idtag in data["diagram_ids"]

    # Reconstruct into a fresh tree
    diagrams_by_id = {d1.idtag: d1, d2.idtag: d2, d3.idtag: d3}
    new_tree = DiagramTree()
    new_tree.parse_data(data, diagrams_by_id)

    assert len(new_tree) == 3
    assert len(new_tree.folders) == 1
    reconstructed_f1 = new_tree.folders[0]
    assert reconstructed_f1.name == "Region A"
    assert len(reconstructed_f1.folders) == 1
    assert len(reconstructed_f1.diagrams) == 1
    assert reconstructed_f1.diagrams[0].idtag == d2.idtag
    assert reconstructed_f1.folders[0].name == "Sub-Area 1"
    assert reconstructed_f1.folders[0].diagrams[0].idtag == d1.idtag
    assert d1.group is reconstructed_f1.folders[0]
    assert d2.group is reconstructed_f1
    assert d3 in new_tree.diagrams


def test_circuit_diagram_tree_persistence_roundtrip(tmp_path: Path):
    """
    Verify a MultiCircuit with nested diagram folders saves and loads with all folders preserved.
    """
    grid = vge.MultiCircuit(name="tree_test_grid")
    b1 = grid.add_bus(vge.Bus(name="B1", Vnom=110.0))
    b2 = grid.add_bus(vge.Bus(name="B2", Vnom=110.0))
    grid.add_line(vge.Line(name="L1", bus_from=b1, bus_to=b2, r=0.1, x=0.2))

    # Add diagrams with folders
    folder_tx = grid.diagrams.add_folder("Transmission")
    folder_400k = folder_tx.add_folder("400kV Grid")

    diag_root = SchematicDiagram(name="Overview")
    diag_tx = SchematicDiagram(name="TX General")
    diag_400k = SchematicDiagram(name="400kV Detail")

    grid.add_diagram(diag_root)
    grid.add_diagram(diag_tx, folder=folder_tx)
    grid.add_diagram(diag_400k, folder=folder_400k)

    assert len(grid.diagrams) == 3
    assert len(grid.diagrams.folders) == 1
    assert len(grid.diagrams.folders[0].folders) == 1

    file_path = tmp_path / "tree_test.veragrid"
    vge.save_file(grid=grid, filename=str(file_path))

    # Load back
    loaded_grid = vge.open_file(filename=str(file_path))
    assert len(loaded_grid.diagrams) == 3
    assert len(loaded_grid.diagrams.folders) == 1
    loaded_tx = loaded_grid.diagrams.folders[0]
    assert loaded_tx.name == "Transmission"
    assert len(loaded_tx.folders) == 1
    loaded_400k = loaded_tx.folders[0]
    assert loaded_400k.name == "400kV Grid"

    # Diagrams check
    assert len(loaded_tx.diagrams) == 1
    assert loaded_tx.diagrams[0].name == "TX General"
    assert len(loaded_400k.diagrams) == 1
    assert loaded_400k.diagrams[0].name == "400kV Detail"
    assert len(loaded_grid.diagrams.diagrams) == 1
    assert loaded_grid.diagrams.diagrams[0].name == "Overview"


def test_legacy_file_diagram_compatibility(tmp_path: Path):
    """
    Verify opening an archive that lacks tree.json loads all diagrams into root level.
    """
    grid = vge.MultiCircuit(name="legacy_grid")
    grid.add_bus(vge.Bus(name="B1", Vnom=10.0))
    d1 = SchematicDiagram(name="Legacy Diagram 1")
    d2 = SchematicDiagram(name="Legacy Diagram 2")
    grid.add_diagram(d1)
    grid.add_diagram(d2)

    file_path = tmp_path / "legacy_test.veragrid"
    vge.save_file(grid=grid, filename=str(file_path))

    # Manually remove tree.json from the zip to simulate a legacy archive
    import zipfile
    clean_zip_path = tmp_path / "legacy_cleaned.veragrid"
    with zipfile.ZipFile(file_path, 'r') as zin:
        with zipfile.ZipFile(clean_zip_path, 'w') as zout:
            for item in zin.infolist():
                if not item.filename.endswith("tree.json"):
                    zout.writestr(item, zin.read(item.filename))

    # Open the simulated legacy file
    loaded_legacy = vge.open_file(filename=str(clean_zip_path))
    assert len(loaded_legacy.diagrams) == 2
    assert len(loaded_legacy.diagrams.folders) == 0
    diagram_names = {d.name for d in loaded_legacy.diagrams}
    assert diagram_names == {"Legacy Diagram 1", "Legacy Diagram 2"}
