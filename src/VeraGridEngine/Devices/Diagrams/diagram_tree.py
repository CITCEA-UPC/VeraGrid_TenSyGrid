# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

import collections.abc
import uuid
from typing import List, Dict, Any, Union, Optional, Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from VeraGridEngine.Devices.Diagrams.base_diagram import BaseDiagram
    from VeraGridEngine.Devices.types import ALL_DEV_TYPES


class DiagramFolder:
    """
    Represents a category/folder of diagrams, which can contain both subfolders
    and diagrams, supporting arbitrary nesting ("folders upon folders").
    """
    __slots__ = (
        'idtag',
        'name',
        'parent',
        'folders',
        'diagrams',
    )

    def __init__(self,
                 name: str = "New Folder",
                 idtag: Optional[str] = None,
                 parent: Optional[DiagramFolder] = None) -> None:
        self.idtag: str = uuid.uuid4().hex if idtag is None else idtag.replace('_', '').replace('-', '')
        self.name: str = name
        self.parent: Optional[DiagramFolder] = parent
        self.folders: List[DiagramFolder] = list()
        self.diagrams: List[BaseDiagram] = list()

    def add_folder(self, folder_or_name: Union[str, DiagramFolder]) -> DiagramFolder:
        """
        Add a child subfolder.

        :param folder_or_name: Folder name or DiagramFolder instance
        :return: Created or added DiagramFolder
        """
        if isinstance(folder_or_name, str):
            folder = DiagramFolder(name=folder_or_name, parent=self)
        else:
            folder = folder_or_name
            folder.parent = self
        if folder not in self.folders:
            self.folders.append(folder)
        return folder

    def remove_folder(self, folder: DiagramFolder) -> None:
        """
        Remove a child subfolder.

        :param folder: DiagramFolder to remove
        """
        if folder in self.folders:
            self.folders.remove(folder)
            folder.parent = None

    def add_diagram(self, diagram: BaseDiagram) -> None:
        """
        Add a diagram to this folder.

        :param diagram: BaseDiagram instance
        """
        if diagram not in self.diagrams:
            self.diagrams.append(diagram)
        diagram.group = self

    def remove_diagram(self, diagram: BaseDiagram) -> None:
        """
        Remove a diagram from this folder.

        :param diagram: BaseDiagram instance
        """
        if diagram in self.diagrams:
            self.diagrams.remove(diagram)
        if getattr(diagram, 'group', None) is self:
            diagram.group = None

    def get_all_diagrams(self) -> List[BaseDiagram]:
        """
        Recursively collect all diagrams in this folder and its subfolders.

        :return: List of all diagrams
        """
        result: List[BaseDiagram] = list(self.diagrams)
        for subfolder in self.folders:
            result.extend(subfolder.get_all_diagrams())
        return result

    def get_all_folders(self) -> List[DiagramFolder]:
        """
        Recursively collect all subfolders.

        :return: List of all subfolders
        """
        result: List[DiagramFolder] = list(self.folders)
        for subfolder in self.folders:
            result.extend(subfolder.get_all_folders())
        return result

    def find_folder(self, idtag: str) -> Optional[DiagramFolder]:
        """
        Find a folder by idtag in this subtree.

        :param idtag: Target folder idtag
        :return: DiagramFolder or None
        """
        if self.idtag == idtag:
            return self
        for subfolder in self.folders:
            found = subfolder.find_folder(idtag)
            if found is not None:
                return found
        return None

    def get_data_dict(self) -> Dict[str, Any]:
        """
        Serialize this folder and its subtree into a dictionary.

        :return: Serializable dictionary
        """
        return {
            'idtag': self.idtag,
            'name': self.name,
            'folders': [f.get_data_dict() for f in self.folders],
            'diagram_ids': [d.idtag for d in self.diagrams],
        }

    def parse_data(self,
                   data: Dict[str, Any],
                   diagrams_by_id: Dict[str, BaseDiagram]) -> None:
        """
        Populate this folder and its subtree from serialized data.

        :param data: Serialized folder dictionary
        :param diagrams_by_id: Lookup dictionary of existing diagram instances by idtag
        """
        self.idtag = data.get('idtag', self.idtag)
        self.name = data.get('name', self.name)
        self.folders = list()
        self.diagrams = list()

        for d_id in data.get('diagram_ids', []):
            d = diagrams_by_id.get(d_id)
            if d is not None:
                self.add_diagram(d)

        for f_dict in data.get('folders', []):
            subfolder = DiagramFolder(name=f_dict.get('name', 'Folder'),
                                      idtag=f_dict.get('idtag'),
                                      parent=self)
            subfolder.parse_data(f_dict, diagrams_by_id)
            self.folders.append(subfolder)

    def copy(self,
             diagram_copy_map: Dict[str, BaseDiagram],
             parent: Optional[DiagramFolder] = None) -> DiagramFolder:
        """
        Deep-copy this folder while re-binding diagrams via diagram_copy_map.

        :param diagram_copy_map: Map of original diagram idtag to copied BaseDiagram
        :param parent: Optional parent folder for the copy
        :return: Copied DiagramFolder
        """
        cpy = DiagramFolder(name=self.name, idtag=self.idtag, parent=parent)
        for d in self.diagrams:
            if d.idtag in diagram_copy_map:
                cpy.add_diagram(diagram_copy_map[d.idtag])
        for f in self.folders:
            cpy.folders.append(f.copy(diagram_copy_map=diagram_copy_map, parent=cpy))
        return cpy

    def __repr__(self) -> str:
        return f"<DiagramFolder {self.name!r} ({len(self.folders)} folders, {len(self.diagrams)} diagrams)>"


class DiagramTree(collections.abc.MutableSequence):
    """
    Manages the complete diagrams hierarchy (root folders and unclassified root diagrams).
    Implements MutableSequence so that Assets._diagrams can BE a DiagramTree directly,
    ensuring 100% retrocompatibility with all existing list-based access patterns without
    any synchronization drift.
    Pure Python: contains zero Qt dependencies.
    """
    __slots__ = (
        'folders',
        'diagrams',
    )

    def __init__(self,
                 folders: Optional[List[DiagramFolder]] = None,
                 diagrams: Optional[List[BaseDiagram]] = None) -> None:
        self.folders: List[DiagramFolder] = list() if folders is None else list(folders)
        self.diagrams: List[BaseDiagram] = list() if diagrams is None else list(diagrams)

    # -------------------------------------------------------------------------
    # List / Sequence interface methods for retrocompatibility
    # -------------------------------------------------------------------------
    def __iter__(self):
        yield from self.get_all_diagrams()

    def __len__(self) -> int:
        return len(self.get_all_diagrams())

    def __getitem__(self, index: Union[int, slice]):
        return self.get_all_diagrams()[index]

    def __setitem__(self, index: int, value: BaseDiagram) -> None:
        all_d = self.get_all_diagrams()
        old_d = all_d[index]
        parent = self.find_diagram_parent(old_d)
        self.remove_diagram(old_d)
        self.add_diagram(value, folder=parent)

    def __delitem__(self, index: int) -> None:
        all_d = self.get_all_diagrams()
        d = all_d[index]
        self.remove_diagram(d)

    def __contains__(self, item: Any) -> bool:
        return item in self.get_all_diagrams()

    def __bool__(self) -> bool:
        return len(self.get_all_diagrams()) > 0 or len(self.folders) > 0

    def append(self, diagram: BaseDiagram, folder: Optional[DiagramFolder] = None) -> None:
        self.add_diagram(diagram, folder=folder)

    def extend(self, diagrams: Iterable[BaseDiagram]) -> None:
        for d in diagrams:
            self.append(d)

    def remove(self, diagram: BaseDiagram) -> None:
        if diagram not in self:
            raise ValueError(f"{diagram} not found in DiagramTree")
        self.remove_diagram(diagram)

    def pop(self, index: int = -1) -> BaseDiagram:
        all_d = self.get_all_diagrams()
        if not all_d:
            raise IndexError("pop from empty DiagramTree")
        d = all_d[index]
        self.remove_diagram(d)
        return d

    def clear(self) -> None:
        self.folders.clear()
        self.diagrams.clear()

    def index(self, diagram: BaseDiagram, *args) -> int:
        return self.get_all_diagrams().index(diagram, *args)

    def count(self, diagram: BaseDiagram) -> int:
        return self.get_all_diagrams().count(diagram)

    def insert(self, index: int, diagram: BaseDiagram) -> None:
        self.remove_diagram(diagram)
        insert_pos = min(max(index, 0), len(self.diagrams))
        self.diagrams.insert(insert_pos, diagram)
        diagram.group = None

    # -------------------------------------------------------------------------
    # Tree management methods
    # -------------------------------------------------------------------------
    def add_folder(self,
                   folder_or_name: Union[str, DiagramFolder],
                   parent: Optional[DiagramFolder] = None) -> DiagramFolder:
        """
        Add a folder at the root level or under the given parent.

        :param folder_or_name: Folder name or DiagramFolder instance
        :param parent: Optional parent DiagramFolder
        :return: Created or added DiagramFolder
        """
        if parent is not None and isinstance(parent, DiagramFolder):
            return parent.add_folder(folder_or_name)
        if isinstance(folder_or_name, str):
            folder = DiagramFolder(name=folder_or_name, parent=None)
        else:
            folder = folder_or_name
            folder.parent = None
        if folder not in self.folders:
            self.folders.append(folder)
        return folder

    def remove_folder(self, folder: DiagramFolder) -> None:
        """
        Remove a folder from the tree.

        :param folder: DiagramFolder to remove
        """
        if folder.parent is not None:
            folder.parent.remove_folder(folder)
        elif folder in self.folders:
            self.folders.remove(folder)
            folder.parent = None

    def add_diagram(self,
                    diagram: BaseDiagram,
                    folder: Optional[DiagramFolder] = None) -> None:
        """
        Add a diagram to the tree, either under a folder or at the root level.

        :param diagram: BaseDiagram to add
        :param folder: Target folder, or None for root level
        """
        self.remove_diagram(diagram)
        if folder is not None:
            folder.add_diagram(diagram)
        else:
            if diagram not in self.diagrams:
                self.diagrams.append(diagram)
            diagram.group = None

    def remove_diagram(self, diagram: BaseDiagram) -> None:
        """
        Remove a diagram from wherever it is in the tree (root or folder).

        :param diagram: BaseDiagram to remove
        """
        if diagram in self.diagrams:
            self.diagrams.remove(diagram)
            diagram.group = None
        for folder in self.get_all_folders():
            if diagram in folder.diagrams:
                folder.remove_diagram(diagram)

    def find_folder(self, idtag: str) -> Optional[DiagramFolder]:
        """
        Find a folder by idtag in the entire tree.

        :param idtag: Target folder idtag
        :return: DiagramFolder or None
        """
        for folder in self.folders:
            found = folder.find_folder(idtag)
            if found is not None:
                return found
        return None

    def find_diagram_parent(self, diagram: BaseDiagram) -> Optional[DiagramFolder]:
        """
        Find the parent folder of a diagram, or None if at root level.

        :param diagram: Diagram to locate
        :return: Parent DiagramFolder or None
        """
        if diagram in self.diagrams:
            return None
        for folder in self.get_all_folders():
            if diagram in folder.diagrams:
                return folder
        return None

    def get_all_folders(self) -> List[DiagramFolder]:
        """
        Recursively collect all folders in the tree.

        :return: List of all folders
        """
        result: List[DiagramFolder] = list(self.folders)
        for folder in self.folders:
            result.extend(folder.get_all_folders())
        return result

    def get_all_diagrams(self) -> List[BaseDiagram]:
        """
        Recursively collect all diagrams in the tree (root diagrams + folder diagrams).

        :return: List of all diagrams in depth-first order
        """
        result: List[BaseDiagram] = list(self.diagrams)
        for folder in self.folders:
            result.extend(folder.get_all_diagrams())
        return result

    def move_diagram(self,
                     diagram: BaseDiagram,
                     target_folder: Optional[DiagramFolder] = None) -> None:
        """
        Move a diagram to a target folder (or root if None).

        :param diagram: Diagram to move
        :param target_folder: Target folder or None
        """
        self.add_diagram(diagram=diagram, folder=target_folder)

    def move_folder(self,
                    folder: DiagramFolder,
                    target_parent: Optional[DiagramFolder] = None) -> None:
        """
        Move a folder to a new parent folder (or root if None).

        :param folder: Folder to move
        :param target_parent: New parent folder or None
        """
        if target_parent is folder or (target_parent is not None and target_parent.find_folder(folder.idtag) is not None):
            raise ValueError("Cannot move a folder into itself or one of its descendants")
        self.remove_folder(folder)
        if target_parent is not None:
            target_parent.add_folder(folder)
        else:
            folder.parent = None
            if folder not in self.folders:
                self.folders.append(folder)

    def get_data_dict(self) -> Dict[str, Any]:
        """
        Serialize the complete tree hierarchy into a dictionary.

        :return: Serializable dictionary
        """
        return {
            'folders': [f.get_data_dict() for f in self.folders],
            'diagram_ids': [d.idtag for d in self.diagrams],
        }

    def parse_data(self,
                   data: Dict[str, Any],
                   diagrams_by_id: Dict[str, BaseDiagram]) -> None:
        """
        Populate the tree from serialized hierarchy data.

        :param data: Serialized tree dictionary
        :param diagrams_by_id: Lookup dictionary of existing diagram instances by idtag
        """
        self.folders = list()
        self.diagrams = list()

        for d_id in data.get('diagram_ids', []):
            d = diagrams_by_id.get(d_id)
            if d is not None:
                self.diagrams.append(d)
                d.group = None

        for f_dict in data.get('folders', []):
            folder = DiagramFolder(name=f_dict.get('name', 'Folder'),
                                   idtag=f_dict.get('idtag'),
                                   parent=None)
            folder.parse_data(f_dict, diagrams_by_id)
            self.folders.append(folder)

        # Any diagram in diagrams_by_id not yet placed in the tree is placed at root level
        all_accounted = set(d.idtag for d in self.get_all_diagrams())
        for d_id, d in diagrams_by_id.items():
            if d_id not in all_accounted:
                self.diagrams.append(d)
                d.group = None

    def copy(self,
             diagram_copy_map: Optional[Dict[str, BaseDiagram]] = None,
             obj_dict: Optional[Dict[str, Dict[str, ALL_DEV_TYPES]]] = None) -> DiagramTree:
        """
        Deep-copy the tree hierarchy.

        :param diagram_copy_map: Optional map of original diagram idtag to copied BaseDiagram
        :param obj_dict: Optional circuit elements dict used to clone diagrams when diagram_copy_map is not given
        :return: Copied DiagramTree
        """
        if diagram_copy_map is None:
            diagram_copy_map = {d.idtag: d.copy(obj_dict=obj_dict) for d in self.get_all_diagrams()}

        cpy = DiagramTree()
        for d in self.diagrams:
            if d.idtag in diagram_copy_map:
                cpy.add_diagram(diagram_copy_map[d.idtag], folder=None)
        for f in self.folders:
            cpy.folders.append(f.copy(diagram_copy_map=diagram_copy_map, parent=None))
        return cpy

    def __repr__(self) -> str:
        return f"<DiagramTree ({len(self.folders)} folders, {len(self)} diagrams)>"
