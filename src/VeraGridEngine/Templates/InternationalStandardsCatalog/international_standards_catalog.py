# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""Typed catalog of international-standard RMS dynamic models."""

from __future__ import annotations

from typing import Sequence

from VeraGridEngine.Devices.Dynamic.rms_template import RmsModelTemplate
from VeraGridEngine.Devices.Dynamic.var_factory import VarFactory
from VeraGridEngine.enumerations import DeviceType, InternationalStandardModel
from VeraGridEngine.Templates.Rms.international_standards import build_international_standard_template


class InternationalStandardTemplateDescriptor:
    """Describe one draggable international-standard RMS model."""

    __slots__ = ("_model", "_display_label", "_category_path", "_module_folder", "_device_type")

    def __init__(
            self,
            model: InternationalStandardModel,
            display_label: str,
            category_path: Sequence[str],
            module_folder: str,
            device_type: DeviceType | None = None,
    ) -> None:
        """Store the stable catalog metadata for one RMS model.

        :param model: Enumerated model used for type-explicit materialization.
        :param display_label: Human-facing Dynamic Editor label.
        :param category_path: Nested branch below International standards.
        :param module_folder: Physical subpackage containing the model module.
        :param device_type: MultiCircuit device type represented by a complete
            device model, or ``None`` when the template is a control component.
        :return: None.
        """
        self._model: InternationalStandardModel = model
        self._display_label: str = display_label
        self._category_path: tuple[str, ...] = tuple(category_path)
        self._module_folder: str = module_folder
        self._device_type: DeviceType | None = device_type

    @property
    def model(self) -> InternationalStandardModel:
        """Return the international-standard model identifier.

        :return: Enumerated model identifier.
        """
        return self._model

    @property
    def display_label(self) -> str:
        """Return the label displayed in the Dynamic Editor library.

        :return: Human-facing model label.
        """
        return self._display_label

    @property
    def category_path(self) -> Sequence[str]:
        """Return the ordered Dynamic Editor category path.

        :return: Immutable category sequence.
        """
        return self._category_path

    @property
    def module_folder(self) -> str:
        """Return the physical model-family package name.

        :return: Subpackage below international_standards.
        """
        return self._module_folder

    @property
    def device_type(self) -> DeviceType | None:
        """Return the MultiCircuit device type represented by the template.

        Control and protection templates return ``None`` because they are
        building blocks inside a device model rather than device templates.

        :return: Associated device type, or ``None`` for non-device models.
        """
        return self._device_type

    @property
    def template_key(self) -> str:
        """Return the stable serialization-independent catalog key.

        :return: Standard model code used by the existing model enum.
        """
        return self._model.value

    @property
    def search_text(self) -> str:
        """Return searchable model, family, and module terms.

        :return: Space-separated case-insensitive search text.
        """
        category_text: str = " ".join(self._category_path)
        return f"{self._display_label} {self._model.name} {self._model.value} {category_text}".strip()

    @property
    def module_relative_path(self) -> str:
        """Return the model path relative to international_standards.

        :return: Slash-separated Python source path.
        """
        return f"{self._module_folder}/{self._model.value}.py"


def load_international_standard_template(
        descriptor: InternationalStandardTemplateDescriptor,
        var_factory: VarFactory,
        name: str | None = None,
) -> RmsModelTemplate:
    """Materialize a fresh RMS model represented by one catalog descriptor.

    :param descriptor: Typed model and catalog metadata.
    :param var_factory: Factory allocating fresh symbolic identities.
    :param name: Optional explicit runtime instance name.
    :return: Materialized international-standard RMS template.
    """
    return build_international_standard_template(
        model=descriptor.model,
        vf=var_factory,
        name=name,
    )
