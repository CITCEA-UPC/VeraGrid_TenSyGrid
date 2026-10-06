# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

class UcteComment:
    """
    UcteComment device holding comment line contents.
    """

    __slots__ = (
        "content",
    )

    def __init__(self) -> None:
        """
        Initialize the UcteComment device.
        """
        self.content: str = ""

    def parse(self, line: str) -> None:
        """
        Parse comment text from line.

        :param line: Raw text line.
        :return: None
        """
        self.content = line.strip()
