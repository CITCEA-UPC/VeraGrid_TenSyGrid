# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
from VeraGridEngine.IO.ucte.devices.ucte_base import sub_str, sub_float, try_float, ucte_split
from VeraGridEngine.basic_structures import Logger


class UcteExchangePower:
    """
    UcteExchangePower device.
    """

    __slots__ = (
        "country1",
        "country2",
        "active_power",
        "comments",
    )

    def __init__(self) -> None:
        """
        Initialize the UcteExchangePower instance.
        """
        self.country1: str = ""  # 0-1: Country 1 (ISO code)
        self.country2: str = ""  # 3-4: Country 2 (ISO code)
        self.active_power: float = 0.0  # 6-12: Scheduled active power exchange (MW)
        self.comments: str = ""  # 14-25: Optional comments

    def parse(self, line: str, logger: Logger) -> None:
        """
        Parse exchange power from line.

        :param line: Raw text line.
        :param logger: Logger instance.
        :return: None
        """
        device: str = "Exchange Power"

        if len(line) == 27:
            self.country1 = sub_str(line, 0, 2, device, "country1", logger)
            self.country2 = sub_str(line, 3, 5, device, "country2", logger)
            self.active_power = sub_float(line, 6, 13, device, "active_power", logger)
            self.comments = sub_str(line, 14, 26, device, "comments", logger)
        else:
            logger.add_warning("Non canonical line length",
                               device_class=device,
                               value=len(line),
                               expected_value=27)

            chunks = ucte_split(line, prefix_lengths=(2, 2), total_fields=4, greedy_tail=True,
                                skip_all_separators=True)

            if len(chunks) >= 1:
                self.country1 = chunks[0]
            else:
                pass

            if len(chunks) >= 2:
                self.country2 = chunks[1]
            else:
                pass

            if len(chunks) >= 3:
                self.active_power = try_float(chunks[2], device, "active_power", logger)
            else:
                pass

            if len(chunks) >= 4:
                self.comments = chunks[3]
            else:
                pass
