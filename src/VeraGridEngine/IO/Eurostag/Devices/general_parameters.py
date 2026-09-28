from __future__ import annotations

from VeraGridEngine.IO.Eurostag.Devices.eurostag_base import slice_float


class EurostagGeneralParameters:
    PREFIX = "9 "
    BASE_POWER = (66, 74)

    __slots__ = (
        "base_power",
    )

    def __init__(self) -> None:
        self.base_power: float = 100.0

    def parse_line(self, line: str) -> None:
        self.base_power = slice_float(line, *self.BASE_POWER, default=100.0)
