# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.  
# SPDX-License-Identifier: MPL-2.0


from VeraGridEngine.basic_structures import CxVec


class ContinuationPowerFlowInput:
    """
    Input data for Continuation Power Flow.
    """
    __slots__ = (
        "Sbase",
        "Starget",
        "Vbase",
        "base_overload_number",
    )

    def __init__(
        self,
        Sbase: CxVec,
        Vbase: CxVec,
        Starget: CxVec,
        base_overload_number: int = 0,
    ) -> None:
        """
        ContinuationPowerFlowInput constructor.

        :param Sbase: Initial power array
        :param Vbase: Initial voltage array
        :param Starget: Final power array
        :param base_overload_number: Number of overloads in the base situation
        """
        self.Sbase: CxVec = Sbase
        self.Starget: CxVec = Starget
        self.Vbase: CxVec = Vbase
        self.base_overload_number: int = base_overload_number

