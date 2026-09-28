# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.  
# SPDX-License-Identifier: MPL-2.0

from typing import Union, Tuple
from VeraGridEngine.Devices.Parents.editable_device import EditableDevice, DeviceType, GCProp
from VeraGridEngine.enumerations import PrpCat


class MarketUnitsGroup(EditableDevice):
    """
    Investments group
    """
    __slots__ = (
        'category',
        'color'
    )

    LOCAL_PROPERTY_DECLARATIONS: Tuple[GCProp, ...] = (
        GCProp(
            prop_name='category',
            units='',
            tpe=str,
            definition='Some tag to category the investment group',
            cat=[PrpCat.INV],
        ),
        GCProp(
            prop_name='color',
            units='',
            tpe=str,
            definition='Color to paint',
            is_color=True,
            cat=[PrpCat.TP],
        ),
    )

    def __init__(self,
                 idtag: Union[str, None] = None,
                 name: str = "MarketUnitGroup",
                 category: str = '',
                 comment: str = "",
                 discount_rate: float = 5.0,
                 CAPEX: float = 0,
                 color: str | None = None):
        """
        Contingency group
        :param idtag: Unique identifier
        :param name: contingency group name
        :param category: tag to category the group
        :param comment: comment
        :param discount_rate: discount rate (%)
        :param CAPEX: Capital Expenditure of the group (added to the individual investments' capex)
        """

        EditableDevice.__init__(self,
                                name=name,
                                idtag=idtag,
                                code='',
                                device_type=DeviceType.MarketUnitsGroupDevice,
                                comment=comment)

        # Contingency type
        self.category: str = category

        self.color = color if color is not None else self.rnd_color()
