# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from datetime import datetime, timedelta
from typing import List, Union

import numpy as np
import pandas as pd

from VeraGridEngine.Devices.Profiles import ProfileFloat
from VeraGridEngine.Devices.Substation.bus import Bus
from VeraGridEngine.Devices.Substation.substation import Substation


def profile_is_missing(profile: ProfileFloat, default_value: float, expected_size: int) -> bool:
    """
    Determine whether a float profile should be considered missing for wizard weather filling.

    :param profile: Profile to inspect.
    :param default_value: Snapshot value associated with the profile.
    :param expected_size: Expected MultiCircuit time profile length.
    :return: True if the profile is missing and can be filled.
    """
    if profile.is_initialized:
        if profile.size() == expected_size:
            profile_array: np.ndarray = profile.toarray()

            if len(profile_array) == expected_size:
                if np.allclose(profile_array, default_value):
                    if np.isclose(default_value, 0.0):
                        return True
                    else:
                        return False
                else:
                    return False
            else:
                return True
        else:
            return True
    else:
        return True


def fill_substation_weather_profiles(bus: Bus,
                                     temperature: Union[np.ndarray, None],
                                     wind_speed: Union[np.ndarray, None],
                                     irradiation: Union[np.ndarray, None],
                                     expected_size: int) -> None:
    """
    Fill missing weather profiles on the substation associated with a bus.

    Existing non-empty weather profiles are preserved. The MultiCircuit time profile length is used as the required
    size for all incoming arrays.

    :param bus: Bus associated with the generated profile.
    :param temperature: Air temperature profile in degrees Celsius.
    :param wind_speed: Wind speed profile in m/s.
    :param irradiation: Solar irradiation profile in W/m2.
    :param expected_size: Expected MultiCircuit time profile length.
    :return: Nothing.
    """
    substation: Union[Substation, None] = bus.substation

    if substation is None:
        return
    else:
        if temperature is None:
            pass
        else:
            if len(temperature) == expected_size:
                if profile_is_missing(profile=substation.temperature_prof,
                                      default_value=substation.temperature,
                                      expected_size=expected_size):
                    substation.temperature_prof = np.asarray(temperature, dtype=float)
                    substation.temperature = float(np.asarray(temperature, dtype=float)[0])
                else:
                    pass
            else:
                pass

        if wind_speed is None:
            pass
        else:
            if len(wind_speed) == expected_size:
                if profile_is_missing(profile=substation.wind_speed_prof,
                                      default_value=substation.wind_speed,
                                      expected_size=expected_size):
                    substation.wind_speed_prof = np.asarray(wind_speed, dtype=float)
                    substation.wind_speed = float(np.asarray(wind_speed, dtype=float)[0])
                else:
                    pass
            else:
                pass

        if irradiation is None:
            pass
        else:
            if len(irradiation) == expected_size:
                if profile_is_missing(profile=substation.irradiation_prof,
                                      default_value=substation.irradiation,
                                      expected_size=expected_size):
                    substation.irradiation_prof = np.asarray(irradiation, dtype=float)
                    substation.irradiation = float(np.asarray(irradiation, dtype=float)[0])
                else:
                    pass
            else:
                pass


def remap_timestamp_to_base_year(ts: Union[datetime, pd.Timestamp],
                                 base_year: int,
                                 start_year: Union[int, None] = None) -> datetime:
    """
    Remap a timestamp to a historical target year based on base_year and start_year,
    preserving month, day (with leap-year handling), and time components.

    :param ts: Source timestamp to remap.
    :param base_year: Base reference year.
    :param start_year: Optional starting year of the profile. If None, ts.year is used.
    :return: Remapped datetime object.
    """
    if start_year is None:
        ref_year: int = int(ts.year)
    else:
        ref_year = int(start_year)

    target_year: int = base_year + int(ts.year - ref_year)
    target_month: int = int(ts.month)
    target_day: int = int(ts.day)

    # Prevent ValueError: day is out of range for month when remapping Feb 29 onto non-leap years
    if target_month == 2 and target_day == 29:
        is_leap: bool = (target_year % 4 == 0) and (target_year % 100 != 0 or target_year % 400 == 0)
        if not is_leap:
            target_day = 28
        else:
            pass
    else:
        pass

    return datetime(year=target_year,
                    month=target_month,
                    day=target_day,
                    hour=int(ts.hour),
                    minute=int(ts.minute),
                    second=int(ts.second),
                    microsecond=int(ts.microsecond))


def build_mapped_time_index(time_index: pd.DatetimeIndex, base_year: int) -> pd.DatetimeIndex:
    """
    Map circuit timestamps to a historical weather year while preserving month, day and time.

    :param time_index: Circuit time index.
    :param base_year: Historical base year used for the mapped timestamps.
    :return: Historical weather time index.
    """
    if len(time_index) == 0:
        return pd.DatetimeIndex(list())
    else:
        start_year: int = int(time_index[0].year)
        mapped_timestamps: List[datetime] = list()

        for ts in time_index:
            mapped_ts: datetime = remap_timestamp_to_base_year(ts=ts,
                                                               base_year=base_year,
                                                               start_year=start_year)
            mapped_timestamps.append(mapped_ts)

        return pd.DatetimeIndex(pd.to_datetime(mapped_timestamps))


def get_longitude_time_offset(longitude: float) -> timedelta:
    """
    Get the local solar time offset from longitude.

    :param longitude: Site longitude in degrees.
    :return: Offset to add to UTC timestamps to obtain local solar time.
    """
    offset_hours: float = float(longitude) / 15.0

    return timedelta(hours=offset_hours)

