"""Parameters for the local STAMP GFOR/GFOL converter models."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi

NOMINAL_FREQUENCY_HZ = 50.0
OMEGA_BASE = 2.0 * pi * NOMINAL_FREQUENCY_HZ


@dataclass(frozen=True)
class StampConverterParameters:
    number: int
    bus: int
    mode: str
    rated_mva: float = 100.0
    p_pu_system: float = 0.85
    q_pu_system: float = -0.11
    voltage_pu: float = 1.025
    transformer_r: float = 0.002
    transformer_x: float = 0.1
    converter_r: float = 0.005
    converter_x: float = 0.15
    capacitor_b: float = 0.15
    damping_r: float = 0.001
    current_settling_time: float = 0.001
    measurement_delay: float = 1.0e-5
    commutation_delay: float = -1.0
    zoh_delay: float = -1.0
    frequency_droop_gain: float = 0.0
    frequency_droop_tau: float = 0.1
    voltage_droop_gain: float = 0.0
    voltage_droop_tau: float = 0.1
    pll_settling_time: float | None = None
    pll_damping: float | None = None
    active_power_tau: float | None = None
    reactive_power_tau: float | None = None
    voltage_settling_time: float | None = None
    voltage_damping: float | None = None
    voltage_feedforward_tau: float | None = None
    current_feedforward_tau: float | None = None

    @property
    def filter_inductance_seconds(self) -> float:
        return self.converter_x / OMEGA_BASE

    @property
    def transformer_inductance_seconds(self) -> float:
        return self.transformer_x / OMEGA_BASE

    @property
    def filter_capacitance_seconds(self) -> float:
        return self.capacitor_b / OMEGA_BASE

    @property
    def current_kp(self) -> float:
        return self.filter_inductance_seconds / self.current_settling_time

    @property
    def current_ki(self) -> float:
        return self.converter_r / self.current_settling_time

    @property
    def pll_gains(self) -> tuple[float, float]:
        if self.pll_settling_time is None or self.pll_damping is None:
            raise ValueError(f"{self.mode} has no PLL")
        natural_frequency = 4.0 / (self.pll_settling_time * self.pll_damping)
        kp = 2.0 * natural_frequency * self.pll_damping
        tau = 2.0 * self.pll_damping / natural_frequency
        return kp, kp / tau

    @property
    def voltage_pi_gains(self) -> tuple[float, float]:
        if self.voltage_settling_time is None or self.voltage_damping is None:
            raise ValueError(f"{self.mode} has no voltage PI")
        natural_frequency = 4.0 / (self.voltage_settling_time * self.voltage_damping)
        kp = 2.0 * self.voltage_damping * natural_frequency * self.filter_capacitance_seconds * 100.0
        ki = natural_frequency**2 * self.filter_capacitance_seconds
        return kp, ki


STAMP_GFOR = StampConverterParameters(
    number=1,
    bus=4,
    mode="GFOR",
    voltage_pu=1.02577,
    frequency_droop_gain=0.05,
    voltage_droop_gain=1.0 / 15.0,
    voltage_settling_time=0.05,
    voltage_damping=0.707,
    voltage_feedforward_tau=0.0001,
    current_feedforward_tau=0.0001,
)

STAMP_GFOL = StampConverterParameters(
    number=2,
    bus=6,
    mode="GFOL",
    voltage_pu=1.03235,
    frequency_droop_gain=0.5,
    voltage_droop_gain=2.0,
    pll_settling_time=0.1,
    pll_damping=0.707,
    active_power_tau=1.0,
    reactive_power_tau=1.0,
)

