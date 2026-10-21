"""Authoritative parameters for STAMP's ``WSCC_SG_GFOR_GFOL`` case.

Values are transcribed from the three workbooks in ``STAMP/01_data/cases``.
Keeping them typed and centralized makes base conversions and model audits
explicit instead of hiding them inside the VeraGrid device objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi


SYSTEM_BASE_MVA = 100.0
NOMINAL_FREQUENCY_HZ = 50.0
OMEGA_BASE = 2.0 * pi * NOMINAL_FREQUENCY_HZ


@dataclass(frozen=True)
class StampSynchronousGeneratorParameters:
    bus: int = 1
    rated_mva: float = 310.0
    terminal_kv: float = 180.0
    p_pu_system: float = 1.63
    q_pu_system: float = 0.07
    voltage_pu: float = 1.02579
    rs: float = 0.0025
    xl: float = 0.2
    xd: float = 1.8
    xd_prime: float = 0.3
    xd_subtransient: float = 0.25
    xq: float = 1.7
    xq_prime: float = 0.55
    xq_subtransient: float = 0.25
    td0_prime: float = 8.0
    td0_subtransient: float = 0.03
    tq0_prime: float = 0.4
    tq0_subtransient: float = 0.05
    transformer_r: float = 0.002
    transformer_x: float = 0.1
    inertia_h: float = 6.5
    damping_d: float = 0.0
    exciter_tr: float = 0.01
    exciter_ka: float = 200.0
    exciter_ta: float = 0.015
    exciter_tb: float = 10.0
    exciter_tc: float = 1.0
    governor_r: float = 0.05
    governor_t1: float = 0.0
    governor_t2: float = 0.0
    governor_t3: float = 0.1
    governor_dt: float = 0.0
    turbine_k1: float = 0.3
    turbine_k2: float = 0.4
    turbine_k3: float = 0.0
    turbine_k4: float = 0.0
    turbine_k5: float = 0.3
    turbine_k6: float = 0.0
    turbine_k7: float = 0.0
    turbine_k8: float = 0.0
    turbine_t4: float = 0.3
    turbine_t5: float = 7.0
    turbine_t6: float = 0.6
    # The processed STAMP case uses a fourth turbine pole at 1/0.3 s.
    turbine_t7: float = 0.3


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

STAMP_SG = StampSynchronousGeneratorParameters()
