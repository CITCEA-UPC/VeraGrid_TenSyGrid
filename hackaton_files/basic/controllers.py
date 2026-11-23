"""Connect native GFM and GFL converter controls to AC generator injections."""
from enum import Enum

import numpy as np

from VeraGridEngine.Devices.Dynamic.var_factory import VarFactory
from VeraGridEngine.Devices.Injections.generator import Generator
from VeraGridEngine.Devices.multi_circuit import MultiCircuit
from VeraGridEngine.Simulations.PowerFlow.power_flow_driver import PowerFlowDriver
from VeraGridEngine.Simulations.PowerFlow.power_flow_options import PowerFlowOptions
from VeraGridEngine.Templates.Rms.voltage_source_template import VoltageSourceBuild
from VeraGridEngine.Templates.Rms.vsc_gfl import build_gfl_converter_model
from VeraGridEngine.Templates.Rms.vsc_gfm import build_gfm_converter_model
from VeraGridEngine.Utils.Symbolic.block import Block
from VeraGridEngine.Utils.Symbolic.symbolic import Var
from VeraGridEngine.Utils.Symbolic.templates_common_functions import set_rms_model
from VeraGridEngine.enumerations import VarPowerFlowReferenceType


class ConverterKind(Enum):
    """Native RMS controller families available to this benchmark."""

    GFM = 'gfm'
    GFL = 'gfl'


def converter_model(factory: VarFactory, kind: ConverterKind, rating_pu: float) -> Block:
    """Wrap a native converter core on its own MVA base.

    :param factory: Shared variable factory for the owning grid.
    :param kind: Controller family, with its existing filters and control loops.
    :param rating_pu: Converter MVA rating divided by the network MVA base.
    :return: Generator-compatible system-base P/Q injection model.
    """
    vm: Var = factory.add_var('Vm', reference=VarPowerFlowReferenceType.Vm)
    va: Var = factory.add_var('Va', reference=VarPowerFlowReferenceType.Va)
    p: Var = factory.add_var('P_converter', reference=VarPowerFlowReferenceType.P)
    q: Var = factory.add_var('Q_converter', reference=VarPowerFlowReferenceType.Q)
    core: Block
    active_power: Var
    reactive_power: Var
    if kind == ConverterKind.GFM:
        core, _, _, active_power, reactive_power, _ = build_gfm_converter_model(
            factory, inputs=[vm, va, -p / rating_pu, -q / rating_pu],
        )
    else:
        core, _, _, active_power, reactive_power, _ = build_gfl_converter_model(
            factory, inputs=[vm, va, factory.add_const(1.0), -p / rating_pu, -q / rating_pu],
        )
    # Only the wrapper maps the external generator injection. Inner controls
    # retain their physical, converter-base equations and initial power inputs.
    core.external_mapping.clear()
    return Block(
        name=f'Basic dynamic {kind.name} converter',
        in_vars=[vm, va], out_vars=[p, q], algebraic_vars=[p, q],
        algebraic_eqs=[p - rating_pu * active_power, q - rating_pu * reactive_power],
        external_mapping={VarPowerFlowReferenceType.Vm: vm, VarPowerFlowReferenceType.Va: va,
                          VarPowerFlowReferenceType.P: p, VarPowerFlowReferenceType.Q: q},
        children=[core],
    )


def attach_converter(grid: MultiCircuit, generator: Generator, kind: ConverterKind, rating_pu: float) -> None:
    """Replace the device's RMS model without changing its AC dispatch.

    :param grid: Circuit owning the generator and its symbolic factory.
    :param generator: Existing source selected for converter representation.
    :param kind: Native controller family to install.
    :param rating_pu: Converter MVA rating on the system base.
    """
    generator.Snom = rating_pu * grid.Sbase
    set_rms_model(generator, converter_model(grid.var_factory, kind, rating_pu), grid.var_factory)


def add_converter_share(grid: MultiCircuit, source: Generator, kind: ConverterKind, share: float) -> None:
    """Assign part of an existing injection to a separately rated converter.

    :param grid: Owning AC network.
    :param source: Synchronous source retaining the remaining injection.
    :param kind: Converter control family.
    :param share: Fraction of the source's remaining dispatch to transfer.
    """
    converter: Generator = Generator(
        name=f'{kind.name} share of {source.name}', P=source.P * share,
        vset=source.Vset, Qmin=source.Qmin * share, Qmax=source.Qmax * share,
        Pmin=source.Pmin * share, Pmax=source.Pmax * share,
    )
    source.P *= 1.0 - share
    source.Qmin *= 1.0 - share
    source.Qmax *= 1.0 - share
    source.Pmin *= 1.0 - share
    source.Pmax *= 1.0 - share
    grid.add_generator(bus=source.bus, api_obj=converter)
    # Size from the solved injection, including reactive support, so the
    # native current limiter is not already saturated at the operating point.
    power_flow: PowerFlowDriver = PowerFlowDriver(grid, options=PowerFlowOptions(tolerance=1e-10, max_iter=100))
    power_flow.run()
    if not bool(power_flow.results.converged):
        raise RuntimeError(f'{grid.name}: converter-share power flow failed')
    else:
        pass
    index: int = len(grid.generators) - 1
    minimum_rating: float = 4.0 if kind == ConverterKind.GFM else 1.0
    if kind == ConverterKind.GFM and (grid.name.startswith('basic_six_bus_') or grid.name.startswith('basic_ieee24_')):
        minimum_rating = 8.0
    else:
        pass
    rating_pu: float = max(minimum_rating, 2.0 * float(np.hypot(power_flow.results.gen_p[index], power_flow.results.gen_q[index])) / grid.Sbase)
    attach_converter(grid, converter, kind, rating_pu)
    # Keep all gains and time constants positive. A low proportional gain and
    # aggressive integral gain destabilize the actual converter control loops.
    converter.rms_model.set_parameter_in_model('Kp_icl', 0.001)
    converter.rms_model.set_parameter_in_model('Ki_icl', 500.0)
    if kind == ConverterKind.GFL:
        converter.rms_model.set_parameter_in_model('Kp_pol', 0.2)
        converter.rms_model.set_parameter_in_model('Ki_pol', 0.2)
        converter.rms_model.set_parameter_in_model('Kp_pll', 0.05)
        converter.rms_model.set_parameter_in_model('Ki_pll', 0.1)
    else:
        pass


def add_variant_controllers(grid: MultiCircuit, variant: int) -> None:
    """Build GFL, GFM or mixed-converter variants with a fixed-grid reference.

    :param grid: Already scaled AC scenario with synchronous machine models.
    :param variant: One through four; zero retains the synchronous exercise.
    """
    if variant == 0:
        pass
    else:
        # The remaining synchronous controls start stable. Restoring their
        # gains therefore cannot resolve the converter's unstable modes.
        generator: Generator
        for generator in grid.generators:
            generator.rms_model.set_parameter_in_model('Kp', -400.0)
        set_rms_model(grid.generators[0], VoltageSourceBuild(grid.var_factory).block, grid.var_factory)
        # Choose a source with meaningful dispatch, retaining a synchronous
        # share for frequency measurements in the existing RMS evaluator.
        source: Generator = grid.generators[1]
        for generator in grid.generators[1:]:
            if abs(generator.P) > abs(source.P):
                source = generator
            else:
                pass
        share: float = 0.65 if variant == 4 else 0.4
        if variant == 1:
            add_converter_share(grid, source, ConverterKind.GFL, share)
        elif variant == 2:
            add_converter_share(grid, source, ConverterKind.GFM, share)
        else:
            add_converter_share(grid, source, ConverterKind.GFL, share)
            add_converter_share(grid, source, ConverterKind.GFM, share)
