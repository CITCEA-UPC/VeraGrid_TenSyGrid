"""Register grid reference events before constructing an explicit EMT ODE.

The grid DAE is assembled once by ``extract``. Each selected runtime parameter
then stays symbolic during reduction and becomes an input of the reduced Block.
The actual VeraGrid events are attached to those reduced Block input variables,
whose UIDs differ from the original grid-template variables.
"""

from __future__ import annotations

from dataclasses import dataclass

from VeraGridEngine.Devices.Events.emt_event import EmtEvent
from VeraGridEngine.Devices.Events.emt_events_group import EmtEventsGroup
from VeraGridEngine.Simulations.EMT.problems.emt_problem_template import EmtProblemTemplate


@dataclass(frozen=True)
class _InputEvent:
    generator: str
    parameter: str
    time: float
    value: float
    name: str
    force_step_alignment: bool


class ExplicitInputEvents:
    """Collect events and expose their references to the one-pass reduction.

    Call ``add_event_to_inputs`` before ``extract``, pass ``references`` to
    ``extract(symbolic_references=...)``, then call ``install`` after creating
    the reduced EmtModelProblem. Several events may target the same input.
    """

    def __init__(self, name="explicit_input_events"):
        self.group = EmtEventsGroup(name=name)
        self._events: list[_InputEvent] = []

    def add_event_to_inputs(self, *, generator, parameter, time, value,
                            name=None, force_step_alignment=True):
        """Schedule a generator runtime reference and retain it in S/Phi."""
        if time < 0:
            raise ValueError("Event time must be nonnegative")
        key = (str(generator), str(parameter))
        self._events.append(_InputEvent(
            *key, float(time), float(value),
            name or f"{generator} {parameter} at {time:g} s",
            bool(force_step_alignment),
        ))

    @property
    def references(self):
        """Unique (generator name, parameter name) pairs in input-mode order."""
        return tuple(dict.fromkeys((event.generator, event.parameter)
                                   for event in self._events))

    @property
    def input_names(self):
        return tuple(f"{generator.replace(' ', '_')}_{parameter}"
                     for generator, parameter in self.references)

    def initial_values(self, full):
        """Read the initialized values of the symbolic grid parameters."""
        if tuple(full.explicit_input_keys) != self.references:
            raise ValueError("Extracted inputs differ from registered event references")
        return tuple(float(full.event_params_values[
            full.uid2idx_event_params[var.uid]])
            for var in full.explicit_input_vars)

    def install(self, full, native):
        """Attach real VeraGrid events to the reduced problem's runtime inputs."""
        if tuple(full.explicit_input_keys) != self.references:
            raise ValueError("Extracted inputs differ from registered event references")
        native.grid = full.grid
        full.grid.add_emt_events_group(self.group)
        for event in self._events:
            generator = next(g for g in full.grid.generators if g.name == event.generator)
            input_name = f"{event.generator.replace(' ', '_')}_{event.parameter}"
            full.grid.add_emt_event(EmtEvent(
                name=event.name, device=generator,
                parameter=native.input_vars_by_name[input_name],
                time=event.time, value=event.value, group=self.group,
                force_step_alignment=event.force_step_alignment,
            ))
        EmtProblemTemplate.set_events_group(native, self.group)
