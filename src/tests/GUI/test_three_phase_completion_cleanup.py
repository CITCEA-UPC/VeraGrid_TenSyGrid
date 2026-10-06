from __future__ import annotations

import pytest

from VeraGrid.Gui.Main.SubClasses.simulations import SimulationsMain
from VeraGridEngine.enumerations import SimulationTypes


class _ProgressLabel:
    """
    Minimal progress-label stand-in for the three-phase post callback.
    """

    __slots__ = ("text",)

    def __init__(self) -> None:
        """
        Build the fake label.

        :return: None.
        """
        self.text: str = ""

    def setText(self, text: str) -> None:
        """
        Store the last progress text written by the callback.

        :param text: Progress text.
        :return: None.
        """
        self.text = text


class _Ui:
    """
    Minimal UI holder for the three-phase post callback.
    """

    __slots__ = ("progress_label",)

    def __init__(self) -> None:
        """
        Build the fake UI holder.

        :return: None.
        """
        self.progress_label: _ProgressLabel = _ProgressLabel()


class _Results:
    """
    Minimal power-flow results stand-in.
    """

    __slots__ = ("converged",)

    def __init__(self) -> None:
        """
        Build converged fake results.

        :return: None.
        """
        self.converged: bool = True


class _Session:
    """
    Minimal session stand-in for a completed three-phase worker.
    """

    __slots__ = ("results",)

    def __init__(self, results: _Results) -> None:
        """
        Build the fake session.

        :param results: Results returned by the session property.
        :return: None.
        """
        self.results: _Results = results

    @property
    def power_flow_3ph(self) -> tuple[None, _Results]:
        """
        Return the completed three-phase results.

        :return: Driver placeholder and results.
        """
        return None, self.results

    @property
    def short_circuit(self) -> tuple[None, _Results]:
        """
        Return the completed short-circuit results.

        :return: Driver placeholder and results.
        """
        return None, self.results

    def is_anything_running(self) -> bool:
        """
        Report that the worker has finished.

        :return: Always ``False``.
        """
        return False


class _Main:
    """
    Small object carrying only the state used by ``post_power_flow3ph``.
    """

    __slots__ = ("session", "stuff_running_now", "ui", "unlocked")

    def __init__(self) -> None:
        """
        Build the fake main window state.

        :return: None.
        """
        self.session: _Session = _Session(results=_Results())
        self.stuff_running_now: list[SimulationTypes] = list()
        self.stuff_running_now.append(SimulationTypes.PowerFlow3ph_run)
        self.ui: _Ui = _Ui()
        self.unlocked: bool = False

    def remove_simulation(self, val: SimulationTypes) -> None:
        """
        Remove the finished simulation marker.

        :param val: Simulation type to remove.
        :return: None.
        """
        while val in self.stuff_running_now:
            self.stuff_running_now.remove(val)

    def update_available_results(self) -> None:
        """
        Simulate a GUI post-processing failure after the worker has finished.

        :return: None.
        """
        raise RuntimeError("post-processing failed")

    def colour_diagrams(self) -> None:
        """
        Stand in for diagram colouring.

        :return: None.
        """
        return None

    def show_info_toast(self, text: str) -> None:
        """
        Stand in for the success toast.

        :param text: Toast text.
        :return: None.
        """
        return None

    def show_warning_toast(self, text: str) -> None:
        """
        Stand in for the warning toast.

        :param text: Toast text.
        :return: None.
        """
        return None

    def sender(self) -> None:
        """
        Stand in for QObject.sender() when the callback is called directly.

        :return: No Qt sender.
        """
        return None

    def tr(self, text: str) -> str:
        """
        Return untranslated text.

        :param text: Source text.
        :return: Same text.
        """
        return text

    def UNLOCK(self) -> None:
        """
        Record that the post callback unlocked the GUI.

        :return: None.
        """
        self.unlocked = True


def test_three_phase_post_cleanup_runs_when_gui_post_processing_fails() -> None:
    """
    The completion callback must clear GUI busy state even if colouring fails.

    :return: None.
    """
    main: _Main = _Main()

    with pytest.raises(RuntimeError, match="post-processing failed"):
        SimulationsMain.post_power_flow3ph(main)

    assert main.stuff_running_now == list()
    assert main.unlocked


def test_short_circuit_post_cleanup_runs_when_gui_post_processing_fails() -> None:
    """
    The short-circuit callback must clear GUI busy state even if colouring fails.

    :return: None.
    """
    main: _Main = _Main()
    main.stuff_running_now.clear()
    main.stuff_running_now.append(SimulationTypes.ShortCircuit_run)

    with pytest.raises(RuntimeError, match="post-processing failed"):
        SimulationsMain.post_short_circuit(main)

    assert main.stuff_running_now == list()
    assert main.unlocked
