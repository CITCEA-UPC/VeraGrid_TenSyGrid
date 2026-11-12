"""Intro control catalog, including the dynamic GFM controller's gains."""
from hackaton_files import additional_grid_evaluator


def make_evaluator():
    """Extend the shared evaluator without changing other hackathon examples.

    These inclusive search ranges are teaching defaults, not stability guarantees.
    Keep physical filter parameters and current limits outside the tuning catalog.
    """
    evaluator = additional_grid_evaluator()
    evaluator.control_ranges = {
        **evaluator.control_ranges,
        "Kdp": (0.001, 0.2),
        "Kdq": (0.001, 0.2),
        "Kp_icl": (0.001, 1.0),
        "Ki_icl": (1.0, 200.0),
        "tau_P": (0.001, 0.1),
        "tau_Q": (0.001, 0.1),
    }
    return evaluator
