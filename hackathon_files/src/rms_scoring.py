"""Public case marks using the hackathon's 35/25/25/15 component weights."""
import math

WEIGHTS = dict(damping=0.35, loss=0.25, robustness=0.25, changes=0.15)


def score_components(metrics, baseline_losses_mw: float) -> dict[str, float]:
    """Score relative to the unchanged case, without spending tuning trials.

    Damping reaches full credit at 10%; stable nonoscillatory cases receive full
    damping credit. A loss increase earns zero loss credit. Electrical limits
    determine robustness together with dynamic/transient stability.
    """
    if not math.isfinite(baseline_losses_mw) or baseline_losses_mw <= 1e-12:
        raise ValueError('Relative-loss scoring requires positive finite baseline losses')
    if not math.isfinite(metrics.active_losses_mw):
        raise ValueError('Candidate losses are not finite')
    if not metrics.stable:
        damping = 0.0
    elif metrics.min_oscillatory_damping is None:
        damping = 1.0
    else:
        if not math.isfinite(metrics.min_oscillatory_damping):
            raise ValueError('Candidate damping is not finite')
        damping = min(1.0, max(0.0, metrics.min_oscillatory_damping / 0.10))
    return dict(
        damping=damping,
        loss=min(1.0, max(0.0, 1.0 - metrics.active_losses_mw / baseline_losses_mw)),
        robustness=float(metrics.stable and metrics.voltage_violations == metrics.thermal_violations == 0),
        changes=metrics.changes_score,
    )


def case_mark(components: dict[str, float]) -> float:
    """Return the weighted mark on a 0–100 scale."""
    return 100.0 * sum(WEIGHTS[name] * components[name] for name in WEIGHTS)
