"""
ui/components/metrics.py
────────────────────────
Reusable metric card renderer for the eval dashboard.

Renders a compact card with: label, value, and a coloured delta vs baseline.
"""

from __future__ import annotations

import streamlit as st


def metric_card(
    label: str,
    value: float,
    baseline: float,
    unit: str = "",
    higher_is_better: bool = True,
    fmt: str = ".3f",
) -> None:
    """
    Render a metric card with a coloured delta compared to a baseline.

    Args:
        label:            Display label (e.g. "NDCG@5").
        value:            Current (hybrid) metric value.
        baseline:         Comparison (naive) metric value.
        unit:             Optional unit suffix (e.g. "s", "%").
        higher_is_better: If True, positive delta is green; else amber.
        fmt:              Python format string for the value.
    """
    delta = value - baseline
    delta_str = f"{'+' if delta >= 0 else ''}{delta:{fmt}}{unit}"

    if higher_is_better:
        delta_color = "normal" if delta >= 0 else "inverse"
    else:
        delta_color = "inverse" if delta >= 0 else "normal"

    st.metric(
        label=label,
        value=f"{value:{fmt}}{unit}",
        delta=delta_str,
        delta_color=delta_color,
    )
