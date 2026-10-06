# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
"""Compatibility imports for the shared dynamic plots handler.

New GUI code must import the implementation from
``VeraGrid.Gui.DynamicModelEditor.Plots.dynamic_plots_handler``. These names
remain available here so existing integrations and tests keep working while the
plot-definition editor is no longer owned by the Results package.
"""

from VeraGrid.Gui.DynamicModelEditor.Plots.dynamic_plots_handler import (
    DynamicDeviceEntryCollection,
    DynamicPlotCandidate,
    DynamicPlotParameter,
    DynamicResultSeries,
    DynamicResultSeriesKey,
    DynamicsDeviceTreeModel,
    DynamicsPlotGroup,
    DynamicsPlotGroups,
    DynamicsPlotsTreeModel,
    DynamicsResultsHandler,
    TreeStateNodeKind,
    TreeStateSnapshot,
    _build_parameter_plot_data_from_events,
    _build_tree_state_key,
    _capture_tree_view_state,
    _restore_tree_view_state,
    build_dynamics_tree_model,
    collect_dynamic_model_plot_parameters,
    ensure_dynamic_plot_event_group,
)
