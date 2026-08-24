"""Unit tests for the EditWidget in the napari plugin."""

from unittest.mock import Mock

import numpy as np
import pytest
from napari.layers.base import ActionType
from qtpy.QtWebEngineWidgets import QWebEngineView
from qtpy.QtWidgets import QCheckBox

from movement.napari.edit_widget import EditWidget


def _unwrap(obj):
    """Unwrap napari's PublicOnlyProxy, mirroring EditWidget's own
    unwrapping so tests compare/mock the same raw objects real napari
    events carry (see EditWidget._on_active_layer_changed).
    """
    return getattr(obj, "__wrapped__", obj)


def test_edit_widget_instantiation(make_napari_viewer_proxy):
    """Test that the widget is properly instantiated with no active layer."""
    viewer = make_napari_viewer_proxy()
    widget = EditWidget(viewer)

    assert widget.findChild(QCheckBox) is not None
    assert widget.findChild(QWebEngineView) is not None
    assert widget.active_layer is None
    assert widget._max_frame == 0


def test_edit_widget_tracks_loaded_points_layer(
    valid_poses_path_and_ds, loaded_data_loader
):
    """Test that the widget picks up an existing, selected Points layer
    created by the DataLoader as its active layer.
    """
    filepath, ds = valid_poses_path_and_ds
    loader = loaded_data_loader(filepath, ds)
    loader.viewer.layers.selection.active = loader.points_layer

    widget = EditWidget(loader.viewer)

    assert widget.active_layer is _unwrap(loader.points_layer)
    assert widget._max_frame == loader.points_layer.metadata["max_frame_idx"]


def test_edit_widget_captures_removed_points(
    valid_poses_path_and_ds, loaded_data_loader
):
    """Test that a REMOVING event snapshots the removed point's identity
    so it can still be drawn as a bar after the row is gone.
    """
    filepath, ds = valid_poses_path_and_ds
    loader = loaded_data_loader(filepath, ds)
    loader.viewer.layers.selection.active = loader.points_layer

    widget = EditWidget(loader.viewer)
    assert widget._removed_points == []

    mock_event = Mock()
    mock_event.source = _unwrap(loader.points_layer)
    mock_event.action = ActionType.REMOVING
    mock_event.data_indices = (0,)

    widget._on_layer_data_changed(mock_event)

    assert len(widget._removed_points) == 1
    frame, individual, color = widget._removed_points[0]
    assert frame == loader.points_layer.data[0, 0]
    assert individual == loader.points_layer.properties["individual"][0]
    np.testing.assert_array_equal(color, loader.points_layer.face_color[0])


def test_edit_widget_ignores_other_layers(
    valid_poses_path_and_ds, loaded_data_loader
):
    """Test that data-change events from a non-active layer are ignored."""
    filepath, ds = valid_poses_path_and_ds
    loader = loaded_data_loader(filepath, ds)
    loader.viewer.layers.selection.active = loader.points_layer

    widget = EditWidget(loader.viewer)

    other_layer = Mock()
    mock_event = Mock()
    mock_event.source = other_layer
    mock_event.action = ActionType.REMOVING

    widget._on_layer_data_changed(mock_event)

    assert widget._removed_points == []


@pytest.mark.parametrize("show_individuals", [False, True])
def test_edit_widget_redraw_bars_does_not_raise(
    valid_poses_path_and_ds, loaded_data_loader, show_individuals
):
    """Test that redrawing bars (with no page yet loaded) just queues the
    corresponding JS instead of raising, for both lane layouts.
    """
    filepath, ds = valid_poses_path_and_ds
    loader = loaded_data_loader(filepath, ds)
    loader.viewer.layers.selection.active = loader.points_layer

    widget = EditWidget(loader.viewer)
    widget._show_individuals = show_individuals

    widget._redraw_bars()

    assert not widget._page_ready
    assert len(widget._pending_js) > 0
