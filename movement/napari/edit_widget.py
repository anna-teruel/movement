"""Widget flagging which frames contain edited points.

Renders the timeline as a Plotly figure inside a ``QWebEngineView``, synced
to the current frame via ``viewer.dims.events.current_step``. Pan and zoom
are Plotly's native drag/scroll behaviour; clicks on a bar are relayed back
from JavaScript to Python over a ``QWebChannel``.

Instantiated by :class:`~movement.napari.meta_widget.MovementMetaWidget`,
which docks it at the bottom of the viewer (rather than nesting it as a
collapsible section) since it's a wide timeline, not a compact control
panel.
"""

import json
import os
import tempfile

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from napari.layers import Points
from napari.layers.base import ActionType
from napari.utils.theme import get_theme
from napari.viewer import Viewer
from qtpy.QtCore import QObject, QTimer, QUrl, Slot
from qtpy.QtGui import QColor
from qtpy.QtWebChannel import QWebChannel
from qtpy.QtWebEngineWidgets import QWebEngineView
from qtpy.QtWidgets import QCheckBox, QVBoxLayout, QWidget

from movement.napari.loader_widgets import (
    POINTS_LAYER_KEY,
    POINTS_PROPERTIES_KEY,
)

PLAYHEAD_COLOR = "#55606E"  # bar indicating current frame
# this is same color as napari slidebar.

PLOT_DIV_ID = "movement-edit-timeline"
BAR_WIDTH_FRAMES = 1  # bar width, in frame units


class _Bridge(QObject):
    """Relays timeline-bar clicks from JavaScript back to Python."""

    def __init__(self, on_bar_click):
        super().__init__()
        self._on_bar_click = on_bar_click

    @Slot(float)
    def on_bar_click(self, frame: float) -> None:
        """Handle a click on a bar, forwarded from the JS side."""
        self._on_bar_click(frame)


class EditWidget(QWidget):
    """Dock widget flagging frames with edited points.

    Draws one lane per individual, with a bar for every frame that
    contains an edited point on the currently active ``movement`` Points
    layer. Bars are coloured to match that point's colour in the
    Points/Tracks layers. A playhead line marks the frame currently shown
    in the viewer. Scroll to zoom in/out on the timeline, drag to pan, and
    click a bar to jump to that frame.
    """

    def __init__(self, napari_viewer: Viewer, parent=None):
        """Initialise the widget and connect it to viewer events."""
        super().__init__(parent=parent)
        self.viewer = napari_viewer
        self.active_layer: Points | None = None
        self._show_individuals = False
        self._removed_points: list = []
        self._max_frame = 0
        self._page_ready = False
        self._pending_js: list[str] = []
        self._html_path: str | None = None

        self._bridge = _Bridge(self._jump_to_frame)
        self.channel = QWebChannel()
        self.channel.registerObject("bridge", self._bridge)

        self.view = QWebEngineView()
        self.view.setMinimumHeight(200)
        self.view.page().setWebChannel(self.channel)
        self.view.loadFinished.connect(self._on_page_loaded)

        self.show_individuals_checkbox = QCheckBox("Display individuals")
        self.show_individuals_checkbox.setChecked(self._show_individuals)
        self.show_individuals_checkbox.toggled.connect(
            self._on_show_individuals_toggled
        )

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.show_individuals_checkbox)
        layout.addWidget(self.view)
        self.setLayout(layout)

        self.viewer.dims.events.current_step.connect(self._on_step_changed)
        self.viewer.layers.selection.events.active.connect(
            self._on_active_layer_changed
        )
        self.viewer.layers.events.inserted.connect(self._on_layer_inserted)
        self.viewer.layers.events.removed.connect(self._on_layer_removed)
        self.viewer.events.theme.connect(self._apply_theme)

        self._load_initial_page()

        for layer in self.viewer.layers:
            self._track_layer(layer)
        self._on_active_layer_changed()

    def _load_initial_page(self) -> None:
        """Build the initial figure and load it into the web view."""
        theme = get_theme(self.viewer.theme)
        background_hex = theme.background.as_hex()
        fig = self._build_initial_figure(theme)
        html = pio.to_html(
            fig,
            include_plotlyjs=True,
            full_html=True,
            div_id=PLOT_DIV_ID,
            config={"scrollZoom": True},
            post_script=self._bootstrap_js(),
        )
        # Load qwebchannel.js (built into QtWebEngine) before the Plotly
        # script runs, so the bootstrap script below can use it. Also zero
        # out the page's default margin/background: left as-is, the 8px
        # default body margin shows the web view's native white background
        # as a border around the (themed, dark) plot.
        html = html.replace(
            "<head>",
            '<head><script src="qrc:///qtwebchannel/qwebchannel.js">'
            "</script>"
            "<style>html, body {margin: 0; padding: 0; height: 100%; "
            f"width: 100%; background: {background_hex};"
            "}</style>",
            1,
        )
        # Also set the native page background, so there's no white flash
        # before the stylesheet above has been parsed.
        self.view.page().setBackgroundColor(QColor(background_hex))
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", delete=False, encoding="utf-8"
        ) as f:
            f.write(html)
            self._html_path = f.name
        self.view.load(QUrl.fromLocalFile(self._html_path))

    def _build_initial_figure(self, theme) -> go.Figure:
        """Build the empty timeline figure, styled for the current theme."""
        fig = go.Figure(
            go.Bar(
                x=[],
                y=[],
                base=[],
                text=[],
                width=BAR_WIDTH_FRAMES,
                marker={"color": []},
                hovertemplate="frame %{x}<br>%{text}<extra></extra>",
            )
        )
        fig.update_layout(
            title={"text": "Edited frames", "font": {"size": 12}},
            xaxis={"title": "frame", "range": [0, max(self._max_frame, 1)]},
            # `visible: False` would suppress tick labels entirely, even
            # once _redraw_bars sets tickvals/ticktext for the
            # "Display individuals" lanes -- so keep the axis itself
            # visible and toggle just `showticklabels` per redraw instead.
            yaxis={
                "range": [0, 1],
                "fixedrange": True,
                "showgrid": False,
                "zeroline": False,
                "showline": False,
                "ticks": "",
                "showticklabels": False,
                "automargin": True,
            },
            dragmode="pan",
            showlegend=False,
            margin={"l": 10, "r": 10, "t": 30, "b": 30},
            paper_bgcolor=theme.background.as_hex(),
            plot_bgcolor=theme.background.as_hex(),
            font={"color": theme.text.as_hex()},
            shapes=[
                {
                    "type": "line",
                    "x0": 0,
                    "x1": 0,
                    "y0": 0,
                    "y1": 1,
                    "xref": "x",
                    "yref": "paper",
                    "line": {"color": PLAYHEAD_COLOR, "width": 1},
                }
            ],
        )
        return fig

    @staticmethod
    def _bootstrap_js() -> str:
        """JS wiring run once the initial figure has been drawn.

        Opens the QWebChannel to reach the Python-side ``bridge``, and
        wires up clicking the timeline (jump to frame) and double-clicking
        it (reset the visible range to the full timeline).

        Click detection is done at the DOM level (mousedown/mouseup pixel
        deltas), not via Plotly's own ``plotly_click``: that event only
        fires on a precise hit against a bar mark, which -- combined with
        ``dragmode: 'pan'`` treating any small pointer movement as the
        start of a pan -- made real clicks unreliable in practice (the
        view would pan by a few pixels instead of registering a click).
        Converting the release pixel straight to a frame via the axis's
        ``p2d`` (pixel-to-data) also means clicking *near* a thin bar
        still jumps to that frame, without needing an exact hit.
        """
        return f"""
        var gd = document.getElementById('{PLOT_DIV_ID}');
        // Overwritten with the real value once the active layer is known
        // (see EditWidget._reset_xlim); 0 is just a safe initial default.
        window._movementMaxFrame = 0;
        new QWebChannel(qt.webChannelTransport, function(channel) {{
            window.movementBridge = channel.objects.bridge;
        }});

        var DRAG_THRESHOLD_PX = 3;
        var DOUBLE_CLICK_DELAY_MS = 250;
        var downX = null, downY = null, dragged = false, clickTimer = null;

        function jumpToPixel(clientX, clientY) {{
            var size = gd._fullLayout._size;
            var rect = gd.getBoundingClientRect();
            var px = clientX - rect.left;
            var py = clientY - rect.top;
            if (px < size.l || px > size.l + size.w ||
                py < size.t || py > size.t + size.h) {{
                return;  // click landed outside the plotting area
            }}
            if (window.movementBridge) {{
                // p2d expects a pixel relative to the plot area's own
                // origin, not the container -- subtract the left margin.
                var frame = gd._fullLayout.xaxis.p2d(px - size.l);
                window.movementBridge.on_bar_click(Math.round(frame));
            }}
        }}

        gd.addEventListener('mousedown', function(e) {{
            downX = e.clientX;
            downY = e.clientY;
            dragged = false;
        }});
        gd.addEventListener('mousemove', function(e) {{
            if (downX === null) return;
            if (Math.abs(e.clientX - downX) > DRAG_THRESHOLD_PX ||
                Math.abs(e.clientY - downY) > DRAG_THRESHOLD_PX) {{
                dragged = true;
            }}
        }});
        gd.addEventListener('mouseup', function(e) {{
            var wasDrag = dragged;
            downX = null;
            downY = null;
            dragged = false;
            if (wasDrag) return;
            if (clickTimer) {{
                // A second click within the window: it's a double-click,
                // handled separately below -- cancel the pending single.
                clearTimeout(clickTimer);
                clickTimer = null;
                return;
            }}
            var clientX = e.clientX, clientY = e.clientY;
            clickTimer = setTimeout(function() {{
                clickTimer = null;
                jumpToPixel(clientX, clientY);
            }}, DOUBLE_CLICK_DELAY_MS);
        }});
        gd.addEventListener('dblclick', function() {{
            if (clickTimer) {{
                clearTimeout(clickTimer);
                clickTimer = null;
            }}
            Plotly.relayout(
                gd, {{'xaxis.range': [0, window._movementMaxFrame]}}
            );
        }});
        """

    def _run_js(self, script: str) -> None:
        """Run JS in the web view, queuing it if the page isn't loaded yet."""
        if self._page_ready:
            self.view.page().runJavaScript(script)
        else:
            self._pending_js.append(script)

    def _on_page_loaded(self, ok: bool) -> None:
        """Flush any JS queued before the page finished loading."""
        self._page_ready = ok
        pending, self._pending_js = self._pending_js, []
        for script in pending:
            self._run_js(script)

    def _apply_theme(self, event=None):
        """Style the plot to match the current napari theme.

        Connected to ``viewer.events.theme`` so the plot updates live
        if the user switches between napari's dark and light themes.
        """
        theme = get_theme(self.viewer.theme)
        background_hex = theme.background.as_hex()
        self.view.page().setBackgroundColor(QColor(background_hex))
        update = {
            "paper_bgcolor": background_hex,
            "plot_bgcolor": background_hex,
            "font.color": theme.text.as_hex(),
        }
        self._run_js(
            f"document.body.style.background = '{background_hex}';"
            f"Plotly.relayout('{PLOT_DIV_ID}', {json.dumps(update)});"
        )

    def _track_layer(self, layer):
        """Connect to a movement Points layer's data-change event."""
        if isinstance(layer, Points) and layer.metadata.get(POINTS_LAYER_KEY):
            layer.events.data.connect(self._on_layer_data_changed)

    def _on_layer_inserted(self, event):
        """Track newly added movement Points layers."""
        self._track_layer(event.value)

    def _on_layer_removed(self, event):
        """Clear the display if the active layer was removed."""
        if event.value is self.active_layer:
            self.active_layer = None
            self._max_frame = 0
            self._removed_points = []
            self._redraw_bars()

    def _on_show_individuals_toggled(self, checked):
        """Switch between one shared lane and one lane per individual."""
        self._show_individuals = checked
        self._redraw_bars()

    def _on_active_layer_changed(self, event=None):
        """Switch to displaying edited frames for the active layer."""
        # viewer.layers.selection.active is accessed through napari's
        # PublicOnlyProxy (wraps viewer access for plugin widgets), which
        # returns a fresh proxy wrapping the real layer on every access.
        # Unwrap it so later `is` comparisons against `event.source`/
        # `event.value` (always the raw layer, since those events are
        # emitted from inside the unwrapped layer/LayerList) succeed.
        active = self.viewer.layers.selection.active
        active = getattr(active, "__wrapped__", active)
        self.active_layer = (
            active
            if isinstance(active, Points)
            and active.metadata.get(POINTS_LAYER_KEY)
            else None
        )
        self._max_frame = (
            self.active_layer.metadata.get("max_frame_idx", 0)
            if self.active_layer is not None
            else 0
        )
        self._removed_points = self._reconstruct_previously_removed_points(
            self.active_layer
        )
        self._redraw_bars()
        self._reset_xlim()

    @staticmethod
    def _reconstruct_previously_removed_points(layer):
        """Restore removed-point bars from a previously-saved session.

        A point removed and saved in an earlier session comes back as a
        NaN row, dropped from the live layer entirely -- so there is no
        ``REMOVING`` event to capture it from this session. Reconstruct
        those bars from ``POINTS_PROPERTIES_KEY``, the full properties
        table (including NaN rows) stashed on the layer at load time.
        Points edited without being removed need no such reconstruction,
        since they're still live rows and ``_redraw_bars`` already reads
        their ``edited`` property directly.
        """
        if layer is None:
            return []
        full_properties = layer.metadata.get(POINTS_PROPERTIES_KEY)
        if full_properties is None or "edited" not in full_properties:
            return []
        previously_removed = full_properties[
            full_properties["position_is_nan"] & full_properties["edited"]
        ]
        if previously_removed.empty:
            return []
        individual_colors = dict(
            zip(
                layer.properties["individual"],
                layer.face_color,
                strict=False,
            )
        )
        return [
            (row.time, row.individual, individual_colors[row.individual])
            for row in previously_removed.itertuples()
            if row.individual in individual_colors
        ]

    def _on_layer_data_changed(self, event):
        """React to point moves and removals on the active layer.

        Connected to ``events.data`` rather than ``events.properties``:
        napari's ``Points.remove()`` (Delete/Backspace) only fires
        ``events.features``, never ``events.properties``, so a
        properties-only listener misses removals entirely.

        For a move (``CHANGED``), ``DataLoader._on_points_data_changed``
        (loader_widgets.py) is also connected to this same event and
        sets the ``edited`` property that ``_redraw_bars`` reads. But
        the two callbacks' relative order isn't guaranteed. Deferring
        the redraw with ``QTimer.singleShot(0, ...)`` runs it on the
        next Qt event-loop tick, after every callback for this event
        has finished, so ``edited`` is always up to date by then.

        For a removal, the row (and its identity) is about to be
        deleted from the layer entirely, so there will be nothing left
        to read afterwards. ``REMOVING`` fires first, while the data is
        still intact, so that's when we snapshot it.
        """
        if event.source is not self.active_layer:
            return
        if event.action == ActionType.REMOVING:
            self._capture_removed_points(event)
            self._redraw_bars()
        elif event.action == ActionType.CHANGED:
            QTimer.singleShot(0, self._redraw_bars)

    def _capture_removed_points(self, event):
        """Snapshot the identity of points about to be removed."""
        layer = event.source
        frames = layer.data[:, 0]
        individuals = layer.properties["individual"]
        colors = layer.face_color
        for idx in event.data_indices:
            self._removed_points.append(
                (frames[idx], individuals[idx], colors[idx])
            )

    def _on_step_changed(self, event=None):
        """Move the playhead line to the current frame."""
        step = self.viewer.dims.current_step
        if not step:
            return
        frame = step[0]
        update = {"shapes[0].x0": int(frame), "shapes[0].x1": int(frame)}
        self._run_js(
            f"Plotly.relayout('{PLOT_DIV_ID}', {json.dumps(update)});"
        )

    @staticmethod
    def _to_rgba_str(color) -> str:
        """Convert an RGBA float array (0-1) to a CSS ``rgba()`` string."""
        r, g, b, a = color
        return f"rgba({int(r * 255)},{int(g * 255)},{int(b * 255)},{a})"

    def _redraw_bars(self):
        """Redraw the per-individual lanes of edited-frame bars."""
        if self.active_layer is None:
            self._run_js(
                f"Plotly.restyle('{PLOT_DIV_ID}', "
                "{x: [[]], y: [[]], base: [[]], text: [[]], "
                "'marker.color': [[]]}, [0]);"
                f"Plotly.relayout('{PLOT_DIV_ID}', "
                "{'yaxis.tickvals': [], 'yaxis.ticktext': [], "
                "'yaxis.showticklabels': false});"
            )
            return

        individuals = self.active_layer.properties["individual"]
        removed_individuals = [ind for _, ind, _ in self._removed_points]
        # Union with removed individuals so a lane survives even if an
        # individual's last remaining point has just been removed.
        unique_individuals = list(
            dict.fromkeys([*individuals, *removed_individuals])
        )

        if self._show_individuals:
            n_lanes = len(unique_individuals)
            lane_of = {ind: i for i, ind in enumerate(unique_individuals)}
            tickvals = [(i + 0.5) / n_lanes for i in range(n_lanes)]
            ticktext = unique_individuals
        else:
            # A single shared lane: one bar per edited frame, regardless
            # of how many (or which) individuals were edited on it.
            n_lanes = 1
            lane_of = dict.fromkeys(unique_individuals, 0)
            tickvals, ticktext = [], []
        lane_height = 1.0 / n_lanes

        # Combine moved (still-live) and removed points into one list of
        # (frame, individual, colour) triples to draw as bars.
        edited = self.active_layer.properties.get("edited")
        moved_points = []
        if edited is not None and edited.any():
            frames = self.active_layer.data[:, 0]
            colors = self.active_layer.face_color
            moved_points = [
                (frames[idx], individuals[idx], colors[idx])
                for idx in np.nonzero(edited)[0]
            ]
        all_points = moved_points + self._removed_points

        bar_x, bar_base, bar_colors, bar_text = [], [], [], []
        if all_points:
            # A frame may have several edited keypoints for the same
            # individual; only draw one bar per (frame, individual) --
            # or, with lanes collapsed, one bar per frame regardless of
            # individual.
            seen = set()
            for frame, individual, color in all_points:
                key = (frame, individual) if self._show_individuals else frame
                if key in seen:
                    continue
                seen.add(key)
                lane = lane_of[individual]
                bar_x.append(float(frame))
                bar_base.append(lane * lane_height)
                bar_colors.append(self._to_rgba_str(color))
                bar_text.append(str(individual))

        bar_y = [lane_height] * len(bar_x)
        self._run_js(
            f"Plotly.restyle('{PLOT_DIV_ID}', "
            + json.dumps(
                {
                    "x": [bar_x],
                    "y": [bar_y],
                    "base": [bar_base],
                    "text": [bar_text],
                    "marker.color": [bar_colors],
                }
            )
            + ", [0]);"
            f"Plotly.relayout('{PLOT_DIV_ID}', "
            + json.dumps(
                {
                    "yaxis.tickvals": tickvals,
                    "yaxis.ticktext": ticktext,
                    "yaxis.showticklabels": self._show_individuals,
                }
            )
            + ");"
        )

    def _reset_xlim(self):
        """Reset the visible frame range to the full extent."""
        full_span = max(self._max_frame, 1)
        self._run_js(
            f"window._movementMaxFrame = {full_span};"
            f"Plotly.relayout('{PLOT_DIV_ID}', "
            f"{{'xaxis.range': [0, {full_span}]}});"
        )

    def _jump_to_frame(self, frame: float) -> None:
        """Move the napari viewer to the given frame, from a bar click."""
        current_step = self.viewer.dims.current_step
        if not current_step:
            return
        self.viewer.dims.current_step = (int(frame),) + current_step[1:]

    def closeEvent(self, event):
        """Clean up the temporary HTML file backing the web view."""
        if self._html_path and os.path.exists(self._html_path):
            os.unlink(self._html_path)
        super().closeEvent(event)
