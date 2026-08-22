from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from basic_systems.renderer.renderer import SystemRenderer

from basic_systems.orbit_pred import Body, Spacecraft
import numpy as np


class ControlManager:
    def __init__(self, renderer: SystemRenderer):
        self.renderer = renderer

    @property
    def r(self):
        return self.renderer

    def setup(self):
        def increase_warp():
            self.r.time_rate_per_s *= 2.0
            self._refresh_overlay()

        def decrease_warp():
            self.r.time_rate_per_s = max(
                0.125,
                self.r.time_rate_per_s / 2.0,
            )
            self._refresh_overlay()

        def toggle_pause():
            if self.r.time_rate_per_s != 0.0:
                self.r._old_rate = self.r.time_rate_per_s
                self.r.time_rate_per_s = 0.0
            else:
                self.r.time_rate_per_s = self.r._old_rate

            self._refresh_overlay()

        self.r.plotter.add_key_event(
            "Up", #type: ignore
            increase_warp,
        )

        self.r.plotter.add_key_event(
            "Down", #type: ignore
            decrease_warp,
        )

        self.r.plotter.add_key_event(
            "space", #type: ignore
            toggle_pause,
        )

        self.r.plotter.add_key_event(
            "n", #type: ignore
            self.focus_next_body,
        )

        self.r.plotter.add_key_event(
            "p", #type: ignore
            self.focus_prev_body,
        )

        self._actor_to_body = {
            actor: body
            for body, actor in self.r.body_actors.items()
        }

        self.r.plotter.enable_mesh_picking(#type: ignore
            callback=self._on_body_picked,
            use_actor=True,
            left_clicking=True,
            show=False,
            show_message=False,
        )

        if self.r.show_timeline_slider:
            self.r.plotter.add_slider_widget(#type: ignore
                callback=self._on_warp_slider,
                rng=[0.0, 4000.0],
                value=self.r.time_rate_per_s,
                title="Time warp (x)",
                pointa=(0.25, 0.93),
                pointb=(0.75, 0.93),
            )

    def _on_warp_slider(self, value):
        value = float(value)

        if not np.isfinite(value):
            return

        self.r.set_time_rate(value)
        self.r.plotter.render()

    def _refresh_overlay(self):
        self.r._update_hud()
        self.r.plotter.render()

    def _on_body_picked(self, actor):
        node_info = self.r.maneuver_node_actors.get(actor)
        if node_info is not None:
            self.r.select_maneuver_node(
                node_info["ticket"],
                node_info["event"],
                node_info["node"],
            )
            return

        body = self._actor_to_body.get(actor)

        if body is not None:
            self.r.selected_info = None
            self.r._update_hud_node_details()
            self.focus_on_body(body)

    def focus_next_body(self):
        bodies = self.r.body_list + self.r.spacecraft_list

        if not bodies:
            return

        if self.r.focused_body not in bodies:
            index = 0
        else:
            index = bodies.index(self.r.focused_body)#type: ignore
            index = (index + 1) % len(bodies)

        self.focus_on_body(bodies[index])

    def focus_prev_body(self):
        bodies = self.r.body_list + self.r.spacecraft_list

        if not bodies:
            return

        if self.r.focused_body not in bodies:
            index = 0
        else:
            index = bodies.index(self.r.focused_body)#type: ignore
            index = (index - 1) % len(bodies)

        self.focus_on_body(bodies[index])

    def focus_on_body(self, body: Body | Spacecraft):
        if body not in self.r.body_actors:
            return

        self.r.focused_body = body

        target_pos = np.array(
            self.r.body_actors[body].position,
            dtype=float,
        )

        old_focal = np.array(
            self.r.plotter.camera.focal_point,
            dtype=float,
        )

        old_pos = np.array(
            self.r.plotter.camera.position,
            dtype=float,
        )

        offset = old_pos - old_focal

        self.r.plotter.camera.focal_point = (
            target_pos.tolist()
        )

        self.r.plotter.camera.position = (
            target_pos + offset
        ).tolist()

        self.r._update_hud()
        self.r.plotter.render()
