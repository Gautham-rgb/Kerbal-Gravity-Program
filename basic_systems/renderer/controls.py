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

        def decrease_warp():
            self.r.time_rate_per_s /= 2.0

        def toggle_pause():
            if self.r.time_rate_per_s != 0.0:
                self.r._old_rate = self.r.time_rate_per_s
                self.r.time_rate_per_s = 0.0
            else:
                self.r.time_rate_per_s = self.r._old_rate

        self.r.plotter.add_key_event("Up", increase_warp) #type: ignore
        self.r.plotter.add_key_event("Down", decrease_warp) #type: ignore
        self.r.plotter.add_key_event("space", toggle_pause) #type: ignore
        self.r.plotter.add_key_event("n", self.focus_next_body) #type: ignore
        self.r.plotter.add_key_event("p", self.focus_prev_body) #type: ignore

        self._actor_to_body = {actor: body for body, actor in self.r.body_actors.items()}
        self.r.plotter.enable_mesh_picking(#type: ignore
            callback=self._on_body_picked,
            use_actor=True,
            left_clicking=True,
            show=False,
            show_message=False,
        )

        if self.r.show_timeline_slider:
            self.r.plotter.add_slider_widget(#type: ignore
                callback=self._on_timeline_slider,
                rng=[0.0, self.r.scene._compute_timeline_max_ut()],
                value=self.r.curr_ut,
                title="UT (s)",
                interaction_event='always',
            )

    def _on_timeline_slider(self, value):
        self.r.curr_ut = float(value)
        self.r.updater.update_all_positions()
        self.r.plotter.render()

    def _on_body_picked(self, actor):
        body = self._actor_to_body.get(actor)
        if body is not None:
            self.focus_on_body(body)

    def focus_next_body(self):
        all_bodies = self.r.body_list + self.r.spacecraft_list
        if not all_bodies:
            return
        self.r._focus_index = (self.r._focus_index + 1) % len(all_bodies)
        self.focus_on_body(all_bodies[self.r._focus_index])

    def focus_prev_body(self):
        all_bodies = self.r.body_list + self.r.spacecraft_list
        if not all_bodies:
            return
        self.r._focus_index = (self.r._focus_index - 1) % len(all_bodies)
        self.focus_on_body(all_bodies[self.r._focus_index])

    def focus_on_body(self, body: Body | Spacecraft):
        if body not in self.r.body_actors:
            return
        
        target_pos = np.array(self.r.body_actors[body].position)
        old_focal = np.array(self.r.plotter.camera.focal_point)
        old_pos = np.array(self.r.plotter.camera.position)
        
        offset = old_pos - old_focal
        new_pos = target_pos + offset
        
        self.r.plotter.camera.focal_point = target_pos.tolist()
        self.r.plotter.camera.position = new_pos.tolist()
        self.r.plotter.render()