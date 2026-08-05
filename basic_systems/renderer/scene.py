from basic_systems.orbit_pred import Body, Spacecraft
import numpy as np
import pyvista as pv
from vtkmodules.vtkRenderingCore import vtkActor2D
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from basic_systems.renderer.renderer import SystemRenderer

class SceneBuilder:
    def __init__(self, renderer: SystemRenderer) -> None:
          self.renderer = renderer

    @property
    def r(self) -> SystemRenderer:
        return self.renderer

    def _flatten_system(self, body: Body):
            if body:
                self.r.body_list.append(body)
                for moon in body.moons:
                    if isinstance(moon, Spacecraft):
                        self.r.spacecraft_list.append(moon)
                    else:
                        self._flatten_system(moon)

    def _compute_scales(self):
        """Dynamically compute scaling factors based on the system's physical dimensions."""
        max_dist = 0.0
        min_radius = float("inf")

        for body in self.r.body_list:
            if body.orbit and body.orbit.semi_major_axis > 0:
                max_dist = max(max_dist, body.orbit.semi_major_axis)
            min_radius = min(min_radius, body.radius)

        if max_dist == 0:
            max_dist = 1e9

        self.r.distance_scale = 1000.0 / max_dist

        if self.r.system.root.radius > 0:
            self.r.body_size_scale = 15.0 / self.r.system.root.radius
            self.r.root_size = (
                self.r.system.root.radius * self.r.body_size_scale
            ) / 4

        if min_radius * self.r.body_size_scale < 0.5:
            self.r.min_render_radius = 0.5
        else:
            self.r.min_render_radius = (
                min_radius * self.r.body_size_scale
        )

    def _compute_timeline_max_ut(self) -> float:
            periods = [b.orbit.period for b in self.r.body_list if b.orbit and b.orbit.period > 0]
            return max(periods) * 1.5 if periods else 3.15e7

    def _create_spacecraft_mesh(self, plotter: pv.Plotter, size: float, spacecraft: Spacecraft) -> tuple[pv.Actor, vtkActor2D]:
            cube = pv.Cube(center=(0, 0, 0), x_length=size, y_length=size, z_length=size)
            
            antenna_len = size * 1.5
            lines = []
            for angle in [0, 90, 180, 270]:
                rad = np.radians(angle)
                start = [0, 0, 0]
                end = [antenna_len * np.cos(rad), antenna_len * np.sin(rad), 0]
                lines.append(pv.Line(start, end))
                
            combined = cube + lines[0] + lines[1] + lines[2] + lines[3]
            
            spacecraft_actor = plotter.add_mesh(combined, color="white", smooth_shading=True)
            
            label_position = [0, 0, size * 0.8]
            label_text = spacecraft.name if spacecraft.name else "Droplet 1"

            cube_label_actor = plotter.add_point_labels(
                [label_position],
                [label_text],
                font_size=16,
                text_color="white",
                always_visible=True,
                shape_opacity=0.3
            )
            
            return spacecraft_actor, cube_label_actor

    def _init_scene_elements(self):
        for body in self.r.body_list:
            scaled_radius = max(
                self.r.min_render_radius,
                body.radius * self.r.body_size_scale,
            )

            if body == self.r.system.root:
                scaled_radius = self.r.root_size
            
            actor = body.to_mesh(self.r.plotter, scaled_radius)
            label = self.r.plotter.add_point_labels(
                    [body.get_absolute_pos_at_ut(self.r.curr_ut) * self.r.distance_scale],
                    [body.name],
                    font_size=16,
                    text_color="white",
                    always_visible=True,
                    shape_opacity=0.3
                )
            self.r.body_actors[body] = actor
            self.r.body_labels[body] = label

            if body.orbit and body.orbit.period > 0 and np.isfinite(body.orbit.period):
                times = np.linspace(0, body.orbit.period, self.r.num_line_segments)
                path_pts = np.array([self.r.updater._get_display_pos_at_ut(body, t) for t in times], dtype=np.float32)
                path_pts *= self.r.distance_scale

                line_poly = pv.lines_from_points(path_pts, close=True)
                self.r.plotter.add_mesh(line_poly, color=body.render_color, line_width=1.5, pickable=False)

        # Spacecraft
        sc_size = self.r.min_render_radius * 1.2
        for sc in self.r.spacecraft_list:
            actor , sc_label = self._create_spacecraft_mesh(self.r.plotter, sc_size, sc)
            self.r.body_labels[sc] = sc_label
            self.r.body_actors[sc] = actor

        self.r.updater.update_all_positions()
        
        self.r.plotter.camera_position = 'iso'
        if self.r.system.root in self.r.body_actors:
            self.r.plotter.set_focus(self.r.body_actors[self.r.system.root].position) #type: ignore
        self.r.plotter.camera.zoom(1.1)

    def build(self):
        self._flatten_system(self.r.system.root)
        self._compute_scales()
        self._init_scene_elements()
