from __future__ import annotations

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
            fuselage = pv.Cylinder(center=(0, 0, 0), direction=(0, 0, 1), radius=size * 0.28, height=size * 1.3)
            nose = pv.Cone(center=(0, 0, size * 0.95), direction=(0, 0, 1), height=size * 0.7, radius=size * 0.3)
            nozzle = pv.Cone(center=(0, 0, -size * 0.82), direction=(0, 0, -1), height=size * 0.38, radius=size * 0.19)
            panel_a = pv.Cube(center=(size * 0.78, 0, 0), x_length=size * 1.15, y_length=size * 0.08, z_length=size * 0.42)
            panel_b = pv.Cube(center=(-size * 0.78, 0, 0), x_length=size * 1.15, y_length=size * 0.08, z_length=size * 0.42)
            combined = fuselage + nose + nozzle + panel_a + panel_b

            spacecraft_actor = plotter.add_mesh(
                combined, color="#d9f6ff", smooth_shading=True, specular=0.65, specular_power=28
            )
            
            label_position = [0, 0, size * 0.8]
            label_text = spacecraft.name if spacecraft.name else "Droplet 1"

            cube_label_actor = plotter.add_point_labels(
                [label_position],
                [label_text],
                font_size=16,
                text_color="white",
                always_visible=True,
                shape_color="#102239",
                shape_opacity=0.55
            )
            
            return spacecraft_actor, cube_label_actor

    def _init_scene_elements(self):
        self._add_starfield()
        for body in self.r.body_list:
            scaled_radius = max(
                self.r.min_render_radius,
                body.radius * self.r.body_size_scale,
            )

            if body == self.r.system.root:
                scaled_radius = self.r.root_size
            
            mesh = pv.Sphere(radius=scaled_radius, theta_resolution=48, phi_resolution=48)
            if body == self.r.system.root:
                actor = self.r.plotter.add_mesh(mesh, color=body.render_color, smooth_shading=True, lighting=False)
            else:
                actor = self.r.plotter.add_mesh(
                    mesh, color=body.render_color, smooth_shading=True, specular=0.25, specular_power=18
                )
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

            if body.atm_height > 0 and body.radius > 0:
                atmosphere_radius = scaled_radius * (1.0 + min(body.atm_height / body.radius, 0.35))
                atmosphere = pv.Sphere(radius=atmosphere_radius, theta_resolution=40, phi_resolution=40)
                atmosphere_actor = self.r.plotter.add_mesh(
                    atmosphere, color="#5ba7ff", opacity=0.13, smooth_shading=True, lighting=False, pickable=False
                )
                self.r.atmosphere_actors[body] = atmosphere_actor

            if body.orbit and body.orbit.period > 0 and np.isfinite(body.orbit.period):
                times = np.linspace(0, body.orbit.period, self.r.num_line_segments)
                path_pts = np.array([self.r.updater._get_display_pos_at_ut(body, t) for t in times], dtype=np.float32)
                path_pts *= self.r.distance_scale

                line_poly = pv.lines_from_points(path_pts, close=True)
                orbit_actor = self.r.plotter.add_mesh(
                    line_poly, color=body.render_color, opacity=0.58, line_width=1.6, pickable=False
                )
                self.r.orbit_actors.append(orbit_actor)

        # Spacecraft
        sc_size = self.r.min_render_radius * 1.2
        for sc in self.r.spacecraft_list:
            actor , sc_label = self._create_spacecraft_mesh(self.r.plotter, sc_size, sc)
            self.r.body_labels[sc] = sc_label
            self.r.body_actors[sc] = actor
            self._add_spacecraft_trajectory(sc, sc_size)

        self._add_maneuver_markers(sc_size)

        self.r.updater.update_all_positions()
        
        self.r.plotter.camera_position = 'iso'
        if self.r.system.root in self.r.body_actors:
            self.r.plotter.set_focus(self.r.body_actors[self.r.system.root].position) #type: ignore
        self.r.plotter.camera.zoom(1.1)

    def _add_starfield(self) -> None:
        """A deterministic sparse star field gives the scene depth without affecting scale."""
        rng = np.random.default_rng(73)
        directions = rng.normal(size=(700, 3))
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        points = directions * rng.uniform(1300, 1900, size=(700, 1))
        stars = pv.PolyData(points)
        self.r.plotter.add_points(stars, color="#cdd8ff", point_size=1.3, render_points_as_spheres=True, pickable=False)

    def _add_spacecraft_trajectory(self, spacecraft: Spacecraft, size: float) -> None:
        period = spacecraft.orbit.period
        if not np.isfinite(period) or period <= 0:
            return
        times = np.linspace(self.r.curr_ut, self.r.curr_ut + period, max(80, self.r.num_line_segments))
        points = np.array([spacecraft.get_absolute_pos_at_ut(time) for time in times]) * self.r.distance_scale
        trajectory = pv.lines_from_points(points.astype(np.float32), close=False)
        self.r.orbit_actors.append(
            self.r.plotter.add_mesh(trajectory, color="#71e7ff", opacity=0.82, line_width=2.4, pickable=False)
        )

    def _add_maneuver_markers(self, size: float) -> None:
        for ticket in self.r.tickets:
            for event in ticket.events:
                node = getattr(event, "node", None)
                if node is None:
                    continue
                pos, _ = ticket.spacecraft.state_at(node.ut)
                marker = pv.Sphere(radius=size * 0.65, center=pos * self.r.distance_scale)
                actor = self.r.plotter.add_mesh(marker, color="#ffb347", lighting=False, pickable=False)
                self.r.maneuver_actors.append(actor)

    def build(self):
        self._flatten_system(self.r.system.root)
        self._compute_scales()
        self._init_scene_elements()
