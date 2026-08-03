import json
import sys
from pathlib import Path
import asyncio

# Add project root to sys.path for robust imports
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from basic_systems.builder import System
from basic_systems.orbit_pred import Body, Orbit, Spacecraft, get_ut_secs
from typing import Any
import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PySide6.QtWidgets import QWidget, QVBoxLayout, QSlider
from PySide6.QtCore import Qt, QTimer
from trame.app import get_server, asynchronous
from trame.ui.vuetify3 import SinglePageLayout
from pyvista.trame.ui import plotter_ui
from vtkmodules.vtkRenderingCore import vtkActor2D


def make_config_system(json_name: str = "planets.json") -> System:
    """Load a planetary system from a JSON configuration file.
    
    The JSON should have a 'system_name' and a 'bodies' dictionary.
    One body in 'bodies' must have a semi-major axis 'a' of 0.0 to be the root.
    """
    target_file = Path(json_name)
    
    # If not found directly, try relative to the script's parent and grandparent
    if not target_file.exists():
        current_dir = Path(__file__).resolve().parent
        # Try basic_systems/planets.json
        target_file = current_dir / json_name
        if not target_file.exists():
            # Try KerbalGravityProg/planets.json
            target_file = current_dir.parent / json_name
            if not target_file.exists():
                # Try relative to CWD
                target_file = Path.cwd() / json_name
                if not target_file.exists():
                    raise FileNotFoundError(f"Configuration file '{json_name}' not found at any expected location.")

    with open(target_file, "r", encoding="utf-8") as f:
        try:
            payload = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON from {target_file}: {e}")

    title = payload.get("system_name", "Unknown System")
    bodies_data = payload.get("bodies", {})
    if not bodies_data:
        raise ValueError("No 'bodies' found in configuration.")

    active_system: System | None = None

    def parse_body(name: str, attrs: dict, parent_node: Body | None = None) -> Body:
        nonlocal active_system
        
        # Validation
        mu = attrs.get("mu", 0.0)
        radius = attrs.get("radius", 1.0)
        
        curr_orbit = Orbit(
            a=attrs.get("a", 0.0), 
            e=attrs.get("e", 0.0), 
            arg_p=attrs.get("arg_p", 0.0),
            lon_of_asc=attrs.get("lon_of_asc", 0.0), 
            MA_at_t0=attrs.get("MA_at_t0", 0.0),
            inclination=attrs.get("inclination", 0.0), 
            parent=parent_node
        )
        
        curr_node = Body(
            name=name, 
            mu=mu, 
            radius=radius,
            atm_height=attrs.get("atm_height", 0.0), 
            orbit=curr_orbit,
            render_color=attrs.get("color", "#808080"),
            identifier=attrs.get("identifier", "X")
        )
        
        if parent_node is None:
            if active_system is not None:
                raise ValueError(f"Multiple root bodies detected. '{name}' and '{active_system.root.name}' both have a=0.")
            active_system = System(name=title, root_obj=curr_node)
        else:
            if active_system is not None:
                active_system.add_child(parent_node, curr_node)

        for moon_name, moon_attrs in attrs.get("moons", {}).items():
            parse_body(moon_name, moon_attrs, curr_node)
            
        return curr_node

    # Find the root body (a=0)
    root_keys = [k for k, v in bodies_data.items() if v.get("a", 0.0) == 0.0]
    if not root_keys:
        raise ValueError("No root body (with a=0.0) found in configuration.")
    
    root_key = root_keys[0]
    root_attrs = bodies_data.pop(root_key)
    
    # Any other top-level bodies are treated as direct children of the root
    # unless they are already in the root's moons list.
    if "moons" not in root_attrs:
        root_attrs["moons"] = {}
    
    root_attrs["moons"].update(bodies_data)
    
    parse_body(root_key, root_attrs, None)
    
    if active_system is None:
        raise ValueError("Failed to initialize system hierarchy.")
        
    return active_system


class SystemRenderer:
    def __init__(self, system: System, title: str = "KGRP render", root: Any | None = None, start_ut: float = 0.0, headless: bool = False, 
                 plotter: pv.Plotter | None = None, show_timeline_slider: bool = True, spacecraft: list[Spacecraft] = []):
        self.headless = headless
        self.root = root
        self.system = system
        self.title = title
        self.show_timeline_slider = show_timeline_slider
        self.plotter = plotter if plotter is not None else pv.Plotter(window_size=[1280, 720], off_screen=self.headless)
        self.plotter.set_background("#04040a") #type: ignore
        
        self.curr_ut = start_ut
        self.time_rate_per_s = 250.0
        self._old_rate = self.time_rate_per_s
        self.num_line_segments = 150
        self.moon_exaggeration = 0.5
        self.moon_exxageration = 5

        self.body_actors = {}
        self.body_labels: dict[Any, vtkActor2D] = {}
        self.body_list: list[Body] = []
        self.spacecraft_list: list[Spacecraft] = spacecraft
        self._focus_index = 0

        self._flatten_system(self.system.root)
        self._compute_scales()
        self._init_scene_elements()
        self._setup_input_controls()

    def _flatten_system(self, body: Body):
        if body:
            self.body_list.append(body)
            for moon in body.moons:
                if isinstance(moon, Spacecraft):
                    self.spacecraft_list.append(moon)
                else:
                    self._flatten_system(moon)

    def _compute_scales(self):
        """Dynamically compute scaling factors based on the system's physical dimensions."""
        max_dist = 0.0
        min_radius = float('inf')
        max_radius = 0.0

        for body in self.body_list:
            if body.orbit and body.orbit.semi_major_axis > 0:
                max_dist = max(max_dist, body.orbit.semi_major_axis)
            min_radius = min(min_radius, body.radius)
            max_radius = max(max_radius, body.radius)

        if max_dist == 0:
            max_dist = 1e9 # Default for single-body systems
        
        # We want the system to fit roughly in a 1000x1000x1000 cube for PyVista
        self.distance_scale = 1000.0 / max_dist
        
        # Body size scaling: root should be prominent, others visible
        # We target the root to be around 10-20 units, and smallest moons to be at least 0.5 units
        self.body_size_scale = 15.0 / self.system.root.radius
        self.root_size = (self.system.root.radius * self.body_size_scale) / 4 
        
        # If smallest moons are too small, we bump the scale a bit
        if min_radius * self.body_size_scale < 0.5:
             # We don't want to make them TOO big, just visible
             self.min_render_radius = 0.5
        else:
             self.min_render_radius = min_radius * self.body_size_scale

    def _get_display_pos_at_ut(self, body: Body, ut: float, _cache: dict | None = None) -> np.ndarray:
        if _cache is None:
            _cache = {}
        if body in _cache:
            return _cache[body]

        parent = body.orbit.parent if body.orbit else None
        if parent is None:
            pos = np.array(body.get_absolute_pos_at_ut(ut), dtype=float)
        else:
            parent_display_pos = self._get_display_pos_at_ut(parent, ut, _cache)
            # Local offset from parent
            local_offset = np.array(body.get_pos_at_ut(ut))
            
            # Exaggerate moon distances if they are too close to parent visually
            factor = self.moon_exaggeration if parent != self.system.root else 1.0
            pos = parent_display_pos + local_offset * factor

        _cache[body] = pos
        return pos

    def _compute_timeline_max_ut(self) -> float:
        periods = [b.orbit.period for b in self.body_list if b.orbit and b.orbit.period > 0]
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
        # Bodies
        for body in self.body_list:
            scaled_radius = max(self.min_render_radius, body.radius * self.body_size_scale)
            if body == self.system.root:
                scaled_radius = self.root_size
            
            actor = body.to_mesh(self.plotter, scaled_radius)
            label = self.plotter.add_point_labels(
                    [body.get_absolute_pos_at_ut(self.curr_ut) * self.distance_scale],
                    [body.name],
                    font_size=16,
                    text_color="white",
                    always_visible=True,
                    shape_opacity=0.3
                )
            self.body_actors[body] = actor
            self.body_labels[body] = label

            if body.orbit and body.orbit.period > 0:
                times = np.linspace(0, body.orbit.period, self.num_line_segments)
                path_pts = np.array([self._get_display_pos_at_ut(body, t) for t in times], dtype=np.float32)
                path_pts *= self.distance_scale

                line_poly = pv.lines_from_points(path_pts, close=True)
                self.plotter.add_mesh(line_poly, color=body.render_color, line_width=1.5, pickable=False)

        # Spacecraft
        sc_size = self.min_render_radius * 1.2
        for sc in self.spacecraft_list:
            actor , sc_label = self._create_spacecraft_mesh(self.plotter, sc_size, sc)
            self.body_labels[sc] = sc_label
            self.body_actors[sc] = actor

        self._update_all_positions()
        
        self.plotter.camera_position = 'iso'
        if self.system.root in self.body_actors:
            self.plotter.set_focus(self.body_actors[self.system.root].position)
        self.plotter.camera.zoom(1.1)

    def _setup_input_controls(self):
        def increase_warp():
            self.time_rate_per_s *= 2.0

        def decrease_warp():
            self.time_rate_per_s /= 2.0

        def toggle_pause():
            if self.time_rate_per_s != 0.0:
                self._old_rate = self.time_rate_per_s
                self.time_rate_per_s = 0.0
            else:
                self.time_rate_per_s = self._old_rate

        self.plotter.add_key_event("Up", increase_warp) #type: ignore
        self.plotter.add_key_event("Down", decrease_warp) #type: ignore
        self.plotter.add_key_event("space", toggle_pause) #type: ignore
        self.plotter.add_key_event("n", self.focus_next_body) #type: ignore
        self.plotter.add_key_event("p", self.focus_prev_body) #type: ignore

        self._actor_to_body = {actor: body for body, actor in self.body_actors.items()}
        self.plotter.enable_mesh_picking(#type: ignore
            callback=self._on_body_picked,
            use_actor=True,
            left_clicking=True,
            show=False,
            show_message=False,
        )

        if self.show_timeline_slider:
            self.plotter.add_slider_widget(#type: ignore
                callback=self._on_timeline_slider,
                rng=[0.0, self._compute_timeline_max_ut()],
                value=self.curr_ut,
                title="UT (s)",
                interaction_event='always',
            )

    def _on_timeline_slider(self, value):
        self.curr_ut = float(value)
        self._update_all_positions()
        self.plotter.render()

    def _on_body_picked(self, actor):
        body = self._actor_to_body.get(actor)
        if body is not None:
            self.focus_on_body(body)

    def focus_next_body(self):
        all_bodies = self.body_list + self.spacecraft_list
        if not all_bodies:
            return
        self._focus_index = (self._focus_index + 1) % len(all_bodies)
        self.focus_on_body(all_bodies[self._focus_index])

    def focus_prev_body(self):
        all_bodies = self.body_list + self.spacecraft_list
        if not all_bodies:
            return
        self._focus_index = (self._focus_index - 1) % len(all_bodies)
        self.focus_on_body(all_bodies[self._focus_index])

    def focus_on_body(self, body: Body | Spacecraft):
        if body not in self.body_actors:
            return
        
        target_pos = np.array(self.body_actors[body].position)
        old_focal = np.array(self.plotter.camera.focal_point)
        old_pos = np.array(self.plotter.camera.position)
        
        offset = old_pos - old_focal
        new_pos = target_pos + offset
        
        self.plotter.camera.focal_point = target_pos.tolist()
        self.plotter.camera.position = new_pos.tolist()
        self.plotter.render()

    def _update_all_positions(self):
        cache: dict = {}
        for body in self.body_list:
            if body in self.body_actors:
                pos = self._get_display_pos_at_ut(body, self.curr_ut, cache)
                world_pos = pos * self.distance_scale
                self.body_actors[body].position = world_pos.tolist()
                
                self.plotter.renderer.SetWorldPoint(world_pos[0], world_pos[1], world_pos[2], 1.0)
                self.plotter.renderer.WorldToDisplay()
                display_point = self.plotter.renderer.GetDisplayPoint()
                self.body_labels[body].SetPosition(int(display_point[0]), int(display_point[1]))
        
        for sc in self.spacecraft_list:
            if sc in self.body_actors:
                pos = sc.get_absolute_pos_at_ut(self.curr_ut)
                self.body_actors[sc].position = (pos * self.distance_scale).tolist()

    def update(self, dt: float = 0.033):
        """Unified update method for the simulation and rendering."""
        self.curr_ut += dt * self.time_rate_per_s
        self._update_all_positions()
        
        if self.root is not None:
            self.plotter.render()
            self.root.after(int(dt * 1000), lambda: self.update(dt))

    def update_simulation(self, *args):
        """Legacy compatibility method."""
        self.update()

    def run(self):
        if self.headless:
            self._update_all_positions()
            self.plotter.render()
        elif self.root is not None:
            self.update()
        else:
            self.plotter.add_timer_event(callback=self.update_simulation, duration=33, max_steps=10000000) #type: ignore
            self.plotter.show(title=self.title)

class PySideVTKAdapter(QWidget):
    _SLIDER_STEPS = 1000

    def __init__(self, system: System, start_ut: float = 0.0, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.qt_interactor = QtInteractor(self)
        layout.addWidget(self.qt_interactor)

        self.renderer = SystemRenderer(system, start_ut=start_ut, plotter=self.qt_interactor, show_timeline_slider=False)
        self._timeline_max_ut = self.renderer._compute_timeline_max_ut()

        self.timeline_slider = QSlider(Qt.Orientation.Horizontal)
        self.timeline_slider.setRange(0, self._SLIDER_STEPS)
        self.timeline_slider.valueChanged.connect(self._on_slider_changed)
        layout.addWidget(self.timeline_slider)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(33)

    def _on_slider_changed(self, value: int):
        frac = value / self._SLIDER_STEPS
        self.renderer.curr_ut = frac * self._timeline_max_ut
        self.renderer._update_all_positions()
        self.renderer.plotter.render()

    def _tick(self):
        # We use a fixed dt here for the timer-based update
        self.renderer.update(0.033)
        if not self.timeline_slider.isSliderDown():
            frac = max(0.0, min(1.0, self.renderer.curr_ut / self._timeline_max_ut))
            self.timeline_slider.blockSignals(True)
            self.timeline_slider.setValue(int(frac * self._SLIDER_STEPS))
            self.timeline_slider.blockSignals(False)

    def closeEvent(self, event):
        self.timer.stop()
        self.renderer.plotter.close()
        event.accept()

class TrameWebAdapter:
    def __init__(self, system: System, start_ut: float = 0.0):
        pv.OFF_SCREEN = True
        self.renderer = SystemRenderer(system, start_ut=start_ut, headless=True)
        self.server = get_server()
        self.state, self.ctrl = self.server.state, self.server.controller#type: ignore
        self.view = None

        with SinglePageLayout(self.server) as layout:
            layout.title.set_text("KRGP Web Render")
            with layout.content:
                self.view = plotter_ui(self.renderer.plotter)

        self.ctrl.on_server_ready.add(self._start_tick_loop)

    def _start_tick_loop(self, **kwargs):
        asynchronous.create_task(self._tick_loop())

    async def _tick_loop(self):
        while True:
            await asyncio.sleep(0.033)
            # headless renderer doesn't use the recursive root.after loop
            self.renderer.curr_ut += 0.033 * self.renderer.time_rate_per_s
            self.renderer._update_all_positions()
            if self.view is not None:
                self.view.update()

    def start(self, port: int = 8080, open_browser: bool = False):
        self.server.start(port=port, open_browser=open_browser)#type: ignore


if __name__ == "__main__":
    planets_path = Path(__file__).resolve().parent.parent / "planets.json"
    system = make_config_system(str(planets_path))
    renderer = SystemRenderer(system)
    renderer.show_timeline_slider = True
    renderer.run()