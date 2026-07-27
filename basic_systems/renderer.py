import json
from pathlib import Path
from builder import System
from orbit_pred import Body, Orbit, Spacecraft, get_ut_secs
from typing import Any
import numpy as np
import pyvista as pv

from PySide6.QtWidgets import QWidget, QVBoxLayout
from PySide6.QtCore import QTimer
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from nicegui import ui, app


def make_config_system(json_name: str = "planets.json") -> System:
    current_dir = Path(__file__).resolve().parent
    target_file = current_dir / json_name
    if not target_file.exists():
        raise FileNotFoundError(f"Configuration file '{json_name}' not found at {target_file}")

    with open(target_file, "r", encoding="utf-8") as f:
        payload = json.load(f)

    title = payload.get("system_name", "UNK")
    data = payload.get("bodies", {})
    active_system: System | None = None

    def parse_bodies(name: str, attrs: dict, parent_node: Body | None = None):
        nonlocal active_system
        curr_orbit = Orbit(
            attrs.get("a", 0.0), attrs.get("e", 0.0), attrs.get("arg_p", 0.0),
            attrs.get("lon_of_asc", 0.0), attrs.get("MA_at_t0", 0.0),
            attrs.get("inclination", 0.0), parent_node
        )
        curr_node = Body(
            name, attrs.get("mu", 0.0), attrs.get("radius", 0.0),
            attrs.get("atm_height", 0.0), curr_orbit,
            render_color=attrs.get("color", "#808080")
        )
        if parent_node is None:
            active_system = System(name=title, root_obj=curr_node)
        else:
            if active_system is not None:
                active_system.add_child(parent_node, curr_node)

        for moon_name, moon_attrs in attrs.get("moons", {}).items():
            parse_bodies(moon_name, moon_attrs, curr_node)

    root_keys = [k for k, v in data.items() if v.get("a", 0.0) == 0.0]
    root_key = root_keys[0]
    root_attrs = data.pop(root_key)
    root_attrs["moons"] = data
    parse_bodies(root_key, root_attrs, None)
    
    if active_system is None:
        raise ValueError("Failed to initialize system hierarchy from configuration.")
    return active_system


class SystemRenderer:
    def __init__(self, system: System, root: Any | None = None, start_ut: float = 0.0, headless = False):
        self.headless = headless
        self.root = root
        self.system = system
        self.plotter = pv.Plotter(window_size=[1280, 720], off_screen = self.headless)
        self.plotter.set_background("#04040a") #type: ignore
        
        self.curr_ut = start_ut
        self.root_size = 2e5
        self.time_rate_per_s = 250.0
        self._old_rate = self.time_rate_per_s
        self.distance_scale = 1.0 / (2.0 * 10**4) 
        self.body_size_scale = 1.0 / 50.0
        self.num_line_segments = 150

        self.body_actors = {}
        self.body_list: list[Body] = []

        self._flatten_system(self.system.root)
        self._init_scene_elements()
        self._setup_input_controls()

    def _flatten_system(self, body):
        if body:
            self.body_list.append(body)
            for moon in getattr(body, "moons", []) or []:
                self._flatten_system(moon)

    def _init_scene_elements(self):
        for body in self.body_list:
            radius = getattr(body, "radius", 1000.0)
            scaled_radius = max(25.0, radius * self.body_size_scale)
            if body == self.system.root:
                scaled_radius = self.root_size
            
            color = getattr(body, "render_color", "#687c98")
            if not isinstance(color, str):
                color = "#687c98"

            actor = body.to_mesh(self.plotter, scaled_radius)
            self.body_actors[body] = actor

            if hasattr(body, "orbit") and body.orbit is not None:
                period = getattr(body.orbit, "period", 1000.0)
                times = np.linspace(0, period, self.num_line_segments)
                path_pts = np.array([body.get_absolute_pos_at_ut(t) for t in times], dtype=np.float32)
                path_pts *= self.distance_scale
                
                line_poly = pv.lines_from_points(path_pts)
                self.plotter.add_mesh(line_poly, color=body.render_color, line_width=1.5)

        self._set_planet_positions()
        self.plotter.camera_position = 'iso'
        root_body = self.system.root
        if root_body in self.body_actors:
            root_actor_pos = self.body_actors[root_body].position
            self.plotter.set_focus(root_actor_pos)
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
                self.time_rate_per_s = getattr(self, "_old_rate", 250.0)

        self.plotter.add_key_event("Up", increase_warp) #type: ignore
        self.plotter.add_key_event("Down", decrease_warp) #type: ignore
        self.plotter.add_key_event("space", toggle_pause) #type: ignore

    def _set_planet_positions(self):
        for body in self.body_list:
            if body in self.body_actors:
                actor = self.body_actors[body]
                pos = body.get_absolute_pos_at_ut(self.curr_ut)
                scaled_pos = [coord * self.distance_scale for coord in pos]
                actor.position = scaled_pos

    def update_simulation(self, *args):
        dt = 0.033  
        self.curr_ut += dt * self.time_rate_per_s
        self._set_planet_positions()
        
        if self.root is not None:
            self.plotter.render()
            self.root.after(33, self.update_simulation)

    def run(self):
        if self.headless:
            self._set_planet_positions()
            self.plotter.render()
        elif self.root is not None:
            self.update_simulation()
        else:
            self.plotter.add_timer_event(callback=self.update_simulation, duration=33, max_steps=10000000) #type: ignore
            self.plotter.show(title="KGRP Render")

class PySideVTKAdapter(QWidget):
    def __init__(self, renderer, parent=None):
        super().__init__(parent)
        self.renderer = renderer

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.vwidget = QVTKRenderWindowInteractor(self)
        layout.addWidget(self.vwidget)

        self.vwidget.SetRenderWindow(self.renderer.plotter.ren_win)
        self.renderer.plotter.iren = self.vwidget

        self.vwidget.Initialize()
        self.vwidget.Start()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.renderer.update_simulation)
        self.timer.start(33)

    def closeEvent(self, event):
        self.timer.stop()
        self.vwidget.Finalize()
        self.renderer.plotter.close()
        event.accept()

class NiceGUIAdapter:
    def __init__(self, renderer):
        self.renderer = renderer
        self.renderer.plotter.off_screen = True
        
        with ui.card().classes('w-full h-[600px] p-0 overflow-hidden items-center justify-center bg-black'):
            self.interactive_view = ui.interactive_image().classes('w-full h-full')
        
        ui.timer(0.033, self.stream_live_frame)

    def stream_live_frame(self):
        self.renderer.update_simulation()
        img_bytes = self.renderer.plotter.screenshot(None)
        encoded = base64.b64encode(img_bytes).decode('utf-8')
        self.interactive_view.set_source(f'data:image/png;base64,{encoded}')


if __name__ == "__main__":
    system = make_config_system(r"C:\Users\kaart\KerbalGravityProg\planets.json")
    renderer = SystemRenderer(system)
    renderer.run()
