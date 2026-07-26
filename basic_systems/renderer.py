import json
from pathlib import Path
from typing import Any
import sys
import typing
import numpy as np
import pyvista as pv
from PySide6.QtWidgets import QWidget, QVBoxLayout, QApplication, QMainWindow
from PySide6.QtCore import Qt, QTimer
from pyvistaqt import QtInteractor
from builder import System
from orbit_pred import Body, Orbit, Spacecraft, get_ut_secs
from nicegui import app, ui



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
    def __init__(self, system: System, start_ut: float = 0.0, notebook=False, plotter: pv.Plotter | None = None):
        self.system = system
        self.plotter = plotter if plotter is not None else pv.Plotter(window_size=[1280, 720], notebook=notebook)
        self.plotter.set_background("#04040a")  # type: ignore
        
        self.curr_ut = start_ut
        self.root_size = 4e5
        self.time_rate_per_s = 50000.0
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
            
            actor = body.to_mesh(self.plotter, scaled_radius)
            self.body_actors[body] = actor

            if hasattr(body, "orbit") and body.orbit is not None:
                period = getattr(body.orbit, "period", 0.0)
                if period > 0.0:
                    times = np.linspace(0, period, self.num_line_segments, endpoint=False)
                    path_pts = np.array([body.get_absolute_pos_at_ut(t) for t in times], dtype=np.float32)
                    
                    if len(path_pts) > 0:
                        path_pts = np.vstack([path_pts, path_pts[0]])
                        
                    path_pts *= self.distance_scale

                    line_poly = pv.lines_from_points(path_pts)
                    self.plotter.add_mesh(line_poly, color="#334477", line_width=1.5)

        self._set_planet_positions()
        self.plotter.camera_position = 'iso'
        root_body = self.system.root
        if root_body in self.body_actors:
            root_actor_pos = self.body_actors[root_body].position
            self.plotter.set_focus(root_actor_pos)
        self.plotter.camera.zoom(1.1)

    def _setup_input_controls(self):
        for key in ["v", "r", "Up", "Down", "space", "w", "s", "a", "d"]:
            try:
                self.plotter.clear_key_event(key)
            except Exception:
                pass

        self.plotter.add_key_event("Up", self.increase_warp)  # type: ignore
        self.plotter.add_key_event("Down", self.decrease_warp)  # type: ignore
        self.plotter.add_key_event("space", self.toggle_pause)  # type: ignore

        self.plotter.add_key_event("w", lambda: self.pan_camera(0, 1.0))  # type: ignore
        self.plotter.add_key_event("s", lambda: self.pan_camera(0, -1.0)) # type: ignore
        self.plotter.add_key_event("a", lambda: self.pan_camera(-1.0, 0)) # type: ignore
        self.plotter.add_key_event("d", lambda: self.pan_camera(1.0, 0))  # type: ignore

    def pan_camera(self, dx: float, dy: float):
        cam = self.plotter.camera
        focal = np.array(cam.focal_point)
        pos = np.array(cam.position)
        
        distance = np.linalg.norm(pos - focal)
        step = distance * 0.05
        
        shift = np.array([dx * step, dy * step, 0.0])
        cam.focal_point = focal + shift
        cam.position = pos + shift

    def increase_warp(self):
        self.time_rate_per_s *= 2.0

    def decrease_warp(self):
        self.time_rate_per_s /= 2.0

    def toggle_pause(self):
        if self.time_rate_per_s != 0.0:
            self._old_rate = self.time_rate_per_s
            self.time_rate_per_s = 0.0
        else:
            self.time_rate_per_s = getattr(self, "_old_rate", 50000.0)

    def _set_planet_positions(self):
        for body in self.body_list:
            if body in self.body_actors:
                actor = self.body_actors[body]
                pos = body.get_absolute_pos_at_ut(self.curr_ut)
                scaled_pos = [coord * self.distance_scale for coord in pos]
                actor.position = scaled_pos

    def update_simulation(self):
        dt = 0.033  
        self.curr_ut += dt * self.time_rate_per_s
        self._set_planet_positions()

    def run(self, title: str = "Kerbal Gravity Program"):
        self.plotter.add_timer_event(callback=lambda: (self.update_simulation(), self.plotter.render()), duration=33, max_steps=1e9)  # type: ignore
        self.plotter.show(title=title)


class PySideAdapter(QWidget):
    def __init__(self, system: System, parent: QWidget | None = None, flags: Qt.WindowType = Qt.WindowType.Widget, start_ut: float = 0.0) -> None:
        super().__init__(parent, flags)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.qt_interactor = QtInteractor(self)
        layout.addWidget(self.qt_interactor)

        self.renderer = SystemRenderer(system, start_ut=start_ut, plotter=self.qt_interactor)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(33)

    def _tick(self) -> None:
        self.renderer.update_simulation()
        self.renderer.plotter.render()


class NiceGUIAdapter(ui.card):
    def __init__(self, renderer: SystemRenderer, render_interval_s: float = 1.0) -> None:
        super().__init__()
        self.classes('w-full items-center justify-center p-4 bg-slate-900 border border-slate-700')
        self.renderer = renderer
        self.render_interval_s = render_interval_s

        self.tmp_dir = Path(__file__).resolve().parent / "tmp"
        self.tmp_dir.mkdir(exist_ok=True)
        self.html_file = self.tmp_dir / "render.html"
        app.add_static_files('/meshes', str(self.tmp_dir))
        self._export_scene()

        with self:
            with ui.row().classes('w-full justify-center gap-4 mb-2'):
                ui.button('Slower (Down)', on_click=self.renderer.decrease_warp).props('color=amber')
                ui.button('Pause/Play (Space)', on_click=self.renderer.toggle_pause).props('color=primary')
                ui.button('Faster (Up)', on_click=self.renderer.increase_warp).props('color=amber')
            
            with ui.row().classes('w-full justify-between px-4 mb-4 text-white text-sm font-mono'):
                self.warp_label = ui.label()
                self.ut_label = ui.label()
                
            self.iframe = ui.html(self._iframe_html())

        ui.timer(0.033, self._tick_simulation)
        ui.timer(self.render_interval_s, self._tick_render)

    def _iframe_html(self) -> str:
        return f'<iframe src="/meshes/render.html?t={self.renderer.curr_ut}" width="1280px" height="720px" style="border:none;"></iframe>'

    def _export_scene(self) -> None:
        self.renderer.plotter.export_html(str(self.html_file))

    def _tick_simulation(self) -> None:
        self.renderer.update_simulation()
        self.warp_label.set_text(f"Warp: {self.renderer.time_rate_per_s:,.1f}x")
        self.ut_label.set_text(f"UT: {self.renderer.curr_ut:,.2f}s")

    def _tick_render(self) -> None:
        self._export_scene()
        self.iframe.set_content(self._iframe_html())

        
if __name__ == "__main__":
    system = make_config_system(Path(__file__).resolve().parent.parent / "planets.json")
    renderer = SystemRenderer(system, start_ut=get_ut_secs(2026, 7, 26, 15, 43))
    renderer.run()