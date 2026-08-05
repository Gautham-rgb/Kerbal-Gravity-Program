from basic_systems.builder import System
from basic_systems.orbit_pred import Body, Orbit, Spacecraft, get_ut_secs
from basic_systems.renderer.loader import load_system as make_config_system
from basic_systems.renderer.scene import SceneBuilder
from basic_systems.renderer.updater import SceneUpdater
from basic_systems.renderer.controls import ControlManager
from typing import Any
import numpy as np
import pyvista as pv
from pathlib import Path

from vtkmodules.vtkRenderingCore import vtkActor2D



class SystemRenderer:
    def __init__(
        self,
        system: System, title: str = "KGRP render", root: Any | None = None, start_ut: float = 0.0, headless: bool = False, 
        plotter: pv.Plotter | None = None, show_timeline_slider: bool = True, spacecraft: list[Spacecraft] | None = None):
        self.headless = headless
        self.root = root
        self.system = system
        self.title = title
        self.show_timeline_slider = show_timeline_slider

        self.plotter = (
            plotter
            if plotter is not None
            else pv.Plotter(
                window_size=[1280, 720],
                off_screen=self.headless,
            )
        )
        self.plotter.set_background("#04040a")  # type: ignore

        # Simulation state
        self.curr_ut = start_ut
        self.time_rate_per_s = 250.0
        self._old_rate = self.time_rate_per_s

        # Rendering settings
        self.num_line_segments = 150
        self.moon_exaggeration = 5

        # Computed by SceneBuilder
        self.distance_scale = 1.0
        self.body_size_scale = 1.0
        self.root_size = 1.0
        self.min_render_radius = 0.5

        # Scene data
        self.body_actors: dict[Any, pv.Actor] = {}
        self.body_labels: dict[Any, vtkActor2D] = {}
        self.body_list: list[Body] = []
        self.spacecraft_list: list[Spacecraft] = (
            spacecraft if spacecraft is not None else []
        )

        # Camera state
        self._focus_index = 0

        # Helpers
        self.scene = SceneBuilder(self)
        self.controls = ControlManager(self)
        self.updater = SceneUpdater(self)
        self.scene.build()
        self.controls.setup()
    

    def update(self, dt: float = 0.033):
        """Unified update method for the simulation and rendering."""
        self.curr_ut += dt * self.time_rate_per_s
        self.updater.update_all_positions()
        
        if self.root is not None:
            self.plotter.render()
            self.root.after(int(dt * 1000), lambda: self.update(dt))

    def update_simulation(self, *args):
        """Legacy compatibility method."""
        self.update()

    def run(self):
        if self.headless:
            self.updater.update_all_positions()
            self.plotter.render()
        elif self.root is not None:
            self.update()
        else:
            self.plotter.add_timer_event(callback=self.update_simulation, duration=33, max_steps=10000000) #type: ignore
            self.plotter.show(title=self.title)




if __name__ == "__main__":
    planets_path = Path(__file__).resolve().parent.parent.parent / "planets.json"
    system = make_config_system(str(planets_path))
    renderer = SystemRenderer(system)
    renderer.show_timeline_slider = True
    renderer.run()