from __future__ import annotations

from typing import Any

import pyvista as pv
from vtkmodules.vtkRenderingCore import vtkActor2D

from basic_systems.builder import System
from basic_systems.orbit_pred import Body, Spacecraft, format_ut
from basic_systems.renderer.scene import SceneBuilder
from basic_systems.renderer.updater import SceneUpdater
from basic_systems.renderer.controls import ControlManager


class SystemRenderer:
    def __init__(
        self,
        system: System,
        title: str = "KGRP",
        root: Any | None = None,
        start_ut: float = 0.0,
        headless: bool = False,
        plotter: pv.Plotter | None = None,
        show_timeline_slider: bool = True,
        spacecraft: list[Spacecraft] | None = None,
        tickets: list[Any] | None = None,
    ):
        self.system = system
        self.title = title
        self.root = root
        self.headless = headless
        self.show_timeline_slider = show_timeline_slider

        self.plotter = (
            plotter
            if plotter is not None
            else pv.Plotter(
                window_size=[1280, 720],
                off_screen=headless,
            )
        )

        self.plotter.set_background( 
            "#030611", #type: ignore
            top="#111a31",
        )


        self.curr_ut = float(start_ut)

        self.time_rate_per_s = 250.0
        self._old_rate = self.time_rate_per_s

        self.timeline_start_ut = 0.0
        self.timeline_end_ut = 3.15e7


        self.num_line_segments = 150
        self.moon_exaggeration = 5

        self.distance_scale = 1.0
        self.body_size_scale = 1.0
        self.root_size = 1.0
        self.min_render_radius = 0.5

        self.body_actors: dict[Any, pv.Actor] = {}
        self.body_labels: dict[Any, vtkActor2D] = {}

        self.body_list: list[Body] = []
        self.spacecraft_list: list[Spacecraft] = (
            spacecraft if spacecraft is not None else []
        )

        self.tickets = tickets if tickets is not None else []

        self.orbit_actors: list[pv.Actor] = []
        self.atmosphere_actors: dict[Any, pv.Actor] = {}
        self.maneuver_actors: list[pv.Actor] = []

        self.hud_actor: Any | None = None

        self._focus_index = 0
        self.focused_body: Body | Spacecraft | None = None

        self.scene = SceneBuilder(self)
        self.updater = SceneUpdater(self)
        self.controls = ControlManager(self)

        # Build scene first.
        self.scene.build()

        # Select first object by default.
        bodies = self.body_list + self.spacecraft_list

        if bodies:
            self.focused_body = bodies[0]

        self.controls.setup()

        self._update_hud()


    def update(self, dt: float = 0.033):
        """
        Advance simulation time and update the entire scene.

        dt is real wall-clock time in seconds.
        """

        if self.time_rate_per_s != 0.0:
            self.curr_ut += dt * self.time_rate_per_s

        # Keep time finite.
        if not __import__("numpy").isfinite(self.curr_ut):
            self.curr_ut = self.timeline_start_ut

        self.updater.set_time(self.curr_ut)

        self._update_hud()

        self.plotter.render()

        if self.root is not None:
            self.root.after(
                int(dt * 1000),
                lambda: self.update(dt),
            )

    def update_simulation(self):
        """PyVista timer callback."""
        self.update()


    def run(self):
        if self.headless:
            self.updater.set_time(self.curr_ut)
            self._update_hud()
            self.plotter.render()
            return

        if self.root is not None:
            self.update()
            return

        self.plotter.add_timer_event( #type: ignore
            callback=self.update_simulation,
            duration=33,
            max_steps=10_000_000,
        )

        self.plotter.show(title=self.title)



    def _update_hud(self):
        focused = (
            self.focused_body.name
            if self.focused_body is not None
            else "System"
        )

        if self.time_rate_per_s == 0.0:
            rate = "PAUSED"
        else:
            rate = f"x{self.time_rate_per_s:g}"

        active_events = 0

        for ticket in self.tickets:
            for event in ticket.events:
                if (
                    not getattr(event, "completed", False)
                    and event.start_ut <= self.curr_ut
                ):
                    active_events += 1

        text = (
            "KERBAL GRAVITY PROGRAM\n"
            f"{format_ut(self.curr_ut)}\n"
            f"Warp {rate}\n"
            f"Focus {focused}\n"
            f"Active events {active_events}"
        )

        if self.hud_actor is None:
            self.hud_actor = self.plotter.add_text(
                text,
                position="upper_left",
                font_size=13,
                color="#dce9ff",
                shadow=True,
                name="mission-hud",
            )
        elif hasattr(self.hud_actor, "SetText"):
            self.hud_actor.SetText(2, text)
        else:
            self.hud_actor.SetInput(text)

if __name__ == "__main__":
    from pathlib import Path

    from basic_systems.renderer.loader import (
        load_system as make_config_system,
    )

    planets_path = (
        Path(__file__).resolve().parent.parent.parent
        / "planets.json"
    )

    system = make_config_system(str(planets_path))

    renderer = SystemRenderer(
        system=system,
        title="KGRP",
        start_ut=0.0,
        show_timeline_slider=True,
    )

    renderer.run()