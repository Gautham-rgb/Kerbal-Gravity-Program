from __future__ import annotations

from typing import Any

import numpy as np
import pyvista as pv
from vtkmodules.vtkRenderingCore import vtkActor2D

from basic_systems.builder import System
from basic_systems.orbit_pred import Body, Spacecraft, format_ut, KERBAL_UNITS, REAL_UNITS
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
        use_kerbal_time: bool = True,
        show_units_km: bool = True,
        event_marker_color: str = "#ffae46",
        moon_exaggeration: int = 10,
    ):
        self.system = system
        self.title = title
        self.root = root
        self.headless = headless
        self.show_timeline_slider = show_timeline_slider
        self.use_kerbal_time = use_kerbal_time
        self.show_units_km = show_units_km
        self.event_marker_color = event_marker_color

        self.plotter = (
            plotter
            if plotter is not None
            else pv.Plotter(
                window_size=[1280, 720],
                off_screen=headless,
            )
        )

        self.plotter.set_background(
            "#030611",
            top="#111a31",
        )


        self.curr_ut = float(start_ut)

        self.time_rate_per_s = 250.0
        self._old_rate = self.time_rate_per_s

        self.timeline_start_ut = 0.0


        self.num_line_segments = 150
        self.moon_exaggeration = moon_exaggeration

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
        self.moon_orbit_links: list[tuple[pv.Actor, Body]] = []
        self.atmosphere_actors: dict[Any, pv.Actor] = {}
        self.maneuver_actors: list[pv.Actor] = []
        self.maneuver_node_actors: dict[Any, dict] = {}
        self.event_marker_actors: dict[int, pv.Actor] = {}

        self.hud_actor: Any | None = None
        self.hud_lower_left: Any | None = None
        self.hud_node_details: Any | None = None
        self.selected_info: str | None = None

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
        self._update_hud_lower()
        self._update_hud_node_details()


    def update(self, dt: float = 0.033):
        """
        Advance simulation time and update the entire scene.

        dt is real wall-clock time in seconds.
        """

        if self.time_rate_per_s != 0.0:
            self.curr_ut += dt * self.time_rate_per_s
            self._advance_tickets()

        # Keep time finite.
        if not np.isfinite(self.curr_ut):
            self.curr_ut = self.timeline_start_ut

        # Keep the animation loop alive even if a frame's HUD/position update
        # raises, so simulation time keeps advancing.
        try:
            self.updater.set_time(self.curr_ut)

            self._update_hud()
            self._update_hud_lower()
            self._update_hud_node_details()
        except Exception:
            pass

        self.plotter.render()

        if self.root is not None:
            self.root.after(
                int(dt * 1000),
                lambda: self.update(dt),
            )

    def _advance_tickets(self):
        """Live playback advances ticket cursors; scrubbing never mutates them.

        This is the single authoritative ``advance_to`` call before the scene
        positions update, so maneuvers are executed once, in time order, and a
        backward slider drag cannot re-run a burn.
        """
        for ticket in self.tickets:
            if ticket.cursor_ut < self.curr_ut:
                ticket.advance_to(self.curr_ut)

    def update_simulation(self):
        """PyVista timer callback."""
        self.update()

    def set_time_rate(self, rate: float) -> None:
        """Set the wall-clock -> sim-time multiplier (warp speed)."""
        rate = float(rate)
        if rate < 0.0:
            rate = 0.0
        self.time_rate_per_s = rate
        if rate != 0.0:
            self._old_rate = rate
        self._update_hud()
        self.plotter.render()

    def run(self):
        if self.headless:
            self.updater.set_time(self.curr_ut)
            self._update_hud()
            self._update_hud_lower()
            self._update_hud_node_details()
            self.plotter.render()
            return

        if self.root is not None:
            self.update()
            return

        # Desktop: expose a live warp control and drive a single timer loop so
        # simulation time (and thus body positions) actually advances.
        self.plotter.add_slider_widget(
            callback=lambda v: self.set_time_rate(float(v)),
            rng=[0.0, 4000.0],
            value=self.time_rate_per_s,
            title="Time warp (x)",
            pointa=(0.25, 0.93),
            pointb=(0.75, 0.93),
        )

        self.plotter.add_timer_event(
            callback=self.update_simulation,
            duration=33,
            max_steps=10_000_000,
        )

        self.plotter.show(title=self.title)

    def screenshot(self, filename: str) -> None:
        """Capture a screenshot to *filename* (PNG/JPG based on extension)."""
        self.plotter.screenshot(filename, transparent=False)
        print(f"[Screenshot] Saved to {filename}")

    def _format_units(self, meters: float) -> str:
        if self.show_units_km and abs(meters) >= 1000.0:
            return f"{meters / 1000.0:.1f} km"
        return f"{meters:.0f} m"

    def _format_time(self, ut: float) -> str:
        if self.use_kerbal_time:
            return format_ut(ut, ker_time=True, units=KERBAL_UNITS)
        return format_ut(ut, units=REAL_UNITS)

    def _speed_to_surface(self, body: Body | Spacecraft) -> float | None:
        """Surface-relative speed of a vessel, or the rotation speed of a planet."""
        if isinstance(body, Spacecraft):
            parent = body.parent
            if parent is None:
                return None
            try:
                pos = np.array(body.get_absolute_pos_at_ut(self.curr_ut))
                parent_pos = np.array(parent.get_absolute_pos_at_ut(self.curr_ut))
                vel = np.array(body.get_absolute_vel_at_ut(self.curr_ut))
                parent_vel = np.array(parent.get_absolute_vel_at_ut(self.curr_ut))
                rel_vel = vel - parent_vel
                rel_pos = pos - parent_pos
                dist = float(np.linalg.norm(rel_pos))
                if dist < 1e-3:
                    return float(np.linalg.norm(rel_vel))
                radial = np.dot(rel_vel, rel_pos) / dist
                tangential = np.sqrt(max(0.0, np.dot(rel_vel, rel_vel) - radial * radial))
                return float(tangential)
            except Exception:
                return None
        if body.rotation_period_s > 0:
            return float(2.0 * np.pi * body.radius / body.rotation_period_s)
        return None

    def _altitude(self, body: Body | Spacecraft) -> float | None:
        """Altitude of a vessel above its reference body (planets have none)."""
        if not isinstance(body, Spacecraft):
            return None
        parent = body.parent
        if parent is None or parent.radius <= 0:
            return None
        try:
            pos = np.array(body.get_absolute_pos_at_ut(self.curr_ut))
            parent_pos = np.array(parent.get_absolute_pos_at_ut(self.curr_ut))
            return float(np.linalg.norm(pos - parent_pos) - parent.radius)
        except Exception:
            return None

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
        upcoming_events = 0

        for ticket in self.tickets:
            for event in ticket.events:
                if (
                    not getattr(event, "completed", False)
                    and event.start_ut <= self.curr_ut
                ):
                    active_events += 1
                elif (
                    not getattr(event, "completed", False)
                    and event.start_ut > self.curr_ut
                ):
                    upcoming_events += 1

        altitude = self._altitude(self.focused_body) if self.focused_body else None
        speed = self._speed_to_surface(self.focused_body) if self.focused_body else None

        text = (
            "KERBAL GRAVITY PROGRAM\n"
            f"{self._format_time(self.curr_ut)}\n"
            f"Warp {rate}\n"
            f"Focus {focused}\n"
            f"Active events {active_events}\n"
            f"Upcoming events {upcoming_events}"
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

    def _update_hud_lower(self):
        focused = self.focused_body
        lines = []

        if focused is not None:
            if isinstance(focused, Spacecraft):
                alt = self._altitude(focused)
                spd = self._speed_to_surface(focused)
                if alt is not None:
                    lines.append(f"ALT    {self._format_units(alt)}")
                if spd is not None:
                    lines.append(f"SPD    {spd:.1f} m/s")
            else:
                rot = self._speed_to_surface(focused)
                if rot is not None:
                    lines.append(f"ROT    {rot:.1f} m/s")

        if not lines:
            lines.append("Focus a body for details")

        text = "\n".join(lines)

        if self.hud_lower_left is None:
            self.hud_lower_left = self.plotter.add_text(
                text,
                position="lower_left",
                font_size=12,
                color="#a8d0ff",
                shadow=True,
                name="hud-lower-left",
            )
        elif hasattr(self.hud_lower_left, "SetText"):
            self.hud_lower_left.SetText(2, text)
        else:
            self.hud_lower_left.SetInput(text)

    def _format_dv_component(self, vec: np.ndarray) -> str:
        mag = float(np.linalg.norm(vec))
        return f"{mag:.1f} m/s"

    def _update_hud_node_details(self):
        """Show maneuver-node details in the upper right when one is selected."""
        info = self.selected_info

        if info is None:
            if self.hud_node_details is not None:
                self.plotter.remove_actor(self.hud_node_details)
                self.hud_node_details = None
            return

        if self.hud_node_details is None:
            self.hud_node_details = self.plotter.add_text(
                info,
                position="upper_right",
                font_size=13,
                color="#ffd27a",
                shadow=True,
                name="hud-node-details",
            )
        else:
            if hasattr(self.hud_node_details, "SetText"):
                self.hud_node_details.SetText(2, info)
            else:
                self.hud_node_details.SetInput(info)

    def select_maneuver_node(self, ticket, event, node) -> None:
        """Register the clicked maneuver node and refresh the HUD."""
        try:
            prograde = self._format_dv_component(node.prograde)
            normal = self._format_dv_component(node.normal)
            radial = self._format_dv_component(node.radial)
        except Exception:
            prograde = normal = radial = "n/a"

        self.selected_info = (
            f"MANEUVER  @ {self._format_time(node.ut)}\n"
            f"Total Δv  {self._format_dv_component(node.delta_v_vector)}\n"
            f"Prograde  {prograde}\n"
            f"Normal    {normal}\n"
            f"Radial    {radial}\n"
            f"(vessel {getattr(ticket.spacecraft, 'name', '?')})"
        )
        self._update_hud_node_details()
        self.plotter.render()

    def clear_selection(self) -> None:
        self.selected_info = None
        self._update_hud_node_details()
        self.plotter.render()

if __name__ == "__main__":
    from basic_systems import example_system_path

    system = System.load(example_system_path("planets_ksp"))

    renderer = SystemRenderer(
        system=system,
        title="KGRP",
        start_ut=0.0,
        show_timeline_slider=True,
    )

    renderer.run()