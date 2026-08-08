from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from basic_systems.renderer.renderer import SystemRenderer

from basic_systems.orbit_pred import Body
import numpy as np

class SceneUpdater:
    def __init__(self, renderer: SystemRenderer):
        self.renderer = renderer

    @property
    def r(self):
        return self.renderer

    def set_time(self, ut: float) -> None:
        """Make ``ut`` the authoritative absolute simulation time.

        Ticket state is rebuilt when moving backwards, so timeline scrubbing
        never leaves a maneuver, stage, or resource transfer applied twice.
        """
        target_ut = float(ut)
        rewinding = target_ut < self.r.curr_ut

        for ticket in self.r.tickets:
            if rewinding:
                ticket.reset()
            if target_ut >= ticket.cursor_ut:
                ticket.advance_to(target_ut)

        self.r.curr_ut = target_ut
        self.update_all_positions()
        self.r._update_hud()

    def update_all_positions(self):
            cache: dict = {}
            for body in self.r.body_list:
                if body in self.r.body_actors:
                    pos = self._get_display_pos_at_ut(body, self.r.curr_ut, cache)
                    world_pos = pos * self.r.distance_scale
                    self.r.body_actors[body].position = world_pos.tolist()
                    
                    self.r.plotter.renderer.SetWorldPoint(world_pos[0], world_pos[1], world_pos[2], 1.0)
                    self.r.plotter.renderer.WorldToDisplay()
                    display_point = self.r.plotter.renderer.GetDisplayPoint()
                    self.r.body_labels[body].SetPosition(int(display_point[0]), int(display_point[1]))
                    if body in self.r.atmosphere_actors:
                        self.r.atmosphere_actors[body].position = world_pos.tolist()
            
            for sc in self.r.spacecraft_list:
                if sc in self.r.body_actors:
                    pos = sc.get_absolute_pos_at_ut(self.r.curr_ut)
                    world_pos = pos * self.r.distance_scale
                    self.r.body_actors[sc].position = world_pos.tolist()
                    self.r.plotter.renderer.SetWorldPoint(world_pos[0], world_pos[1], world_pos[2], 1.0)
                    self.r.plotter.renderer.WorldToDisplay()
                    display_point = self.r.plotter.renderer.GetDisplayPoint()
                    self.r.body_labels[sc].SetPosition(int(display_point[0]), int(display_point[1]))
    
        
    
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
            factor = self.r.moon_exaggeration if parent != self.r.system.root else 1.0
            pos = parent_display_pos + local_offset * factor

        _cache[body] = pos
        return pos
