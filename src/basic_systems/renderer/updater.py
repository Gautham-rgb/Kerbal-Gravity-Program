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

    def set_time(self, ut: float):
        ut = float(ut)

        if not np.isfinite(ut):
            raise ValueError(f"set_time received invalid UT: {ut}")

        self.r.curr_ut = ut

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
                    pos = self.spacecraft_display_pos(sc, self.r.curr_ut, cache)
                    world_pos = pos * self.r.distance_scale
                    self.r.body_actors[sc].position = world_pos.tolist()
                    self.r.plotter.renderer.SetWorldPoint(world_pos[0], world_pos[1], world_pos[2], 1.0)
                    self.r.plotter.renderer.WorldToDisplay()
                    display_point = self.r.plotter.renderer.GetDisplayPoint()
                    self.r.body_labels[sc].SetPosition(int(display_point[0]), int(display_point[1]))

            # Moon orbit guides are drawn in their parent's local frame, so they
            # must be re-anchored to the parent's (moving) display position.
            scale = self.r.distance_scale
            for actor, parent in self.r.moon_orbit_links:
                parent_disp = self._get_display_pos_at_ut(parent, self.r.curr_ut, cache)
                actor.position = (parent_disp * scale).tolist()

    def spacecraft_display_pos(self, sc, ut, cache=None) -> np.ndarray:
        """Display position for a spacecraft, consistent with the (possibly
        exaggerated) body positions.

        A craft orbiting an exaggerated moon is offset from that moon by the
        same factor, otherwise it would render detached from the moon it flies
        around.
        """
        sc_abs = np.array(sc.get_absolute_pos_at_ut(ut), dtype=float)
        ref = getattr(sc, "parent", None)
        if (
            ref is not None
            and getattr(ref, "orbit", None) is not None
            and ref.orbit.parent is not None
            and ref.orbit.parent != self.r.system.root
        ):
            ref_disp = self._get_display_pos_at_ut(ref, ut, cache)
            ref_true = np.array(ref.get_absolute_pos_at_ut(ut), dtype=float)
            return ref_disp + (sc_abs - ref_true) * self.r.moon_exaggeration
        return sc_abs


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
