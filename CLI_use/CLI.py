import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import click as cl
from basic_systems.orbit_pred import Body, Orbit, ManeuverNode, ManeuverPlanner, LambertSolver, Spacecraft
from basic_systems.builder import System
from basic_systems.RKF45 import RKF45
from basic_systems.renderer import SystemRenderer, make_config_system
import numpy as np

def j2_grav_system(pos_sc: np.ndarray, root_mu: float, ut: float, system: System):
    accel_vec = np.zeros(3)
    nodes_to_check = [system.root]

    while nodes_to_check:
        curr_body = nodes_to_check.pop(0)

        if hasattr(curr_body, "mu"):
            if curr_body == system.root:
                pos = np.zeros(3)
            elif hasattr(curr_body, "get_absolute_pos_at_ut"):
                pos = curr_body.get_absolute_pos_at_ut(ut)
            else:
                pos = np.zeros(3)

            r_vector = pos - pos_sc
            dist = np.linalg.norm(r_vector)

            if dist > 1e-5:
                accel_vec += (curr_body.mu / dist ** 3) * r_vector

        for moon in getattr(curr_body, "moons", []):
            nodes_to_check.append(moon)
            
    return accel_vec

class Ticket:
    def __init__(self, spacecraft: Spacecraft, system: System, bodies: list[Body], tof: float, start_ut: float, parking_orbit: Orbit, end_orbit: Orbit) -> None:
        self.spacecraft = spacecraft
        self.system = system
        self.bodies = bodies
        self.tof = tof
        self.start_ut = start_ut
        self.parking_orbit = parking_orbit
        self.end_orbit = end_orbit

        self.rkf45 = RKF45(spacecraft = self.spacecraft, gravity_equation = lambda pos, root_mu, ut: j2_grav_system(pos, root_mu, ut, self.system),
                           tolerance = 1e-3)

        self.lambert = LambertSolver(self.system.root.mu)
