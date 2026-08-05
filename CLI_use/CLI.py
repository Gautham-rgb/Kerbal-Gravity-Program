import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import click as cl
from basic_systems.orbit_pred import Body, Orbit, ManeuverNode, ManeuverPlanner, LambertSolver, Spacecraft
from basic_systems.builder import System
from basic_systems.RKF45 import RKF45
from basic_systems.renderer.renderer import SystemRenderer, make_config_system
import numpy as np


class Ticket:
    def __init__(self, spacecraft: Spacecraft, system: System, bodies: list[Body], tof: float, start_ut: float, parking_orbit: Orbit, end_orbit: Orbit) -> None:
        self.spacecraft = spacecraft
        self.system = system
        self.bodies = bodies
        self.tof = tof
        self.start_ut = start_ut
        self.parking_orbit = parking_orbit
        self.end_orbit = end_orbit

        self.rkf45 = RKF45(spacecraft = self.spacecraft, gravity_equation = lambda pos, root_mu, ut: RKF45.n_body_grav_system(pos, root_mu, ut, self.system),
                           tolerance = 1e-3, system = self.system)

        self.lambert = LambertSolver(self.system.root.mu)

