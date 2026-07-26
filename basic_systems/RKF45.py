from basic_systems.orbit_pred import Body, Spacecraft, Orbit, LambertSolver
import numpy as np
import scipy as sp
from collections.abc import Callable

class RKF45:
    def __init__(self, spacecraft: Spacecraft, gravity_equation: Callable[[np.ndarray, float, float], np.ndarray], tolerence: float):
        self.spacecraft = spacecraft
        self.root_mu = self.spacecraft.get_root_of_system().mu
        self.gravity_func = gravity_equation
        self.tol = tolerence
    
    def _derivatives(self, state: np.ndarray, ut: float):
        pos, vel = state[0:3], state[3:6]
        accel = self.gravity_func(pos, self.root_mu, ut)
        return np.concatenate([vel, accel])
    
    def step(self, state: np.ndarray, h: float, ut: float):
        der = self._derivatives
        k1 = h * der(state, ut)
        k2 = h * der(state + k1 / 4, ut + h / 4)
        k3 = h * der(state + 3*k1/32 + 9*k2/32, ut + 3*h/8)
        k4 = h * der(state + 1982*k1/2197 - 7200*k2/2197 + 7296*k3/2197, ut + 12*h/13)
        k5 = h * der(state + 439*k1/216 - 8*k2 + 3680*k3/513 - 845*k4/4104, ut + h)
        k6 = h * der(state - 8*k1/27 + 2*k2 - 3544*k3/2565 + 1859*k4/4104 - 11*k5/40, ut + h/2)

        rk5_res = state + 16*k1/135 + 6656*k3/12825 + 28561*k4/56430 - 9*k5/50 + 2*k6/55
        rk4_res = state + 25*k1/216 + 1408*k3/2565 + 2197*k4/4104 - k5/5

        pos_error = np.linalg.norm(rk5_res[0:3] - rk4_res[0:3])
        scale = 2.0 if pos_error == 0 else 0.84 * (self.tol / pos_error) ** 0.25
        next_h = h * max(0.1, min(scale, 4.0))

        if pos_error <= self.tol:
            return True, next_h, rk4_res
        return False, next_h, state
    
    def propagate(self, duration: float, h0: float, start_ut: float, maneuver_nodes: list | None = None):
        curr_ut = start_ut
        end_ut = start_ut + duration
        h = h0

        pos = self.spacecraft.get_absolute_pos_at_ut(start_ut)
        vel = self.spacecraft.get_absolute_vel_at_ut(start_ut)
        state = np.concatenate([pos, vel])

        traj_hist = [(curr_ut, state[0:3].copy(), state[3:6].copy())]
        nodes = sorted(maneuver_nodes or [], key = lambda node: node.ut)

        while curr_ut < end_ut:
            next_node_ut = nodes[0].ut if len(nodes) > 0 else None
            target_time = end_ut

            if next_node_ut is not None and next_node_ut < end_ut:
                target_time = next_node_ut
            if curr_ut + h > target_time:
                h = target_time - curr_ut

            if h <= 1e-6:
                if next_node_ut is not None and abs(curr_ut - next_node_ut) < 1e-4:
                    node = nodes.pop(0)
                    state[3:6] += node.delta_v_vector
                    traj_hist.append((curr_ut, state[0:3].copy(), state[3:6].copy())) #type: ignore
                    h = h0
                    continue
                else:
                    break
            
            accepted, next_h, next_state = self.step(state, float(h), float(curr_ut))
            if accepted:
                state = next_state
                curr_ut += h
                traj_hist.append((curr_ut, state[0:3].copy(), state[3:6].copy())) #type: ignore
            
            h = next_h

        self.spacecraft.set_absolute_pos_at_ut(state[0:3], curr_ut) #type: ignore
        self.spacecraft.set_absolute_vel_at_ut(state[3:6], curr_ut) #type: ignore

        return traj_hist