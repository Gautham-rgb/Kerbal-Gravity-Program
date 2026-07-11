from kerbal_solver import Body, Spacecraft, Orbit, ManeuverNode, LambertSolver
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
                    traj_hist.append((curr_ut, state[0:3].copy(), state[3:6].copy()))
                    h = h0
                    continue
                else:
                    break
            
            accepted, next_h, next_state = self.step(state, float(h), float(curr_ut))
            if accepted:
                state = next_state
                curr_ut += h
                traj_hist.append((curr_ut, state[0:3].copy(), state[3:6].copy()))
            
            h = next_h

        self.spacecraft.set_absolute_pos_at_ut(state[0:3], curr_ut)
        self.spacecraft.set_absolute_vel_at_ut(state[3:6], curr_ut)

        return traj_hist

if __name__ == "__main__":
    kerbol = Body("Kerbol", mu=1.1723328e18, radius=261600000)
    kerbin = Body("Kerbin", mu=3.5316000e12, radius=600000, orbit=Orbit(13599840256.0, 0.01, 0.0, 0.785398, 0.0, 0.174533, kerbol))
    duna = Body("Duna", mu=3.0136321e11, radius=320000, orbit=Orbit(20726155264.0, 0.051, 0.0, 2.364921, 3.1415926535, 0.010472, kerbol))

    start_ut, tof = 14148000.0, 200.0 * 21600.0
    r_start = kerbin.get_pos_at_ut(start_ut)
    r_target = duna.get_pos_at_ut(start_ut + tof)

    solver = LambertSolver(root_mu=kerbol.mu)
    v_departure_req, _ = solver.solve(r_start, r_target, tof, long_way=False)
    ker_vel = kerbin.get_absolute_vel_at_ut(start_ut)

    v_infinity_vector = v_departure_req - ker_vel
    v_infinity_mag = float(np.linalg.norm(v_infinity_vector))

    print("=====================================================")
    print("=== HYPERBOLIC EXCESS VELOCITY (V_INFINITY) ===")
    print("=====================================================")
    print(f"Heliocentric Departure Req:  {np.round(v_departure_req, 2)} m/s")
    print(f"Kerbin Solar Velocity Vector: {np.round(ker_vel, 2)} m/s")
    print(f"Net 3D V_Infinity Vector:     {np.round(v_infinity_vector, 2)} m/s")
    print(f"V_Infinity Magnitude:         {v_infinity_mag:.2f} m/s")

    ship_lambert = Spacecraft("Lambert_Pure", r_start, v_departure_req, start_ut, kerbol)
    halfway_ut = start_ut + (tof / 2.0)
    r_halfway_lamb = ship_lambert.get_pos_at_ut(halfway_ut)
    v_actual_halfway_lamb = ship_lambert.get_vel_at_ut(halfway_ut)

    v_required_halfway_lamb, _ = solver.solve(r_halfway_lamb, r_target, tof / 2.0, long_way=False)
    delta_v_lamb = v_required_halfway_lamb - v_actual_halfway_lamb

    u_pro_l = v_actual_halfway_lamb / np.linalg.norm(v_actual_halfway_lamb)
    u_norm_l = np.cross(r_halfway_lamb, v_actual_halfway_lamb)
    u_norm_l /= np.linalg.norm(u_norm_l)
    u_rad_l = np.cross(u_pro_l, u_norm_l)

    print("=====================================================")
    print("=== PATH 1: PURE CONIC LAMBERT PREDICTION VALUE ===")
    print("=====================================================")
    print(f"Actual Ship Speed at Halfway:   {np.linalg.norm(v_actual_halfway_lamb):.2f} m/s")
    print(f"Required Target Speed at Halfway: {np.linalg.norm(v_required_halfway_lamb):.2f} m/s")
    print(f"Net 3D Delta-V Vector: {np.round(delta_v_lamb, 2)} m/s")
    print(f"Prograde Burn:         {np.dot(delta_v_lamb, u_pro_l):.4f} m/s")
    print(f"Normal Burn:           {np.dot(delta_v_lamb, u_norm_l):.4f} m/s")
    print(f"Radial Burn:           {np.dot(delta_v_lamb, u_rad_l):.4f} m/s")
    print(f"Total Required Burn:   {np.linalg.norm(delta_v_lamb):.4f} m/s\n")

    r_parking = kerbin.radius + 100000.0
    v_orbit_mag = np.sqrt(kerbin.mu / r_parking)
    v_periap_mag = np.sqrt(v_infinity_mag**2 + (2.0 * kerbin.mu / r_parking))
    dv_burn_mag = v_periap_mag - v_orbit_mag

    e_escape = 1.0 + (r_parking * v_infinity_mag**2) / kerbin.mu
    theta_inf = np.arccos(-1.0 / e_escape)
    
    v_inf_unit = v_infinity_vector / v_infinity_mag
    h_kerbin = np.cross(kerbin.get_pos_at_ut(start_ut), kerbin.get_vel_at_ut(start_ut))
    z_dir = h_kerbin / np.linalg.norm(h_kerbin)
    
    y_dir = np.cross(z_dir, v_inf_unit)
    y_dir /= np.linalg.norm(y_dir)
    
    beta = np.pi - theta_inf
    r_periap_unit = (np.cos(beta) * v_inf_unit) + (np.sin(beta) * y_dir)
    v_periap_unit = np.cross(z_dir, r_periap_unit)
    
    r_lko_abs = r_start + (r_parking * r_periap_unit)
    v_lko_abs = ker_vel + (v_periap_mag * v_periap_unit)

    ship_rk45 = Spacecraft("RKF45_Hyperbolic_Escape", r_lko_abs, v_lko_abs, start_ut, kerbol)

    def full_n_body_gravity(pos: np.ndarray, mu: float, ut: float) -> np.ndarray:
        r_sun = np.linalg.norm(pos)
        accel = -mu * pos / r_sun**3
        
        pos_kerbin = pos - kerbin.get_pos_at_ut(ut)
        accel += -kerbin.mu * pos_kerbin / np.linalg.norm(pos_kerbin)**3
        
        pos_duna = pos - duna.get_pos_at_ut(ut)
        accel += -duna.mu * pos_duna / np.linalg.norm(pos_duna)**3
        return accel

    rk_propagator = RKF45(ship_rk45, full_n_body_gravity, 1e-6)
    rk_propagator.propagate(duration=tof / 2.0, h0=60.0, start_ut=start_ut)
    
    r_halfway_rk = ship_rk45.r0
    v_actual_halfway_rk = ship_rk45.v0

    v_required_halfway_rk, _ = solver.solve(r_halfway_rk, r_target, tof / 2.0, long_way=False)
    delta_v_rk = v_required_halfway_rk - v_actual_halfway_rk

    u_pro_rk = v_actual_halfway_rk / np.linalg.norm(v_actual_halfway_rk)
    u_norm_rk = np.cross(r_halfway_rk, v_actual_halfway_rk)
    u_norm_rk /= np.linalg.norm(u_norm_rk)
    u_rad_rk = np.cross(u_pro_rk, u_norm_rk)

    print("=====================================================")
    print("=== PATH 2: REALISTIC RKF45 NUMERICAL PREDICTION ===")
    print("=====================================================")
    print(f"Injection Burn Size (Delta-V): {dv_burn_mag:.2f} m/s")
    print(f"Actual Ship Speed at Halfway:   {np.linalg.norm(v_actual_halfway_rk):.2f} m/s")
    print(f"Required Target Speed at Halfway: {np.linalg.norm(v_required_halfway_rk):.2f} m/s")
    print(f"Net 3D Delta-V Vector: {np.round(delta_v_rk, 2)} m/s")
    print(f"Prograde Burn:         {np.dot(delta_v_rk, u_pro_rk):.4f} m/s")
    print(f"Normal Burn:           {np.dot(delta_v_rk, u_norm_rk):.4f} m/s")
    print(f"Radial Burn:           {np.dot(delta_v_rk, u_rad_rk):.4f} m/s")
    print(f"Total Required Burn:   {np.linalg.norm(delta_v_rk):.4f} m/s")



