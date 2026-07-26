import numpy as np
import datetime as dtime
from dataclasses import dataclass
from pyvista import Sphere, Plotter
from typing import Literal

@dataclass
class Constants:
    KER_YEAR_DAY: int = 426
    KER_DAY_HOUR: int = 6
    KER_HOUR_MIN: int = 60
    KER_MIN_SEC: int = 60
    KER_YEAR_SEC: float = 9201600.0
    KER_DAY_SEC: float = 21600.0

def get_ut_secs(year: int, month: int, day: int, hour: int, minute: int, seconds: int = 0, ker_time = False, am_pm: None | Literal["AM", "PM", "am", "pm"] = None):

    if am_pm is not None:
        am_pm_upper = am_pm.upper()
        if am_pm_upper == "PM" and hour < 12:
            hour += 12
        elif am_pm_upper == "AM" and hour == 12:
            hour = 0

    if ker_time:
        elapsed_years = year - 1
        elapsed_days = day - 1
        
        time_conversions = (
            (elapsed_years, Constants.KER_YEAR_SEC),
            (elapsed_days, Constants.KER_DAY_SEC),
            (hour, Constants.KER_HOUR_MIN * Constants.KER_MIN_SEC),
            (minute, Constants.KER_MIN_SEC),
            (seconds, 1)
        )

        return float(sum(val * const for val, const in time_conversions))
    else:
        j2000_epoch = dtime.datetime(2000, 1, 1, 12, 0, 0, tzinfo=dtime.timezone.utc)
        base_date = dtime.datetime(year, month, 1, tzinfo=dtime.timezone.utc)
        target_date = base_date + dtime.timedelta(
            days=day - 1, 
            hours=hour, 
            minutes=minute, 
            seconds=seconds
        )
        return (target_date - j2000_epoch).total_seconds()

def solve_anomaly(mean_anomaly: float, eccen: float):
    epsilon = 1e-12
    max_iter = 100
    m = np.fmod(mean_anomaly, 2 * np.pi)
    if m > np.pi: m -= 2 * np.pi
    if m < -np.pi: m += 2 * np.pi
    
    if eccen < 1.0:
        E = m
        for _ in range(max_iter):
            delta_E = (E - eccen * np.sin(E) - m) / (1 - eccen * np.cos(E))
            E -= delta_E
            if abs(delta_E) < epsilon:
                break
        return E
    elif eccen > 1.0:
        m = mean_anomaly
        H = 0.0
        if abs(m) > 0.1:
            H = np.asinh(m / eccen)
        
        for _ in range(max_iter):
            delta_H = (eccen * np.sinh(H) - H - m) / (eccen * np.cosh(H) - 1)
            H -= delta_H
            if abs(delta_H) < epsilon:
                break
        return H
    else:
        e_eff = 0.9999999999
        E = m
        for _ in range(max_iter):
            delta_E = (E - e_eff * np.sin(E) - m) / (1 - e_eff * np.cos(E))
            E -= delta_E
            if abs(delta_E) < epsilon:
                break
        return E

class Orbit:
    def __init__(self, a = 0.0, e = 0.0, arg_p = 0.0, lon_of_asc = 0.0, MA_at_t0 = 0.0, inclination: float = 0.0, parent: Body|None = None):
        self.parent = parent
        self.semi_major_axis = a
        self.eccen = e
        self.arg_periapsis = arg_p
        self.lon_of_asc = lon_of_asc
        self.mean_anomaly_at_t0 = MA_at_t0
        self.inclination = inclination
        if isinstance(parent, (Body, Spacecraft)):
            self.period = 2 * np.pi * np.sqrt((self.semi_major_axis ** 3 / self.parent.mu))

class Body:
    def __init__(self, name: str, mu: float, radius: float, atm_height: float = 0.0, 
                 orbit = None, moons = None, render_color: str | None = None) -> None:
        self.name = name
        self.mu = mu
        self.radius = radius
        self.atm_height = atm_height
        self.orbit = orbit if orbit else Orbit()
        self.moons = moons if moons is not None else []
        self.render_color = render_color
        self.period = 0.0
        

    def to_mesh(self, plotter, scaled_radius):
        color = getattr(self, "render_color", "#687c98")
        if not isinstance(color, str):
            color = "#687c98"
            
        mesh = Sphere(
            radius=scaled_radius, 
            theta_resolution=32, 
            phi_resolution=32
        )
        return plotter.add_mesh(mesh, color=color, smooth_shading=True)


    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Body):
            return False
        return self.__dict__ == value.__dict__

    def __hash__(self) -> int:
        return hash(frozenset((k, v) for k, v in self.__dict__.items() if k != "parent" and not isinstance(v, (list, set, dict))))
    
    def get_root_of_system(self):
        if self.orbit.parent == None:
            return self
        return self.orbit.parent.get_root_of_system()
    
    def get_all_obj_in_system(self) -> dict[Body, list[Body | Spacecraft]]:
        root = self.get_root_of_system()
        system_dict = {}
        def traverse(root: Body):
            if not root: 
                return
            system_dict[root] = [] 
            for moon in root.moons:
                traverse(moon)
        traverse(root)
        return system_dict 

    def get_pos_at_ut(self, ut: float):
        if self.orbit.parent is None:
            return np.zeros(3)
        
        a, e, mu = self.orbit.semi_major_axis, self.orbit.eccen, self.orbit.parent.mu
        n = np.sqrt(np.abs(mu / a ** 3))
        mean_anomaly = self.orbit.mean_anomaly_at_t0 + n * ut
        anomaly = solve_anomaly(mean_anomaly, e)

        if e < 1.0:
            cosA, sinA = np.cos(anomaly), np.sin(anomaly)
            x_perifocal = a * (cosA - e)
            y_perifocal = a * np.sqrt(1.0 - e**2) * sinA
        elif e > 1.0:
            a_abs = np.abs(a)
            coshA, sinhA = np.cosh(anomaly), np.sinh(anomaly)
            x_perifocal = a_abs * (e - coshA)
            y_perifocal = a_abs * np.sqrt(e**2 - 1.0) * sinhA
        else:
            cosA, sinA = np.cos(anomaly), np.sin(anomaly)
            x_perifocal = a * (cosA - e)
            y_perifocal = a * sinA

        pos_perifocal = np.array([x_perifocal, y_perifocal, 0.], np.float64)

        cos_lan, sin_lan = np.cos(self.orbit.lon_of_asc), np.sin(self.orbit.lon_of_asc)
        cos_inc, sin_inc = np.cos(self.orbit.inclination), np.sin(self.orbit.inclination)
        cos_ap, sin_ap = np.cos(self.orbit.arg_periapsis), np.sin(self.orbit.arg_periapsis)

        R_lan = np.array([[cos_lan, -sin_lan, 0.], [sin_lan, cos_lan, 0.], [0., 0., 1.]], np.float64)
        R_inc = np.array([[1., 0., 0.], [0., cos_inc, -sin_inc], [0., sin_inc, cos_inc]], np.float64)
        R_ap = np.array([[cos_ap, -sin_ap, 0.], [sin_ap, cos_ap, 0.], [0., 0., 1.]], np.float64)

        return R_lan @ R_inc @ R_ap @ pos_perifocal
    
    def get_absolute_pos_at_ut(self, ut: float):
        if self.orbit.parent is None:
            return np.zeros(3)
        return self.get_pos_at_ut(ut) + self.orbit.parent.get_absolute_pos_at_ut(ut)
    
    def get_vel_at_ut(self, ut: float):
        if self.orbit.parent is None:
            return np.zeros(3)
        
        a, e, mu = self.orbit.semi_major_axis, self.orbit.eccen, self.orbit.parent.mu
        n = np.sqrt(np.abs(mu / a ** 3))
        mean_anomaly = self.orbit.mean_anomaly_at_t0 + n * ut
        anomaly = solve_anomaly(mean_anomaly, e)

        if e < 1.0:
            cosA, sinA = np.cos(anomaly), np.sin(anomaly)
            v_coef = np.sqrt(mu * a) / (a * (1.0 - e * cosA))
            vx_perifocal = -v_coef * sinA
            vy_perifocal = v_coef * np.sqrt(1.0 - e**2) * cosA
        elif e > 1.0:
            a_abs = np.abs(a)
            coshA, sinhA = np.cosh(anomaly), np.sinh(anomaly)
            v_coef = np.sqrt(mu * a_abs) / (a_abs * (e * coshA - 1.0))
            vx_perifocal = -v_coef * sinhA
            vy_perifocal = v_coef * np.sqrt(e**2 - 1.0) * coshA
        else:
            cosA, sinA = np.cos(anomaly), np.sin(anomaly)
            v_coef = np.sqrt(mu / a)
            vx_perifocal = -v_coef * sinA
            vy_perifocal = v_coef * cosA

        vel_perifocal = np.array([vx_perifocal, vy_perifocal, 0.], np.float64)

        cos_lan, sin_lan = np.cos(self.orbit.lon_of_asc), np.sin(self.orbit.lon_of_asc)
        cos_inc, sin_inc = np.cos(self.orbit.inclination), np.sin(self.orbit.inclination)
        cos_ap, sin_ap = np.cos(self.orbit.arg_periapsis), np.sin(self.orbit.arg_periapsis)

        R_lan = np.array([[cos_lan, -sin_lan, 0.], [sin_lan, cos_lan, 0.], [0., 0., 1.]], np.float64)
        R_inc = np.array([[1., 0., 0.], [0., cos_inc, -sin_inc], [0., sin_inc, cos_inc]], np.float64)
        R_ap = np.array([[cos_ap, -sin_ap, 0.], [sin_ap, cos_ap, 0.], [0., 0., 1.]], np.float64)

        return R_lan @ R_inc @ R_ap @ vel_perifocal
    
    def get_absolute_vel_at_ut(self, ut: float):
        if self.orbit.parent is None:
            return np.zeros(3)
        return self.get_vel_at_ut(ut) + self.orbit.parent.get_absolute_vel_at_ut(ut)

class Barycenter(Body):
    def __init__(self, sys_name: str, a: Body, b: Body, total_sma: float, ecc: float, inc: float, lan: float, arg_p: float):
        super().__init__(sys_name, mu=(a.mu + b.mu), radius=0.0)
        self.child_A, self.child_B = a, b
        a_sma = total_sma * (self.child_B.mu / self.mu)
        b_sma = total_sma * (self.child_A.mu / self.mu)
        self.child_A.orbit = Orbit(a=a_sma, e=ecc, arg_p=arg_p, inclination=inc, lon_of_asc=lan, MA_at_t0=0.0, parent=self)
        self.child_B.orbit = Orbit(a=b_sma, e=ecc, arg_p=arg_p, inclination=inc, lon_of_asc=lan, MA_at_t0=np.pi, parent=self)
        self.moons.extend([self.child_A, self.child_B])

class LambertSolver:
    def __init__(self, root_mu: float):
        self.mu = root_mu
        self.solved = False

    @staticmethod
    def _stumpff(z: float) -> tuple[float, float]:
        if z > 1e-8:
            root_z = np.sqrt(z)
            return ((1.0 - np.cos(root_z)) / z,
                    (root_z - np.sin(root_z)) / root_z**3)
        if z < -1e-8:
            root_neg_z = np.sqrt(-z)
            return ((np.cosh(root_neg_z) - 1.0) / -z,
                    (np.sinh(root_neg_z) - root_neg_z) / root_neg_z**3)
        return 0.5, 1.0 / 6.0

    def _tof_for_z(self, z: float, r1_norm: float, r2_norm: float, A: float) -> float:
        C, S = self._stumpff(z)
        if C <= 0.0:
            return np.inf
        y = r1_norm + r2_norm + A * (z * S - 1.0) / np.sqrt(C)
        if y < 0.0:
            return np.inf
        x = np.sqrt(y / C)
        return (x**3 * S + A * np.sqrt(y)) / np.sqrt(self.mu)

    def _solve_universal_z(self, r1_norm: float, r2_norm: float, A: float, tof: float) -> float:
        z = 0.0
        for _ in range(100):
            trial_tof = self._tof_for_z(z, r1_norm, r2_norm, A)
            if not np.isfinite(trial_tof):
                z += 0.1
                continue
            if abs(trial_tof - tof) < 1e-8:
                return z

            step_size = max(1e-5, abs(z) * 1e-5)
            high_tof = self._tof_for_z(z + step_size, r1_norm, r2_norm, A)
            low_tof = self._tof_for_z(z - step_size, r1_norm, r2_norm, A)
            derivative = (high_tof - low_tof) / (2.0 * step_size)
            if not np.isfinite(derivative) or abs(derivative) < 1e-12:
                z += 0.1 if trial_tof < tof else -0.1
                continue

            step = np.clip((trial_tof - tof) / derivative, -1.0, 1.0)
            z -= step

        raise RuntimeError("Lambert solve did not converge.")

    def solve(self, r1: np.ndarray, r2: np.ndarray, tof: float, long_way: bool = False) -> tuple[np.ndarray, np.ndarray]:
        r1_norm, r2_norm = np.linalg.norm(r1), np.linalg.norm(r2)
        if r1_norm == 0.0 or r2_norm == 0.0:
            raise ValueError("Lambert solve requires non-zero position vectors.")
        if tof <= 0.0:
            raise ValueError("Lambert solve requires a positive time of flight.")

        cos_nu = np.clip(np.dot(r1, r2) / (r1_norm * r2_norm), -1.0, 1.0)
        sin_nu_mag = np.linalg.norm(np.cross(r1, r2)) / (r1_norm * r2_norm)
        sin_nu = -sin_nu_mag if long_way else sin_nu_mag
        if abs(sin_nu) < 1e-12:
            raise ValueError("Lambert solve is undefined for collinear transfer positions.")

        A = sin_nu * np.sqrt((r1_norm * r2_norm) / (1.0 - cos_nu))
        if abs(A) < 1e-12:
            raise ValueError("Lambert solve has degenerate transfer geometry.")

        z = self._solve_universal_z(float(r1_norm), float(r2_norm), A, tof)

        C, S = self._stumpff(z)
        y = r1_norm + r2_norm + A * (z * S - 1.0) / np.sqrt(C)
        if y < 0.0:
            raise RuntimeError("Lambert solve failed to find a physical transfer.")

        f = 1.0 - y / r1_norm
        g = A * np.sqrt(y / self.mu)
        gdot = 1.0 - y / r2_norm
        if abs(g) < 1e-12:
            raise RuntimeError("Lambert solve produced a singular transfer.")

        self.solved = True
        return ((r2 - f * r1) / g, (gdot * r2 - r1) / g)

class Spacecraft(Body):
    def __init__(self, name: str, r0: np.ndarray, v0: np.ndarray, t0: float, parent: Body):
        self.name = name
        self.parent = parent
        self.r0 = r0.copy()
        self.v0 = v0.copy()
        self.t0 = t0
        self.moons = []
        self._recalculate_orbit(self.r0, self.v0, self.t0)

    def set_absolute_pos_at_ut(self, r: np.ndarray, ut: float) -> None:
        self.r0 = r.copy()
        self.t0 = ut
        self._recalculate_orbit(self.r0, self.v0, self.t0)

    def set_absolute_vel_at_ut(self, v: np.ndarray, ut: float) -> None:
        self.v0 = v.copy()
        self.t0 = ut
        self._recalculate_orbit(self.r0, self.v0, self.t0)

    def _recalculate_orbit(self, r: np.ndarray, v: np.ndarray, ut: float) -> None:
        mu = self.parent.mu
        r_mag, v_mag = np.linalg.norm(r), np.linalg.norm(v)
        h_vec = np.cross(r, v)
        h_mag = np.linalg.norm(h_vec)
        energy = (v_mag**2 / 2.0) - (mu / r_mag)
        a = -mu / (2.0 * energy)
        e_vec = (np.cross(v, h_vec) / mu) - (r / r_mag)
        e = np.linalg.norm(e_vec)
        inc = np.arccos(np.clip(h_vec[2] / h_mag, -1.0, 1.0))
        n_vec = np.cross(np.array([0.0, 0.0, 1.0]), h_vec)
        n_mag = np.linalg.norm(n_vec)

        lan = np.arccos(np.clip(n_vec[0] / n_mag, -1.0, 1.0)) if n_mag != 0.0 else 0.0
        if n_mag != 0.0 and n_vec[1] < 0: 
            lan = 2.0 * np.pi - lan

        arg_p = np.arccos(np.clip(np.dot(n_vec, e_vec) / (n_mag * e), -1.0, 1.0)) if e > 1e-11 and n_mag != 0.0 else 0.0
        if e > 1e-11 and n_mag != 0.0 and e_vec[2] < 0: 
            arg_p = 2.0 * np.pi - arg_p

        nu0 = np.arccos(np.clip(np.dot(e_vec, r) / (e * r_mag), -1.0, 1.0)) if e > 1e-11 else 0.0
        if e > 1e-11 and np.dot(r, v) < 0: 
            nu0 = 2.0 * np.pi - nu0

        if e < 1.0:
            cos_nu0, sin_nu0 = np.cos(nu0), np.sin(nu0)
            E0 = np.arctan2(np.sqrt(1.0 - e**2) * sin_nu0, e + cos_nu0)
            ma0 = E0 - e * np.sin(E0)
        elif e > 1.0:
            cos_nu0, sin_nu0 = np.cos(nu0), np.sin(nu0)
            H0 = np.arctan2(np.sqrt(e**2 - 1.0) * sin_nu0, e + cos_nu0)
            ma0 = e * np.sinh(H0) - H0
        else:
            ma0 = 0.0

        n = np.sqrt(np.abs(mu / a**3))
        
        self.orbit = Orbit(
            a=float(a), 
            e=float(e), 
            arg_p=float(arg_p), 
            lon_of_asc=float(lan), 
            MA_at_t0=float(ma0 - n * ut), 
            inclination=float(inc), 
            parent=self.parent
        )

@dataclass
class ManeuverNode:
    ut: float                  
    delta_v_vector: np.ndarray 
    prograde: np.ndarray       
    normal: np.ndarray         
    radial: np.ndarray         
    total_mag: float           

class ManeuverPlanner:
    def calculate_maneuver(self, ut: float, v_curr: np.ndarray, v_req: np.ndarray, r_curr: np.ndarray) -> ManeuverNode:
        dv_vect = v_req - v_curr
        total_mag = float(np.linalg.norm(dv_vect))
        speed = np.linalg.norm(v_curr)
        
        if speed == 0.0:
            raise ValueError("Cannot calculate maneuver components for a zero velocity vector.")

        prograde_unit = v_curr / speed
        angular_momentum = np.cross(r_curr, v_curr)
        h_mag = np.linalg.norm(angular_momentum)
        
        if h_mag == 0.0:
            raise ValueError("Cannot calculate maneuver components for collinear position and velocity.")

        normal_unit = angular_momentum / h_mag
        radial_unit = np.cross(prograde_unit, normal_unit)
        
        return ManeuverNode(
            ut,
            dv_vect,
            (dv_vect @ prograde_unit) * prograde_unit,
            (dv_vect @ normal_unit) * normal_unit,
            (dv_vect @ radial_unit) * radial_unit,
            total_mag
        )