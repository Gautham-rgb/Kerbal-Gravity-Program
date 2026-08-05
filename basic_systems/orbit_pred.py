import numpy as np
import datetime as dtime
from dataclasses import dataclass
from pyvista import Sphere, Cube
from typing import Literal
import itertools

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

def ut_secs_to_date_components(ut: float, ker_time: bool = False):
    """Convert UT seconds back to date components (Y, M, D, H, M, S)."""
    if ker_time:
        y = int(ut // Constants.KER_YEAR_SEC) + 1
        rem = ut % Constants.KER_YEAR_SEC
        d = int(rem // Constants.KER_DAY_SEC) + 1
        rem %= Constants.KER_DAY_SEC
        h = int(rem // (Constants.KER_HOUR_MIN * Constants.KER_MIN_SEC))
        rem %= (Constants.KER_HOUR_MIN * Constants.KER_MIN_SEC)
        m = int(rem // Constants.KER_MIN_SEC)
        s = int(rem % Constants.KER_MIN_SEC)
        return y, 0, d, h, m, s # Month is 0 for Kerbal time
    else:
        j2000_epoch = dtime.datetime(2000, 1, 1, 12, 0, 0, tzinfo=dtime.timezone.utc)
        dt = j2000_epoch + dtime.timedelta(seconds=ut)
        return dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second

def format_ut(ut: float, ker_time: bool = False) -> str:
    """Format UT seconds as a human-readable string."""
    y, mon, d, h, m, s = ut_secs_to_date_components(ut, ker_time)
    if ker_time:
        return f"Year {y}, Day {d}, {h:02d}:{m:02d}:{s:02d}"
    else:
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        return f"{y}-{months[mon-1]}-{d:02d} {h:02d}:{m:02d}:{s:02d} UTC"

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
    def __init__(self, a: float = 0.0, e: float = 0.0, arg_p: float = 0.0, lon_of_asc: float = 0.0, 
                 MA_at_t0: float = 0.0, inclination: float = 0.0, parent: Body|None = None):
        self.parent = parent
        self.semi_major_axis = a
        self.eccen = e
        self.arg_periapsis = arg_p
        self.lon_of_asc = lon_of_asc
        self.mean_anomaly_at_t0 = MA_at_t0
        self.inclination = inclination
        if parent and parent.mu > 0 and self.semi_major_axis > 0 and self.eccen < 1:
            self.period = 2*np.pi*np.sqrt(self.semi_major_axis**3/parent.mu)
        else:
            self.period = np.inf

class Body:

    _id_counter = itertools.count()

    def __init__(self, name: str, mu: float, identifier: str, radius: float, atm_height: float = 0.0, 
                 orbit: Orbit | None = None, moons: list[Body] | None = None, render_color: str | None = None) -> None:
        self.name = name
        self.identifier = identifier
        self.mu = mu
        self.radius = radius
        self.atm_height = atm_height
        self.orbit = orbit if orbit else Orbit()
        self.moons = moons if moons is not None else []
        self.render_color = render_color
        self.period = 0.0
        self._uid = next(Body._id_counter)
        

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


    def __hash__(self):
        return hash(self._uid)

    def __eq__(self, other):
        return isinstance(other, Body) and self._uid == other._uid
        
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
        super().__init__(sys_name, mu=(a.mu + b.mu), radius=0.0, identifier = f"{a.identifier}_{b.identifier}_barycenter", orbit=None, moons=[], render_color="#ffffff")
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
    def __init__(self, name: str, r0: np.ndarray, v0: np.ndarray, t0: float, parent: Body, 
                 dry_mass: float, wet_mass: float, hull_mesh, identifier: str = ""):
        self.dry_mass = dry_mass
        self.fuel_mass = wet_mass - dry_mass
        self.engines = []
        self.hull_mesh = hull_mesh
        self.active_actors = []
        
        super().__init__(
            name=name, 
            mu=6.6743e-11 * wet_mass, 
            identifier=identifier or f"SC_{name.replace(' ', '_').upper()}", 
            radius=0.01, 
            atm_height=0.0, 
            render_color="#ffffff"
        )
        
        self.name = name
        self.parent = parent
        self.r0, self.v0, self.t0 = r0.copy(), v0.copy(), t0
        self.moons = []
        self._recalculate_orbit(self.r0, self.v0, self.t0)

    @property
    def mass(self) -> float:
        return self.dry_mass + self.fuel_mass

    def add_engine(self, thrust: float, isp: float, offset: np.ndarray, mesh=None) -> None:
        self.engines.append({"thrust": thrust, "isp": isp, "offset": offset.copy(), "mesh": mesh, "active": True})

    def set_engine_state(self, index: int, active: bool) -> None:
        if 0 <= index < len(self.engines):
            self.engines[index]["active"] = active

    def get_combined_engine_specs(self) -> tuple[float, float]:
        active = [e for e in self.engines if e["active"]]
        if not active: return 0.0, 0.0
        total_thrust = sum(e["thrust"] for e in active)
        total_flow = sum(e["thrust"] / (e["isp"] * 9.80665) for e in active)
        return total_thrust, total_thrust / (total_flow * 9.80665)

    def burn_fuel(self, amount: float) -> None:
        self.fuel_mass = max(0.0, self.fuel_mass - amount)
        self.mu = 6.6743e-11 * self.mass

    def execute_maneuver(self, delta_v_vec: np.ndarray, ut: float) -> bool:
        delta_v = np.linalg.norm(delta_v_vec)
        if delta_v == 0.0: return True
        _, combined_isp = self.get_combined_engine_specs()
        if combined_isp == 0.0: return False
        
        fuel_needed = self.mass * (1.0 - np.exp(-delta_v / (combined_isp * 9.80665)))
        if fuel_needed > self.fuel_mass: return False
        
        self.burn_fuel(fuel_needed)
        self.r0 = self.get_absolute_pos_at_ut(ut) - self.parent.get_absolute_pos_at_ut(ut)
        self.v0 = (self.get_absolute_vel_at_ut(ut) + delta_v_vec) - self.parent.get_absolute_vel_at_ut(ut)
        self.t0 = ut
        self._recalculate_orbit(self.r0, self.v0, self.t0)
        return True

    def clear_visualization(self, plotter) -> None:
        for actor in self.active_actors:
            plotter.remove_actor(actor)
        self.active_actors.clear()

    def render(self, plotter, ut: float) -> None:
        self.clear_visualization(plotter)
        pos = self.get_absolute_pos_at_ut(ut)
        
        hull_copy = self.hull_mesh.copy()
        hull_copy.translate(pos, inplace=True)
        self.active_actors.append(plotter.add_mesh(hull_copy, color="white"))
        
        for e in self.engines:
            if e["mesh"] is not None:
                engine_copy = e["mesh"].copy()
                engine_copy.translate(pos + e["offset"], inplace=True)
                color = "#ff4500" if e["active"] else "#555555"
                self.active_actors.append(plotter.add_mesh(engine_copy, color=color))
                
        label_actor = plotter.add_point_labels(
            [pos], 
            [f"{self.name}"],
            always_visible=True,
            point_size=0,
            font_size=12,
            text_color="cyan"
        )
        self.active_actors.append(label_actor)

    def set_absolute_pos_at_ut(self, r: np.ndarray, ut: float) -> None:
        self.r0, self.t0 = r.copy(), ut
        self._recalculate_orbit(self.r0, self.v0, self.t0)

    def set_absolute_vel_at_ut(self, v: np.ndarray, ut: float) -> None:
        self.v0, self.t0 = v.copy(), ut
        self._recalculate_orbit(self.r0, self.v0, self.t0)

    def _recalculate_orbit(self, r: np.ndarray, v: np.ndarray, ut: float) -> None:
        mu = self.parent.mu

        r_mag = np.linalg.norm(r)
        v_mag = np.linalg.norm(v)

        if r_mag < 1e-12:
            raise ValueError("Position vector magnitude is zero.")

        # Specific angular momentum
        h_vec = np.cross(r, v)
        h_mag = np.linalg.norm(h_vec)

        # Specific orbital energy
        energy = v_mag**2 / 2.0 - mu / r_mag

        # Semi-major axis
        if abs(energy) < 1e-12:
            a = np.inf  # Parabolic
        else:
            a = -mu / (2.0 * energy)

        # Eccentricity vector
        e_vec = (np.cross(v, h_vec) / mu) - (r / r_mag)
        e = np.linalg.norm(e_vec)

        # Inclination
        if h_mag < 1e-12:
            inc = 0.0
        else:
            inc = np.arccos(np.clip(h_vec[2] / h_mag, -1.0, 1.0))

        # Node vector
        k = np.array([0.0, 0.0, 1.0])
        n_vec = np.cross(k, h_vec)
        n_mag = np.linalg.norm(n_vec)

        # Longitude of ascending node
        if n_mag > 1e-12:
            lan = np.arccos(np.clip(n_vec[0] / n_mag, -1.0, 1.0))
            if n_vec[1] < 0:
                lan = 2.0 * np.pi - lan
        else:
            lan = 0.0

        # Argument of periapsis
        if e > 1e-10 and n_mag > 1e-12:
            arg_p = np.arccos(
                np.clip(np.dot(n_vec, e_vec) / (n_mag * e), -1.0, 1.0)
            )
            if e_vec[2] < 0:
                arg_p = 2.0 * np.pi - arg_p
        else:
            arg_p = 0.0

        # True anomaly
        if e > 1e-10:
            nu = np.arccos(
                np.clip(np.dot(e_vec, r) / (e * r_mag), -1.0, 1.0)
            )
            if np.dot(r, v) < 0:
                nu = 2.0 * np.pi - nu
        else:
            # Circular orbit
            if n_mag > 1e-12:
                nu = np.arccos(
                    np.clip(np.dot(n_vec, r) / (n_mag * r_mag), -1.0, 1.0)
                )
                if r[2] < 0:
                    nu = 2.0 * np.pi - nu
            else:
                nu = np.arctan2(r[1], r[0])

        # Mean anomaly at epoch
        if e < 1.0 - 1e-10:
            # Elliptic
            E = 2.0 * np.arctan2(
                np.sqrt(1.0 - e) * np.sin(nu / 2.0),
                np.sqrt(1.0 + e) * np.cos(nu / 2.0),
            )
            ma0 = E - e * np.sin(E)

        elif e > 1.0 + 1e-10:
            # Hyperbolic
            H = 2.0 * np.arctanh(
                np.sqrt((e - 1.0) / (e + 1.0)) * np.tan(nu / 2.0)
            )
            ma0 = e * np.sinh(H) - H

        else:
            # Near-parabolic
            ma0 = 0.0

        # Store orbit
        if np.isfinite(a):
            n = np.sqrt(mu / abs(a) ** 3)
            ma_at_t0 = ma0 - n * ut
        else:
            ma_at_t0 = ma0

        self.orbit = Orbit(
            a=float(a),
            e=float(e),
            arg_p=float(arg_p),
            lon_of_asc=float(lan),
            MA_at_t0=float(ma_at_t0),
            inclination=float(inc),
            parent=self.parent,
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