import numpy as np
import datetime as dtime
class Constants:
    KER_YEAR_DAY = 426
    KER_DAY_HOUR = 6
    KER_HOUR_MIN = 60
    KER_MIN_SEC = 60
    KER_YEAR_SEC = 9201600.0
    KER_DAY_SEC = 21600.0

def get_ut_secs(year: int, day: int, hour: int, minute: int, seconds: int, ker_time = False):
    """Calculate elapsed seconds since the epoch tracker.

    Uses the KSP savefile start for Kerbal time, or the J2000 epoch 
    for real-world time.

    Args:
        year: Year of the target timestamp.
        day: Day of the year (1-indexed).
        hour: Hour of the day (0-23).
        minute: Minute of the hour (0-59).
        seconds: Seconds of the minute (0-59).
        ker_time: True to use Kerbin calendar rules, False for Earth.

    Returns:
        float: Total elapsed seconds.
    """
    
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
        j2000_epoch = dtime.datetime(2000, 1, 1, 12, 0, 0, tzinfo = dtime.timezone.utc)
        base_date = dtime.datetime(year, 1, 1, tzinfo = dtime.timezone.utc)

        target_date = base_date + dtime.timedelta(day - 1, hours = hour, minutes = minute, seconds = seconds)

        return (target_date - j2000_epoch).total_seconds()

def solve_eccen_anomaly(mean_anomaly: float, eccen: float):
    m = np.fmod(mean_anomaly, 2 * np.pi)
    if m > np.pi: m -= 2 * np.pi
    if m < -np.pi: m += 2 * np.pi

    E = m
    epsilon = 1e-12
    max_iter = 100
    i = 0

    while i < max_iter:
        delta_E = (E - eccen * np.sin(E) - m) / (1 - eccen * np.cos(E))
        E -= delta_E

        if abs(delta_E) < epsilon:
            break

        i+=1
    
    return E

class Orbit:
    def __init__(self, a = 0.0, e = 0.0, arg_p = 0.0, lon_of_asc = 0.0, MA_at_t0 = 0.0, inclination: float = 0.0, parent: Body | None = None):
        self.parent = parent
        self.semi_major_axis = a
        self.eccen = e
        self.arg_periapsis = arg_p
        self.lon_of_asc = lon_of_asc
        self.mean_anomaly_at_t0 = MA_at_t0
        self.inclination = inclination

class Body:
    def __init__(self, name: str, mu: float, radius: float, atm_height: float = 0.0, 
                 orbit: Orbit | None = None, moons: list[Body] | None = None) -> None:
        self.name = name
        self.mu = mu
        self.radius = radius
        self.atm_height = atm_height
        self.orbit = orbit if orbit else Orbit()
        self.moons = moons
    
    def get_pos_at_ut(self, ut: float):
        if self.orbit.parent == None:
            return np.zeros(3)
        
        n = np.sqrt(self.orbit.parent.mu / self.orbit.semi_major_axis ** 3)
        mean_anomaly = self.orbit.mean_anomaly_at_t0 + n * ut

        E = solve_eccen_anomaly(mean_anomaly, self.orbit.eccen)
        cosE, sinE = np.cos(E), np.sin(E)

        x_perifocal, y_perifocal = self.orbit.semi_major_axis * (cosE - self.orbit.eccen), \
        self.orbit.semi_major_axis * np.sqrt(1.0 - self.orbit.eccen * self.orbit.eccen) * sinE

        pos_perifocal = np.array([x_perifocal, y_perifocal, 0.], np.float64)

        cos_lan = np.cos(self.orbit.lon_of_asc)
        sin_lan = np.sin(self.orbit.lon_of_asc)
        cos_inc = np.cos(self.orbit.inclination)
        sin_inc = np.sin(self.orbit.inclination)
        cos_ap = np.cos(self.orbit.arg_periapsis)
        sin_ap = np.sin(self.orbit.arg_periapsis)

        R_lan = np.array([
            [cos_lan, -sin_lan, 0.],
            [sin_lan, cos_lan, 0.],
            [0., 0., 1.]
        ])

        R_inc = np.array([
            [1., 0., 0.],
            [0., cos_inc, -sin_inc],
            [0., sin_inc, cos_inc]
        ])

        R_ap = np.array([
            [cos_ap, -sin_ap, 0.],
            [sin_ap, cos_ap, 0.],
            [0.,0.,1.]
        ])

        return pos_perifocal @ (R_lan * R_ap * R_inc)
    
    def get_absolute_pos_at_ut(self, ut: float):
        if self.orbit.parent == None:
            return np.zeros(3)
        
        return self.get_pos_at_ut(ut) + self.orbit.parent.get_absolute_pos_at_ut(ut)

def main():
    sun = Body("Sun", 1.1723328e18, 261600000)
    kerbin_orbit = Orbit(
        13599840256.0, 0.01, 0.0, 0.174533, 0.785398, 0.0, sun
    )
    kerbin = Body("Kerbin", 3.5316000e12, 600000, orbit = kerbin_orbit)
    pos = kerbin.get_pos_at_ut(10000.0)
    print(f"{pos[0]} {pos[1]} {pos[2]}")

if __name__ == "__main__":
    main()