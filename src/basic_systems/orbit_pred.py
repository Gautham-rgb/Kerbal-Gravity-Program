"""Orbital mechanics simulation core for the mission planner.

This module is the importable simulation layer: it defines the body/orbit
hierarchy (:class:`Body`, :class:`Orbit`), spacecraft state and manoeuvres
(:class:`Spacecraft`, :class:`ManeuverNode`, :class:`ManeuverPlanner`), the
Kerbal/real calendars (:class:`TimeUnits`, :data:`KERBAL_UNITS`,
:data:`REAL_UNITS`), time parsing/formatting helpers and the universal-variable
:class:`LambertSolver`. Distances are in metres, times in seconds (UT, seconds
since the calendar epoch) and velocities in m/s unless noted otherwise.
"""

from __future__ import annotations

import numpy as np
import datetime as dtime
import copy
import re
from dataclasses import dataclass, field
try:
    from pyvista import Sphere, Cube
except ModuleNotFoundError:  # Keep the simulation and CLI usable without GUI extras.
    Sphere = Cube = None
from typing import Any, Literal
import itertools

@dataclass
class Constants:
    """Stock Kerbal calendar divisors used as defaults by :class:`TimeUnits`."""

    KER_YEAR_DAY: int = 426
    KER_DAY_HOUR: int = 6
    KER_HOUR_MIN: int = 60
    KER_MIN_SEC: int = 60
    KER_YEAR_SEC: float = 9201600.0
    KER_DAY_SEC: float = 21600.0

@dataclass(frozen=True)
class TimeUnits:
    """A custom calendar for converting between (year, day, hour, minute,
    second) and raw UT seconds.

    Pass an instance as ``units`` to :func:`get_ut_secs`,
    :func:`ut_secs_to_date_components` or :func:`format_ut` to override the
    stock Kerbal calendar (``KERBAL_UNITS``). A 365-day, 24-hour calendar is
    just ``TimeUnits(year_days=365, day_hours=24)``.
    """
    year_days: int = Constants.KER_YEAR_DAY
    day_hours: int = Constants.KER_DAY_HOUR
    hour_minutes: int = Constants.KER_HOUR_MIN
    minute_seconds: int = Constants.KER_MIN_SEC

    @property
    def day_seconds(self) -> float:
        return float(self.day_hours * self.hour_minutes * self.minute_seconds)

    @property
    def year_seconds(self) -> float:
        return float(self.year_days * self.day_seconds)


KERBAL_UNITS = TimeUnits()  # Stock Kerbal calendar (426-day year, 6-hour day).
REAL_UNITS = TimeUnits(year_days=365, day_hours=24, hour_minutes=60, minute_seconds=60)  # Gregorian-style 365-day calendar.

def get_ut_secs(year: int, month: int, day: int, hour: int, minute: int, seconds: int = 0, ker_time = False, am_pm: None | Literal["AM", "PM", "am", "pm"] = None, units: TimeUnits | None = None):
    """Convert a calendar timestamp into raw UT seconds.

    With ``ker_time`` (or a non-default ``units`` calendar) the arguments are
    interpreted against that calendar's year/day/hour/minute divisions. Otherwise
    they are treated as a real (Gregorian) date from J2000 and returned as
    seconds since the J2000 epoch. ``am_pm`` optionally shifts a 12-hour clock.
    """

    if am_pm is not None:
        am_pm_upper = am_pm.upper()
        if am_pm_upper == "PM" and hour < 12:
            hour += 12
        elif am_pm_upper == "AM" and hour == 12:
            hour = 0

    if ker_time or units is not None:
        u = units or KERBAL_UNITS
        elapsed_years = year - 1
        elapsed_days = day - 1

        time_conversions = (
            (elapsed_years, u.year_seconds),
            (elapsed_days, u.day_seconds),
            (hour, u.hour_minutes * u.minute_seconds),
            (minute, u.minute_seconds),
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

def ut_secs_to_date_components(ut: float, ker_time: bool = False, units: TimeUnits | None = None):
    """Convert UT seconds back to date components (Y, M, D, H, M, S)."""
    if ker_time or units is not None:
        u = units or KERBAL_UNITS
        y = int(ut // u.year_seconds) + 1
        rem = ut % u.year_seconds
        d = int(rem // u.day_seconds) + 1
        rem %= u.day_seconds
        h = int(rem // (u.hour_minutes * u.minute_seconds))
        rem %= (u.hour_minutes * u.minute_seconds)
        m = int(rem // u.minute_seconds)
        s = int(rem % u.minute_seconds)
        return y, 0, d, h, m, s # Month is 0 for Kerbal time
    else:
        j2000_epoch = dtime.datetime(2000, 1, 1, 12, 0, 0, tzinfo=dtime.timezone.utc)
        dt = j2000_epoch + dtime.timedelta(seconds=ut)
        return dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second

def format_ut(ut: float, ker_time: bool = False, units: TimeUnits | None = None) -> str:
    """Format UT seconds as a human-readable string."""
    y, mon, d, h, m, s = ut_secs_to_date_components(ut, ker_time, units)
    if ker_time or units is not None:
        return f"Year {y}, Day {d}, {h:02d}:{m:02d}:{s:02d}"
    else:
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        return f"{y}-{months[mon-1]}-{d:02d} {h:02d}:{m:02d}:{s:02d} UTC"


_TIME_TOKEN_RE = re.compile(r"^([0-9]+(?:\.[0-9]+)?)\s*([a-zA-Z]+)$")
_CLOCK_RE = re.compile(r"^(\d{1,2}):(\d{1,2})(?::(\d{1,2}))?$")
_YEAR_UNITS = {"y", "yr", "year", "years"}
_MONTH_UNITS = {"mo", "mon", "month", "months"}
_DAY_UNITS = {"d", "day", "days"}
_HOUR_UNITS = {"h", "hr", "hour", "hours"}
_MINUTE_UNITS = {"m", "min", "mins", "minute", "minutes"}
_SECOND_UNITS = {"s", "sec", "secs", "second", "seconds"}


def _time_multiplier(unit: str, units: TimeUnits) -> float | None:
    key = unit.lower()
    if key in _YEAR_UNITS:
        return units.year_seconds
    if key in _MONTH_UNITS:
        return units.year_seconds / 12.0
    if key in _DAY_UNITS:
        return units.day_seconds
    if key in _HOUR_UNITS:
        return float(units.hour_minutes * units.minute_seconds)
    if key in _MINUTE_UNITS:
        return float(units.minute_seconds)
    if key in _SECOND_UNITS:
        return 1.0
    return None


# --- Friendly (but still useful) error quirks --------------------------------
#
# Time parsing is the single most common place a new user faceplants, so the
# errors there get to be a little silly. The quips are chosen deterministically
# from the offending unit so the same typo always gets the same gentle ribbing.

_KNOWN_TIME_UNITS = ("s", "m", "h", "d", "mo", "y")


def _quirk_index(word: str) -> int:
    return sum(ord(ch) for ch in word) % 3


def _unknown_time_unit_quip(unit: str) -> str:
    quips = [
        f"Unknown time unit '{unit}'. We measure in s, m, h, d, mo, y — '{unit}' is still "
        f"awaiting approval from the Kerbal Calendar Committee.",
        f"Time unit '{unit}'? Ambitious. The accepted ones are s, m, h, d, mo, y, and "
        f"'{unit}' didn't make the cut.",
        f"'{unit}' isn't a unit of time we recognise. Try s, m, h, d, mo, y "
        f"(the universe hasn't reached '{unit}' yet).",
    ]
    return quips[_quirk_index(unit)]


def _missing_unit_quip(number: str) -> str:
    quips = [
        f"'{number}' on its own is ambiguous. Tag on a unit (s/m/h/d/mo/y), e.g. '{number}d', "
        f"or just type '{number}' as the whole input to mean raw UT seconds.",
        f"Almost! '{number}' needs a unit. Try '{number}s' for seconds or '{number}d' for days "
        f"(s/m/h/d/mo/y all work).",
        f"Bare number '{number}' spotted with no unit. We're not mind-readers — add s/m/h/d/mo/y, "
        f"or pass '{number}' alone to mean raw UT seconds.",
    ]
    return quips[_quirk_index(number)]


def parse_time_string(text: str, units: TimeUnits | None = None, base: float = 0.0) -> float:
    """Parse a human time string into UT seconds.

    Accepted forms (case-insensitive, commas/spaces optional):

    * A bare number: raw UT seconds (e.g. ``"21600"``).
    * A duration of tokens: ``"1y 2mo 3d 4h 5m 6s"``, ``"2d 5h"``, ``"90s"``,
      ``"2 days 3 hours"``. Returned as ``base`` plus the elapsed seconds
      (``base`` defaults to 0). Years and days use the given calendar (Kerbal
      by default); a month is one twelfth of a year.
    * A calendar timestamp: ``"Year 2, Day 100, 12:30:00"``, ``"2y 100d 12:30"``.
      Interpreted as an absolute UT from the start of the calendar (``base``
      ignored).
    """
    u = units or KERBAL_UNITS
    if base is None:
        base = 0.0
    text = str(text).strip()
    if not text:
        raise ValueError("Empty time string.")

    try:
        return float(text)
    except ValueError:
        pass

    calendar_match = _CALENDAR_RE.match(text)
    if calendar_match:
        year, day = int(calendar_match.group(1)), int(calendar_match.group(2))
        hours = int(calendar_match.group(3) or 0.0)
        minutes = int(calendar_match.group(4) or 0.0)
        seconds = int(calendar_match.group(5) or 0.0)
        return float(get_ut_secs(year, 1, day, hours, minutes, seconds, ker_time=True, units=u))

    tokens = [token for token in re.split(r"[,\s]+", _normalize_time_labels(text)) if token]
    year, day = 1, 1
    hours = minutes = seconds = 0.0
    duration = 0.0
    calendar = False

    for idx, token in enumerate(tokens):
        clock = _CLOCK_RE.match(token)
        if clock:
            calendar = True
            hours += float(clock.group(1))
            minutes += float(clock.group(2))
            if clock.group(3):
                seconds += float(clock.group(3))
            continue

        match = _TIME_TOKEN_RE.match(token)
        if match is None:
            # A bare number with no unit (e.g. "5" in "5 bananas"/"5 eras") is a common slip.
            if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", token):
                nxt = tokens[idx + 1] if idx + 1 < len(tokens) else None
                if nxt and re.fullmatch(r"[A-Za-z]+", nxt):
                    raise ValueError(
                        _unknown_time_unit_quip(nxt)
                        + f"  (Looks like you typed '{token} {nxt}' — if you meant a duration, "
                        f"smoosh them together with a real unit, e.g. '{token}d' or '{token}mo'.)"
                    )
                raise ValueError(_missing_unit_quip(token))
            raise ValueError(
                _unknown_time_unit_quip(token)
                if re.search(r"[A-Za-z]", token)
                else f"Unrecognized time token '{token}'. Example: 'advance 2d 5h' or 'time 21600'."
            )
        value = float(match.group(1))
        unit = match.group(2).lower()

        if unit in _YEAR_UNITS:
            calendar = True
            year = int(value)
        elif unit in _DAY_UNITS:
            if calendar:
                day = int(value)
            else:
                duration += value * u.day_seconds
        elif unit in _MONTH_UNITS:
            duration += value * (u.year_seconds / 12.0)
        else:
            multiplier = _time_multiplier(unit, u)
            if multiplier is None:
                raise ValueError(_unknown_time_unit_quip(unit))
            if calendar:
                if unit in _HOUR_UNITS:
                    hours += value
                elif unit in _MINUTE_UNITS:
                    minutes += value
                else:
                    seconds += value
            else:
                duration += value * multiplier

    if calendar:
        absolute = get_ut_secs(year, 1, day, int(hours), minutes, seconds, ker_time=True, units=u)
        return float(absolute + duration)
    return float(base + duration)


_CALENDAR_RE = re.compile(
    r"^(?:(?:year|y)[,\s]*)?([0-9]+)[,\s]+(?:day|d)[,\s]*([0-9]+)"
    r"(?:[,\s]+([0-9]{1,2}):([0-9]{1,2})(?::([0-9]{1,2}))?)?$",
    re.IGNORECASE,
)

_LABEL_TO_SHORT = [
    ("years", "y"), ("year", "y"), ("months", "mo"), ("month", "mo"),
    ("days", "d"), ("day", "d"), ("hours", "h"), ("hour", "h"),
    ("hrs", "h"), ("hr", "h"), ("minutes", "m"), ("minute", "m"),
    ("mins", "m"), ("min", "m"), ("seconds", "s"), ("second", "s"),
    ("secs", "s"), ("sec", "s"),
]


def _normalize_time_labels(text: str) -> str:
    """Turn ``"2 days 3 hours"`` into token form ``"2d 3h"``."""
    for label, short in _LABEL_TO_SHORT:
        text = re.sub(rf"([0-9]+(?:\.[0-9]+)?)\s+\b{label}\b", rf"\1{short}", text, flags=re.IGNORECASE)
    return text

def solve_elliptic_anomaly(mean_anomaly: float, eccen: float) -> float:
    epsilon = 1e-12
    max_iter = 100

    if not np.isfinite(mean_anomaly):
        raise ValueError(f"Invalid mean anomaly: {mean_anomaly}")

    m = np.fmod(mean_anomaly, 2.0 * np.pi)

    if m > np.pi:
        m -= 2.0 * np.pi
    elif m < -np.pi:
        m += 2.0 * np.pi

    # Good starting point for Newton iteration.
    E = m if eccen < 0.8 else np.pi

    for _ in range(max_iter):
        f = E - eccen * np.sin(E) - m
        fp = 1.0 - eccen * np.cos(E)

        if abs(fp) < epsilon:
            raise RuntimeError("Elliptic Kepler solver became singular.")

        delta = f / fp
        E -= delta

        if abs(delta) < epsilon:
            return E

    raise RuntimeError(
        f"Elliptic Kepler solver did not converge: "
        f"M={mean_anomaly}, e={eccen}"
    )


def solve_hyperbolic_anomaly(mean_anomaly: float, eccen: float) -> float:
    epsilon = 1e-12
    max_iter = 100

    if not np.isfinite(mean_anomaly):
        raise ValueError(f"Invalid mean anomaly: {mean_anomaly}")

    H = np.arcsinh(mean_anomaly / eccen)

    for _ in range(max_iter):
        sinh_h = np.sinh(H)
        cosh_h = np.cosh(H)

        f = eccen * sinh_h - H - mean_anomaly
        fp = eccen * cosh_h - 1.0

        if abs(fp) < epsilon:
            raise RuntimeError("Hyperbolic Kepler solver became singular.")

        delta = f / fp
        H -= delta

        if abs(delta) < epsilon:
            return H

    raise RuntimeError(
        f"Hyperbolic Kepler solver did not converge: "
        f"M={mean_anomaly}, e={eccen}"
    )


def solve_parabolic_anomaly(parabolic_mean_anomaly: float) -> float:
    """
    Barker's equation:

        M_p = D + D^3 / 3

    where D = tan(nu / 2).
    """

    if not np.isfinite(parabolic_mean_anomaly):
        raise ValueError(
            f"Invalid parabolic mean anomaly: "
            f"{parabolic_mean_anomaly}"
        )

    M = parabolic_mean_anomaly

    # Real cube roots are important here.
    A = 1.5 * M
    B = np.sqrt(1.0 + 2.25 * M * M)

    return np.cbrt(A + B) + np.cbrt(A - B)

PARABOLIC_EPS = 1e-8

def solve_anomaly(mean_anomaly: float, eccen: float) -> float:
    epsilon = PARABOLIC_EPS

    if not np.isfinite(eccen):
        raise ValueError(f"Invalid eccentricity: {eccen}")

    if eccen < 1.0 - epsilon:
        return solve_elliptic_anomaly(mean_anomaly, eccen)

    if eccen > 1.0 + epsilon:
        return solve_hyperbolic_anomaly(mean_anomaly, eccen)

    # e ≈ 1
    return solve_parabolic_anomaly(mean_anomaly)

class Orbit:
    """Keplerian orbital elements of a body around ``parent`` (for internal use).

    Distances are in metres and angles in radians. ``semi_major_axis`` is ``a``
    (metres), ``eccen`` the eccentricity, ``arg_p`` the argument of periapsis,
    ``lon_of_asc`` the longitude of the ascending node, ``MA_at_t0`` the mean
    anomaly at epoch and ``inclination`` the orbital inclination. ``period`` is
    derived for closed (elliptical) orbits and is ``inf`` otherwise.
    """

    def __init__(self, a: float = 0.0, e: float = 0.0, arg_p: float = 0.0, lon_of_asc: float = 0.0,
                 MA_at_t0: float = 0.0, inclination: float = 0.0, parent: Body|None = None,
                 periapsis: float | None = None):
        self.parent = parent
        self.semi_major_axis = a
        self.eccen = e
        self.arg_p = arg_p
        self.lon_of_asc = lon_of_asc
        self.MA_at_t0 = MA_at_t0
        self.inclination = inclination
        self.periapsis = periapsis
        if parent and parent.mu > 0 and self.semi_major_axis > 0 and self.eccen < 1:
            self.period = 2*np.pi*np.sqrt(self.semi_major_axis**3/parent.mu)
        else:
            self.period = np.inf

class Body:
    """A celestial body (planet, moon, star) in the system hierarchy.

    ``mu`` is the gravitational parameter (m^3/s^2), ``radius`` the equatorial
    radius (metres) and ``atm_height`` the atmosphere height (metres). Bodies
    orbit a ``parent`` body via an :class:`Orbit`; the root body has no parent.
    ``moons`` lists orbiting children and ``rotation_period_s`` is the sidereal
    rotation period in seconds. Equality/hashing is by an internal unique id,
    not by name, so two distinct Body instances are never equal.
    """

    _id_counter = itertools.count()

    def __init__(self, name: str, mu: float, identifier: str, radius: float, atm_height: float = 0.0, 
                 orbit: Orbit | None = None, moons: list[Body] | None = None, render_color: str | None = None,
                 rotation_period_s: float = 0.0) -> None:
        self.name = name
        self.identifier = identifier
        self.mu = mu
        self.radius = radius
        self.atm_height = atm_height
        self.rotation_period_s = rotation_period_s
        self.orbit = orbit if orbit else Orbit()
        self.moons = moons if moons is not None else []
        self.render_color = render_color
        self.period = 0.0
        self._uid = next(Body._id_counter)
        

    def synchronous_radius(self) -> float | None:
        """Radius of a circular synchronous orbit (in m), or None if unknown."""
        if self.mu > 0 and self.rotation_period_s > 0:
            T = self.rotation_period_s
            return (self.mu * T * T / (4.0 * np.pi ** 2)) ** (1.0 / 3.0)
        return None

    def soi_radius(self) -> float | None:
        """Hill-sphere SOI radius (m) of this body around its parent."""
        parent = self.orbit.parent
        if parent is None:
            return None
        a = self.orbit.semi_major_axis
        if a <= 0.0 or not np.isfinite(a):
            return None
        return a * (self.mu / parent.mu) ** (2.0 / 5.0)

    def to_mesh(self, plotter, scaled_radius):
        if Sphere is None:
            raise RuntimeError("Rendering requires the optional 'pyvista' dependency.")
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

        if e < 1.0 - PARABOLIC_EPS:
            n = np.sqrt(np.abs(mu / a ** 3))
            mean_anomaly = self.orbit.MA_at_t0 + n * ut
            anomaly = solve_anomaly(mean_anomaly, e)
            cosA, sinA = np.cos(anomaly), np.sin(anomaly)
            x_perifocal = a * (cosA - e)
            y_perifocal = a * np.sqrt(1.0 - e**2) * sinA
        elif e > 1.0 + PARABOLIC_EPS:
            n = np.sqrt(np.abs(mu / a ** 3))
            mean_anomaly = self.orbit.MA_at_t0 + n * ut
            anomaly = solve_anomaly(mean_anomaly, e)
            a_abs = np.abs(a)
            coshA, sinhA = np.cosh(anomaly), np.sinh(anomaly)
            x_perifocal = a_abs * (e - coshA)
            y_perifocal = a_abs * np.sqrt(e**2 - 1.0) * sinhA
        else:
            q = self.orbit.periapsis
            if q is None or not np.isfinite(q) or q <= 0.0:
                q = a * (1.0 - e) if e <= 1.0 else abs(a) * (e - 1.0)
            if q > 0.0 and np.isfinite(q):
                n = np.sqrt(mu / (2.0 * q ** 3))
                mean_anomaly = self.orbit.MA_at_t0 + n * ut
                D = solve_parabolic_anomaly(mean_anomaly)
                x_perifocal = q * (1.0 - D * D)
                y_perifocal = 2.0 * q * D
            else:
                x_perifocal = 0.0
                y_perifocal = 0.0

        pos_perifocal = np.array([x_perifocal, y_perifocal, 0.], np.float64)

        cos_lan, sin_lan = np.cos(self.orbit.lon_of_asc), np.sin(self.orbit.lon_of_asc)
        cos_inc, sin_inc = np.cos(self.orbit.inclination), np.sin(self.orbit.inclination)
        cos_ap, sin_ap = np.cos(self.orbit.arg_p), np.sin(self.orbit.arg_p)

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

        if e < 1.0 - PARABOLIC_EPS:
            n = np.sqrt(np.abs(mu / a ** 3))
            mean_anomaly = self.orbit.MA_at_t0 + n * ut
            anomaly = solve_anomaly(mean_anomaly, e)
            cosA, sinA = np.cos(anomaly), np.sin(anomaly)
            v_coef = np.sqrt(mu * a) / (a * (1.0 - e * cosA))
            vx_perifocal = -v_coef * sinA
            vy_perifocal = v_coef * np.sqrt(1.0 - e**2) * cosA
        elif e > 1.0 + PARABOLIC_EPS:
            n = np.sqrt(np.abs(mu / a ** 3))
            mean_anomaly = self.orbit.MA_at_t0 + n * ut
            anomaly = solve_anomaly(mean_anomaly, e)
            a_abs = np.abs(a)
            coshA, sinhA = np.cosh(anomaly), np.sinh(anomaly)
            v_coef = np.sqrt(mu * a_abs) / (a_abs * (e * coshA - 1.0))
            vx_perifocal = -v_coef * sinhA
            vy_perifocal = v_coef * np.sqrt(e**2 - 1.0) * coshA
        else:
            q = self.orbit.periapsis
            if q is None or not np.isfinite(q) or q <= 0.0:
                q = a * (1.0 - e) if e <= 1.0 else abs(a) * (e - 1.0)
            if q > 0.0 and np.isfinite(q):
                n = np.sqrt(mu / (2.0 * q ** 3))
                mean_anomaly = self.orbit.MA_at_t0 + n * ut
                D = solve_parabolic_anomaly(mean_anomaly)
                v_coef = np.sqrt(mu / (2.0 * q)) * 2.0 / (1.0 + D * D)
                vx_perifocal = -v_coef * D
                vy_perifocal = v_coef
            else:
                vx_perifocal = 0.0
                vy_perifocal = 0.0

        vel_perifocal = np.array([vx_perifocal, vy_perifocal, 0.], np.float64)

        cos_lan, sin_lan = np.cos(self.orbit.lon_of_asc), np.sin(self.orbit.lon_of_asc)
        cos_inc, sin_inc = np.cos(self.orbit.inclination), np.sin(self.orbit.inclination)
        cos_ap, sin_ap = np.cos(self.orbit.arg_p), np.sin(self.orbit.arg_p)

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
    """Universal-variable Lambert solver for a two-point boundary transfer.

    Given an initial and final position (metres) and a time of flight (seconds)
    around a body of gravitational parameter ``root_mu`` (m^3/s^2), ``solve``
    returns the required initial and final velocity vectors (m/s). Set
    ``long_way=True`` to take the transfer arc going the long way around.
    """

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

@dataclass
class OperationResult:
    success: bool
    reason: str = ""
    consumed: dict[str, float] = field(default_factory=dict)
    spawned_vessel: "Spacecraft | None" = None
    detached_parts: list[str] = field(default_factory=list)


@dataclass
class ResourceTank:
    resource: str
    amount: float
    capacity: float
    crossfeed: bool = True

    def __post_init__(self) -> None:
        if self.capacity < 0 or self.amount < 0 or self.amount > self.capacity:
            raise ValueError("Resource tank amount must be between zero and capacity.")


@dataclass
class Engine:
    name: str
    max_thrust: float
    vacuum_isp: float
    propellants: dict[str, float] = field(default_factory=lambda: {"LiquidFuel": 1.0})
    atmospheric_isp: float | None = None
    active: bool = True
    throttle: float = 1.0
    gimbal_limit: float = 0.0
    offset: np.ndarray = field(default_factory=lambda: np.zeros(3))
    mesh: Any = None

    def __post_init__(self) -> None:
        if self.max_thrust < 0 or self.vacuum_isp <= 0:
            raise ValueError("Engine thrust and specific impulse must be positive.")
        if not self.propellants or any(amount <= 0 for amount in self.propellants.values()):
            raise ValueError("An engine needs positive propellant ratios.")
        self.throttle = float(np.clip(self.throttle, 0.0, 1.0))
        self.offset = np.asarray(self.offset, dtype=float).copy()

    def isp_at_pressure(self, pressure: float = 0.0) -> float:
        if self.atmospheric_isp is None:
            return self.vacuum_isp
        # One atmosphere is the simple interpolation boundary for now.
        return self.vacuum_isp + (self.atmospheric_isp - self.vacuum_isp) * np.clip(pressure / 101325.0, 0.0, 1.0)

    def available_thrust(self, pressure: float = 0.0) -> float:
        del pressure
        return self.max_thrust * self.throttle if self.active else 0.0

    def mass_flow(self, pressure: float = 0.0) -> float:
        thrust = self.available_thrust(pressure)
        return thrust / (self.isp_at_pressure(pressure) * 9.80665) if thrust else 0.0


@dataclass
class Part:
    identifier: str
    dry_mass: float
    tanks: list[ResourceTank] = field(default_factory=list)
    engines: list[Engine] = field(default_factory=list)
    attached: bool = True

    @property
    def mass(self) -> float:
        return self.dry_mass + sum(tank.amount for tank in self.tanks)


@dataclass
class GuidanceState:
    mode: str = "inertial"
    attitude: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    target: str | None = None


@dataclass
class VesselSnapshot:
    ut: float
    parent: Body
    position: np.ndarray
    velocity: np.ndarray
    parts: list[Part]
    guidance: GuidanceState
    surface_body: Body | None
    science_data: float


class Spacecraft(Body):
    """A controllable vessel, modelled as a Body with parts, fuel and engines.

    Constructed from an initial relative position ``r0`` (metres) and velocity
    ``v0`` (m/s) at epoch ``t0`` (seconds) around its ``parent`` body, plus
    ``dry_mass``/``wet_mass`` (kg). It derives its own :class:`Orbit` from the
    state and tracks :class:`Part`\\ s, :class:`ResourceTank`\\ s and
    :class:`Engine`\\ s. Use the ``apply_impulse``/``advance_burn``/``snapshot``/
    ``restore`` methods to fly it; never mutate its state directly.
    """

    def __init__(self, name: str, r0: np.ndarray, v0: np.ndarray, t0: float, parent: Body,
                 dry_mass: float, wet_mass: float, hull_mesh, identifier: str = "", render_color: str = "#ffffff"):
        if dry_mass < 0 or wet_mass < dry_mass:
            raise ValueError("Wet mass must be greater than or equal to dry mass.")
        self.parts = [
            Part(
                identifier="core",
                dry_mass=float(dry_mass),
                tanks=[ResourceTank("LiquidFuel", float(wet_mass - dry_mass), float(wet_mass - dry_mass))],
            )
        ]
        self.guidance = GuidanceState()
        self.surface_body: Body | None = None
        self.science_data = 0.0
        self.docked_to: set[Spacecraft] = set()
        self.hull_mesh = hull_mesh
        self.active_actors = []
        
        super().__init__(
            name=name, 
            mu=6.6743e-11 * wet_mass, 
            identifier=identifier or f"SC_{name.replace(' ', '_').upper()}", 
            radius=0.01, 
            atm_height=0.0, 
            render_color=render_color
        )
        
        self.name = name
        self.parent = parent
        self.r0, self.v0, self.t0 = r0.copy(), v0.copy(), t0
        self.moons = []
        self._recalculate_orbit(self.r0, self.v0, self.t0)

    @property
    def mass(self) -> float:
        return sum(part.mass for part in self.parts if part.attached)

    @property
    def dry_mass(self) -> float:
        return sum(part.dry_mass for part in self.parts if part.attached)

    @property
    def fuel_mass(self) -> float:
        return self.resource_amount("LiquidFuel")

    @property
    def engines(self) -> list[Engine]:
        return [engine for part in self.parts if part.attached for engine in part.engines]

    def _refresh_mass(self) -> None:
        self.mu = 6.6743e-11 * self.mass

    def add_part(self, part: Part) -> None:
        if any(existing.identifier == part.identifier for existing in self.parts):
            raise ValueError(f"Part '{part.identifier}' already exists.")
        self.parts.append(part)
        self._refresh_mass()

    def add_resource_tank(
        self, resource: str, capacity: float, amount: float | None = None, part_id: str = "core"
    ) -> None:
        part = self._part(part_id)
        part.tanks.append(ResourceTank(resource, capacity if amount is None else amount, capacity))
        self._refresh_mass()

    def resource_amount(self, resource: str) -> float:
        return sum(tank.amount for part in self.parts if part.attached for tank in part.tanks if tank.resource == resource)

    def resource_capacity(self, resource: str) -> float:
        return sum(tank.capacity for part in self.parts if part.attached for tank in part.tanks if tank.resource == resource)

    def consume_resource(self, resource: str, amount: float) -> float:
        if amount < 0:
            raise ValueError("Resource consumption cannot be negative.")
        remaining = amount
        for part in self.parts:
            for tank in part.tanks:
                if part.attached and tank.crossfeed and tank.resource == resource:
                    used = min(tank.amount, remaining)
                    tank.amount -= used
                    remaining -= used
                    if remaining <= 1e-12:
                        self._refresh_mass()
                        return amount
        self._refresh_mass()
        return amount - remaining

    def add_resource(self, resource: str, amount: float) -> float:
        if amount < 0:
            raise ValueError("Resource addition cannot be negative.")
        remaining = amount
        for part in self.parts:
            for tank in part.tanks:
                if part.attached and tank.resource == resource:
                    added = min(tank.capacity - tank.amount, remaining)
                    tank.amount += added
                    remaining -= added
                    if remaining <= 1e-12:
                        self._refresh_mass()
                        return amount
        self._refresh_mass()
        return amount - remaining

    def _part(self, identifier: str) -> Part:
        for part in self.parts:
            if part.identifier == identifier and part.attached:
                return part
        raise ValueError(f"Attached part '{identifier}' was not found.")

    def add_engine(
        self, thrust: float, isp: float, offset: np.ndarray, mesh=None, *, name: str | None = None,
        propellants: dict[str, float] | None = None, part_id: str = "core"
    ) -> None:
        part = self._part(part_id)
        part.engines.append(Engine(name or f"Engine {len(self.engines) + 1}", thrust, isp, propellants or {"LiquidFuel": 1.0}, offset=offset, mesh=mesh))

    def remove_engine(self, index: int) -> Engine:
        engines = self.engines
        if not 0 <= index < len(engines):
            raise IndexError(f"Engine index {index} is out of range (0-{len(engines) - 1}).")
        target = engines[index]
        for part in self.parts:
            if part.attached and target in part.engines:
                part.engines.remove(target)
                self._refresh_mass()
                return target
        raise ValueError("Engine not found.")

    def set_engine_state(self, index: int, active: bool) -> None:
        if 0 <= index < len(self.engines):
            self.engines[index].active = active

    def set_throttle(self, throttle: float, engine_indices: list[int] | None = None) -> None:
        indices = engine_indices if engine_indices is not None else list(range(len(self.engines)))
        for index in indices:
            if not 0 <= index < len(self.engines):
                raise IndexError(f"Engine index {index} is out of range.")
            self.engines[index].throttle = float(np.clip(throttle, 0.0, 1.0))

    def get_combined_engine_specs(self) -> tuple[float, float]:
        active = [engine for engine in self.engines if engine.active and engine.throttle > 0]
        if not active: return 0.0, 0.0
        total_thrust = sum(engine.available_thrust() for engine in active)
        total_flow = sum(engine.mass_flow() for engine in active)
        return total_thrust, total_thrust / (total_flow * 9.80665)

    def burn_fuel(self, amount: float) -> None:
        if self.consume_resource("LiquidFuel", amount) + 1e-12 < amount:
            raise ValueError("Insufficient LiquidFuel.")

    def execute_maneuver(self, node: ManeuverNode, ut: float) -> bool:
        return self.apply_impulse(node.delta_v_vector, ut).success

    def _active_engines(self) -> list[Engine]:
        return [engine for engine in self.engines if engine.active and engine.throttle > 0]

    def _propellant_requirements(self, duration: float, pressure: float = 0.0) -> dict[str, float]:
        requirements: dict[str, float] = {}
        for engine in self._active_engines():
            flow = engine.mass_flow(pressure) * duration
            ratio_sum = sum(engine.propellants.values())
            for resource, ratio in engine.propellants.items():
                requirements[resource] = requirements.get(resource, 0.0) + flow * ratio / ratio_sum
        return requirements

    def _consume_requirements(self, requirements: dict[str, float]) -> bool:
        if any(self.resource_amount(resource) + 1e-9 < amount for resource, amount in requirements.items()):
            return False
        for resource, amount in requirements.items():
            self.consume_resource(resource, amount)
        return True

    def apply_impulse(self, delta_v_vector: np.ndarray, ut: float) -> OperationResult:
        """Apply an instantaneous manoeuvre using the active engines' propellant mix."""
        delta_v = float(np.linalg.norm(delta_v_vector))
        if delta_v == 0.0:
            return OperationResult(True)
        _, isp = self.get_combined_engine_specs()
        if isp == 0.0:
            return OperationResult(False, "No active engine is available.")
        propellant_mass = self.mass * (1.0 - np.exp(-delta_v / (isp * 9.80665)))
        per_second = self._propellant_requirements(1.0)
        total_flow = sum(per_second.values())
        requirements = {resource: propellant_mass * amount / total_flow for resource, amount in per_second.items()}
        if not self._consume_requirements(requirements):
            return OperationResult(False, "Insufficient propellant for manoeuvre.")
        position, velocity = self.state_at(ut)
        self.set_state(position, velocity + np.asarray(delta_v_vector, dtype=float), ut)
        return OperationResult(True, consumed=requirements)

    def apply_rcs_impulse(self, delta_v_vector: np.ndarray, ut: float, isp: float = 260.0) -> OperationResult:
        """Apply a small translation impulse using MonoPropellant."""
        delta_v = float(np.linalg.norm(delta_v_vector))
        if delta_v == 0.0:
            return OperationResult(True)
        if isp <= 0:
            return OperationResult(False, "RCS specific impulse must be positive.")
        required = self.mass * (1.0 - np.exp(-delta_v / (isp * 9.80665)))
        if self.resource_amount("MonoPropellant") + 1e-9 < required:
            return OperationResult(False, "Insufficient MonoPropellant.")
        self.consume_resource("MonoPropellant", required)
        position, velocity = self.state_at(ut)
        self.set_state(position, velocity + np.asarray(delta_v_vector, dtype=float), ut)
        return OperationResult(True, consumed={"MonoPropellant": required})

    def advance_burn(
        self, duration: float, direction: np.ndarray, start_ut: float, pressure: float = 0.0
    ) -> OperationResult:
        """Perform a finite burn with constant thrust and direction over ``duration``."""
        if duration <= 0:
            return OperationResult(False, "Burn duration must be positive.")
        unit_direction = np.asarray(direction, dtype=float)
        magnitude = np.linalg.norm(unit_direction)
        if magnitude == 0:
            return OperationResult(False, "Burn direction cannot be zero.")
        thrust = sum(engine.available_thrust(pressure) for engine in self._active_engines())
        if thrust == 0:
            return OperationResult(False, "No active engine is available.")
        requested = self._propellant_requirements(duration, pressure)
        possible_duration = duration
        for resource, amount in requested.items():
            if amount > 0:
                possible_duration = min(possible_duration, duration * self.resource_amount(resource) / amount)
        if possible_duration <= 1e-9:
            return OperationResult(False, "Insufficient propellant to start burn.")
        requirements = self._propellant_requirements(possible_duration, pressure)
        initial_mass = self.mass
        if not self._consume_requirements(requirements):
            return OperationResult(False, "Propellant feed failed.")
        final_mass = self.mass
        _, isp = self.get_combined_engine_specs()
        delta_v = isp * 9.80665 * np.log(initial_mass / final_mass)
        end_ut = start_ut + possible_duration
        position, velocity = self.state_at(end_ut)
        self.set_state(position, velocity + unit_direction / magnitude * delta_v, end_ut)
        reason = "" if possible_duration == duration else "Burn ended early: propellant depleted."
        return OperationResult(True, reason, requirements)

    def set_guidance(self, mode: str, attitude: np.ndarray | None = None, target: str | None = None) -> None:
        self.guidance.mode = mode
        self.guidance.target = target
        if attitude is not None:
            attitude = np.asarray(attitude, dtype=float)
            if attitude.shape != (4,) or np.linalg.norm(attitude) == 0:
                raise ValueError("Attitude must be a non-zero quaternion.")
            self.guidance.attitude = attitude / np.linalg.norm(attitude)

    def stage(self, part_ids: list[str]) -> OperationResult:
        if not part_ids:
            return OperationResult(False, "Select at least one part to stage.")
        if "core" in part_ids:
            return OperationResult(False, "The core part cannot be staged.")
        detached: list[str] = []
        for identifier in part_ids:
            part = self._part(identifier)
            part.attached = False
            detached.append(identifier)
        self._refresh_mass()
        return OperationResult(True, detached_parts=detached)

    def transfer_resource(self, other: "Spacecraft", resource: str, amount: float) -> OperationResult:
        if amount <= 0:
            return OperationResult(False, "Transfer amount must be positive.")
        moved = min(amount, self.resource_amount(resource), other.resource_capacity(resource) - other.resource_amount(resource))
        if moved <= 0:
            return OperationResult(False, "No transferable resource capacity is available.")
        self.consume_resource(resource, moved)
        other.add_resource(resource, moved)
        return OperationResult(True, consumed={resource: moved})

    def change_reference_body(self, new_parent: Body, ut: float) -> OperationResult:
        if new_parent is self.parent:
            return OperationResult(True)
        position, velocity = self.state_at(ut)
        self.parent = new_parent
        self.set_state(position, velocity, ut)
        return OperationResult(True)

    def time_to_soi_exit(self, ut: float | None = None) -> float | None:
        """UT when craft crosses the parent body's SOI boundary, or None if bound."""
        t0 = self.t0 if ut is None else float(ut)
        orb = self.orbit
        parent = orb.parent
        if parent is None:
            return None
        r_soi = parent.soi_radius()
        if r_soi is None or r_soi <= 0.0:
            return None
        mu = parent.mu
        e = orb.eccen
        q = orb.periapsis
        if e >= 1.0 - PARABOLIC_EPS and q is not None and q > 0.0 and np.isfinite(q):
            D_exit = np.sqrt(r_soi / q - 1.0)
            n_p = np.sqrt(mu / (2.0 * q ** 3))
            M_p_exit = D_exit + D_exit ** 3 / 3.0
            dt = (M_p_exit - orb.MA_at_t0) / n_p
            return t0 + dt
        if e > 1.0 + PARABOLIC_EPS:
            a = orb.semi_major_axis
            if a >= 0.0 or not np.isfinite(a):
                return None
            cosh_H = (r_soi / a + 1.0) / e
            if cosh_H < 1.0:
                return None
            H = np.arccosh(cosh_H)
            M_h = e * np.sinh(H) - H
            n_h = np.sqrt(mu / (-a) ** 3)
            dt = (M_h - orb.MA_at_t0) / n_h
            return t0 + dt
        return None

    def dock(self, other: Spacecraft, ut: float) -> OperationResult:
        if other is self:
            return OperationResult(False, "A vessel cannot dock with itself.")
        if other.parent is not self.parent:
            return OperationResult(False, "Vessels must share a reference body before docking.")
        position_a, velocity_a = self.state_at(ut)
        position_b, velocity_b = other.state_at(ut)
        if np.linalg.norm(position_a - position_b) > max(self.radius + other.radius, 1.0):
            return OperationResult(False, "Vessels are not within docking distance.")
        mass_a, mass_b = self.mass, other.mass
        self.parts.extend(copy.deepcopy(other.parts))
        self._refresh_mass()
        self.set_state((position_a * mass_a + position_b * mass_b) / (mass_a + mass_b), (velocity_a * mass_a + velocity_b * mass_b) / (mass_a + mass_b), ut)
        other.docked_to.add(self)
        self.docked_to.add(other)
        return OperationResult(True)

    def snapshot(self, ut: float) -> VesselSnapshot:
        position, velocity = self.state_at(ut)
        return VesselSnapshot(ut, self.parent, position.copy(), velocity.copy(), copy.deepcopy(self.parts), copy.deepcopy(self.guidance), self.surface_body, self.science_data)

    def restore(self, snapshot: VesselSnapshot) -> None:
        self.parent = snapshot.parent
        self.parts = copy.deepcopy(snapshot.parts)
        self.guidance = copy.deepcopy(snapshot.guidance)
        self.surface_body = snapshot.surface_body
        self.science_data = snapshot.science_data
        self._refresh_mass()
        self.set_state(snapshot.position, snapshot.velocity, snapshot.ut)

    def collect_science(self, amount: float) -> OperationResult:
        if amount <= 0:
            return OperationResult(False, "Science amount must be positive.")
        self.science_data += amount
        return OperationResult(True)

    def transmit_science(self, amount: float | None = None) -> OperationResult:
        sent = self.science_data if amount is None else min(amount, self.science_data)
        if sent <= 0:
            return OperationResult(False, "No science data is available.")
        self.science_data -= sent
        return OperationResult(True, consumed={"Science": sent})

    def land(self, body: Body) -> OperationResult:
        self.surface_body = body
        return OperationResult(True)

    def launch(self) -> OperationResult:
        if self.surface_body is None:
            return OperationResult(False, "Vessel is not landed.")
        self.surface_body = None
        return OperationResult(True)

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
            if e.mesh is not None:
                engine_copy = e.mesh.copy()
                engine_copy.translate(pos + e.offset, inplace=True)
                color = "#ff4500" if e.active else "#555555"
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

    def state_at(self, ut: float) -> tuple[np.ndarray, np.ndarray]:
        return (
            self.get_absolute_pos_at_ut(ut),
            self.get_absolute_vel_at_ut(ut),
        )

    def set_state(
        self,
        r_abs: np.ndarray,
        v_abs: np.ndarray,
        ut: float,
    ):
        parent_r = self.parent.get_absolute_pos_at_ut(ut)
        parent_v = self.parent.get_absolute_vel_at_ut(ut)

        self.r0 = r_abs - parent_r
        self.v0 = v_abs - parent_v
        self.t0 = ut

        self._recalculate_orbit(self.r0, self.v0, self.t0)

    def set_absolute_pos_at_ut(self, r: np.ndarray, ut: float) -> None:
        parent_r = self.parent.get_absolute_pos_at_ut(ut)
        self.r0, self.t0 = r - parent_r, ut
        self._recalculate_orbit(self.r0, self.v0, self.t0)

    def set_absolute_vel_at_ut(self, v: np.ndarray, ut: float) -> None:
        parent_v = self.parent.get_absolute_vel_at_ut(ut)
        self.v0, self.t0 = v - parent_v, ut
        self._recalculate_orbit(self.r0, self.v0, self.t0)

    def undock(self, part_ids: list[str], name: str, ut: float) -> OperationResult:
        if not part_ids or "core" in part_ids:
            return OperationResult(False, "Undocking requires one or more non-core parts.")
        
        detached = [self._part(identifier) for identifier in part_ids]
        for part in detached:
            part.attached = False
        self._refresh_mass()

        pos_abs, vel_abs = self.state_at(ut)
        parent_pos = self.parent.get_absolute_pos_at_ut(ut)
        parent_vel = self.parent.get_absolute_vel_at_ut(ut)

        # Compute initial mass specs from detached parts
        d_dry = sum(p.dry_mass for p in detached)
        d_wet = d_dry + sum(t.amount for p in detached for t in p.tanks if t.resource == "LiquidFuel")

        spawned = Spacecraft(
            name=name,
            r0=pos_abs - parent_pos,
            v0=vel_abs - parent_vel,
            t0=ut,
            parent=self.parent,
            dry_mass=d_dry,
            wet_mass=d_wet,
            hull_mesh=self.hull_mesh,
        )
        spawned.parts = copy.deepcopy(detached)
        for part in spawned.parts:
            part.attached = True

        spawned._refresh_mass()
        spawned._recalculate_orbit(spawned.r0, spawned.v0, ut)
        return OperationResult(True, spawned_vessel=spawned, detached_parts=part_ids)

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

        q = h_mag ** 2 / (mu * (1.0 + e))

        # Mean anomaly at epoch
        if e < 1.0 - PARABOLIC_EPS:
            # Elliptic
            E = 2.0 * np.arctan2(
                np.sqrt(1.0 - e) * np.sin(nu / 2.0),
                np.sqrt(1.0 + e) * np.cos(nu / 2.0),
            )
            ma0 = E - e * np.sin(E)

        elif e > 1.0 + PARABOLIC_EPS:
            # Hyperbolic
            H = 2.0 * np.arctanh(
                np.sqrt((e - 1.0) / (e + 1.0)) * np.tan(nu / 2.0)
            )
            ma0 = e * np.sinh(H) - H

        else:
            D = np.tan(nu / 2.0)
            ma0 = D + D * D * D / 3.0

        # Store orbit
        if e < 1.0 - PARABOLIC_EPS or e > 1.0 + PARABOLIC_EPS:
            n = np.sqrt(mu / abs(a) ** 3)
        elif q > 0.0 and np.isfinite(q):
            n = np.sqrt(mu / (2.0 * q ** 3))
        else:
            n = 0.0
        ma_at_t0 = ma0 - n * ut

        self.orbit = Orbit(
            a=float(a),
            e=float(e),
            arg_p=float(arg_p),
            lon_of_asc=float(lan),
            MA_at_t0=float(ma_at_t0),
            inclination=float(inc),
            parent=self.parent,
            periapsis=float(q),
        )

@dataclass
class ManeuverNode:
    """A planned instantaneous burn at time ``ut`` (seconds).

    ``delta_v_vector`` is the raw delta-v (m/s); ``prograde``/``normal``/``radial``
    are its components resolved into the local orbital frame at the burn, and
    ``total_mag`` is the burn magnitude in m/s.
    """

    ut: float
    delta_v_vector: np.ndarray
    prograde: np.ndarray
    normal: np.ndarray
    radial: np.ndarray
    total_mag: float

class ManeuverPlanner:
    """Builds a :class:`ManeuverNode` from current and required velocities."""

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
