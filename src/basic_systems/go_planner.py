"""Orbit presets and two-burn transfer planning for the ``go`` command.

This module is pure simulation: it never touches the console and never
permanently mutates a spacecraft (planning runs on a snapshot and restores
the state afterwards). The REPL/API layer in ``CLI_use/go.py`` turns a plan
into ticket events.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import copy
import json
from typing import Callable

import numpy as np

from basic_systems.orbit_pred import Body, Orbit, Spacecraft, ManeuverNode, ManeuverPlanner

MU_G = 9.80665


@dataclass
class OrbitSpec:
    """A target orbit expressed in radii (not altitudes)."""

    name: str
    periapsis: float
    apoapsis: float
    inclination: float = 0.0
    arg_periapsis: float = 0.0
    lan: float = 0.0
    escape: bool = False

    def to_orbit(self, parent: Body, mean_anomaly: float = 0.0) -> Orbit:
        a = (self.periapsis + self.apoapsis) / 2.0
        e = (self.apoapsis - self.periapsis) / (self.apoapsis + self.periapsis)
        return Orbit(
            a=float(a),
            e=float(e),
            arg_p=float(self.arg_periapsis),
            lon_of_asc=float(self.lan),
            MA_at_t0=float(mean_anomaly),
            inclination=float(self.inclination),
            parent=parent,
        )

    def describe(self, body: Body | None = None) -> str:
        label = self.name
        radius = body.radius if body is not None else 0.0
        peri_alt = self.periapsis - radius
        apo_alt = self.apoapsis - radius
        if self.escape:
            return f"{label} (escape)"
        if abs(peri_alt - apo_alt) < 1.0:
            return f"{label} ({peri_alt:,.0f}m circular, {np.degrees(self.inclination):.1f} deg)"
        return (
            f"{label} (pe {peri_alt:,.0f}m / ap {apo_alt:,.0f}m, "
            f"{np.degrees(self.inclination):.1f} deg)"
        )


@dataclass
class CustomPreset:
    """A user-defined, editable orbit preset (stored in the preset library)."""

    key: str
    label: str
    description: str = ""
    peri_alt: float = 200_000.0
    apo_alt: float = 200_000.0
    inclination_deg: float = 0.0
    arg_p_deg: float = 0.0
    lan_deg: float = 0.0

    def resolve(self, body: Body) -> OrbitSpec:
        return OrbitSpec(
            name=self.label,
            periapsis=body.radius + self.peri_alt,
            apoapsis=body.radius + self.apo_alt,
            inclination=np.radians(self.inclination_deg),
            arg_periapsis=np.radians(self.arg_p_deg),
            lan=np.radians(self.lan_deg),
        )


DEFAULT_ARCHETYPES: list[CustomPreset] = [
    CustomPreset("leo", "LEO", "Low circular orbit (200 km)", 200_000.0, 200_000.0, 0.0),
    CustomPreset("polar", "Polar", "Polar circular orbit (200 km)", 200_000.0, 200_000.0, 90.0),
    CustomPreset("sso", "SSO", "Sun-synchronous-ish retrograde (98.2 deg)", 200_000.0, 200_000.0, 98.2),
    CustomPreset("geo", "GEO", "5,000 km circular placeholder; use the 'sync' preset for true synchronous altitude", 5_000_000.0, 5_000_000.0, 0.0),
    CustomPreset("heo", "HEO", "Highly eccentric orbit", 200_000.0, 10_000_000.0, 0.0),
    CustomPreset("molniya", "Molniya", "Molniya-class inclined ellipse (63.4 deg)", 200_000.0, 7_800_000.0, 63.4),
    CustomPreset("tundra", "Tundra", "Tundra-class near-circular (63.4 deg)", 6_000_000.0, 8_000_000.0, 63.4),
]


class PresetLibrary:
    """Editable collection of user orbit presets, persisted as JSON."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else Path("presets.json")
        self.presets: dict[str, CustomPreset] = {}
        for preset in DEFAULT_ARCHETYPES:
            self.presets[preset.key] = copy.deepcopy(preset)
        if self.path.exists():
            self.load(self.path)

    def load(self, path: str | Path | None = None) -> None:
        target = Path(path) if path else self.path
        if not target.exists():
            return
        data = json.loads(target.read_text(encoding="utf-8"))
        for item in data.get("presets", []):
            try:
                preset = CustomPreset(
                    key=item["key"],
                    label=item.get("label", item["key"]),
                    description=item.get("description", ""),
                    peri_alt=float(item.get("peri_alt", 200_000.0)),
                    apo_alt=float(item.get("apo_alt", 200_000.0)),
                    inclination_deg=float(item.get("inclination_deg", 0.0)),
                    arg_p_deg=float(item.get("arg_p_deg", 0.0)),
                    lan_deg=float(item.get("lan_deg", 0.0)),
                )
            except (KeyError, TypeError, ValueError):
                continue
            self.presets[preset.key] = preset

    def save(self, path: str | Path | None = None) -> Path:
        target = Path(path) if path else self.path
        data = {
            "presets": [
                {
                    "key": p.key,
                    "label": p.label,
                    "description": p.description,
                    "peri_alt": p.peri_alt,
                    "apo_alt": p.apo_alt,
                    "inclination_deg": p.inclination_deg,
                    "arg_p_deg": p.arg_p_deg,
                    "lan_deg": p.lan_deg,
                }
                for p in self.presets.values()
            ]
        }
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return target

    def add(self, preset: CustomPreset) -> None:
        if not preset.key:
            raise ValueError("Preset key is required.")
        self.presets[preset.key] = preset

    def remove(self, key: str) -> CustomPreset | None:
        return self.presets.pop(key, None)

    def get(self, key: str) -> CustomPreset | None:
        return self.presets.get(key)


class Preset:
    """A selectable orbit preset entry with a body/spacecraft-dependent resolve."""

    def __init__(
        self,
        key: str,
        label: str,
        description: str,
        resolve: Callable[[Body, Spacecraft], OrbitSpec],
    ) -> None:
        self.key = key
        self.label = label
        self.description = description
        self._resolve = resolve

    def resolve(self, body: Body, spacecraft: Spacecraft) -> OrbitSpec:
        return self._resolve(body, spacecraft)


def _spec_from_orbit(name: str, orbit: Orbit) -> OrbitSpec:
    a, e = orbit.semi_major_axis, orbit.eccen
    if a <= 0.0 or e >= 1.0:
        return OrbitSpec(name, 0.0, 0.0, orbit.inclination, orbit.arg_p, orbit.lon_of_asc, escape=True)
    return OrbitSpec(
        name,
        periapsis=a * (1.0 - e),
        apoapsis=a * (1.0 + e),
        inclination=orbit.inclination,
        arg_periapsis=orbit.arg_p,
        lan=orbit.lon_of_asc,
    )


def _escape_spec(spacecraft: Spacecraft) -> OrbitSpec:
    base = _spec_from_orbit("Escape", spacecraft.orbit)
    return OrbitSpec(
        base.name,
        periapsis=base.periapsis,
        apoapsis=base.apoapsis,
        inclination=base.inclination,
        arg_periapsis=base.arg_periapsis,
        lan=base.lan,
        escape=True,
    )


def orbit_presets(body: Body, spacecraft: Spacecraft, library: PresetLibrary) -> list[Preset]:
    """All selectable orbit presets for ``body`` (dynamic + user library)."""
    entries: list[Preset] = []

    def entry(key: str, label: str, description: str, resolve) -> None:
        entries.append(Preset(key, label, description, resolve))

    entry(
        "current", "Current", "Stay in the current orbit",
        lambda b, sc: _spec_from_orbit("Current", sc.orbit),
    )

    radius = body.radius
    for mult, label, desc in (
        (0.05, "Low", "Low circular orbit"),
        (0.2, "Medium", "Medium circular orbit"),
        (0.6, "High", "High circular orbit"),
    ):
        r = radius * (1.0 + mult)
        entry(
            f"circular:{mult:.2f}".replace(".", ""),
            label,
            f"{desc} ({radius * mult:,.0f}m altitude)",
            lambda b, sc, r=r, label=label: OrbitSpec(label, r, r),
        )

    if body.rotation_period_s > 0:
        sync_r = body.synchronous_radius()
        if sync_r is not None:
            entry(
                "sync", "Synchronous", f"Circular orbit matching rotation ({body.rotation_period_s:,.0f}s)",
                lambda b, sc, r=sync_r: OrbitSpec("Synchronous", r, r),
            )

    entry(
        "atm-edge", "Atmosphere Edge", "Circular orbit at the atmosphere limit",
        lambda b, sc: OrbitSpec("Atmosphere Edge", b.radius + b.atm_height, b.radius + b.atm_height),
    )

    entry(
        "escape", "Escape", "Single burn to escape velocity",
        lambda b, sc: _escape_spec(sc),
    )

    entry(
        "impact", "Impact", "Descend to a surface impact trajectory",
        lambda b, sc: OrbitSpec("Impact", b.radius, b.radius * 2.0),
    )

    for moon in body.moons:
        orbit = moon.orbit
        if orbit.parent is None or orbit.semi_major_axis <= 0.0 or orbit.eccen >= 1.0:
            continue
        key = f"moon:{moon.identifier}"
        entry(
            key, f"Moon: {moon.name}", f"Match {moon.name}'s orbit around {body.name}",
            lambda b, sc, moon=moon, orbit=orbit: _spec_from_orbit(f"{moon.name} orbit", orbit),
        )

    for preset in library.presets.values():
        entry(
            f"preset:{preset.key}",
            preset.label,
            preset.description or f"User preset {preset.key}",
            lambda b, sc, p=preset: p.resolve(b),
        )

    return entries


def find_preset(body: Body, spacecraft: Spacecraft, library: PresetLibrary, key: str) -> Preset | None:
    """Resolve ``key`` to a preset, tolerating ``moon:`` / ``preset:`` prefixes."""
    key = key.strip()
    for preset in orbit_presets(body, spacecraft, library):
        if preset.key == key:
            return preset
        if preset.key.split(":", 1)[-1] == key or preset.label.lower() == key.lower():
            return preset
    return None


# --- Transfer planning -------------------------------------------------------

@dataclass
class PlannedStep:
    """A single coast or manoeuvre segment within a :class:`TransferPlan`.

    ``kind`` is ``"coast"`` or ``"maneuver"``; ``start``/``end`` are UT seconds.
    For a burn, ``ut`` is the burn time, ``dv`` the delta-v vector (m/s) and
    ``node`` the corresponding :class:`~basic_systems.orbit_pred.ManeuverNode`.
    ``note`` is a short human description.
    """

    kind: str  # "coast" | "maneuver"
    start: float
    end: float
    ut: float = 0.0
    dv: np.ndarray | None = None
    node: ManeuverNode | None = None
    note: str = ""


@dataclass
class TransferPlan:
    """Result of :func:`plan_transfer`: a sequence of burn/coast steps.

    ``body`` is the reference body; ``steps`` lists the burns and coasts;
    ``total_dv`` is the summed delta-v magnitude (m/s) and ``burn_count`` the
    number of burns. ``start_ut``/``end_ut`` bound the plan (seconds),
    ``achieved`` describes the resulting orbit and ``warnings`` holds notes.
    """

    body: Body
    steps: list[PlannedStep]
    total_dv: float
    burn_count: int
    start_ut: float
    end_ut: float
    achieved: dict
    warnings: list[str] = field(default_factory=list)


def _normal_from_spec(spec: OrbitSpec) -> np.ndarray:
    i, o = spec.inclination, spec.lan
    return np.array([np.sin(i) * np.sin(o), -np.sin(i) * np.cos(o), np.cos(i)], dtype=float)


def _tangential(normal: np.ndarray, r: np.ndarray, reference: np.ndarray) -> np.ndarray:
    t = np.cross(normal, r)
    magnitude = np.linalg.norm(t)
    if magnitude < 1e-9:
        raise ValueError("Burn point is on the target orbit's pole; cannot resolve a tangential direction.")
    t = t / magnitude
    if np.dot(t, reference) < 0.0:
        t = -t
    return t


def _vis_viva(mu: float, r: float, a: float) -> float:
    if a <= 0.0:
        raise ValueError("Transfer orbit is not elliptical; cannot plan an elliptical rendezvous.")
    return float(np.sqrt(mu * (2.0 / r - 1.0 / a)))


def _time_to_mean_anomaly(sc: Spacecraft, ut: float, target_ma: float) -> float:
    orbit = sc.orbit
    a, e = orbit.semi_major_axis, orbit.eccen
    if a <= 0.0 or e >= 1.0:
        raise ValueError("Cannot plan from a non-elliptical orbit.")
    n = np.sqrt((orbit.parent.mu / a ** 3)) #type: ignore
    mean_anomaly = orbit.MA_at_t0 + n * ut
    delta = (target_ma - mean_anomaly) % (2.0 * np.pi)
    return delta / n


def _sim_impulse(sc: Spacecraft, dv: np.ndarray, ut: float) -> None:
    position, velocity = sc.state_at(ut)
    sc.set_state(position, velocity + np.asarray(dv, dtype=float), ut)


def _planner_node(ut: float, v_curr: np.ndarray, v_req: np.ndarray, r: np.ndarray):
    return ManeuverPlanner().calculate_maneuver(ut, v_curr, v_req, r)


def plan_transfer(
    spacecraft: Spacecraft,
    body: Body,
    start_spec: OrbitSpec,
    end_spec: OrbitSpec,
    ut: float = 0.0,
) -> TransferPlan:
    """Plan a two-burn (or single-burn escape) transfer without mutating the craft."""
    snapshot = spacecraft.snapshot(ut)
    try:
        return _plan(spacecraft, body, start_spec, end_spec, ut)
    finally:
        spacecraft.restore(snapshot)


def _plan(spacecraft: Spacecraft, body: Body, start_spec: OrbitSpec, end_spec: OrbitSpec, ut: float) -> TransferPlan:
    mu = body.mu
    steps: list[PlannedStep] = []
    warnings: list[str] = []

    orbit = spacecraft.orbit
    a = orbit.semi_major_axis
    e = orbit.eccen
    if a <= 0.0 or e >= 1.0:
        raise ValueError("Planning requires an elliptical starting orbit.")
    r_p1 = a * (1.0 - e)
    r_a1 = a * (1.0 + e)

    if end_spec.escape:
        steps, total_dv = _plan_escape(spacecraft, body, ut, r_p1)
        return TransferPlan(body, steps, total_dv, len(steps), ut, ut, _achieved(spacecraft, body), warnings)

    end_peri, end_apo = end_spec.periapsis, end_spec.apoapsis
    if end_peri <= 0.0 or end_apo < end_peri:
        raise ValueError(f"Invalid target orbit '{end_spec.name}': periapsis and apoapsis radii are invalid.")

    ascending = end_apo >= r_a1 - 1.0

    if ascending:
        burn1_ut = ut + _time_to_mean_anomaly(spacecraft, ut, 0.0)
        steps, total_dv = _plan_raise(spacecraft, body, end_spec, ut, burn1_ut, r_p1, end_apo, mu, steps, warnings)
    else:
        burn1_ut = ut + _time_to_mean_anomaly(spacecraft, ut, np.pi)
        steps, total_dv = _plan_lower(spacecraft, body, end_spec, ut, burn1_ut, r_a1, end_peri, mu, steps, warnings)

    end_ut = steps[-1].end if steps else ut
    maneuvers = [s for s in steps if s.kind == "maneuver"]
    if maneuvers and all(np.linalg.norm(s.dv if s.dv is not None else np.zeros(3)) < 1e-6 for s in maneuvers):
        steps = []
        total_dv = 0.0
    return TransferPlan(body, steps, total_dv, sum(1 for s in steps if s.kind == "maneuver"), ut, end_ut, _achieved(spacecraft, body), warnings)


def _plan_raise(spacecraft, body, end_spec, ut, burn1_ut, r_p1, transfer_apo, mu, steps, warnings):
    total_dv = 0.0

    if burn1_ut - ut > 1e-6:
        steps.append(PlannedStep("coast", ut, burn1_ut, note="Wait to periapsis"))

    r1 = spacecraft.get_pos_at_ut(burn1_ut)
    v_curr1 = spacecraft.get_vel_at_ut(burn1_ut)
    a_trans = (r_p1 + transfer_apo) / 2.0
    speed1 = _vis_viva(mu, r_p1, a_trans)
    t2 = _tangential(_normal_from_spec(end_spec), r1, v_curr1)
    v_req1 = speed1 * t2
    dv1 = v_req1 - v_curr1
    node1 = _planner_node(burn1_ut, v_curr1, v_req1, r1)
    steps.append(PlannedStep("maneuver", burn1_ut, burn1_ut, burn1_ut, dv1, node1, "Raise apoapsis to transfer"))
    total_dv += float(np.linalg.norm(dv1))
    _sim_impulse(spacecraft, dv1, burn1_ut)

    transfer = spacecraft.orbit
    a_trans = transfer.semi_major_axis
    if a_trans <= 0.0 or transfer.eccen >= 1.0:
        warnings.append("First burn produced a non-elliptical orbit; second burn skipped.")
        steps.append(PlannedStep("coast", burn1_ut, burn1_ut, note="No transfer possible"))
        return steps, total_dv
    burn2_ut = burn1_ut + transfer.period / 2.0
    steps.append(PlannedStep("coast", burn1_ut, burn2_ut, note="Coast to apoapsis"))

    r2 = spacecraft.get_pos_at_ut(burn2_ut)
    v_curr2 = spacecraft.get_vel_at_ut(burn2_ut)
    a_end = (end_spec.periapsis + end_spec.apoapsis) / 2.0
    speed2 = _vis_viva(mu, np.linalg.norm(r2), a_end)
    t2b = _tangential(_normal_from_spec(end_spec), r2, v_curr2)
    v_req2 = speed2 * t2b
    dv2 = v_req2 - v_curr2
    node2 = _planner_node(burn2_ut, v_curr2, v_req2, r2)
    steps.append(PlannedStep("maneuver", burn2_ut, burn2_ut, burn2_ut, dv2, node2, "Circularize / match target"))
    total_dv += float(np.linalg.norm(dv2))
    _sim_impulse(spacecraft, dv2, burn2_ut)
    return steps, total_dv


def _plan_lower(spacecraft, body, end_spec, ut, burn1_ut, r_a1, transfer_peri, mu, steps, warnings):
    total_dv = 0.0

    if burn1_ut - ut > 1e-6:
        steps.append(PlannedStep("coast", ut, burn1_ut, note="Wait to apoapsis"))

    r1 = spacecraft.get_pos_at_ut(burn1_ut)
    v_curr1 = spacecraft.get_vel_at_ut(burn1_ut)
    a_trans = (transfer_peri + r_a1) / 2.0
    speed1 = _vis_viva(mu, r_a1, a_trans)
    t2 = _tangential(_normal_from_spec(end_spec), r1, v_curr1)
    v_req1 = speed1 * t2
    dv1 = v_req1 - v_curr1
    node1 = _planner_node(burn1_ut, v_curr1, v_req1, r1)
    steps.append(PlannedStep("maneuver", burn1_ut, burn1_ut, burn1_ut, dv1, node1, "Lower periapsis to transfer"))
    total_dv += float(np.linalg.norm(dv1))
    _sim_impulse(spacecraft, dv1, burn1_ut)

    transfer = spacecraft.orbit
    a_trans = transfer.semi_major_axis
    if a_trans <= 0.0 or transfer.eccen >= 1.0:
        warnings.append("First burn produced a non-elliptical orbit; second burn skipped.")
        steps.append(PlannedStep("coast", burn1_ut, burn1_ut, note="No transfer possible"))
        return steps, total_dv
    burn2_ut = burn1_ut + transfer.period / 2.0
    steps.append(PlannedStep("coast", burn1_ut, burn2_ut, note="Coast to periapsis"))

    r2 = spacecraft.get_pos_at_ut(burn2_ut)
    v_curr2 = spacecraft.get_vel_at_ut(burn2_ut)
    a_end = (end_spec.periapsis + end_spec.apoapsis) / 2.0
    speed2 = _vis_viva(mu, np.linalg.norm(r2), a_end)
    t2b = _tangential(_normal_from_spec(end_spec), r2, v_curr2)
    v_req2 = speed2 * t2b
    dv2 = v_req2 - v_curr2
    node2 = _planner_node(burn2_ut, v_curr2, v_req2, r2)
    steps.append(PlannedStep("maneuver", burn2_ut, burn2_ut, burn2_ut, dv2, node2, "Match target orbit"))
    total_dv += float(np.linalg.norm(dv2))
    _sim_impulse(spacecraft, dv2, burn2_ut)
    return steps, total_dv


def _plan_escape(spacecraft: Spacecraft, body: Body, ut: float, r_p1: float) -> tuple[list[PlannedStep], float]:
    mu = body.mu
    steps: list[PlannedStep] = []
    burn_ut = ut + _time_to_mean_anomaly(spacecraft, ut, 0.0)
    if burn_ut - ut > 1e-6:
        steps.append(PlannedStep("coast", ut, burn_ut, note="Wait to periapsis"))
    r1 = spacecraft.get_pos_at_ut(burn_ut)
    v_curr = spacecraft.get_vel_at_ut(burn_ut)
    v_esc = np.sqrt(2.0 * mu / np.linalg.norm(r1))
    t2 = v_curr / np.linalg.norm(v_curr)
    v_req = v_esc * t2
    dv = v_req - v_curr
    node = _planner_node(burn_ut, v_curr, v_req, r1)
    steps.append(PlannedStep("maneuver", burn_ut, burn_ut, burn_ut, dv, node, "Escape burn"))
    _sim_impulse(spacecraft, dv, burn_ut)
    return steps, float(np.linalg.norm(dv))


def _achieved(spacecraft: Spacecraft, body: Body) -> dict:
    orbit = spacecraft.orbit
    a, e = orbit.semi_major_axis, orbit.eccen
    return {
        "semi_major_axis": a,
        "eccentricity": e,
        "periapsis": a * (1.0 - e) if a > 0.0 else None,
        "apoapsis": a * (1.0 + e) if a > 0.0 else None,
        "inclination_deg": np.degrees(orbit.inclination),
        "parent": body.name,
    }
