"""Programmatic API for the ``go`` command.

``apply_go`` / ``go_plan`` are importable so missions can be planned from code,
not just from the REPL.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from basic_systems.go_planner import (
    CustomPreset,
    OrbitSpec,
    Preset,
    PresetLibrary,
    TransferPlan,
    find_preset,
    orbit_presets,
    plan_transfer,
)
from CLI_use.ticket import CoastEvent, ManeuverEvent, Ticket, TicketEvent

DEFAULT_PRESET_PATH = Path("presets.json")


def preset_library(path: str | Path = DEFAULT_PRESET_PATH) -> PresetLibrary:
    """Load the user orbit preset library from ``path`` (``presets.json`` default)."""
    return PresetLibrary(path)


@dataclass
class GoPlan:
    """A planned in-system transfer, ready to inspect or commit.

    Built by :func:`go_plan`/:func:`apply_go`. Holds the start/target orbit
    specs and the :class:`~basic_systems.go_planner.TransferPlan`; ``events`` are
    the scheduled :class:`~CLI_use.ticket.TicketEvent`\\ s (coasts + burns) and
    ``warnings`` any planning notes. ``burn_count``/``total_dv``/``end_ut``
    summarise the plan; ``summary_lines`` renders it for display.
    """

    body_name: str
    start_label: str
    end_label: str
    start_spec: OrbitSpec
    end_spec: OrbitSpec
    transfer: TransferPlan
    events: list[TicketEvent] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def burn_count(self) -> int:
        return self.transfer.burn_count

    @property
    def total_dv(self) -> float:
        return self.transfer.total_dv

    @property
    def end_ut(self) -> float:
        return self.transfer.end_ut

    def summary_lines(self) -> list[str]:
        lines = [
            f"  {self.start_label}  ->  {self.end_label}  (about {self.body_name})",
            f"  Burns: {self.burn_count}   Total delta-v: {self.total_dv:,.1f} m/s   Duration: {self.transfer.end_ut - self.transfer.start_ut:,.1f}s",
        ]
        for step in self.transfer.steps:
            if step.kind == "coast":
                lines.append(f"    coast  {step.start:,.1f}s -> {step.end:,.1f}s  {step.note}")
            else:
                lines.append(
                    f"    burn   @ {step.ut:,.1f}s  dv={np.linalg.norm(step.dv):,.1f} m/s  {step.note}" #type: ignore
                )
        achieved = self.transfer.achieved
        if achieved.get("apoapsis"):
            lines.append(
                f"  Resulting orbit: pe={achieved['periapsis']:,.0f}m ap={achieved['apoapsis']:,.0f}m "
                f"i={achieved['inclination_deg']:.1f} deg around {achieved['parent']}"
            )
        for warning in self.transfer.warnings + self.warnings:
            lines.append(f"  [Warning] {warning}")
        return lines


def _orbit_suggestions(body, spacecraft, library: PresetLibrary) -> str:
    presets = orbit_presets(body, spacecraft, library)
    keys = [p.key for p in presets]
    if not keys:
        return ""
    return "  Known orbits: " + ", ".join(keys)


def _unknown_orbit_quip(role: str, key: str, body, spacecraft, library: PresetLibrary) -> str:
    quips = [
        f"Unknown {role} orbit '{key}'. Did the KSC rename it when we weren't looking?",
        f"No '{key}' in the orbit catalogue for {body.name}. The navigator is confused.",
        f"'{key}' isn't a known {role} orbit around {body.name}. Try 'current' or run 'go' with no args for the picker.",
    ]
    idx = (sum(ord(c) for c in key) + len(role)) % len(quips)
    return quips[idx] + _orbit_suggestions(body, spacecraft, library)


def _resolve_pair(
    body, spacecraft, library: PresetLibrary, start_key: str, end_key: str
) -> tuple[Preset, OrbitSpec, Preset, OrbitSpec]:
    start_preset = find_preset(body, spacecraft, library, start_key)
    end_preset = find_preset(body, spacecraft, library, end_key)
    if start_preset is None:
        raise ValueError(_unknown_orbit_quip("start", start_key, body, spacecraft, library))
    if end_preset is None:
        raise ValueError(_unknown_orbit_quip("target", end_key, body, spacecraft, library))
    return (
        start_preset,
        start_preset.resolve(body, spacecraft),
        end_preset,
        end_preset.resolve(body, spacecraft),
    )


def apply_overrides(spec: OrbitSpec, body, overrides: dict[str, str]) -> tuple[OrbitSpec, list[str]]:
    """Apply per-use ``key=value`` overrides to an OrbitSpec without saving them.

    Supported keys: ``peri_alt``, ``apo_alt`` (meters above the body surface),
    ``incl`` / ``inclination``, ``arg_p``, ``lan`` (degrees).
    """
    import copy

    out = copy.copy(spec)
    warnings: list[str] = []
    for key, raw in overrides.items():
        key_l = key.strip().lower()
        try:
            value = float(raw)
        except (TypeError, ValueError):
            raise ValueError(f"Override '{key}' expects a number, got '{raw}'.")
        if key_l in ("peri_alt", "pe_alt"):
            out.periapsis = body.radius + value
        elif key_l in ("apo_alt", "ap_alt"):
            out.apoapsis = body.radius + value
        elif key_l in ("incl", "inclination", "inc"):
            out.inclination = np.radians(value)
        elif key_l in ("arg_p", "argp", "argument"):
            out.arg_periapsis = np.radians(value)
        elif key_l in ("lan", "raan", "node"):
            out.lan = np.radians(value)
        else:
            raise ValueError(
                f"Unknown override '{key}'. Supported: peri_alt, apo_alt, incl, arg_p, lan"
            )
    if out.apoapsis < out.periapsis:
        warnings.append(
            f"Override set apoapsis ({out.apoapsis:,.0f}m) below periapsis "
            f"({out.periapsis:,.0f}m); swapped them."
        )
        out.periapsis, out.apoapsis = out.apoapsis, out.periapsis
    return out, warnings


def go_plan(
    ticket: Ticket,
    start_key: str,
    end_key: str,
    ut: float | None = None,
    overrides: dict[str, str] | None = None,
) -> GoPlan:
    """Plan a transfer for ``ticket``'s spacecraft without scheduling anything."""
    spacecraft = ticket.spacecraft
    body = spacecraft.parent
    library = preset_library()
    start_preset, start_spec, end_preset, end_spec = _resolve_pair(body, spacecraft, library, start_key, end_key)

    override_warnings: list[str] = []
    if overrides:
        end_spec, override_warnings = apply_overrides(end_spec, body, overrides)

    start_ut = ticket.cursor_ut if ut is None else float(ut)
    transfer = plan_transfer(spacecraft, body, start_spec, end_spec, start_ut)

    warnings = list(override_warnings)
    same = transfer.burn_count == 0
    if same:
        warnings.append("Start and target orbits match; no maneuver needed.")

    events: list[TicketEvent] = []
    for step in transfer.steps:
        if step.kind == "coast":
            events.append(CoastEvent(step.start, step.end))
        else:
            events.append(ManeuverEvent(step.node)) #type: ignore

    plan = GoPlan(
        body_name=body.name,
        start_label=start_spec.describe(body),
        end_label=end_spec.describe(body),
        start_spec=start_spec,
        end_spec=end_spec,
        transfer=transfer,
        events=events,
        warnings=warnings,
    )
    return plan


def apply_go(
    ticket: Ticket,
    start_key: str,
    end_key: str,
    ut: float | None = None,
    mode: str = "gradual",
    overrides: dict[str, str] | None = None,
) -> GoPlan:
    """Plan a transfer and commit it to the ticket.

    ``mode="gradual"`` schedules coast + maneuver events to be executed later
    with ``advance``. ``mode="instant"`` also runs the ticket immediately so
    the vessel is already in the target orbit at the next prompt. ``overrides``
    are per-use ``key=value`` tweaks applied to the target orbit (not saved).
    """
    plan = go_plan(ticket, start_key, end_key, ut, overrides)

    if plan.burn_count > 0:
        for event in plan.events:
            ticket.add_event(event)

    if mode == "instant" and plan.events:
        ticket.advance_to(plan.end_ut)
    elif mode not in ("gradual", "instant"):
        raise ValueError(f"Unknown apply mode '{mode}'. Use 'gradual' or 'instant'.")

    return plan


@dataclass
class InterplanetaryPlan:
    """A planned *direct* interplanetary transfer (escape + heliocentric leg).

    Built by :func:`go_plan_interplanetary`. ``escape_plan`` is the burn to leave
    the origin body's SOI, ``helio_plan`` the transfer in the parent's frame
    (``None`` if only escape is needed), ``soi_exit_ut`` the time the vessel
    leaves the SOI and ``grandparent_name`` the new reference body. ``events`` is
    the full scheduled timeline; ``total_dv``/``end_ut``/``burn_count`` summarise
    it.
    """

    escape_plan: GoPlan
    helio_plan: GoPlan | None
    soi_exit_ut: float
    grandparent_name: str
    events: list[TicketEvent] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def total_dv(self) -> float:
        return self.escape_plan.total_dv + (self.helio_plan.total_dv if self.helio_plan else 0.0)

    @property
    def end_ut(self) -> float:
        if self.helio_plan:
            return self.helio_plan.end_ut
        return self.soi_exit_ut

    @property
    def burn_count(self) -> int:
        return self.escape_plan.burn_count + (self.helio_plan.burn_count if self.helio_plan else 0)

    def summary_lines(self) -> list[str]:
        lines = [
            f"  Escape from {self.escape_plan.body_name}  ->  SOI exit  ->  {self.grandparent_name} frame",
            f"  Total delta-v: {self.total_dv:,.1f} m/s   Burns: {self.burn_count}",
            f"    [{self.escape_plan.body_name}] escape burn  dv={self.escape_plan.total_dv:,.1f} m/s",
        ]
        lines.append(f"    coast to SOI exit @ {self.soi_exit_ut:,.0f}s")
        lines.append(f"    refbody -> {self.grandparent_name}")
        if self.helio_plan and self.helio_plan.burn_count > 0:
            for step in self.helio_plan.transfer.steps:
                if step.kind == "coast":
                    lines.append(f"    coast  {step.start:,.1f}s -> {step.end:,.1f}s  {step.note}")
                else:
                    lines.append(f"    burn   @ {step.ut:,.1f}s  dv={np.linalg.norm(step.dv):,.1f} m/s  {step.note}")
        for w in self.warnings:
            lines.append(f"  [Warning] {w}")
        return lines


def go_plan_interplanetary(
    ticket: Ticket,
    target_key: str,
    ut: float | None = None,
    overrides: dict[str, str] | None = None,
) -> InterplanetaryPlan:
    """Plan a *direct* interplanetary transfer: escape, coast to SOI exit, then transfer in the parent frame.

    No events are scheduled on the ticket — this just builds the plan so it can
    be inspected (and committed with :func:`apply_go_interplanetary`).
    """
    from CLI_use.ticket import ReferenceBodyEvent
    from basic_systems.orbit_pred import Spacecraft as _SC
    from basic_systems.go_planner import _spec_from_orbit as _spec_from

    spacecraft = ticket.spacecraft
    body = spacecraft.parent
    grandparent = body.orbit.parent
    if grandparent is None:
        raise ValueError(
            f"{body.name} has no parent body to escape to — it's the top of the food chain. "
            f"There's no 'interplanetary' from here, just 'interstellar' (unsupported)."
        )

    start_ut = ticket.cursor_ut if ut is None else float(ut)

    escape_plan = go_plan(ticket, "current", "escape", ut)
    if escape_plan.burn_count == 0:
        raise ValueError(
            "You're already on an escape trajectory — congratulations, you're space junk in a "
            "hurry. No escape burn needed (try 'go -i <target>' to plan the heliocentric leg)."
        )

    from basic_systems.orbit_pred import Spacecraft as _SC_sim
    _burn_node = escape_plan.transfer.steps[-1].node
    _burn_pos_abs, _burn_vel_abs = spacecraft.state_at(_burn_node.ut)
    _body_pos = body.get_absolute_pos_at_ut(_burn_node.ut)
    _body_vel = body.get_absolute_vel_at_ut(_burn_node.ut)
    _sim = _SC_sim(
        name="_soi_sim",
        r0=_burn_pos_abs - _body_pos,
        v0=_burn_vel_abs - _body_vel + _burn_node.delta_v_vector,
        t0=_burn_node.ut,
        parent=body,
        dry_mass=spacecraft.dry_mass,
        wet_mass=spacecraft.mass,
        hull_mesh=None,
    )

    soi_exit_ut = _sim.time_to_soi_exit(_burn_node.ut)
    if soi_exit_ut is None or soi_exit_ut <= start_ut:
        raise ValueError("Cannot compute SOI exit time.")

    library = preset_library()
    target_preset = find_preset(grandparent, spacecraft, library, target_key)
    if target_preset is None:
        raise ValueError(
            f"Unknown target '{target_key}' in {grandparent.name}'s frame. "
            f"Use moon:<identifier> for moons (e.g. moon:3 for Kerbin). "
            f"The deep-space navigation charts don't list '{target_key}'."
        )

    abs_pos, abs_vel = _sim.state_at(soi_exit_ut)
    if grandparent.orbit.parent is not None:
        gp_pos = grandparent.get_absolute_pos_at_ut(soi_exit_ut)
        gp_vel = grandparent.get_absolute_vel_at_ut(soi_exit_ut)
    else:
        gp_pos = np.zeros(3)
        gp_vel = np.zeros(3)
    rel_pos = abs_pos - gp_pos
    rel_vel = abs_vel - gp_vel

    temp_sc = _SC(
        name="_interplanetary_tmp",
        r0=rel_pos,
        v0=rel_vel,
        t0=soi_exit_ut,
        parent=grandparent,
        dry_mass=spacecraft.dry_mass,
        wet_mass=spacecraft.mass,
        hull_mesh=None,
    )

    if overrides:
        end_spec, _ = apply_overrides(target_preset.resolve(grandparent, temp_sc), grandparent, overrides)
    else:
        end_spec = target_preset.resolve(grandparent, temp_sc)

    start_spec = _spec_from("Heliocentric", temp_sc.orbit)
    helio_transfer = plan_transfer(temp_sc, grandparent, start_spec, end_spec, soi_exit_ut)

    events: list[TicketEvent] = []
    for step in escape_plan.events:
        events.append(step)
    events.append(CoastEvent(start_ut, soi_exit_ut))
    events.append(ReferenceBodyEvent(soi_exit_ut, grandparent))
    for step in helio_transfer.steps:
        if step.kind == "coast":
            events.append(CoastEvent(step.start, step.end))
        else:
            events.append(ManeuverEvent(step.node))

    helio_plan = GoPlan(
        body_name=grandparent.name,
        start_label=start_spec.describe(grandparent),
        end_label=end_spec.describe(grandparent),
        start_spec=start_spec,
        end_spec=end_spec,
        transfer=helio_transfer,
    )

    return InterplanetaryPlan(
        escape_plan=escape_plan,
        helio_plan=helio_plan,
        soi_exit_ut=soi_exit_ut,
        grandparent_name=grandparent.name,
        events=events,
    )


def apply_go_interplanetary(
    ticket: Ticket,
    target_key: str,
    ut: float | None = None,
    mode: str = "gradual",
    overrides: dict[str, str] | None = None,
    plan: InterplanetaryPlan | None = None,
) -> InterplanetaryPlan:
    """Plan a *direct* interplanetary transfer and commit it to the ticket.

    ``mode="gradual"`` schedules escape + coast + SOI-exit + heliocentric
    transfer events to run later with ``advance``. ``mode="instant"`` also runs
    the ticket immediately. ``plan`` may be a pre-built plan from
    :func:`go_plan_interplanetary` to avoid re-planning.
    """
    if plan is None:
        plan = go_plan_interplanetary(ticket, target_key, ut, overrides)

    events = list(plan.events)
    if mode == "instant":
        for event in events:
            ticket.add_event(event)
        ticket.advance_to(plan.end_ut)
    elif mode == "gradual":
        for event in events:
            ticket.add_event(event)
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'gradual' or 'instant'.")

    return plan
