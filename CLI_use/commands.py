"""Declarative registry for constructing TicketEvents from CLI input.

Adding a new ticket event type to the REPL requires two things:

1. The event class already exists in ``CLI_use.ticket``.
2. A single entry in :data:`TICKET_COMMANDS` describing its arguments and a
   small builder function.

The REPL then automatically gets prompting, ``key=value`` parsing, help text
and error handling for that event type - no new handler function needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from basic_systems.orbit_pred import ManeuverPlanner
from CLI_use.ticket import (
    AttitudeEvent,
    CheckpointEvent,
    CoastEvent,
    DockEvent,
    EngineControlEvent,
    FiniteBurnEvent,
    ManeuverEvent,
    PropagationEvent,
    RKF45Event,
    RCSTranslationEvent,
    ReferenceBodyEvent,
    ResourceTransferEvent,
    ScienceEvent,
    StageEvent,
    SurfaceOperationEvent,
    UndockEvent,
    Ticket,
    TicketEvent,
)

# Type of a single command argument: one of "float", "int", "str", "bool",
# "vec3", "vec4", "body", "vessel" or "int_list".
ArgKind = str


@dataclass
class ArgSpec:
    """Specification of one argument accepted by a ticket command.

    ``name`` is the CLI key, ``kind`` an :data:`ArgKind`, ``required`` whether it
    must be supplied, ``default`` the value used otherwise, ``prompt`` the REPL
    prompt text and ``help`` a short description shown to the user.
    """

    name: str
    kind: ArgKind = "float"
    required: bool = True
    default: Any = None
    prompt: str = ""
    help: str = ""


@dataclass
class TicketCommand:
    """Declarative description of one REPL ticket command.

    ``name`` is the command word (e.g. ``"maneuver"``), ``summary`` a one-line
    description, ``args`` the list of :class:`ArgSpec` it accepts and ``build``
    a callable that, given a :class:`~CLI_use.ticket.Ticket` and the parsed
    argument values, returns the corresponding :class:`TicketEvent`. Registered
    in :data:`TICKET_COMMANDS`.
    """

    name: str
    summary: str
    args: list[ArgSpec]
    build: Callable[..., TicketEvent]

    def usage(self) -> str:
        parts = [f"add {self.name}"]
        for arg in self.args:
            label = (
                f"<{arg.name}>"
                if arg.required
                else f"[{arg.name}={arg.default}]"
            )
            parts.append(label)
        return " ".join(parts)


def _f(value: Any) -> float:
    return float(value)


def _vec(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=float)


def _int_list(value: Any) -> list[int]:
    if isinstance(value, list):
        return [int(v) for v in value]
    return [int(v) for v in str(value).replace(" ", "").split(",") if v]


# --- Builders ---------------------------------------------------------------

def _coast(ticket: Ticket, start=0.0, end=0.0, **_):
    return CoastEvent(_f(start), _f(end))


def _propagate(ticket: Ticket, start=0.0, end=0.0, **_):
    return PropagationEvent(_f(start), _f(end))


def _rkf45(ticket: Ticket, start=0.0, end=0.0, tol=1e-6, **_):
    return RKF45Event(_f(start), _f(end), _f(tol))


_MANEUVER_PLANNER = ManeuverPlanner()


def _maneuver(ticket: Ticket, ut=0.0, dv=None, **_):
    sc = ticket.spacecraft
    pos, vel = sc.state_at(_f(ut))
    node = _MANEUVER_PLANNER.calculate_maneuver(_f(ut), vel, vel + _vec(dv), pos)
    return ManeuverEvent(node)


def _finite_burn(ticket: Ticket, start=0.0, end=0.0, direction=None, pressure=0.0, **_):
    return FiniteBurnEvent(_f(start), _f(end), _vec(direction), _f(pressure))


def _engine_control(ticket: Ticket, ut=0.0, active=None, throttle=None, indices=None, **_):
    active = None if active is None else bool(active)
    throttle = None if throttle is None else _f(throttle)
    return EngineControlEvent(_f(ut), active, throttle, _int_list(indices) if indices else None)


def _attitude(ticket: Ticket, ut=0.0, mode="inertial", attitude=None, target=None, **_):
    quat = None if attitude is None else _vec(attitude)
    return AttitudeEvent(_f(ut), str(mode), quat, target)


def _rcs(ticket: Ticket, ut=0.0, dv=None, isp=260.0, **_):
    return RCSTranslationEvent(_f(ut), _vec(dv), _f(isp))


def _stage(ticket: Ticket, ut=0.0, parts="", **_):
    part_ids = [p.strip() for p in str(parts).split(",") if p.strip()]
    return StageEvent(_f(ut), part_ids)


def _resource_transfer(ticket: Ticket, ut=0.0, target=None, resource="LiquidFuel", amount=0.0, **_):
    return ResourceTransferEvent(_f(ut), target, str(resource), _f(amount)) #type: ignore


def _dock(ticket: Ticket, ut=0.0, target=None, **_):
    return DockEvent(_f(ut), target) #type: ignore


def _undock(ticket: Ticket, ut=0.0, parts="", name="Spawned", **_):
    part_ids = [p.strip() for p in str(parts).split(",") if p.strip()]
    return UndockEvent(_f(ut), part_ids, str(name))


def _reference_body(ticket: Ticket, ut=0.0, body=None, **_):
    return ReferenceBodyEvent(_f(ut), body) #type: ignore


def _surface_op(ticket: Ticket, ut=0.0, operation="land", body=None, **_):
    return SurfaceOperationEvent(_f(ut), str(operation), body)


def _science(ticket: Ticket, ut=0.0, operation="collect", amount=None, **_):
    op = str(operation)
    if op == "collect" and amount is None:
        amount = 100.0
    return ScienceEvent(_f(ut), op, None if amount is None else _f(amount))


def _checkpoint(ticket: Ticket, ut=0.0, name="checkpoint", **_):
    return CheckpointEvent(_f(ut), str(name))


# --- Registry ---------------------------------------------------------------

# Registry of every ticket command the REPL can build. Maps the command word
# (e.g. "coast", "maneuver", "refbody") to its TicketCommand description. Add a
# new entry here (plus the event class in CLI_use.ticket) to expose a new
# scheduled event to the REPL; prompting, key=value parsing and help are
# generated automatically.
TICKET_COMMANDS: dict[str, TicketCommand] = {
    "coast": TicketCommand(
        name="coast",
        summary="Coast without changing state until end UT.",
        args=[
            ArgSpec("start", "float", True, 0.0, "Start UT (s) — when the coast begins"),
            ArgSpec("end", "float", True, 0.0, "End UT (s) — when the coast ends"),
        ],
        build=_coast,
    ),
    "propagate": TicketCommand(
        name="propagate",
        summary="Keplerian two-body propagation from start to end UT.",
        args=[
            ArgSpec("start", "float", True, 0.0, "Start UT (s)"),
            ArgSpec("end", "float", True, 0.0, "End UT (s)"),
        ],
        build=_propagate,
    ),
    "rkf45": TicketCommand(
        name="rkf45",
        summary="Full n-body propagation using the RKF45 integrator.",
        args=[
            ArgSpec("start", "float", True, 0.0, "Start UT (s)"),
            ArgSpec("end", "float", True, 0.0, "End UT (s)"),
            ArgSpec("tol", "float", False, 1e-6, "Integration tolerance"),
        ],
        build=_rkf45,
    ),
    "maneuver": TicketCommand(
        name="maneuver",
        summary="Instantaneous delta-v impulse at a given UT.",
        args=[
            ArgSpec("ut", "float", True, 0.0, "Burn UT (s) — when the impulse is applied"),
            ArgSpec("dv", "vec3", True, None, "Delta-v vector in m/s, e.g. dv=10,0,0 (prograde/retrograde/normal/radial)"),
        ],
        build=_maneuver,
    ),
    "burn": TicketCommand(
        name="burn",
        summary="Finite constant-direction burn from start to end UT.",
        args=[
            ArgSpec("start", "float", True, 0.0, "Burn start UT (s)"),
            ArgSpec("end", "float", True, 0.0, "Burn end UT (s)"),
            ArgSpec("direction", "vec3", True, None, "Thrust direction as a 3-vector, e.g. direction=0,1,0"),
            ArgSpec("pressure", "float", False, 0.0, "Ambient pressure (Pa) — affects Isp"),
        ],
        build=_finite_burn,
    ),
    "engine": TicketCommand(
        name="engine",
        summary="Toggle engine(s) active/throttle at a UT.",
        args=[
            ArgSpec("ut", "float", True, 0.0, "UT (s) at which the engine state changes"),
            ArgSpec("active", "bool", False, None, "Turn engines on/off (y/n)"),
            ArgSpec("throttle", "float", False, None, "Throttle 0-1 (fraction of max thrust)"),
            ArgSpec("indices", "int_list", False, None, "Which engines (comma separated, e.g. 0,1); default all"),
        ],
        build=_engine_control,
    ),
    "attitude": TicketCommand(
        name="attitude",
        summary="Set guidance mode/attitude at a UT.",
        args=[
            ArgSpec("ut", "float", True, 0.0, "UT (s)"),
            ArgSpec("mode", "str", False, "inertial", "Guidance mode"),
            ArgSpec("attitude", "vec4", False, None, "Attitude quaternion"),
            ArgSpec("target", "str", False, None, "Optional target name"),
        ],
        build=_attitude,
    ),
    "rcs": TicketCommand(
        name="rcs",
        summary="Apply a small MonoPropellant translation impulse.",
        args=[
            ArgSpec("ut", "float", True, 0.0, "Impulse UT (s)"),
            ArgSpec("dv", "vec3", True, None, "Delta-v vector in m/s, e.g. dv=0,0,1"),
            ArgSpec("isp", "float", False, 260.0, "RCS specific impulse (s)"),
        ],
        build=_rcs,
    ),
    "stage": TicketCommand(
        name="stage",
        summary="Detach a comma-separated list of part identifiers.",
        args=[
            ArgSpec("ut", "float", True, 0.0, "UT (s)"),
            ArgSpec("parts", "str", True, None, "Part ids (comma separated)"),
        ],
        build=_stage,
    ),
    "transfer": TicketCommand(
        name="transfer",
        summary="Transfer a resource to another vessel.",
        args=[
            ArgSpec("ut", "float", True, 0.0, "UT (s)"),
            ArgSpec("target", "vessel", True, None, "Target vessel"),
            ArgSpec("resource", "str", False, "LiquidFuel", "Resource name"),
            ArgSpec("amount", "float", True, 0.0, "Amount to transfer"),
        ],
        build=_resource_transfer,
    ),
    "dock": TicketCommand(
        name="dock",
        summary="Dock with another vessel.",
        args=[
            ArgSpec("ut", "float", True, 0.0, "UT (s)"),
            ArgSpec("target", "vessel", True, None, "Target vessel"),
        ],
        build=_dock,
    ),
    "undock": TicketCommand(
        name="undock",
        summary="Spawn a new vessel from the given parts.",
        args=[
            ArgSpec("ut", "float", True, 0.0, "UT (s)"),
            ArgSpec("parts", "str", True, None, "Part ids (comma separated)"),
            ArgSpec("name", "str", False, "Spawned", "New vessel name"),
        ],
        build=_undock,
    ),
    "refbody": TicketCommand(
        name="refbody",
        summary="Change the reference body.",
        args=[
            ArgSpec("ut", "float", True, 0.0, "UT (s)"),
            ArgSpec("body", "body", True, None, "New reference body"),
        ],
        build=_reference_body,
    ),
    "surface": TicketCommand(
        name="surface",
        summary="Land on or launch from a body.",
        args=[
            ArgSpec("ut", "float", True, 0.0, "UT (s)"),
            ArgSpec("operation", "str", False, "land", "land or launch"),
            ArgSpec("body", "body", False, None, "Body (required for land)"),
        ],
        build=_surface_op,
    ),
    "science": TicketCommand(
        name="science",
        summary="Collect or transmit science data.",
        args=[
            ArgSpec("ut", "float", True, 0.0, "UT (s) at which the science action happens"),
            ArgSpec("operation", "str", False, "collect", "collect or transmit science data"),
            ArgSpec("amount", "float", False, None, "Amount to transmit (required for transmit)"),
        ],
        build=_science,
    ),
    "checkpoint": TicketCommand(
        name="checkpoint",
        summary="Record a vessel snapshot at a UT.",
        args=[
            ArgSpec("ut", "float", True, 0.0, "UT (s)"),
            ArgSpec("name", "str", False, "checkpoint", "Checkpoint name"),
        ],
        build=_checkpoint,
    ),
}
