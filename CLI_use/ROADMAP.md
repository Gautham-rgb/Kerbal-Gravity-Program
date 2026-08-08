# REPL and renderer delivery plan

## Goal

Provide a dependable, scriptable REPL for planning missions, then make the
native renderer the visual companion to that same simulation state.  The CLI
must never create a separate model from the renderer: vessels and tickets are
shared objects. This document is a design contract only; implementation remains
with the project owner.

## REPL boundaries

Keep the REPL thin. It should parse input and present output, while a session
service owns loaded system data, vessels, the selected UT, and a ticket
registry. Ticket classes remain the domain model; renderer classes only receive
read-only scene state plus a time-update callback. This prevents CLI commands,
scheduled events, and frame updates from advancing time in competing ways.

The startup command should accept a system file and optional mission file. A
mission file stores the selected UT, vessels, engines, tickets, events, and
renderer preferences; planet definitions remain in the existing system JSON.

## Command contract

| Area | Commands |
| --- | --- |
| System | `system info`, `system bodies`, `system validate` |
| Vessels | `vessel list`, `vessel create`, `vessel delete`, `vessel engine ...` |
| Tickets | `ticket list/create/show/delete/clear/run` |
| Events | `ticket event list/add/remove` for `coast`, `maneuver`, `propagation`, and `rkf45` |
| Visuals | `render [ticket]`, `time <UT>`, `exit` |

Every mutation is confirmed in concise human-readable output; invalid command
syntax, body/vessel names, time ordering, and failed burns produce clear errors
without ending the session.  Standard `help` and tab completion make the REPL
discoverable from a blank prompt.

### Ticket actions and semantics

| Action | Expected behaviour |
| --- | --- |
| Create | Bind a named ticket to exactly one vessel and initialise its cursor at the vessel epoch. |
| Inspect/list | Show ID, vessel, cursor, ordered events, time range, and execution state. |
| Add event | Reject a time range before the cursor; order events by UT with stable ordering for ties. |
| Remove/clear | Only remove unexecuted events; require explicit reset or a new ticket for historical edits. |
| Run/advance | Advance monotonically to a target UT; process every due event, including events whose start was passed between frames. |
| Reset | Restore a saved vessel snapshot and set every event back to pending. |
| Delete | Remove only the ticket, never its vessel. |

All four current event kinds are part of the contract:

- `coast`: advances time without modifying the analytic orbit.
- `maneuver`: applies a node once, reporting a failed burn if no usable engine
  or fuel is available.
- `propagation`: delegates to a named propagator, initially the two-body
  analytic propagator.
- `rkf45`: performs numerical n-body integration with an explicit tolerance.

Before the REPL consumes them, correct the ticket interface: `active`,
`finished`, and `update` must be real `TicketEvent` methods (not nested inside
its constructor); every event should share `update(spacecraft, ut)`; and ticket
advancement must consider due events rather than only those currently active.
These are integration prerequisites, not changes made by this plan.

## Renderer quality bar

The renderer remains a native PyVista scene and receives the same spacecraft
objects used by tickets.  The first integration delivers accurate current-time
positions, trajectories, body labels, focus controls, timeline controls, and
ticket-driven updates.  The next visual pass should add maneuver-node markers,
burn/trajectory colouring, a compact HUD, and screenshot/export support.  Keep
visual scaling isolated from physical coordinates, as it is today, so the
render never changes simulation outcomes.

### Visual design and performance

The finished renderer should communicate both scale and intent at a glance:

- Dark, high-contrast scene with physically named bodies, restrained orbit
  lines, and distinct vessel/trajectory colours.
- A compact HUD for UT, time warp, focus target, velocity, altitude, and active
  ticket/event. Units should switch cleanly between metres/kilometres and
  seconds/minutes/days.
- Maneuver nodes visible on the trajectory with UT, total delta-v, and
  prograde/normal/radial breakdown on selection.
- A timeline with event ticks, scrub preview that does not mutate the mission,
  play/pause, warp controls, body focus, and screenshot export.
- Cached orbit polylines and label placement updates; recompute physical state
  only when UT or a ticket changes. Keep display scaling and moon exaggeration
  as renderer-only transforms.

Renderer integration should use one authoritative `advance_to(ut)` call before
the scene positions update. A scrub interaction uses a preview state or an
immutable snapshot, while live playback uses the ticket cursor. This is the
critical safeguard against executing a maneuver repeatedly when moving the
slider backwards and forwards.

## Delivery order

1. Add serialisable mission/session data types around the implemented vessel and
   ticket model.
2. Build the command parser, help, completion, and non-GUI ticket/vessel flows.
3. Connect one session instance to renderer launch and live time advancement.
4. Add node markers, the HUD, timeline event ticks, camera polish, and export.
5. Add end-to-end mission fixtures and performance checks on the supplied solar
   system.

## Verification

Automated smoke checks cover command parsing, ticket creation, every event type,
and non-GUI rendering construction.  Manual acceptance is: create a vessel,
give it an engine, schedule a maneuver and propagation, run the ticket, and
open `render` to see the same vessel at the selected UT.
