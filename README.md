# Kerbal Gravity Program (KGRP)

KGRP is a **mission planner for Kerbal Space Program** that runs *outside* the
game. You define a spacecraft, stack events (coast, maneuver, burn, engine
toggle, science, and more) on a timeline, and KGRP simulates them with real
orbital mechanics. Plan in KGRP, then fly the plan in KSP.

It is built for KSP players who want to work out transfers, delta-v budgets,
and timelines before launching — and it is flexible enough that you can model
arbitrary spacecraft and solar systems (KSP stock, OPM, or your own).

> Status: pre-release (v0.1.0). The core works, but some math is still coarse

---

## Features

- **Round-trip mission planner** — define a vessel, schedule a timeline of
  events, and step the sim forward with `advance`.
- **`go` transfer planner** — pick a start and end orbit and KGRP computes the
  burns, coast phases, and total delta-v for you.
- **Editable orbit presets** — low orbit, synchronous, escape, custom orbits.
- **Full event timeline** — coast, propagate, rkf45, maneuver, burn, engine
  control, attitude, rcs, stage, resource transfer, dock, undock, reference-body
  change, surface (land/launch), science, checkpoint.
- **3D renderer** — a PyVista/Qt window to watch the mission unfold.
- **Save/load** — systems, tickets, and whole missions as JSON.

---

## Installation

Requires Python 3.10+.

```bash
# From the terminal (using pip):
pip install kgrp

# Or using uv
uv tool install kgrp

# This installs the `kgrp` command (entry point -> CLI_use.CLI:main).
```

Verify it works:

```bash
kgrp
```

You should see the KGRP prompt (`krgp >`). Type `help` and press Enter.

---

## Quick start: your first mission

KGRP is a REPL. Everything below is typed at the `krgp >` prompt.

```text
# 1. Load the stock KSP system (or `new system` to build your own)
load system planets_ksp.json

# 2. Create a spacecraft + its first ticket (a wizard walks you through it)
new ticket

# 3. Plan a transfer to the Mun with the `go` wizard
go

# 4. KGRP shows the plan, then asks how to apply it.
#    Choose "gradual" to schedule the burns, then run the timeline:
advance <end-ut shown by go>

# 5. Watch it happen
render
```

That is the whole loop: **load a system -> new ticket -> plan with `go` ->
`advance` the timeline -> `render`**.

### A concrete example

```text
krgp > load system planets_ksp.json
krgp > new ticket
# ... answer the wizard prompts (name, body = Kerbin, defaults are fine) ...
krgp > go current moon:3a
Transfer plan:
  Current (700,000m circular, 0.0 deg)  ->  Mun orbit (11,400,000m circular, ...)  (about Kerbin)
  Burns: 2   Total delta-v: 1,112.7 m/s   Duration: 27,130.1s
    burn   @ 0.0s  dv=764.2 m/s  Raise apoapsis to transfer
    coast  0.0s -> 27,130.1s  Coast to apoapsis
    burn   @ 27,130.1s  dv=348.5 m/s  Circularize / match target
krgp > advance 27130
# vessel is now on the transfer; advance further to reach the Mun
krgp > render
```

---

## Command overview

Type `help` for the full list, and `help <command>` (e.g. `help go`,
`help add`) for details. The headline commands:

| Command | What it does |
| --- | --- |
| `load` / `save` | Load/save a `system`, `ticket`, or whole `mission` from JSON. |
| `new` | Creation wizards for a `system` or a `ticket`. |
| `status` | Show the active system and ticket. |
| `tree` | Print the body hierarchy (`-v` for orbital elements). |
| `system` | Inspect the loaded system (`info`, `bodies`, `validate`). |
| `vessel` | Manage vessels (`list`, `create`, `delete`, `engine`). |
| `edit` | Tweak a body or vessel's fields (mass, orbit elements, ...). |
| `add` | Schedule a ticket event (`add <type> [key=value ...]`). |
| `events` / `remove` | List / remove scheduled events. |
| `advance` | Run the timeline forward to a UT (`advance 2d 5h`). |
| `time` | Scrub the timeline to a UT without simulating. |
| `reset` | Restore the ticket to its initial state. |
| `go` | Plan a transfer between two orbits. |
| `go_i` | Interplanetary transfer (escape + heliocentric transfer). |
| `preset` | Manage editable orbit presets. |
| `render` | Open the 3D renderer window. |
| `help` / `clear` / `exit` | Help, clear screen, quit. |

### Time input is human-friendly

You rarely need raw seconds:

```text
advance 2d 5h          # a duration, added to the current UT
advance 21600          # or a raw UT in seconds
time "Year 2, Day 100" # a calendar timestamp
go current preset:molniya incl=30   # per-use orbit overrides
```

Valid time units: `y` (year), `mo` (month), `d` (day), `h` (hour), `m`
(minute), `s` (second). Words like `days`/`hours` work too.

---

## Tips for new players

- **Start from a preset, not a blank page.** `new ticket` hands you a vessel
  with an engine and fuel already attached — you do not start with a
  paperweight.
- **Let `go` do the math.** Instead of hand-computing burns, run `go` and read
  the delta-v budget it prints.
- **Use `advance` to walk time**, one coast/burn at a time. `time` jumps the
  cursor without simulating (handy for inspection).
- **Save your work.** `save mission` writes a single file with the system,
  ticket, and vessels; `load mission` brings it all back.
- **Stuck?** `help` and `help <command>` are always available, and `system
  validate` will tell you if your system has problems.

---

## Project layout

- `CLI_use/` — the REPL, command parsing, completion, and the `go` planner.
- `basic_systems/` — physics: bodies, orbits, integrators, and the renderer.
- `planets_ksp.json` / `planets_ksp_opm.json` — stock and OPM system data.
- `tests/` — pytest suite (`python -m pytest`).

## Running the tests

```bash
python -m pytest
```

## License

Pre-release; see the repository for license terms.
