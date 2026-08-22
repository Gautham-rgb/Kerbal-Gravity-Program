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

## Demo

Watch KGRP plan and fly a mission right in the terminal:

[![asciicast](https://asciinema.org/a/fBPRPZsKrolGCy2I.svg)](https://asciinema.org/a/fBPRPZsKrolGCy2I)

On sites that allow raw HTML, embed the player directly:

```html
<script id="asciinema-player-fBPRPZsKrolGCy2I" src="https://asciinema.org/a/fBPRPZsKrolGCy2I.js" async></script>
```

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

## Quickstart

KGRP is a REPL. The whole loop is five commands, typed at the `krgp >`
prompt (launch with `kgrp`):

```text
load system planets_ksp.json   # or: new system  (build your own)
new ticket                     # wizard creates a vessel + first ticket
go                             # pick two orbits; KGRP plans the burns
advance <end-ut>               # run the timeline (UT is printed by go)
render                         # optional 3D view
```

`help` lists every command; `help <command>` (e.g. `help go`) explains one.
That is it: **load a system -> new ticket -> plan with `go` -> `advance` the
timeline -> `render`**.

---

## Workflow example

### 1. A local transfer (Kerbin -> Mun)

```text
krgp > load system planets_ksp.json
krgp > new ticket
# ... answer the wizard prompts (name, body = Kerbin, defaults are fine) ...
krgp > go current moon:3a
Transfer plan:
  Current (~800,000m circular, 0.0 deg)  ->  Mun orbit (12,000,000m circular)  (about Kerbin)
  Burns: 2   Total delta-v: 1,126.6 m/s   Duration: 27,066.6s
    burn   @ 0.0s  dv=775.9 m/s  Raise apoapsis to transfer
    coast  0.0s -> 27,066.6s  Coast to apoapsis
    burn   @ 27,066.6s  dv=350.7 m/s  Circularize / match target
krgp > advance 27067
# vessel is now on the transfer; advance further to reach the Mun
krgp > render
```

`go` takes `<start-orbit> <end-orbit>` plus optional `key=value` overrides
(`peri_alt=`, `apo_alt=`, `incl=`, `arg_p=`, `lan=`). Or run `go` with no
arguments to pick both orbits from a menu.

### 2. An interplanetary transfer (Kerbin -> Duna)

Use `go -i <target>` (alias `go_i <target>`). It plans the escape burn, the
heliocentric leg, and the arrival burn in one shot:

```text
krgp > go_i duna
Interplanetary transfer plan (Kerbin -> Duna):
  Escape from Kerbin -> SOI exit -> Kerbol frame
  Total delta-v: 2,583.5 m/s   Burns: 3
    [Kerbin] escape burn  dv=870.3 m/s
    coast to SOI exit @ 196,411s
    refbody -> Kerbol
    burn   @ 1,994,868.8s  dv=946.1 m/s  Raise apoapsis to transfer
    coast  1,994,868.8s -> 8,719,637.1s
    burn   @ 8,719,637.1s  dv=767.2 m/s  Circularize / match target
krgp > advance 8719637    # or 'advance 8.7M' if your fingers object
krgp > render
```

Override the arrival orbit with the same `key=value` overrides, e.g.
`go_i duna peri_alt=200000 apo_alt=200000`. `go -i` / `go_i` work from any body
that orbits a star (stock or OPM).

### 3. Save the mission and come back

```text
krgp > save mission duna_trip.json
krgp > exit
# later:
kgrp
krgp > load mission duna_trip.json
```

`save mission` writes one file with the system, ticket, and vessels; `load
mission` brings it all back. You can also `save system` / `load system` and
`save ticket` / `load ticket` separately.

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
- `planets.json` - Actual Solar System data
- `tests/` — pytest suite (`python -m pytest`).

## Running the tests

```bash
python -m pytest
```

---

## Credits

Built by Gautham, powered by an unreasonable amount of Kerbal optimism.

- **Jebediah Kerman** — chief test pilot. Assumes every plan works until
  physically proven otherwise (and sometimes after).
- **Bill & Bob** — held the ladder. Reviewed nothing.
- **The math** — probably fine. Mostly. We checked it, like, twice.
- **The deep-space navigation charts** — still don't list your spelling
  mistakes, but they do list Duna now.
- **You** — for reading this far instead of just launching and hoping.

KGRP plans the mission. Flying it is still gloriously your problem.

## License

GNU GPLv3 — see the `LICENSE` file. Go forth and transfer.

