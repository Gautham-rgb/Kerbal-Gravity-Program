# Kerbal Gravity Program

Kerbal Gravity Program (KGRP or KGrP) is a mission planning software for any and all space games or heck, even the real solar system.

## The Reason I made This

See, people in KSP make kind of dumb decisions (sorry Matt Lowne), like getting entire crafts to somewhere like Jool or Sarnus, just as refueling stations.
plus NASA (yup, the space agency) takes way too much time to make cool missions like the Voyagers or Cassini-Huygens (well, for one they have to perfectly make the spacecraft and find a window, but I am ignoring that). So I researched about how do people plan stuff like Grand Tours or something.

So, i came up to my computer and searched for "KSP planning software" and found random stuff like injection burn predictors, window finders, and other stuff, so I set out to make a tool to allow for anyone to plan cool stuff, and that is how KGrP was made.

here is an example of me importing the Real world system (`planets.json`) and making a trip (called a ticket).
[![asciicast](https://asciinema.org/a/fBPRPZsKrolGCy2I.svg)](https://asciinema.org/a/fBPRPZsKrolGCy2I)

---


---

## Installation

Requires Python 3.10+, and pip (python package manager) or uv (another package manager, but I recommend you to use pip instead of uv tools, reason will be said later).

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

KGrP is a REPL. The whole loop is five commands, typed at the `krgp >`
prompt (launch with `kgrp`):

```text
load system planets_ksp.json   
new ticket                     
go                             
advance <end-ut>               
render    # this can only work when you install the kgrp[render] package, which adds PyVista                     
```

`help` lists every command; `help <command>` (e.g. `help go`) explains one.
the loop is just: **load a system -> new ticket -> plan with `go` -> `advance` the
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
that orbits a star that is within the system.

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

### Time input

You rarely need raw seconds:

```text
advance 2d 5h
advance 21600         
time "Year 2, Day 100" 
go current preset:molniya incl=30   
```

Valid time units: `y` (year), `mo` (month), `d` (day), `h` (hour), `m`
(minute), `s` (second). Words like `days`/`hours` work too.

---

## Tips 

- **Use presets.** `new ticket` hands you a vessel
  with an engine and fuel already attached.
- **Let `go` do the math.** `go` is one of, if not the most useful command when you are planning something.
- **Use `advance` to walk time**, one coast/burn at a time. `time` jumps the
  cursor without simulating (handy for inspection).
- **Save your work.** `save mission` writes a single file with the system,
  ticket, and vessels; `load mission` brings it all back.
- **Stuck?** `help` and `help <command>` are always available, and `system
  validate` will tell you if your system has problems.
- **Use it via code** KGrP can be used via python code (just `import kgrp`)
---

## Some things to keep in mind about KGrP (both code and REPL)

1. for basic orbital mechanics, an RKF45 and some other basic stuff, type `import kgrp.basic_systems` at the top of your script
2. for a little complex stuff like a ticket or subclassing the REPL or adding a new command, type `import kgrp.CLI_use` at the top of your script
3. if you find any bugs or cool features you want to add, please use the `feedback -y` command, allowing me to see your message, the OS and the version of KGrP you are on

## Running the tests

```bash
python -m pytest
```

---

## Credits

Made by Gautham, and way too much coffee

As it turns out (unsurprisingly) that this needs way too much math, so I did use some AI to help me understand the math (for example: Runge - Kutta - Fehler (4)5 optimizers) but other than that, I did use AI to fix bugs like reference body bugs, rendering stuff and random math bugs that I didn't see coming, but i did use AI kind of sparingly (**"Kind of"**).

## License

GNU GPLv3 — see the `LICENSE` file. Go forth and transfer. 
