from __future__ import annotations
import os, asyncio
import shlex
from pathlib import Path
import numpy as np
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.styles import Style
import prompt_toolkit
import subprocess

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from basic_systems import __version__
from basic_systems.builder import System
from basic_systems.orbit_pred import Body, Spacecraft, Orbit, format_ut, parse_time_string
from CLI_use.commands import ArgSpec, TICKET_COMMANDS
from CLI_use.completion import REPLCompleter
from CLI_use.go import CustomPreset, apply_go, go_plan, orbit_presets, preset_library
from CLI_use.ticket import Ticket, TICKET_EXTENSION, _vessel_to_json, _vessel_from_json
from CLI_use.validation import validate_system

MISSION_EXTENSION = ".mission.json"

_CONSOLE = Console()

_COMMAND_CATEGORIES = {
    "Setup": ["load", "save", "new", "system", "vessel", "ticket"],
    "Timeline": ["add", "events", "remove", "advance", "time", "reset", "cursor"],
    "Planning": ["go", "go_i", "preset"],
    "Inspect": ["status", "tree", "render"],
    "Help": ["help", "clear", "exit"],
}

# A short, friendly one-liner shown after the welcome banner and on `help`.
_QUICKSTART = (
    "Quick start:  load system planets_ksp.json  ->  new ticket  ->  go  ->  advance <ut>  ->  render\n"
    "Type 'help <command>' (e.g. 'help go') for details on any command."
)


def _did_you_mean(word: str, candidates: list[str]) -> str | None:
    """Return the closest command name to ``word`` using simple edit distance."""
    word = word.lower()
    best = None
    best_dist = 99
    for cand in candidates:
        c = cand.lower()
        if c == word:
            continue
        # Levenshtein-ish: only consider small edits.
        if c.startswith(word) or word.startswith(c):
            dist = abs(len(c) - len(word))
        else:
            dist = sum(1 for a, b in zip(c, word)) + abs(len(c) - len(word))
            dist += sum(1 for ch in c[len(word):]) + sum(1 for ch in word[len(c):])
        # crude but good enough for short command names
        if dist < best_dist and dist <= 3:
            best_dist = dist
            best = cand
    return best

class REPL:
    def __init__(self, system_path: str | None = None, ticket_path: str | None = None) -> None:
        self.running = True
        self.system = None
        self.system_path = None
        self.current_ut = 0.0
        self.ticket = None
        self.vessels: dict[str, Spacecraft] = {}
        self.renderer_prefs = {"use_kerbal_time": True, "show_units_km": True}
        self.commands = {
            "exit": self.exit,
            "quit": self.exit,
            "help": self.help,
            "clear": self.clear,
            "status": self.status,
            "load": self.load,
            "save": self.save,
            "tree": self.tree,
            "new": self.new,
            "edit": self.edit,
            "add": self.add,
            "events": self.events,
            "remove": self.remove,
            "reset": self.reset,
            "advance": self.advance,
            "delete": self.delete,
            "system": self.system_cmd,
            "ticket": self.ticket_cmd,
            "vessel": self.vessel_cmd,
            "render": self.render_cmd,
            "time": self.time_cmd,
            "go": self.go_cmd,
            "go_i": self.go_interplanetary_cmd,
            "preset": self.preset_cmd,
            "cursor": self.cursor,
            "version": self.version,
            "feedback": self.feedback_cmd
        }

        self.command_help = {
            "exit": "Exit the program",
            "quit": "Exit the program",
            "help": "Show this help; 'help <command>' shows subcommand help",
            "clear": "Clear the terminal",
            "status": "Show the active system and ticket",
            "load": "load <system|ticket> <path>",
            "save": "save <system|ticket> [path]",
            "tree": "tree [body] [-v|--verbose] - show the body hierarchy",
            "new": "new <system|ticket> - creation wizard",
            "edit": "edit <body|vessel> <field> <value> - edit body/orbit/vessel values",
            "add": "add <event-type> [key=value ...] - schedule a ticket event",
            "events": "events - list scheduled ticket events",
            "remove": "remove <index> - remove a scheduled event",
            "reset": "reset - restore the ticket to its initial state",
            "advance": "advance <ut|duration> - run the ticket forward to UT (e.g. 'advance 2d 5h')",
            "delete": "delete <ticket|system> [file] - delete with confirmation",
            "system": "system <info|bodies|validate> [-v|--verbose] - inspect the loaded system",
            "ticket": "ticket <show|list|clear|delete> - manage tickets",
            "vessel": "vessel <list|create|delete|engine> - manage vessels",
            "render": "render - open 3D renderer window",
            "time": "time <ut|duration> - scrub the timeline to UT (e.g. 'time 1y 2mo 3d 4h 5m 6s')",
            "go": "go [<start-orbit> <end-orbit> [key=value ...]] - plan a transfer; add -i/--interplanetary <target> for a direct interplanetary A->B (target: body name like 'duna' or 'moon:<id>'; overrides: peri_alt, apo_alt, incl, arg_p, lan)",
            "go_i": "go_i <target> - alias for 'go -i <target>' (direct interplanetary transfer: escape + SOI exit + heliocentric leg; target: body name like 'duna' or 'moon:<id>')",
            "preset": "preset <list|new|edit|delete|save|load> - manage editable orbit presets",
            "cursor": "cursor <seconds>- give the current ut in human readable units or in seconds",
            "feedback": "feedback [-y|--yes] <message> - send feedback as a public GitHub issue (includes kgrp version + OS; -y to actually submit)"
        }

        self.subcommand_help = {
            "add": "add <event-type> [key=value ...]\n"
                   "  Types: coast, propagate, rkf45, maneuver, burn, engine, attitude,\n"
                   "         rcs, stage, transfer, dock, undock, refbody, surface, science, checkpoint\n"
                   "  Run 'help add' for each type's arguments.",
            "system": "system <info|bodies|validate> [-v|--verbose]\n"
                      "  info      - system name, body count, root, timeline\n"
                      "  bodies    - list bodies (brief, or -v for the full orbital table)\n"
                      "  validate  - check the system for errors",
            "ticket": "ticket <show|list|clear|delete> [file]\n"
                      "  show    - details of the active ticket\n"
                      "  list    - find ticket files on disk\n"
                      "  clear   - drop unexecuted events\n"
                      "  delete  - remove the active ticket",
            "vessel": "vessel <list|create|delete|engine> ...\n"
                      "  list                    - show loaded vessels\n"
                      "  create                  - vessel creation wizard\n"
                      "  delete <name|id>        - remove a vessel\n"
                      "  engine <name|id> [op]   - manage engines (add/on/off/throttle/del)",
            "go": "go [<start-orbit> <end-orbit> [key=value ...]]\n"
                  "  No arguments opens the orbit picker wizard.\n"
                  "  Overrides: peri_alt, apo_alt, incl, arg_p, lan\n"
                  "  Example: go current preset:molniya incl=30\n"
                  "  Interplanetary: go -i <target>  (alias go_i <target>)\n"
                  "    Direct A->B transfer: escape burn + coast to SOI exit + heliocentric leg.\n"
                  "    target is a body name (e.g. 'duna') or moon:<id> (e.g. 'moon:4').\n"
                  "    Example: go -i duna   |   go -i moon:4 peri_alt=200000",
            "preset": "preset <list|new|edit|delete|save|load> [args]\n"
                      "  list                - show all presets\n"
                      "  new                 - create a preset\n"
                      "  edit <key> [f] [v]  - edit a preset (or show it)\n"
                      "  delete <key>        - remove a preset\n"
                      "  save [path]         - persist presets to disk\n"
                      "  load <path>         - load presets from a file",
            "tree": "tree [body] [-v|--verbose]\n"
                    "  Show the body hierarchy; -v adds orbital elements.",
            "edit": "edit <body> <field> <value>\n"
                    "  Body fields:  mu radius atm_height rotation_period_s\n"
                    "  Orbit fields: a e inc arg_p lon_of_asc MA_at_t0",
            "advance": "advance <ut|duration>\n"
                       "  Run the ticket forward to a UT (e.g. 'advance 2d 5h', 'advance 21600').",
            "time": "time <ut|duration>\n"
                    "  Scrub the timeline to a UT (e.g. 'time 1y 2mo 3d', 'time \"Year 2, Day 100, 12:30:00\"').",
            "cursor": "cursor <seconds>- give the current ut in human readable units or in seconds",
            "load": "load <system|ticket|mission> <path>",
            "save": "save <system|ticket|mission> [path]",
            "new": "new <system|ticket>  - creation wizard",
            "delete": "delete <ticket|system> [file]",
            "events": "events  - list scheduled ticket events",
            "remove": "remove <index>  - remove a scheduled event",
            "reset": "reset  - restore the ticket to its initial state",
            "render": "render  - open the 3D renderer window",
            "status": "status  - show the active system and ticket",
            "clear": "clear  - clear the terminal",
            "exit": "exit  - leave the REPL",
            "quit": "quit  - leave the REPL",
            "help": "help [command]  - show help for a command (or all commands)",
            "feedback": "feedback [-y|--yes] <message>\n"
                        "  Opens a PUBLIC GitHub issue in Gautham-rgb/Kerbal-Gravity-Program.\n"
                        "  Collected: your message, kgrp version, and OS info.\n"
                        "  Without -y/--yes it only prints a preview and the submission URL.\n"
                        "  Example: feedback -y the interplanetary planner crashed on go_i duna",
        }

        self.style = Style.from_dict({
            "prompt": "#00ff87 bold",
            "bottom-toolbar": "bg:#222222 #aaaaaa",
        })

        self.history = InMemoryHistory()
        self.word_completer = REPLCompleter(self)

        self.session = PromptSession(
            history=self.history,
            completer=self.word_completer,
            style=self.style,
        )

    async def exit(self, args):
        self.running = False

    async def cursor(self, args):
        if not self.ticket:
            print("please add a ticket by using the 'load ticket <path>' or the 'new ticket' commands")
            return
        
        if not self.ticket.events:
            print("Add some events to the ticket using 'add'")

        return self.ticket.cursor_ut    

    def get_toolbar(self):
        system = self.system.name if self.system else "None"
        ticket = self.ticket.name if self.ticket else "None"

        return HTML(
            f"  System: <b>{system}</b>"
            f"  |  Ticket: <b>{ticket}</b>"
        )

    async def status(self, args):
        system = self.system.name if self.system else "None"
        ticket = self.ticket.name if self.ticket else "None"
        print(f"System: {system}", end=" " * 10)
        print(f"Ticket: {ticket}")
    
    async def help(self, args):
        if not args:
            self._help_overview()
            return

        target = args[0].lower()
        if target == "add":
            self.print_event_types()
            return
        if target in self.subcommand_help:
            print(self.subcommand_help[target])
            return
        if target in self.command_help:
            _CONSOLE.print(f"[bold]{target}[/bold] — {self.command_help[target]}")
            detail = self.subcommand_help.get(target)
            if detail:
                print(detail)
            return
        suggestion = _did_you_mean(target, list(self.commands))
        if suggestion:
            print(f"Unknown command '{target}'. Did you mean '{suggestion}'?")
        else:
            print(f"Unknown command '{target}'. Run 'help' to list commands.")

    def _help_overview(self) -> None:
        _CONSOLE.print(Panel(
            "[bold cyan]Kerbal Gravity Program[/bold cyan] — mission planner for KSP\n\n"
            + _QUICKSTART,
            title="Help",
            border_style="cyan",
        ))
        for category, names in _COMMAND_CATEGORIES.items():
            table = Table(title=category, title_justify="left",
                          show_header=True, header_style="bold",
                          show_lines=False, expand=False)
            table.add_column("Command", style="green", no_wrap=True)
            table.add_column("What it does", style="white")
            seen: set[str] = set()
            for name in names:
                if name in seen:
                    continue
                seen.add(name)
                table.add_row(name, self.command_help.get(name, ""))
            _CONSOLE.print(table)
        print("\nRun 'help <command>' for details (e.g. 'help go'). "
              "Run 'help add' to see every ticket event type.")

    async def clear(self, args):
        if os.name == "nt":
            subprocess.run(["cmd", "/c", "cls"])
        else:
            subprocess.run(["clear"])  # Unreachable on Windows; kept for POSIX shells.

    async def tree(self, args):
        if self.system is None:
            print("Please import a system using the 'load' command")
            return

        verbose = "--verbose" in args or "-v" in args
        filtered = [a for a in args if a not in ("-v", "--verbose")]

        if filtered:
            body = self.system.get(filtered[0])

            if body is None:
                print(f"Celestial Body {filtered[0]} not found in the current system")
                return

        else:
            body = None

        if verbose:
            print(self._tree_verbose(body))
        else:
            print(self.system.tree(body))

    def _tree_verbose(self, start_body: Body | None = None) -> str:
        if self.system is None:
            return "System(empty)"

        start = start_body if start_body else self.system.root

        def format_orbit(o: Orbit) -> str:
            if o.parent:
                a_str = f"{o.semi_major_axis:.0f}m" if o.semi_major_axis > 0 else "N/A"
                e_str = f"{o.eccen:.6f}"
                inc_str = f"{np.degrees(o.inclination):.2f}°"
                return f"↷{o.parent.name}  a={a_str}  e={e_str}  i={inc_str}"
            return "(root)"

        def build_tree(body: Body, prefix: str = "", is_last: bool = True) -> list[str]:
            connector = "└── " if is_last else "├── "
            orbit_str = format_orbit(body.orbit)
            label = f"{body.name} ({body.identifier})  μ={body.mu:.3e}  R={body.radius:.0f}m  {orbit_str}  {self._rotation_str(body)}"
            lines = [prefix + connector + label]
            child_prefix = prefix + ("    " if is_last else "│   ")
            for i, moon in enumerate(body.moons):
                lines.extend(build_tree(moon, child_prefix, i == len(body.moons) - 1))
            return lines

        tree_lines = build_tree(start)
        tree_lines[0] = tree_lines[0][4:]
        return "\n".join(tree_lines)

    async def save(self, args):
        if not args:
            print("Usage: save <system|ticket|mission> [path]")
            return

        target = args[0].lower()
        path_arg = args[1] if len(args) > 1 else None

        if target == "ticket":
            if self.ticket is None:
                print("Please create a new ticket using the 'new' command")
                return
            self.ticket.save(path_arg if path_arg else Path.cwd())

        elif target == "system":
            if self.system is None:
                print("Please create a system using the 'new' command")
                return
            self.system.save(path_arg if path_arg else Path.cwd())

        elif target == "mission":
            self._save_mission(path_arg)
        else:
            print(f"Invalid argument: {target}")

    async def load(self, args):
        if len(args) != 2:
            print("Usage: load <system|ticket|mission> <path>")
            return

        target = args[0].lower()

        if target == "system":
            path = self._resolve_example_path(args[1])
            self.system = System.load(path)
            if not self._validate_and_report(self.system):
                print("System rejected: fix the errors and try again.")
                self.system = None
            else:
                self.system_path = str(path)
        elif target == "ticket":
            if self.system is None:
                print("Please import your system first using load system path/to/file.json")
            else:
                self.ticket = Ticket.load(args[1], self.system, Ticket.collect_bodies(self.system.root))
                self.vessels[self.ticket.spacecraft.identifier] = self.ticket.spacecraft
        elif target == "mission":
            self._load_mission(args[1])
        else:
            print(f"Invalid argument: {target}")

    def _save_mission(self, path_arg: str | None = None):
        if self.system is None:
            print("No system loaded. Save a system or load one first.")
            return

        path = Path(path_arg) if path_arg else Path.cwd()
        if path.is_dir() or path_arg is None:
            path = path / f"mission{MISSION_EXTENSION}"
        if not path.name.endswith(MISSION_EXTENSION):
            path = path.with_name(path.stem + MISSION_EXTENSION)

        if self.system_path:
            system_ref = self.system_path
        else:
            system_ref = path.with_name(path.stem + ".system.json").name
            self.system.save(str(path.parent / system_ref))

        ticket_ids = {self.ticket.spacecraft.identifier} if self.ticket else set()
        standalone = [
            _vessel_to_json(v, v.snapshot(self.current_ut), None)
            for vid, v in self.vessels.items()
            if vid not in ticket_ids
        ]

        payload = {
            "format": "kgrp-mission",
            "version": 1,
            "system": system_ref,
            "current_ut": self.current_ut,
            "renderer_prefs": self.renderer_prefs,
            "tickets": [self.ticket.to_json()] if self.ticket else [],
            "vessels": standalone,
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(__import__("json").dumps(payload, indent=2))
        print(f"[Saved mission] {path}")

    def _load_mission(self, path_arg: str):
        import json

        path = Path(path_arg)
        if path.is_dir():
            path = path / f"mission{MISSION_EXTENSION}"
        if not path.exists():
            print(f"No mission file at {path}")
            return

        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            print(f"[Failed] invalid mission file: {e}")
            return

        if data.get("format") != "kgrp-mission":
            print(f"[Failed] '{path}' is not a kgrp mission file.")
            return

        system_path = Path(data["system"])
        if not system_path.is_absolute():
            system_path = path.parent / system_path

        if not system_path.exists():
            print(f"[Failed] system file not found: {system_path}")
            return

        self.system = System.load(str(system_path))
        if not self._validate_and_report(self.system):
            print("System rejected: fix the errors and try again.")
            self.system = None
            return
        self.system_path = str(system_path)

        bodies = Ticket.collect_bodies(self.system.root)
        vessels: dict[str, Spacecraft] = {}
        tickets = []
        for ticket_data in data.get("tickets", []):
            try:
                tickets.append(Ticket.from_dict(ticket_data, self.system, bodies, vessels))
            except Exception as e:
                print(f"[Failed] could not restore ticket '{ticket_data.get('identifier', '?')}': {e}")
                return
            for t in tickets:
                vessels[t.spacecraft.identifier] = t.spacecraft

        for vessel_data in data.get("vessels", []):
            try:
                vessel = _vessel_from_json(vessel_data, bodies, None)
                vessels[vessel.identifier] = vessel
            except Exception as e:
                print(f"[Failed] could not restore vessel '{vessel_data.get('identifier', '?')}': {e}")
                return

        self.vessels = vessels
        self.ticket = tickets[0] if tickets else None
        self.current_ut = float(data.get("current_ut", 0.0))
        self.renderer_prefs = dict(data.get("renderer_prefs") or self.renderer_prefs)
        print(f"[Loaded mission] {path} (system '{self.system.name}', {len(self.vessels)} vessel(s), {len(tickets)} ticket(s))")

    async def time_cmd(self, args):
        if not args:
            if self.ticket is not None:
                print(f"Current UT: {self.ticket.cursor_ut:g}s ({format_ut(self.ticket.cursor_ut, ker_time=True)})")
            else:
                print(f"Current UT: {self.current_ut:g}s ({format_ut(self.current_ut, ker_time=True)})")
            return

        base = self.ticket.cursor_ut if self.ticket is not None else self.current_ut
        try:
            ut = parse_time_string(" ".join(args), base=base)
        except ValueError as e:
            print(f"[Failed] {e}")
            return

        if self.ticket is not None:
            try:
                self.ticket.advance_to(ut)
            except (ValueError, RuntimeError) as e:
                print(f"[Failed] {e}")
                return
        self.current_ut = float(ut)
        print(f"Timeline scrubbed to UT {ut:g}s ({format_ut(ut, ker_time=True)})")

    # --- go -----------------------------------------------------------------

    async def _pick_preset(self, body, spacecraft, title: str) -> str | None:
        entries = orbit_presets(body, spacecraft, preset_library())
        selection = await prompt_toolkit.shortcuts.radiolist_dialog(
            title=title,
            text=f"Choose an orbit around {body.name}:",
            values=[(entry.key, f"{entry.label}  -  {entry.description}") for entry in entries],
        ).run_async()
        return selection

    async def go_cmd(self, args):
        if self.ticket is None:
            print("Please create a ticket first: 'new ticket'")
            return

        # `go -i` / `go --interplanetary` plans a *direct* interplanetary
        # transfer (escape + SOI exit + heliocentric leg) to a target body,
        # instead of an in-system orbit-to-orbit transfer.
        interplanetary = False
        clean_args: list[str] = []
        for token in args:
            if token in ("-i", "--interplanetary"):
                interplanetary = True
            else:
                clean_args.append(token)
        args = clean_args

        if interplanetary:
            await self._go_interplanetary_flow(args)
            return

        body = self.ticket.spacecraft.parent
        spacecraft = self.ticket.spacecraft

        overrides: dict[str, str] = {}
        if not args:
            start_key = await self._pick_preset(body, spacecraft, "Start orbit")
            if start_key is None:
                print("Cancelled.")
                return
            end_key = await self._pick_preset(body, spacecraft, "Target orbit")
            if end_key is None:
                print("Cancelled.")
                return
            raw = await self._prompt_str(
                "Override target orbit (key=value, space/comma separated) [none]: ", ""
            )
            for token in raw.replace(",", " ").split():
                if "=" in token:
                    key, value = token.split("=", 1)
                    overrides[key.strip()] = value.strip()
        else:
            positional = []
            for token in args:
                if "=" in token:
                    key, value = token.split("=", 1)
                    overrides[key.strip()] = value.strip()
                else:
                    positional.append(token)
            if len(positional) != 2:
                print("Usage: go <start-orbit> <end-orbit> [key=value ...]")
                print("       (run 'go' with no arguments for the wizard)")
                print("Overrides: peri_alt=, apo_alt=, incl=, arg_p=, lan=")
                return
            start_key, end_key = positional[0], positional[1]

        try:
            plan = go_plan(self.ticket, start_key, end_key, overrides=overrides or None)
        except (ValueError, RuntimeError) as e:
            print(f"[Failed] {e}")
            return

        print("Transfer plan:")
        for line in plan.summary_lines():
            print(line)

        if plan.burn_count == 0:
            print("Nothing to do - the vessel is already in the target orbit.")
            return

        mode = "gradual"
        if plan.events and (plan.burn_count > 1 or len(plan.events) > 1):
            print(
                "\nThis plan schedules multiple events over time "
                f"({plan.transfer.start_ut:,.0f}s -> {plan.transfer.end_ut:,.0f}s)."
            )
            raw = await self._prompt_str(
                "Apply gradually (schedule events) or all at once now? [gradual/instant/cancel]: ",
                "gradual",
            )
            mode = raw.strip().lower()
            if mode not in ("gradual", "instant"):
                print("Cancelled.")
                return

        try:
            applied = apply_go(self.ticket, start_key, end_key, mode=mode, overrides=overrides or None)
        except (ValueError, RuntimeError) as e:
            print(f"[Failed] {e}")
            return

        if mode == "instant":
            print(f"[Done] Vessel is now in the target orbit (UT {applied.end_ut:,.1f}s).")
        else:
            print(
                f"[Scheduled] {applied.burn_count} burn(s) scheduled. "
                f"Run 'advance {applied.end_ut:,.0f}' to execute, or 'reset' to undo."
            )

    async def _go_interplanetary_flow(self, args):
        """Plan + (optionally) commit a direct interplanetary transfer.

        Invoked by ``go -i <target> [key=value ...]``. Kept separate so ``go_i``
        can stay as a thin alias for muscle-memory compatibility.
        """
        if not args:
            print("Usage: go -i <target> [key=value ...]   (alias: go_i <target>)")
            print("  <target> is a body name (e.g. 'duna') or moon:<id> (e.g. 'moon:4').")
            print("  e.g. go -i duna                (Kerbin -> Duna, escape + heliocentric leg)")
            print("  e.g. go -i moon:4 peri_alt=200000")
            print("  Direct A->B transfer (escape burn + coast to SOI exit + heliocentric leg).")
            return

        from CLI_use.go import apply_go_interplanetary, go_plan_interplanetary
        positional = []
        overrides: dict[str, str] = {}
        for token in args:
            if "=" in token:
                key, value = token.split("=", 1)
                overrides[key.strip()] = value.strip()
            else:
                positional.append(token)

        if len(positional) < 1:
            print("Usage: go -i <target>  (transfer target, e.g. moon:3)")
            return
        target_key = positional[0]

        try:
            plan = go_plan_interplanetary(self.ticket, target_key, overrides=overrides or None) #type: ignore
        except (ValueError, RuntimeError) as e:
            print(f"[Failed] {e}")
            return

        print("Interplanetary transfer plan:")
        for line in plan.summary_lines():
            print(line)

        if plan.burn_count == 0:
            print("Nothing to do.")
            return

        raw = await self._prompt_str(
            f"\nApply gradually (schedule {len(plan.events)} events) or all at once now? [gradual/instant/cancel]: ",
            "gradual",
        )
        mode = raw.strip().lower()
        if mode not in ("gradual", "instant"):
            print("Cancelled.")
            return

        try:
            apply_go_interplanetary(
                self.ticket, target_key, mode=mode, overrides=overrides or None, plan=plan #type: ignore
            )
        except (ValueError, RuntimeError) as e:
            print(f"[Failed] {e}")
            return

        if mode == "instant":
            print(f"[Done] Vessel is now in orbit around {plan.grandparent_name} (UT {plan.end_ut:,.1f}s).")
        else:
            print(
                f"[Scheduled] {plan.burn_count} burn(s) + coast + refbody scheduled. "
                f"Run 'advance {plan.end_ut:,.0f}' to execute, or 'reset' to undo."
            )

    async def go_interplanetary_cmd(self, args):
        # Deprecated alias: prefer `go -i ...`. Kept for muscle memory.
        await self.go_cmd(["-i", *args])

    # --- preset -------------------------------------------------------------

    async def preset_cmd(self, args):
        op = args[0].lower() if args else "list"
        lib = preset_library()

        if op == "list":
            if not lib.presets:
                print("No presets defined.")
                return
            for key, preset in sorted(lib.presets.items()):
                print(
                    f"  {key:10s} {preset.label:12s} "
                    f"pe={preset.peri_alt:,.0f}m ap={preset.apo_alt:,.0f}m "
                    f"i={preset.inclination_deg:.1f} deg"
                )
            return

        if op == "new":
            key = await self._prompt_str("Preset key [e.g. station]: ", "")
            if not key:
                print("Cancelled.")
                return
            label = await self._prompt_str("Label: ", key)
            peri_alt = await self._prompt_float(f"Periapsis altitude (m) [Default: 200000]: ", 200000.0)
            apo_alt = await self._prompt_float(f"Apoapsis altitude (m) [Default: {peri_alt:g}]: ", peri_alt)
            inclination = await self._prompt_float(f"Inclination (deg) [Default: 0]: ", 0.0)
            arg_p = await self._prompt_float(f"Argument of periapsis (deg) [Default: 0]: ", 0.0)
            lan = await self._prompt_float(f"Longitude of ascending node (deg) [Default: 0]: ", 0.0)
            lib.add(CustomPreset(
                key=key, label=label, peri_alt=peri_alt, apo_alt=apo_alt,
                inclination_deg=inclination, arg_p_deg=arg_p, lan_deg=lan,
            ))
            lib.save()
            print(f"[Saved] Preset '{key}' added to {lib.path}.")
            return

        if op == "delete":
            if len(args) < 2:
                print("Usage: preset delete <key>")
                return
            removed = lib.remove(args[1])
            if removed is None:
                print(f"Preset '{args[1]}' not found.")
                return
            lib.save()
            print(f"[Removed] Preset '{args[1]}'.")
            return

        if op == "save":
            path = args[1] if len(args) > 1 else None
            target = lib.save(path)
            print(f"[Saved] {target}")
            return

        if op == "load":
            if len(args) < 2:
                print("Usage: preset load <path>")
                return
            lib.load(args[1])
            print(f"[Loaded] {len(lib.presets)} presets from {args[1]}.")
            return

        if op == "edit":
            if len(args) < 2:
                print("Usage: preset edit <key> [field] [value]")
                return
            preset = lib.get(args[1])
            if preset is None:
                print(f"Preset '{args[1]}' not found.")
                return
            if len(args) >= 4:
                field, value = args[2], args[3]
                if field in ("label", "description"):
                    setattr(preset, field, value)
                elif field in ("peri_alt", "apo_alt", "inclination_deg", "arg_p_deg", "lan_deg"):
                    try:
                        setattr(preset, field, float(value))
                    except ValueError:
                        print("Invalid numeric value.")
                        return
                else:
                    print(f"Unknown field '{field}'.")
                    return
                lib.save()
                print(f"[Updated] {args[1]}.{field} = {value}")
            else:
                print(
                    f"  {preset.key}: {preset.label}  pe={preset.peri_alt:,.0f}m ap={preset.apo_alt:,.0f}m "
                    f"i={preset.inclination_deg:.1f} deg\n"
                    f"  Edit with: preset edit {preset.key} <field> <value>\n"
                    f"  Fields: label, description, peri_alt, apo_alt, inclination_deg, arg_p_deg, lan_deg"
                )
            return

        print("Usage: preset <list|new|edit|delete|save|load>")

    async def _prompt_float(self, prompt_text: str, default_val: float, validator=lambda v: True, err_msg: str = "Invalid input.") -> float:
        while True:
            val_str = await self.session.prompt_async(prompt_text)
            if not val_str.strip():
                return default_val
            try:
                val = float(val_str)
                if validator(val):
                    return val
                print(err_msg)
            except ValueError:
                print("Invalid input. Please enter a valid number.")
    
    async def _prompt_str(self, prompt_text: str, default_val: str) -> str:
        val_str = await self.session.prompt_async(prompt_text)
        return val_str.strip() or default_val

    async def _prompt_time(self, prompt_text: str, default_val: float) -> float:
        while True:
            val_str = await self.session.prompt_async(prompt_text)
            if not val_str.strip():
                return default_val
            try:
                if any(char.isalpha() for char in val_str):
                    return parse_time_string(val_str)
                return float(val_str)
            except ValueError:
                print("Invalid input. Please enter seconds or a Kerbal time string (e.g. '2d 3h').")

    async def select_body_hierarchical(self, current_body):
        while True:
            choices = [("select", f"-> Select '{current_body.name}' as Parent")] #type: ignore

            children = getattr(current_body, "moons", [])
            for child in children:
                choices.append((child.identifier, f"Orbit {child.name}"))

            parent = getattr(getattr(current_body, "orbit", None), "parent", None)
            if parent is not None:
                choices.append(("..", f"^ Go up to parent ({parent.name})"))

            selection = await prompt_toolkit.shortcuts.radiolist_dialog(
                title=f"Current Focus: {current_body.name}", #type: ignore
                text="Choose to select this body or navigate to one of its children/parent:",
                values=choices,
            ).run_async()

            if selection is None:
                return None

            if selection == "select":
                return current_body

            if selection == "..":
                if parent is None:
                    print(f"[Failed] '{current_body.name}' has no parent to go up to.")
                    continue
                current_body = parent
                continue

            child = next((c for c in children if c.identifier == selection), None)
            if child is None:
                print(f"[Failed] Could not resolve body '{selection}'.")
                continue
            current_body = child


    def _resolve_example_path(self, arg: str) -> str:
        """Resolve a ``load system`` argument to a file path.

        If the literal path doesn't exist, fall back to a bundled example
        system of the same name (e.g. ``planets_ksp`` -> the shipped
        ``planets_ksp.json``). This lets a fresh install run the quickstart
        without hunting for data files.
        """
        if os.path.exists(arg):
            return arg
        try:
            from basic_systems import example_system_path
        except Exception:
            return arg
        candidate = example_system_path(arg)
        return candidate if os.path.exists(candidate) else arg

    def _validate_and_report(self, system) -> bool:
        result = validate_system(system)
        for warning in result.warnings:
            print(f"[Warning] {warning}")
        for error in result.errors:
            print(f"[Failed] {error}")
        if result.errors:
            print(f"Validation failed with {len(result.errors)} error(s).")
        return result.ok

    @staticmethod
    def _parse_vec(raw: str, count: int) -> np.ndarray:
        parts = raw.replace(",", " ").split()
        values = [float(p) for p in parts]
        if len(values) != count:
            raise ValueError(f"Expected {count} numbers, got {len(values)}.")
        return np.asarray(values, dtype=float)

    def _resolve_vessel(self, name: str) -> Spacecraft:
        name = name.strip().lower()
        if name in ("me", "self", "current", ""):
            if self.ticket is None:
                raise ValueError("No active ticket to reference.")
            return self.ticket.spacecraft
        for identifier, vessel in self.vessels.items():
            if identifier.lower() == name or vessel.name.lower() == name:
                return vessel
        raise ValueError(f"Vessel '{name}' not found.")

    async def _convert_arg(self, arg: ArgSpec, raw: str):
        kind = arg.kind
        if kind == "float":
            if any(char.isalpha() for char in raw):
                return parse_time_string(raw)
            return float(raw)
        if kind == "int":
            return int(raw)
        if kind == "str":
            return raw
        if kind == "bool":
            return raw.strip().lower() in {"y", "yes", "1", "true", "on"}
        if kind == "vec3":
            return self._parse_vec(raw, 3)
        if kind == "vec4":
            return self._parse_vec(raw, 4)
        if kind == "body":
            body = self.system.get(raw) if self.system else None
            if body is None:
                raise ValueError(f"Body '{raw}' not found.")
            return body
        if kind == "vessel":
            return self._resolve_vessel(raw)
        if kind == "int_list":
            return [int(v) for v in raw.replace(" ", "").split(",") if v]
        raise ValueError(f"Unknown argument kind '{kind}'.")

    async def _prompt_arg(self, arg: ArgSpec):
        kind, prompt, default = arg.kind, arg.prompt, arg.default

        if kind == "float":
            return await self._prompt_float(f"{prompt} [Default: {default:g}]: ", float(default))
        
        if kind == "int":
            while True:
                raw = await self.session.prompt_async(f"{prompt} [Default: {default}]: ")
                if not raw.strip():
                    return int(default)
                try:
                    return int(raw)
                except ValueError:
                    print("Invalid integer.")

        if kind == "str":
            return await self._prompt_str(f"{prompt} [Default: {default}]: ", str(default))
        
        if kind == "bool":
            raw = await self._prompt_str(f"{prompt} (y/n) [Default: {'y' if default else 'n'}]: ", "y" if default else "n")
            return raw.strip().lower() in {"y", "yes", "1", "true", "on"}
        
        if kind in ("vec3", "vec4"):
            count = 3 if kind == "vec3" else 4
            default_str = np.array2string(np.asarray(default, dtype=float), separator=" ").strip("[]") if default is not None else ""
            raw = await self._prompt_str(f"{prompt} ({count} numbers, space/comma separated) [Default: {default_str}]: ", "")
            if not raw.strip():
                if default is None:
                    raise ValueError(f"'{prompt}' is required.")
                return np.asarray(default, dtype=float)
            return self._parse_vec(raw, count)
        
        if kind == "body":
            body = await self.select_body_hierarchical(self.system.root) #type: ignore
            if body is None:
                raise ValueError(f"'{prompt}' is required.")
            return body
        
        if kind == "vessel":
            return await self._prompt_vessel(prompt, default)
        
        if kind == "int_list":
            raw = await self._prompt_str(f"{prompt} (comma separated) [Default: {default}]: ", "")
            if not raw.strip() and default is not None:
                return default
            return [int(v) for v in raw.replace(" ", "").split(",") if v]
        raise ValueError(f"Unknown argument kind '{kind}'.")

    async def _prompt_vessel(self, prompt: str, default):
        if not self.vessels:
            return default
        names = list(self.vessels)
        selection = await prompt_toolkit.shortcuts.radiolist_dialog(
            title=prompt,
            text="Choose a vessel:",
            values=[(name, f"{name} ({self.vessels[name].name})") for name in names],
        ).run_async()
        if selection is None:
            return default
        return self.vessels[selection]

    def print_event_types(self):
        print("\nTicket event types (usage: add <type> [key=value ...]):")
        for spec in TICKET_COMMANDS.values():
            print(f"  {spec.usage()}")
            if spec.summary:
                print(f"      {spec.summary}")
            for arg in spec.args:
                req = "required" if arg.required else f"default={arg.default}"
                help_text = f" — {arg.help}" if arg.help else ""
                print(f"        {arg.name} ({arg.kind}, {req}){help_text}")
        print("\nTip: required args can be given as key=value (e.g. 'add coast start=0 end=60') "
              "or you'll be prompted for them.")

    async def add(self, args):
        if self.ticket is None:
            print("Please create a ticket first: 'new ticket'")
            return
        if not args:
            self.print_event_types()
            return

        name = args[0].lower()
        spec = TICKET_COMMANDS.get(name)
        if spec is None:
            print(f"Unknown event type '{name}'. See 'help add' for the full list.")
            return

        kv = {}
        for token in args[1:]:
            if "=" in token:
                key, value = token.split("=", 1)
                kv[key.strip()] = value.strip()
            else:
                print(f"Expected key=value pairs after the event type (got '{token}').")
                return

        values = {}
        for arg in spec.args:
            raw = kv.pop(arg.name, None)
            if raw is None:
                if not arg.required:
                    values[arg.name] = arg.default
                    continue
                values[arg.name] = await self._prompt_arg(arg)
            else:
                try:
                    values[arg.name] = await self._convert_arg(arg, raw)
                except ValueError as e:
                    print(f"[Failed] {e}")
                    return

        if kv:
            print(f"Unknown argument(s): {', '.join(kv)}")
            return

        try:
            event = spec.build(self.ticket, **values)
            self.ticket.add_event(event)
            print(f"[Added] {event.type} @ {event.start_ut:g}s")
        except Exception as e:
            print(f"[Failed] {e}")

    async def events(self, args):
        if self.ticket is None:
            print("No active ticket.")
            return
        if not self.ticket.events:
            print("No events scheduled. Use 'add <type>' to schedule one.")
            return
        for i, event in enumerate(self.ticket.events):
            if event.completed:
                state = "done"
            elif event.start_ut <= self.ticket.cursor_ut:
                state = "active"
            else:
                state = "pending"
            print(f"{i:3d}  {state:7s}  {event.type:16s}  t={event.start_ut:12.3g} -> {event.end_ut:12.3g}")
        print(f"Cursor: {self.ticket.cursor_ut:g}")

    async def remove(self, args):
        if self.ticket is None:
            print("No active ticket.")
            return
        if not args:
            print("Usage: remove <index>")
            return
        try:
            index = int(args[0])
        except ValueError:
            print("Index must be an integer.")
            return
        try:
            event = self.ticket.remove_event(index)
            print(f"[Removed] {event.type} @ {event.start_ut:g}")
        except (IndexError, ValueError) as e:
            print(e)

    async def reset(self, args):
        if self.ticket is None:
            print("No active ticket.")
            return
        self.ticket.reset()
        self.current_ut = self.ticket.cursor_ut
        print("Ticket reset to its initial state.")

    async def advance(self, args):
        if self.ticket is None:
            print("No active ticket.")
            return
        if not args:
            print("Usage: advance <ut|duration>")
            return
        try:
            ut = parse_time_string(" ".join(args), base=self.ticket.cursor_ut)
        except ValueError as e:
            print(f"[Failed] {e}")
            return
        try:
            self.ticket.advance_to(ut)
            self.current_ut = float(ut)
            print(f"Advanced to UT {ut:g}s ({format_ut(ut, ker_time=True)}) (cursor {self.ticket.cursor_ut:g})")
        except (ValueError, RuntimeError) as e:
            print(f"[Failed] {e}")

    async def _confirm(self, message: str) -> bool:
        answer = await self._prompt_str(f"{message}. Proceed? (y/n) [n]: ", "n")
        return answer.strip().lower() in {"y", "yes", "1", "true", "on"}

    def _delete_file(self, file_arg: str, default_name: str):
        path = Path(file_arg)
        if path.is_dir():
            path = path / default_name
        if path.exists():
            path.unlink()
            print(f"[Deleted file] {path}")
        else:
            print(f"[Skipped] no file at {path}")

    async def delete(self, args):
        if not args:
            print("Usage: delete <ticket|system> [file]")
            return

        target = args[0].lower()
        file_arg = args[1] if len(args) > 1 else None

        if target == "ticket":
            if self.ticket is None:
                print("No active ticket to delete.")
                return
            if not await self._confirm(
                f"Delete ticket '{self.ticket.name}' (id '{self.ticket.identifier}')"
            ):
                print("Deletion cancelled.")
                return
            if file_arg:
                self._delete_file(file_arg, f"{self.ticket.identifier}{TICKET_EXTENSION}")
            self.vessels.pop(self.ticket.spacecraft.identifier, None)
            self.ticket = None
            print("[Deleted] active ticket.")

        elif target == "system":
            if self.system is None:
                print("No active system to delete.")
                return
            if not await self._confirm(f"Delete system '{self.system.name}'"):
                print("Deletion cancelled.")
                return
            if file_arg:
                self._delete_file(file_arg, "planets.json")
            self.system = None
            self.ticket = None
            self.vessels.clear()
            print("[Deleted] active system (and any loaded ticket/vessels).")

        else:
            print(f"Invalid argument: {target}")

    async def version(self, args):
        if not args:
            print(f"v{__version__} of KGRP (Kerbal GRavity Program)")
            return
        flag = args[0]
        if flag in ("-u", "--update", "update"):
            await self._check_update()
            return
        if flag in ("-c", "--credits"):
            print(f"v{__version__} of KGRP (Kerbal GRavity Program)")
            print("Made by Gautham_rgb (systemic_speed on PyPi)")
            print("LICENSE: GNU GPLv3")
            print("Repository link: https://github.com/Gautham-rgb/Kerbal-Gravity-Program")
            print("Thank you for installing! If you delete this app, I swear")
            print("Uncle Roger will delete you like a Jamie Oliver video. Chilli jam")
            print("does NOT belong in fried rice, and deleting KGRP is a sin like that")
            print("Don't be a stupid, weak-sauce chilli jam bastard.")
            return
        print(f"v{__version__} of KGRP (Kerbal GRavity Program)")

    async def _check_update(self) -> None:
        """Check PyPI for a newer release and tell the user how to upgrade."""
        from CLI_use.updates import get_latest_version, is_update_available

        print(f"You have v{__version__}.")
        try:
            latest = get_latest_version()
        except Exception as exc:  # network/JSON errors must never break the CLI
            print(f"Could not reach PyPI to check for updates ({exc}).")
            return

        if is_update_available(__version__, latest):
            print(f"A newer version is available: v{latest}")
            print("Upgrade with:  pip install --upgrade kgrp")
        else:
            print(f"You're up to date (latest on PyPI is v{latest}).")

    async def feedback_cmd(self, args):
        send = False
        rest = []
        for tok in args:
            if tok in ("-y", "--yes"):
                send = True
            else:
                rest.append(tok)
        message = " ".join(rest).strip()
        if not message:
            print("Usage: feedback [-y|--yes] <your message>")
            print("  Sends feedback as a PUBLIC GitHub issue to the developer.")
            print("  Add -y/--yes to actually submit (otherwise a preview is shown).")
            print("  Collected: your message, kgrp version, and OS info.")
            return

        import platform, shutil, subprocess, webbrowser, urllib.parse

        version = __version__
        sys_info = (
            f"OS:      {platform.system()} {platform.release()} ({platform.machine()})\n"
            f"Python:  {platform.python_version()}\n"
            f"kgrp:    {version}"
        )
        body = (
            f"**Message**\n\n{message}\n\n"
            f"**Environment**\n\n```\n{sys_info}\n```\n\n"
            f"_Submitted via the `kgrp feedback` command._"
        )
        title = f"Feedback: {message[:60]}{'...' if len(message) > 60 else ''}"
        repo = "Gautham-rgb/Kerbal-Gravity-Program"

        print()
        print(f"This creates a PUBLIC GitHub issue in {repo} with:")
        print("  - your message")
        print(f"  - kgrp version: {version}")
        print(f"  - OS: {platform.system()} {platform.release()} ({platform.machine()})")
        print()
        print("Title:\n  " + title)
        print("Body:\n  " + body.replace("\n", "\n  "))
        print()

        if not send:
            print("Preview only. Re-run with -y/--yes to submit:")
            if shutil.which("gh"):
                print(f'  gh issue create --repo {repo} --title "{title}" --body "<message>"')
            print(f"  https://github.com/{repo}/issues/new?"
                  f"title={urllib.parse.quote(title)}&body={urllib.parse.quote(body)}")
            return

        if shutil.which("gh"):
            try:
                result = subprocess.run(
                    ["gh", "issue", "create", "--repo", repo,
                     "--title", title, "--body", body],
                    capture_output=True, text=True, timeout=30,
                )
                if result.returncode == 0:
                    print("[OK] Feedback submitted. Issue created at:")
                    print(result.stdout.strip())
                    return
                print(f"[warn] `gh` returned an error:\n{result.stderr.strip() or result.stdout.strip()}")
            except subprocess.TimeoutExpired:
                print("[warn] `gh` timed out (possibly waiting for auth). Falling back to browser.")
            except Exception as e:
                print(f"[warn] Could not run `gh`: {e}")
        else:
            print("[info] `gh` CLI not found; opening GitHub in your browser instead.")

        url = (f"https://github.com/{repo}/issues/new?title="
               f"{urllib.parse.quote(title)}&body={urllib.parse.quote(body)}")
        print("\nOpening GitHub so you can submit the issue manually:")
        print(url)
        try:
            webbrowser.open(url)
            print("[OK] Browser opened. Review and click 'Submit new issue'.")
        except Exception:
            print("[info] Could not open a browser automatically; copy the URL above.")

    async def edit(self, args):
        if self.system is None:
            print("Load or create a system first.")
            return
        if len(args) < 3:
            print("Usage: edit <body|vessel> <field> <value>")
            print("Body fields:   mu radius atm_height rotation_period_s | orbit: a e inc arg_p lon_of_asc MA_at_t0")
            print("Vessel fields: wet_mass dry_mass fuel t0 name render_color | orbit: a e inc arg_p lon_of_asc MA_at_t0")
            return

        body = self.system.get(args[0])
        vessel = None
        if body is None:
            try:
                vessel = self._resolve_vessel(args[0])
            except ValueError:
                vessel = None
        if body is None and vessel is None:
            print(f"Body or vessel '{args[0]}' not found.")
            return

        field = args[1].lower()

        if vessel is not None:
            self._edit_vessel(vessel, field, args[2])
            return

        try:
            value = float(args[2])
        except ValueError:
            print("Value must be a number.")
            return

        if field in ("mu", "radius", "atm_height", "rotation_period_s"):
            setattr(body, field, value)
        elif field in ("a", "e", "inc", "arg_p", "lon_of_asc", "MA_at_t0"):
            orbit = getattr(body, "orbit", None)
            if orbit is None:
                print(f"Body '{body.name}' has no orbit.") #type: ignore
                return
            mapping = {"a": "semi_major_axis", "inc": "inclination"}
            setattr(orbit, mapping.get(field, field), value)
        else:
            print(f"Unknown field '{field}'.")
            return

        if not self._validate_and_report(self.system):
            print("Note: the system now fails validation.")
        print(f"[Edited] {body.name}.{field} = {value:g}") #type: ignore

    def _edit_vessel(self, vessel: Spacecraft, field: str, raw: str) -> None:
        field = field.lower()
        try:
            value = float(raw)
            is_number = True
        except ValueError:
            value = raw
            is_number = False

        if field == "wet_mass" and is_number:
            new_fuel = value - vessel.dry_mass #type: ignore
            if new_fuel < 0:
                print(f"[Failed] wet_mass ({value:g}kg) below dry_mass ({vessel.dry_mass:g}kg).")
                return
            for tank in vessel._part("core").tanks:
                if tank.resource == "LiquidFuel":
                    tank.capacity = new_fuel
                    tank.amount = new_fuel
                    break
            vessel._refresh_mass()
            self._sync_ticket_snapshot(vessel)
            print(f"[Edited] {vessel.name}.wet_mass = {value:g}  (fuel budget {new_fuel:g}kg)")
            return
        if field == "dry_mass" and is_number:
            core = vessel._part("core")
            other_dry = vessel.dry_mass - core.dry_mass
            core.dry_mass = value - other_dry #type: ignore
            if core.dry_mass < 0:
                print(f"[Failed] dry_mass ({value:g}kg) below the vessel's other parts ({other_dry:g}kg).")
                return
            vessel._refresh_mass()
            self._sync_ticket_snapshot(vessel)
            print(f"[Edited] {vessel.name}.dry_mass = {value:g}")
            return
        if field == "fuel" and is_number:
            target = value
            for part in vessel.parts:
                for tank in part.tanks:
                    if tank.resource == "LiquidFuel":
                        tank.amount = min(tank.capacity, max(target, 0.0)) #type: ignore
                        target -= tank.amount #type: ignore
            vessel._refresh_mass()
            self._sync_ticket_snapshot(vessel)
            print(f"[Edited] {vessel.name}.fuel = {vessel.fuel_mass:g}kg")
            return
        if field == "t0" and is_number:
            if float(value) < 0:
                print("[Failed] t0 cannot be negative.")
                return
            vessel.t0 = value  #type: ignore
            vessel._recalculate_orbit(vessel.r0, vessel.v0, value)  #type: ignore
            self._sync_ticket_snapshot(vessel)
            print(f"[Edited] {vessel.name}.t0 = {value:g}s")
            return
        if field == "name":
            vessel.name = str(value)
            print(f"[Edited] {vessel.name}.name = {value}")
            return
        if field == "render_color":
            vessel.render_color = str(value)
            print(f"[Edited] {vessel.name}.render_color = {value}")
            return
        if field in ("a", "e", "inc", "arg_p", "lon_of_asc", "MA_at_t0") and is_number:
            mapping = {"a": "semi_major_axis", "inc": "inclination", "e": "eccen"}
            setattr(vessel.orbit, mapping.get(field, field), value)
            vessel.r0 = vessel.get_pos_at_ut(vessel.t0)
            vessel.v0 = vessel.get_vel_at_ut(vessel.t0)
            self._sync_ticket_snapshot(vessel)
            print(f"[Edited] {vessel.name}.{field} = {value:g}")
            return
        if is_number and hasattr(vessel, field):
            try:
                setattr(vessel, field, value)
            except (AttributeError, TypeError):
                print(f"[Failed] field '{field}' is read-only on '{vessel.name}'.")
                return
            self._sync_ticket_snapshot(vessel)
            print(f"[Edited] {vessel.name}.{field} = {value:g}")
            return
        print(f"Unknown vessel field '{field}'. Use wet_mass dry_mass fuel t0 name render_color, or an orbit element (a e inc arg_p lon_of_asc MA_at_t0).")

    def _sync_ticket_snapshot(self, vessel: Spacecraft) -> None:
        if self.ticket is not None and self.ticket.spacecraft is vessel:
            self.ticket.initial_snapshot = vessel.snapshot(vessel.t0)

    async def new(self, args):
        if not args:
            print("Usage: new <system|ticket>")
            return

        target = args[0].lower()

        if target == "system":
            print("System Creation Wizard")

            root_type = await prompt_toolkit.shortcuts.radiolist_dialog(
                title="Root Body type",
                text="What is the type of body that you want to create?",
                values=[
                    ("Barycenter", "Virtual origin, used for sims, R = 0"),
                    ("Object", "An actual tangible planet/star, R > 0"),
                ],
            ).run_async()

            if root_type is None:
                print("System creation cancelled.")
                return

            default_name = "Barycenter" if root_type == "Barycenter" else "Kerbol"
            root_name = await self.session.prompt_async(
                f"Root Name [Default: {default_name}]: "
            )
            root_name = root_name.strip() or default_name

            mu = await self._prompt_float(
                "Gravitational Parameter mu (m^3/s^2) [Default: 3.5316e12]: ",
                3.5316e12,
            )

            if root_type == "Barycenter":
                radius = 0.0
                print("Selected Barycenter: Radius automatically set to 0.0")
                rotation_period_s = 0.0
            else:
                radius = await self._prompt_float(
                    "Physical Radius in meters [Default: 600000]: ",
                    600000.0,
                )
                rotation_period_s = await self._prompt_float(
                    "Rotation period (s) [Default: 0, 0 = unknown]: ",
                    0.0,
                    validator=lambda v: v >= 0,
                    err_msg="Rotation period must be non-negative.",
                )

            root_body = Body(
                name=root_name,
                mu=mu,
                radius=radius,
                identifier=root_name.lower().replace(" ", "_"),
                rotation_period_s=rotation_period_s,
            )

            self.system = System(name=f"{root_name} System", root_obj=root_body)
            self._validate_and_report(self.system)

            print(
                f"\n[Success] Created System '{self.system.name}' with root '{root_name}' "
                f"(type={root_type}, mu={mu:.3e}, R={radius})"
            )

        elif target == "ticket":
            if self.system is None:
                print(
                    "Please create or load a system first using 'new system' or 'load system'."
                )
                return

            print("Ticket Creation Wizard")

            ticket_name = await self._prompt_str("Ticket name [Default: Mission-1]: ", "Mission-1")
            ticket_id = await self._prompt_str("Ticket identifier [Default: M-1]: ", "M-1")
            parent_body = await self.select_body_hierarchical(self.system.root)

            if parent_body is None:
                print("Ticket creation cancelled.")
                return

            if parent_body.rotation_period_s > 0:
                sync_r = parent_body.synchronous_radius()
                if sync_r is not None:
                    print(
                        f"  [Note] '{parent_body.name}' synchronous orbit: "
                        f"{sync_r - parent_body.radius:.0f}m altitude "
                        f"(rotation {parent_body.rotation_period_s:,.0f}s)"
                    )

            spacecraft_name = await self._prompt_str("Spacecraft name [Default: Voyager-1]: ", "Voyager-1")

            t0 = await self._prompt_time(
                f"First existence time (UT) of {spacecraft_name} [Default: {self.current_ut:g}s]: ",
                self.current_ut,
            )

            print("Initial State (Relative to parent)\n")
            print("  Tip: just press Enter on each to get a near-circular orbit at the suggested altitude.\n")

            default_r = parent_body.radius * 0.7 + parent_body.atm_height
            rx = await self._prompt_float(f"X position of {spacecraft_name} [Default: {default_r:.0f}m]: ", default_r)
            ry = await self._prompt_float(f"Y position of {spacecraft_name} [Default: 0m]: ", 0.0)
            rz = await self._prompt_float(f"Z position of {spacecraft_name} [Default: 0m]: ", 0.0)
            vx = await self._prompt_float(f"X velocity of {spacecraft_name} [Default: 0m/s]: ", 0.0)
            default_v = np.sqrt(parent_body.mu / rx) if rx > 0 else 0.0
            vy = await self._prompt_float(f"Y velocity of {spacecraft_name} [Default: {default_v:.0f}m/s]: ", default_v)
            vz = await self._prompt_float(f"Z velocity of {spacecraft_name} [Default: 0m/s]: ", 0.0)
            dry_mass = await self._prompt_float(f"Dry mass of {spacecraft_name} [Default: 1000kg]: ", 1000.0)
            wet_mass = await self._prompt_float(f"Wet mass of {spacecraft_name} [Default: 4000kg]: ", 4000.0)

            spacecraft = Spacecraft(
                name=spacecraft_name,
                r0=np.array([rx, ry, rz]),
                v0=np.array([vx, vy, vz]),
                t0=t0,
                parent=parent_body,
                dry_mass=dry_mass,
                wet_mass=wet_mass,
                hull_mesh=None,
            )
            spacecraft.add_engine(50000.0, 300.0, np.zeros(3), name="Default Engine")

            self.ticket = Ticket(
                identifier=ticket_id,
                spacecraft=spacecraft,
                system=self.system,
                name=ticket_name,
            )
            self.vessels[spacecraft.identifier] = spacecraft
            self.current_ut = self.ticket.cursor_ut

            print(
                f"\n[Success] Created Ticket '{self.ticket.name}' for '{spacecraft_name}' "
                f"orbiting '{parent_body.name}' (UT {t0:g}s). "
                f"Schedule events with 'add'."
            )


    async def execute(self, text):
        parts = shlex.split(text)

        if not parts:
            return

        command = parts[0].lower()
        args = parts[1:]

        handler = self.commands.get(command)

        if not handler:
            print(f"Invalid Command: {command}")
            suggestion = _did_you_mean(command, list(self.commands))
            if suggestion:
                print(f"  Did you mean '{suggestion}'? Run 'help' to list commands.")
            else:
                print("  Run 'help' to list commands.")
            return

        await handler(args)

    async def _run(self):
        while self.running:
            try:
                text = await self.session.prompt_async(HTML("<prompt>krgp &gt; </prompt>"), bottom_toolbar=self.get_toolbar)
                if not text.strip():
                    continue
                await self.execute(text)
            except KeyboardInterrupt:
                print()
            except EOFError:
                break

    def load_from_paths(self, system_path: str | None = None, ticket_path: str | None = None):
        if system_path:
            self.system = System.load(system_path)
            if not self._validate_and_report(self.system):
                print("System rejected: fix the errors and try again.")
                self.system = None
            else:
                self.system_path = str(system_path)

        if ticket_path:
            if self.system is None:
                print("Cannot load ticket without a system; please load a system first.")
                return
            self.ticket = Ticket.load(ticket_path, self.system, Ticket.collect_bodies(self.system.root))
            self.vessels[self.ticket.spacecraft.identifier] = self.ticket.spacecraft

    def run(self):
        self._print_welcome()
        asyncio.run(self._run())

    def _print_welcome(self) -> None:
        _CONSOLE.print(Panel(
            f"[bold cyan]Kerbal Gravity Program[/bold cyan]  [dim](v{__version__} — mission planner for KSP)[/dim]\n\n"
            + _QUICKSTART,
            border_style="cyan",
        ))
        self._print_update_notice()

    def _print_update_notice(self) -> None:
        """Best-effort: tell the user if a newer version is on PyPI.

        Network failures are swallowed silently so startup never blocks.
        """
        from CLI_use.updates import get_latest_version, is_update_available

        try:
            latest = get_latest_version()
        except Exception:
            return
        if is_update_available(__version__, latest):
            _CONSOLE.print(
                f"[yellow]A new KGRP version is available: v{latest} "
                f"(you have v{__version__}). Upgrade: pip install --upgrade kgrp[/yellow]"
            )

    async def system_cmd(self, args):
        if self.system is None:
            print("No system loaded. Use 'load system <path>' or 'new system'.")
            return

        subcommand = args[0].lower() if args else "info"

        if subcommand == "info":
            print(f"System: {self.system.name}")
            body_count = len(list(self.system.get_all_obj_in_system()))
            print(f"Bodies: {body_count}")
            root = self.system.root
            print(f"Root: {root.name} ({root.identifier}), μ={root.mu:.3e}, R={root.radius:.0f}m, {self._rotation_str(root)}")
            if self.ticket is not None and self.ticket.events:
                horizon = max(ev.end_ut for ev in self.ticket.events)
                print(f"Timeline: UT 0 to {horizon:g}s (last scheduled event)")
            else:
                print("Timeline: open-ended (no fixed end time; scrub with 'time')")

        elif subcommand == "bodies":
            verbose = "--verbose" in args or "-v" in args
            if verbose:
                self._system_bodies_verbose()
            else:
                self._system_bodies_brief()

        elif subcommand == "validate":
            if self._validate_and_report(self.system):
                print("System is valid.")
            else:
                print("System has errors — see above.")

        else:
            print(f"Unknown subcommand '{subcommand}'. Use: system <info|bodies|validate>")

    def _system_bodies_brief(self):
        print(f"Bodies in {self.system.name}:") #type: ignore
        for body in self.body_iter():
            orbit_info = ""
            if body.orbit and body.orbit.parent:
                orbit_info = f"  orbits {body.orbit.parent.name} (a={body.orbit.semi_major_axis:.0f}m, e={body.orbit.eccen:.4f})"
            print(f"  {body.name:12s} ({body.identifier:4s})  μ={body.mu:.3e}  R={body.radius:.0f}m{orbit_info}  {self._rotation_str(body)}")

    def _system_bodies_verbose(self):
        print(f"Bodies in {self.system.name}:") #type: ignore
        header = (f"  {'Name':14s} {'ID':5s} {'μ (m³/s²)':>12s} {'R (m)':>10s} {'a (m)':>14s} {'e':>8s} "
                  f"{'i (deg)':>9s} {'LAN':>8s} {'ω (deg)':>8s} {'MA':>8s} {'Period':>12s} {'Rot':>12s} "
                  f"{'Sync alt':>12s} {'Parent':>12s}")
        print(header)
        print("  " + "-" * (len(header) - 2))
        for body in self.body_iter():
            o = body.orbit
            parent_name = o.parent.name if o.parent else "None"
            if o.semi_major_axis > 0:
                a_str = f"{o.semi_major_axis:.0f}"
                e_str = f"{o.eccen:.6f}"
                inc_str = f"{np.degrees(o.inclination):.2f}"
                lan_str = f"{np.degrees(o.lon_of_asc):.2f}"
                ap_str = f"{np.degrees(o.arg_p):.2f}"
                ma_str = f"{np.degrees(o.MA_at_t0):.2f}"
                if np.isinf(o.period):
                    period_str = "∞"
                else:
                    period_str = self._format_period(o.period)
            else:
                a_str = "N/A"
                e_str = "N/A"
                inc_str = "N/A"
                lan_str = "N/A"
                ap_str = "N/A"
                ma_str = "N/A"
                period_str = "N/A"
            rot = getattr(body, "rotation_period_s", 0.0)
            rot_str = self._format_period(rot) if rot > 0 else "unknown"
            sync = body.synchronous_radius()
            sync_str = f"{sync - body.radius:,.0f}m" if sync is not None else "-"
            print(f"  {body.name:14s} {body.identifier:5s} {body.mu:12.3e} {body.radius:10.0f} {a_str:>14s} {e_str:>8s} {inc_str:>9s} {lan_str:>8s} {ap_str:>8s} {ma_str:>8s} {period_str:>12s} {rot_str:>12s} {sync_str:>12s} {parent_name:>12s}")

    def _format_period(self, period_seconds: float) -> str:
        from basic_systems.orbit_pred import KERBAL_UNITS
        day_sec = KERBAL_UNITS.day_seconds
        if period_seconds >= day_sec:
            days = int(period_seconds // day_sec)
            rem = period_seconds % day_sec
            hours = int(rem // (KERBAL_UNITS.hour_minutes * KERBAL_UNITS.minute_seconds))
            if hours > 0:
                return f"{days}d {hours}h"
            return f"{days}d"
        else:
            minutes = int(period_seconds // KERBAL_UNITS.minute_seconds)
            seconds = period_seconds % KERBAL_UNITS.minute_seconds
            return f"{minutes}m {seconds:.0f}s"

    def _rotation_str(self, body) -> str:
        rot = getattr(body, "rotation_period_s", 0.0)
        if rot <= 0:
            return "rotation=unknown"
        s = f"rotation={self._format_period(rot)}"
        sync = body.synchronous_radius()
        if sync is not None:
            s += f", sync={sync - body.radius:,.0f}m alt"
        return s

    def body_iter(self):
        def visit(body):
            yield body
            for moon in body.moons:
                yield from visit(moon)
        yield from visit(self.system.root) #type: ignore

    async def ticket_cmd(self, args):
        subcommand = args[0].lower() if args else "show"

        if subcommand == "show":
            if self.ticket is None:
                print("No active ticket. Use 'new ticket' or 'load ticket <path>'.")
                return
            t = self.ticket
            print(f"Ticket: {t.name} (id: {t.identifier})")
            print(f"  Vessel: {t.spacecraft.name} (id: {t.spacecraft.identifier})")
            print(f"  Cursor UT: {t.cursor_ut:.1f}s")
            if t.events:
                first_start = min(e.start_ut for e in t.events)
                last_end = max(e.end_ut for e in t.events)
                print(f"  Time range: {first_start:.1f}s -> {last_end:.1f}s")
            print(f"  Events: {len(t.events)}")
            for i, event in enumerate(t.events):
                state = "done" if event.completed else ("active" if event.start_ut <= t.cursor_ut else "pending")
                node = getattr(event, "node", None)
                dv_info = f"  Δv={node.total_mag:.1f}m/s" if node else ""
                print(f"  [{i}] {state:7s}  {event.type:16s}  t={event.start_ut:.1f}->{event.end_ut:.1f}{dv_info}")

        elif subcommand == "list":
            self._list_tickets()

        elif subcommand == "clear":
            if self.ticket is None:
                print("No active ticket.")
                return
            try:
                self.ticket.clear()
                print(f"[Cleared] removed all unexecuted events from '{self.ticket.name}'.")
            except ValueError as e:
                print(f"[Failed] {e}")

        elif subcommand == "delete":
            file_arg = args[1] if len(args) > 1 else None
            if self.ticket is None:
                print("No active ticket to delete.")
                return
            if not await self._confirm(f"Delete ticket '{self.ticket.name}' (id '{self.ticket.identifier}')"):
                print("Deletion cancelled.")
                return
            if file_arg:
                self._delete_file(file_arg, f"{self.ticket.identifier}{TICKET_EXTENSION}")
            self.vessels.pop(self.ticket.spacecraft.identifier, None)
            self.ticket = None
            print("[Deleted] active ticket.")

        else:
            print(f"Unknown subcommand '{subcommand}'. Use 'ticket show|list|clear|delete'.")

    def _list_tickets(self):
        search_dirs = [Path.cwd()]
        if self.system and Path(self.system.name).is_dir():
            search_dirs.append(Path(self.system.name))

        found = []
        for search_dir in search_dirs:
            if search_dir.is_dir():
                for f in search_dir.glob(f"*{TICKET_EXTENSION}"):
                    found.append(f)

        if not found:
            print("No ticket files found.")
            return

        print("Available ticket files:")
        for f in sorted(set(found)):
            print(f"  {f}")

    async def vessel_cmd(self, args):
        subcommand = args[0].lower() if args else "list"

        if subcommand == "list":
            if not self.vessels:
                print("No vessels loaded.")
                return
            print("Loaded vessels:")
            for identifier, vessel in self.vessels.items():
                pos = vessel.get_absolute_pos_at_ut(self.current_ut)
                speed = np.linalg.norm(vessel.get_absolute_vel_at_ut(self.current_ut))
                print(f"  {vessel.name:16s} ({identifier})  pos={pos[0]:.0f},{pos[1]:.0f},{pos[2]:.0f}m  |v|={speed:.1f}m/s")

        elif subcommand == "create":
            if self.system is None:
                print("Please create or load a system first using 'new system' or 'load system'.")
                return
            await self._create_vessel()

        elif subcommand == "delete":
            if not args[1:]:
                print("Usage: vessel delete <name|id>")
                return
            try:
                vessel = self._resolve_vessel(args[1])
            except ValueError as e:
                print(e)
                return
            if self.ticket is not None and vessel is self.ticket.spacecraft:
                print(f"Cannot delete '{vessel.name}': it is the active ticket's vessel. Use 'delete ticket' first.")
                return
            if not await self._confirm(f"Delete vessel '{vessel.name}' (id '{vessel.identifier}')"):
                print("Deletion cancelled.")
                return
            self.vessels.pop(vessel.identifier, None)
            print(f"[Deleted] vessel '{vessel.name}'.")

        elif subcommand == "engine":
            if not args[1:]:
                print("Usage: vessel engine <name|id> [index on|off|throttle <0-1>|add <thrust> <isp>|del]")
                return
            try:
                vessel = self._resolve_vessel(args[1])
            except ValueError as e:
                print(e)
                return
            await self._vessel_engine(vessel, args[2:])

        else:
            print(f"Unknown subcommand '{subcommand}'. Use 'vessel list|create|delete|engine'.")

    async def _vessel_engine(self, vessel: Spacecraft, args):
        if not args:
            self._list_engines(vessel)
            return

        action = args[0].lower()

        if action == "add":
            if len(args) < 3:
                print("Usage: vessel engine <name> add <thrust> <isp> [name]")
                return
            try:
                thrust = float(args[1])
                isp = float(args[2])
            except ValueError:
                print("Thrust and ISP must be numbers.")
                return
            engine_name = args[3] if len(args) > 3 else None
            vessel.add_engine(thrust, isp, offset=np.zeros(3), name=engine_name)
            vessel._refresh_mass()
            print(f"[Added] engine to '{vessel.name}'. Active engines: {len(vessel.engines)}")
            self._list_engines(vessel)
            return

        try:
            index = int(args[0])
        except ValueError:
            print(f"Unknown action '{action}'. Use an engine index, 'add', 'del', or 'list'.")
            return

        if not 0 <= index < len(vessel.engines):
            print(f"Engine index {index} out of range (0-{len(vessel.engines) - 1}).")
            return

        engine = vessel.engines[index]
        if len(args) < 2:
            self._list_engines(vessel)
            return

        op = args[1].lower()

        if op == "on":
            vessel.set_engine_state(index, True)
            print(f"[Engine {index}] '{engine.name}' activated.")
        elif op == "off":
            vessel.set_engine_state(index, False)
            print(f"[Engine {index}] '{engine.name}' deactivated.")
        elif op == "throttle":
            if len(args) < 3:
                print("Usage: vessel engine <name> <index> throttle <0-1>")
                return
            try:
                throttle = float(args[2])
            except ValueError:
                print("Throttle must be a number between 0 and 1.")
                return
            vessel.set_throttle(throttle, [index])
            print(f"[Engine {index}] '{engine.name}' throttle set to {throttle:g}.")
        elif op in ("del", "remove"):
            try:
                removed = vessel.remove_engine(index)
            except (IndexError, ValueError) as e:
                print(f"[Failed] {e}")
                return
            print(f"[Deleted] engine [{index}] '{removed.name}' from '{vessel.name}'.")
        else:
            print(f"Unknown operation '{op}'. Use on|off|throttle|del.")
        self._list_engines(vessel)

    def _list_engines(self, vessel: Spacecraft):
        if not vessel.engines:
            print(f"  '{vessel.name}' has no engines. Use 'vessel engine <name> add <thrust> <isp>'.")
            return
        print(f"  Engines on '{vessel.name}':")
        for i, engine in enumerate(vessel.engines):
            state = "ON" if engine.active else "OFF"
            print(f"    [{i}] {engine.name}  thrust={engine.max_thrust:g}N  Isp={engine.vacuum_isp:g}s  throttle={engine.throttle:g}  {state}")

    async def _create_vessel(self):
        print("Vessel Creation Wizard")

        spacecraft_name = await self._prompt_str("Spacecraft name [Default: Voyager-1]: ", "Voyager-1")
        parent_body = await self.select_body_hierarchical(self.system.root) #type: ignore

        if parent_body is None:
            print("Vessel creation cancelled.")
            return

        vessel_id = await self._prompt_str("Vessel identifier [Default: auto]: ", "")

        print("Initial State (Relative to parent)\n")
        print("  Tip: just press Enter on each to get a near-circular orbit at the suggested altitude.\n")

        sync_r = None
        if parent_body.rotation_period_s > 0:
            sync_r = parent_body.synchronous_radius()
            if sync_r is not None:
                print(
                    f"  [Note] '{parent_body.name}' synchronous orbit: "
                    f"{sync_r - parent_body.radius:.0f}m altitude "
                    f"(default initial distance)"
                )

        default_r = sync_r if sync_r is not None else parent_body.radius * 0.7 + parent_body.atm_height
        rx = await self._prompt_float(f"X position of {spacecraft_name} [Default: {default_r:.0f}m]: ", default_r)
        ry = await self._prompt_float(f"Y position of {spacecraft_name} [Default: 0m]: ", 0.0)
        rz = await self._prompt_float(f"Z position of {spacecraft_name} [Default: 0m]: ", 0.0)
        vx = await self._prompt_float(f"X velocity of {spacecraft_name} [Default: 0m/s]: ", 0.0)
        default_v = np.sqrt(parent_body.mu / rx) if rx > 0 else 0.0
        vy = await self._prompt_float(f"Y velocity of {spacecraft_name} [Default: {default_v:.0f}m/s]: ", default_v)
        vz = await self._prompt_float(f"Z velocity of {spacecraft_name} [Default: 0m/s]: ", 0.0)
        dry_mass = await self._prompt_float(f"Dry mass of {spacecraft_name} [Default: 1000kg]: ", 1000.0)
        wet_mass = await self._prompt_float(f"Wet mass of {spacecraft_name} [Default: 1500kg]: ", 1500.0)

        t0 = self.current_ut

        spacecraft = Spacecraft(
            name=spacecraft_name,
            r0=np.array([rx, ry, rz]),
            v0=np.array([vx, vy, vz]),
            t0=t0,
            parent=parent_body,
            dry_mass=dry_mass,
            wet_mass=wet_mass,
            hull_mesh=None,
            identifier=vessel_id if vessel_id else "",
        )

        self.vessels[spacecraft.identifier] = spacecraft
        print(f"\n[Success] Created vessel '{spacecraft_name}' (id: '{spacecraft.identifier}') orbiting '{parent_body.name}' at UT {t0:g}s.")

    async def render_cmd(self, args):
        if self.system is None:
            print("No system loaded. Use 'load system <path>' or 'new system'.")
            return

        try:
            from basic_systems.renderer.renderer import SystemRenderer
        except ImportError:
            print("Renderer unavailable — install pyvista.")
            return

        tickets = [self.ticket] if self.ticket else []
        spacecraft = list(self.vessels.values())

        renderer = SystemRenderer(
            system=self.system,
            title="KGRP",
            start_ut=self.current_ut,
            spacecraft=spacecraft,
            tickets=tickets,
            use_kerbal_time=self.renderer_prefs.get("use_kerbal_time", True),
            show_units_km=self.renderer_prefs.get("show_units_km", True),
            moon_exaggeration=self.renderer_prefs.get("moon_exaggeration", 10),
        )

        print(f"Launching renderer at UT {renderer.curr_ut:.0f} ...")
        renderer.run()

def main():
    import argparse

    from basic_systems import __version__

    parser = argparse.ArgumentParser(
        prog="kgrp",
        description="Kerbal Gravity Program — mission planning REPL",
    )
    parser.add_argument(
        "--version", "-V",
        action="version",
        version=f"kgrp {__version__}",
    )
    parser.add_argument(
        "--system", "-s",
        metavar="PATH",
        help="Path to a system JSON file to load at startup.",
    )
    parser.add_argument(
        "--ticket", "-t",
        metavar="PATH",
        help="Path to a ticket file to load at startup (requires --system).",
    )
    parser.add_argument(
        "--mission", "-m",
        metavar="PATH",
        help="Path to a mission file to load at startup (overrides --system/--ticket).",
    )
    args = parser.parse_args()

    repl = REPL()
    repl.load_from_paths(args.system, args.ticket)
    if args.mission:
        repl._load_mission(args.mission)
    repl.run()
