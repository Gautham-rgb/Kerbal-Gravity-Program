"""Context-aware autocompletion for the KGRP REPL."""

from __future__ import annotations

from typing import Iterable, Iterator

from prompt_toolkit.completion import Completer, Completion

from CLI_use.commands import TICKET_COMMANDS
from CLI_use.go import preset_library


def _preset_keys() -> list[str]:
    try:
        return list(preset_library().presets.keys())
    except Exception:
        return []


def _yield(candidates: Iterable[str], partial: str) -> Iterator[Completion]:
    partial = partial.lower()
    seen: set[str] = set()
    for cand in candidates:
        if cand in seen:
            continue
        seen.add(cand)
        if cand.lower().startswith(partial):
            yield Completion(cand, start_position=-len(partial))


class REPLCompleter(Completer):
    def __init__(self, repl) -> None:
        self._repl = repl

    def _command_names(self) -> list[str]:
        return list(self._repl.commands.keys())

    def _vessel_names(self) -> list[str]:
        vessels = getattr(self._repl, "vessels", {})
        names = list(vessels.keys())
        names += [vessel.name for vessel in vessels.values()]
        return names

    def _body_names(self) -> list[str]:
        system = getattr(self._repl, "system", None)
        if system is None:
            return []
        try:
            names: list[str] = []
            for body in self._repl.body_iter():
                names.append(body.name)
                if body.identifier != body.name:
                    names.append(body.identifier)
            return names
        except Exception:
            return []

    def _moon_keys(self) -> list[str]:
        """Interplanetary transfer targets: ``moon:<identifier>`` for every moon."""
        system = getattr(self._repl, "system", None)
        if system is None or getattr(self._repl, "ticket", None) is None:
            return []
        try:
            body = self._repl.ticket.spacecraft.parent
        except Exception:
            return []
        if body is None:
            return []
        keys: list[str] = []
        for moon in getattr(body, "moons", []):
            keys.append(f"moon:{moon.identifier}")
        return keys

    def get_completions(self, document, complete_event) -> Iterator[Completion]:
        text = document.text_before_cursor
        if not text.strip():
            for name in self._command_names():
                yield Completion(name, start_position=0)
            return

        if text.endswith(" "):
            tokens = text.split()
            index = len(tokens)
            partial = ""
        else:
            tokens = text.split()
            index = len(tokens) - 1
            partial = tokens[-1] if tokens else ""

        yield from self._suggest(index, tokens, partial)

    def _suggest(self, index: int, tokens: list[str], partial: str) -> Iterator[Completion]:
        if index == 0:
            yield from _yield(self._command_names(), partial)
            return

        command = tokens[0].lower()

        if command == "add":
            if index == 1:
                yield from _yield(TICKET_COMMANDS.keys(), partial)
                return
            event_type = tokens[1].lower() if len(tokens) > 1 else ""
            spec = TICKET_COMMANDS.get(event_type)
            if spec is None:
                return
            if "=" in partial:
                arg_name = partial.split("=", 1)[0]
                value = partial.split("=", 1)[1]
                arg = next((a for a in spec.args if a.name == arg_name), None)
                if arg is not None and arg.kind == "body":
                    yield from _yield(self._body_names(), value)
                elif arg is not None and arg.kind == "vessel":
                    yield from _yield(self._vessel_names(), value)
                return
            used = {token.split("=", 1)[0] for token in tokens[2:] if "=" in token}
            for arg in spec.args:
                if arg.name not in used:
                    yield from _yield([f"{arg.name}="], partial)
            return

        if command in ("save", "load"):
            if index == 1:
                yield from _yield(["system", "ticket", "mission"], partial)
            return

        if command == "delete":
            if index == 1:
                yield from _yield(["ticket", "system"], partial)
            return

        if command == "new":
            if index == 1:
                yield from _yield(["system", "ticket"], partial)
            return

        if command == "system":
            if index == 1:
                yield from _yield(["info", "bodies", "validate", "-v", "--verbose"], partial)
            return

        if command == "ticket":
            if index == 1:
                yield from _yield(["show", "list", "clear", "delete"], partial)
            return

        if command == "vessel":
            if index == 1:
                yield from _yield(["list", "create", "delete", "engine"], partial)
                return
            subcommand = tokens[1].lower() if len(tokens) > 1 else ""
            if subcommand in ("delete", "engine") and index == 2:
                yield from _yield(self._vessel_names(), partial)
            elif subcommand == "engine" and index == 3:
                yield from _yield(["on", "off", "throttle", "add", "del"], partial)
            return

        if command == "tree":
            if index == 1:
                yield from _yield([*self._body_names(), "-v", "--verbose"], partial)
            return

        if command == "edit":
            if index == 1:
                yield from _yield(self._body_names(), partial)
            elif index == 2:
                yield from _yield(
                    ["mu", "radius", "atm_height", "rotation_period_s", "a", "e", "inc", "arg_p", "lon_of_asc", "MA_at_t0"],
                    partial,
                )
            return

        if command == "go":
            interplanetary = any(
                t in ("-i", "--interplanetary") for t in tokens
            )
            if index == 1 and not interplanetary and "=" not in partial:
                yield from _yield(["-i", "--interplanetary", *self._moon_keys(), "current", "escape", "impact", "atm-edge"], partial)
                return
            if interplanetary:
                # After `go -i` the next positional is a target body.
                if index >= 1 and "=" not in partial:
                    yield from _yield([*self._moon_keys(), *self._body_names()], partial)
                    if index >= 2 or (index == 1 and partial):
                        yield from _yield(["peri_alt=", "apo_alt=", "incl=", "arg_p=", "lan="], partial)
                return
            if index == 1 or (index == 2 and len(tokens) > 1 and "=" not in tokens[1]):
                yield from _yield([*_preset_keys(), "current", "escape", "impact", "atm-edge"], partial)
            elif index >= 3 or (index >= 2 and "=" in tokens[-1]):
                yield from _yield(["peri_alt=", "apo_alt=", "incl=", "arg_p=", "lan="], partial)
            return

        if command == "preset":
            if index == 1:
                yield from _yield(["list", "new", "edit", "delete", "save", "load"], partial)
            elif index == 2 and tokens[1].lower() in ("edit", "delete"):
                yield from _yield(_preset_keys(), partial)
            return

        if command == "help":
            if index == 1:
                yield from _yield(self._command_names(), partial)
            return
