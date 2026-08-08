import os
import sys
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.styles import Style
from prompt_toolkit.formatted_text import HTML

from basic_systems.renderer.loader import load_system

theme = Style.from_dict({
    'cli.error':    '#ff005f bold',
    'cli.success':  '#00ff00 bold',
    'prompt.arrow': '#00ff87 bold',
    'file.name':    '#00d7ff italic',
})

def print_usage():
    """Prints standard CLI help when they type wrong arguments."""
    print("Usage:  krgp repl <abs_or_rel_path_to_file>")
    print("Example: kgrp repl ./config.json\n")

def main():
    if len(sys.argv) < 3:
        print("Missing arguments, please check the command")
        print_usage()
        sys.exit(1)

    if sys.argv[1] != "repl":
        print(f"plase recheck your command ({sys.argv[1]}), did you mean 'repl'??")
        print_usage()
        