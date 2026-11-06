"""Status labels for people reading a terminal: a symbol, a word, and a colour.

Open code, used by doctor, init, baseline approve and gate. Styling is for people only:
machine output (the gate's decision JSON on stdout, doctor --json) is never styled, and
styled text always carries the word, so the meaning never depends on colour or the symbol.

A stream is styled when NO_COLOR is unset and it is a terminal, or when GITHUB_ACTIONS is
"true" (the job log renders colour though it isn't a terminal) unless the caller opts out.
A stream whose encoding isn't UTF-8 gets ASCII symbols. Unstyled, a label is just the word,
so plain output is exactly what it was before styling existed. Uses click only.
"""

import codecs
import os
from dataclasses import dataclass
from typing import IO

import click

# word: (symbol, ASCII symbol, colour)
STATUSES: dict[str, tuple[str, str, str]] = {
    "OK": ("✓", "+", "green"),
    "WARN": ("⚠", "!", "yellow"),
    "FAIL": ("✗", "x", "red"),
    "PASS": ("✓", "+", "green"),
    "CONDITIONAL_PASS": ("◐", "~", "yellow"),
    "REJECT": ("✗", "x", "red"),
    "INCONCLUSIVE": ("?", "?", "magenta"),
}


def _is_utf8(stream: IO[str]) -> bool:
    encoding = getattr(stream, "encoding", None)
    if not encoding:
        return False
    try:
        return codecs.lookup(encoding).name == "utf-8"
    except LookupError:
        return False


def _is_tty(stream: IO[str]) -> bool:
    try:
        return stream.isatty()
    except (AttributeError, ValueError):  # no isatty, or a closed stream
        return False


@dataclass(frozen=True)
class Style:
    """How to write labels to one stream. The default is plain."""

    color: bool = False
    unicode: bool = True

    @classmethod
    def for_stream(cls, stream: IO[str], *, github_actions: bool = True) -> "Style":
        """The style for this stream, from NO_COLOR, the terminal and GITHUB_ACTIONS.

        github_actions=False keeps a non-terminal stream plain even in GitHub Actions
        (doctor's stdout, which scripts may read).
        """
        if os.environ.get("NO_COLOR"):
            return cls()
        in_ci = github_actions and os.environ.get("GITHUB_ACTIONS") == "true"
        return cls(color=_is_tty(stream) or in_ci, unicode=_is_utf8(stream))

    @property
    def mark_width(self) -> int:
        """Columns a label's symbol adds before the word (0 when plain)."""
        return 2 if self.color else 0

    def label(self, word: str, width: int = 0) -> str:
        """The word padded to width; styled, with its symbol before it and in its colour."""
        padded = f"{word:<{width}}"
        if not self.color:
            return padded
        symbol, ascii_symbol, colour = STATUSES[word]
        mark = symbol if self.unicode else ascii_symbol
        return f"{click.style(mark, fg=colour)} {click.style(padded, fg=colour, bold=True)}"

    def paint(self, text: str, word: str) -> str:
        """Text in the colour of a status word, when styled."""
        return click.style(text, fg=STATUSES[word][2]) if self.color else text

    def echo(self, text: str, *, err: bool = False) -> None:
        """click.echo that keeps colour codes exactly when this style is coloured."""
        click.echo(text, err=err, color=self.color)
