"""Terminal status labels (agent/calbredge/term.py).

Styled only on a terminal with NO_COLOR unset, or in GitHub Actions; ASCII symbols when the
stream isn't UTF-8; unstyled, a label is exactly the word.
"""

import io

import click
import pytest
from calbredge.term import STATUSES, Style

pytestmark = pytest.mark.unit


class _Stream(io.StringIO):
    def __init__(self, *, tty: bool, encoding: str = "utf-8"):
        super().__init__()
        self._tty = tty
        self._encoding = encoding

    def isatty(self) -> bool:
        return self._tty

    @property
    def encoding(self) -> str:  # type: ignore[override]
        return self._encoding


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)


def test_a_terminal_is_styled():
    assert Style.for_stream(_Stream(tty=True)) == Style(color=True, unicode=True)


def test_a_pipe_or_file_is_plain():
    assert Style.for_stream(_Stream(tty=False)) == Style(color=False, unicode=True)


def test_no_color_turns_a_terminal_plain(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    assert Style.for_stream(_Stream(tty=True)).color is False


def test_an_empty_no_color_is_ignored(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "")
    assert Style.for_stream(_Stream(tty=True)).color is True


def test_github_actions_is_styled_without_a_terminal(monkeypatch):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    assert Style.for_stream(_Stream(tty=False)).color is True
    # ...unless the caller keeps that stream plain (doctor's stdout).
    assert Style.for_stream(_Stream(tty=False), github_actions=False).color is False


def test_no_color_wins_over_github_actions(monkeypatch):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("NO_COLOR", "1")
    assert Style.for_stream(_Stream(tty=False)).color is False


@pytest.mark.parametrize("encoding", ["ascii", "latin-1", "cp1252"])
def test_a_non_utf8_terminal_gets_ascii_symbols(encoding):
    style = Style.for_stream(_Stream(tty=True, encoding=encoding))
    assert style == Style(color=True, unicode=False)
    assert click.unstyle(style.label("FAIL")) == "x FAIL"
    style.label("CONDITIONAL_PASS").encode(encoding)  # nothing it can't print


def test_a_utf8_alias_is_utf8():
    assert Style.for_stream(_Stream(tty=True, encoding="UTF8")).unicode is True


def test_plain_labels_are_just_the_padded_word():
    assert Style().label("OK", 4) == "OK  "
    assert Style().paint("Ready.", "OK") == "Ready."
    assert Style().mark_width == 0


@pytest.mark.parametrize(
    ("word", "shown"),
    [
        ("OK", "✓ OK"),
        ("WARN", "⚠ WARN"),
        ("FAIL", "✗ FAIL"),
        ("PASS", "✓ PASS"),
        ("CONDITIONAL_PASS", "◐ CONDITIONAL_PASS"),
        ("REJECT", "✗ REJECT"),
        ("INCONCLUSIVE", "? INCONCLUSIVE"),
    ],
)
def test_styled_labels_carry_symbol_word_and_colour(word, shown):
    label = Style(color=True).label(word)
    assert click.unstyle(label) == shown
    assert "\x1b[" in label
    assert Style(color=True).mark_width == 2


def test_every_status_has_an_ascii_symbol():
    for symbol, ascii_symbol, _ in STATUSES.values():
        ascii_symbol.encode("ascii")
        assert symbol != ascii_symbol or symbol == "?"
