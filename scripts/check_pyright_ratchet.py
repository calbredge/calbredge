"""Hold pyright's per-file error counts at or below the committed ones.

Usage: python scripts/check_pyright_ratchet.py --counts <counts file> --report <pyright JSON>
           [--root <dir>] [--summary <file>] [--update [--allow-rise]]

The report comes from

    pyright --outputjson <directories> > <report>

and the counts file records the pyright version and, for every file with errors, how many.
Warnings aren't counted. The check fails when:

- a file has more errors than recorded;
- a file with errors isn't recorded;
- a file has fewer errors than recorded, or a recorded file has none or is gone: the fall is
  kept by recording it, with --update;
- the report's pyright version isn't the recorded one.

--update rewrites the counts file from the report. It refuses to raise a count or to add a
file unless --allow-rise is given, and leaves the file unchanged when it refuses.

--root is the checkout the report was made in (default: the current directory); the report's
absolute paths are recorded relative to it. --summary appends the pyright version, the Python
version, the files analysed and the error total to a file, as Markdown.

Exit 0 when the check passes or the counts were rewritten, 1 when it fails or the rewrite is
refused (one line per problem), 2 on a usage error, an unreadable file or a report in which
pyright analysed nothing. Standard library only.
"""

import argparse
import json
import platform
import sys
from pathlib import Path, PurePosixPath


def count_errors(report: dict, root: str) -> dict[str, int]:
    """Errors per file, each path relative to root, sorted by path."""
    counts: dict[str, int] = {}
    for diagnostic in report["generalDiagnostics"]:
        if diagnostic["severity"] != "error":
            continue
        path = PurePosixPath(diagnostic["file"])
        try:
            relative = str(path.relative_to(root))
        except ValueError:
            raise ValueError(
                f"{path} is not under {root}: pass --root, the checkout the report was made in"
            ) from None
        counts[relative] = counts.get(relative, 0) + 1
    return dict(sorted(counts.items()))


def files_analysed(report: dict) -> int:
    analysed = report["summary"]["filesAnalyzed"]
    if not isinstance(analysed, int) or analysed < 1:
        raise ValueError(f"pyright analysed no files (filesAnalyzed: {analysed!r})")
    return analysed


def load_counts(text: str) -> tuple[str, dict[str, int]]:
    """The recorded pyright version and per-file counts."""
    recorded = json.loads(text)
    version, errors = recorded["pyright"], recorded["errors"]
    if not isinstance(version, str) or not isinstance(errors, dict):
        raise ValueError("the counts file needs a 'pyright' version and an 'errors' object")
    for path, count in errors.items():
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ValueError(f"{path}: the recorded count must be a positive integer: {count!r}")
    return version, errors


def dump_counts(version: str, errors: dict[str, int]) -> str:
    return json.dumps({"pyright": version, "errors": dict(sorted(errors.items()))}, indent=2) + "\n"


def _errors(count: int) -> str:
    return "1 error" if count == 1 else f"{count} errors"


def rises(recorded: dict[str, int], current: dict[str, int]) -> list[str]:
    """A problem for each file with more errors than recorded, or with errors and no record."""
    problems = []
    for path, count in current.items():
        if path not in recorded:
            problems.append(f"{path}: {_errors(count)}, and the file is not recorded")
        elif count > recorded[path]:
            problems.append(f"{path}: {_errors(count)}, {recorded[path]} recorded")
    return problems


def falls(recorded: dict[str, int], current: dict[str, int]) -> list[str]:
    """A problem for each recorded file that now has fewer errors, none, or is gone."""
    problems = []
    for path, count in recorded.items():
        now = current.get(path, 0)
        if now < count:
            problems.append(
                f"{path}: {_errors(now)}, {count} recorded; record the fall with --update"
            )
    return problems


def check(recorded_version: str, recorded: dict[str, int], report: dict, root: str) -> list[str]:
    """Every problem; an empty list passes."""
    problems = []
    if report["version"] != recorded_version:
        problems.append(
            f"pyright {report['version']} made the report, but the counts are from"
            f" pyright {recorded_version}"
        )
    current = count_errors(report, root)
    return problems + rises(recorded, current) + falls(recorded, current)


def summary(report: dict, root: str) -> str:
    total = sum(count_errors(report, root).values())
    return (
        "### pyright ratchet\n\n"
        "| | |\n|---|---|\n"
        f"| pyright | {report['version']} |\n"
        f"| Python | {platform.python_version()} |\n"
        f"| Files analysed | {files_analysed(report)} |\n"
        f"| Errors | {total} |\n"
    )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Fail when pyright's per-file error counts differ from the committed ones."
    )
    parser.add_argument("--counts", required=True, type=Path, help="the committed counts file")
    parser.add_argument("--report", required=True, type=Path, help="pyright's --outputjson report")
    parser.add_argument("--root", default=None, help="the checkout the report was made in")
    parser.add_argument("--summary", type=Path, default=None, help="append a Markdown summary")
    parser.add_argument("--update", action="store_true", help="rewrite the counts file")
    parser.add_argument(
        "--allow-rise",
        action="store_true",
        help="with --update: accept a higher count and a new file",
    )
    args = parser.parse_args(argv)
    if args.allow_rise and not args.update:
        parser.error("--allow-rise only applies with --update")
    root = args.root if args.root is not None else str(Path.cwd())

    try:
        report = json.loads(args.report.read_text())
        files_analysed(report)
        current = count_errors(report, root)
        version = report["version"]
        if not isinstance(version, str):
            raise ValueError(f"the report's pyright version is not a string: {version!r}")
        if args.summary is not None:
            with args.summary.open("a") as handle:
                handle.write(summary(report, root))
        if args.update:
            recorded = load_counts(args.counts.read_text())[1] if args.counts.exists() else {}
            problems = [] if args.allow_rise else rises(recorded, current)
        else:
            recorded_version, recorded = load_counts(args.counts.read_text())
            problems = check(recorded_version, recorded, report, root)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"check_pyright_ratchet: {error!r}", file=sys.stderr)
        return 2

    total = sum(current.values())
    if args.update:
        if problems:
            print(f"{args.counts} not rewritten: a count would rise (--allow-rise accepts it):")
            for problem in problems:
                print(f"  {problem}")
            return 1
        args.counts.write_text(dump_counts(version, current))
        print(f"{args.counts} rewritten: {total} errors in {len(current)} files")
        return 0
    if problems:
        print(f"pyright ratchet failed against {args.counts}:")
        for problem in problems:
            print(f"  {problem}")
        return 1
    print(f"pyright ratchet passed: {total} errors in {len(current)} files, as recorded")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
