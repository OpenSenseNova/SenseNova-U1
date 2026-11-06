#!/usr/bin/env python3
"""Render pip-friendly direct dependency lists from the two pyprojects."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECTS = (
    (REPO_ROOT / "pyproject.toml", None, REPO_ROOT / "requirements.txt"),
    (REPO_ROOT / "training" / "pyproject.toml", None, REPO_ROOT / "training" / "requirements.txt"),
    (
        REPO_ROOT / "training" / "pyproject.toml",
        "flash-build",
        REPO_ROOT / "training" / "requirements-flash-build.txt",
    ),
    (
        REPO_ROOT / "training" / "pyproject.toml",
        "flash",
        REPO_ROOT / "training" / "requirements-flash.txt",
    ),
)


def requirement_name(requirement: str) -> str:
    match = re.match(r"[A-Za-z0-9_.-]+", requirement)
    if match is None:
        raise ValueError(f"Cannot determine package name from requirement: {requirement!r}")
    return match.group(0).lower().replace("_", "-")


def render_project_requirements(pyproject: Path, extra: str | None) -> str:
    with pyproject.open("rb") as file:
        project = tomllib.load(file)

    if extra is None:
        dependencies = project["project"]["dependencies"]
        source = "[project].dependencies"
    else:
        dependencies = project["project"]["optional-dependencies"][extra]
        source = f"[project.optional-dependencies].{extra}"

    uv = project.get("tool", {}).get("uv", {})
    indexes = {index["name"]: index["url"] for index in uv.get("index", [])}
    configured_sources = {name.lower().replace("_", "-"): config for name, config in uv.get("sources", {}).items()}
    index_urls = []
    for dependency in dependencies:
        package_source = configured_sources.get(requirement_name(dependency), {})
        index_name = package_source.get("index")
        if index_name is not None and indexes[index_name] not in index_urls:
            index_urls.append(indexes[index_name])

    relative_pyproject = pyproject.relative_to(REPO_ROOT)
    lines = [
        "# This file is generated from pyproject.toml by scripts/render_requirements.py.",
        f"# Source: {relative_pyproject} {source}",
        "# Direct dependencies only; use uv.lock for an exact reproducible environment.",
    ]
    if index_urls:
        lines.extend(["", *(f"--extra-index-url {url}" for url in index_urls)])
    lines.extend(["", *dependencies, ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail if generated files are stale")
    args = parser.parse_args()

    stale = []
    for pyproject, extra, output in PROJECTS:
        rendered = render_project_requirements(pyproject, extra)
        if args.check:
            if not output.is_file() or output.read_text() != rendered:
                stale.append(output.relative_to(REPO_ROOT))
        else:
            output.write_text(rendered)

    if stale:
        parser.error("stale generated requirements: " + ", ".join(map(str, stale)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
