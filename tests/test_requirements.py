from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


PROJECT_ROOT = Path(__file__).parents[1]
ROOT_REQUIREMENTS = PROJECT_ROOT / "requirements.txt"


def _parse_requirement_tree(
    path: Path,
    requirements: dict[str, Requirement],
    constraints: dict[str, Requirement],
    *,
    constraint_file: bool = False,
    visited: set[tuple[Path, bool]] | None = None,
) -> None:
    if visited is None:
        visited = set()
    key = (path.resolve(), constraint_file)
    if key in visited:
        return
    visited.add(key)

    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        directive = parts[0]
        if directive in {"-r", "--requirement", "-c", "--constraint"}:
            assert len(parts) == 2, f"{path}:{line_number}: missing path"
            child_is_constraint = directive in {"-c", "--constraint"}
            _parse_requirement_tree(
                path.parent / parts[1],
                requirements,
                constraints,
                constraint_file=child_is_constraint,
                visited=visited,
            )
            continue
        if directive.startswith("--"):
            # Package indexes affect resolution, not the locked package set.
            continue

        parsed = Requirement(line)
        assert parsed.url is None, (
            f"{path}:{line_number}: direct URL/path dependencies are not "
            "reproducible"
        )
        specifiers = list(parsed.specifier)
        assert len(specifiers) == 1 and specifiers[0].operator == "==", (
            f"{path}:{line_number}: {parsed.name} must use one exact == pin"
        )
        destination = constraints if constraint_file else requirements
        normalized_name = canonicalize_name(parsed.name)
        previous = destination.setdefault(normalized_name, parsed)
        assert str(previous.specifier) == str(parsed.specifier), (
            f"{path}:{line_number}: conflicting pins for {parsed.name}"
        )


def _locked_requirements() -> tuple[
    dict[str, Requirement],
    dict[str, Requirement],
]:
    requirements: dict[str, Requirement] = {}
    constraints: dict[str, Requirement] = {}
    _parse_requirement_tree(
        ROOT_REQUIREMENTS,
        requirements,
        constraints,
    )
    return requirements, constraints


def _installed_version(requirement: Requirement) -> str:
    try:
        return version(requirement.name)
    except PackageNotFoundError as error:
        raise AssertionError(
            f"Required distribution {requirement.name!r} is not installed"
        ) from error


def test_requirements_use_only_exact_reproducible_pins() -> None:
    requirements, constraints = _locked_requirements()

    assert requirements
    assert constraints


def test_environment_matches_locked_requirements() -> None:
    requirements, constraints = _locked_requirements()

    for requirement in requirements.values():
        installed = _installed_version(requirement)
        assert requirement.specifier.contains(installed, prereleases=True), (
            f"{requirement.name} {installed} does not match "
            f"{requirement.specifier}"
        )

    # Constraints do not install packages themselves. If a constrained
    # transitive dependency is present, however, its installed version must
    # exactly match the lock.
    for requirement in constraints.values():
        try:
            installed = version(requirement.name)
        except PackageNotFoundError:
            continue
        assert requirement.specifier.contains(installed, prereleases=True), (
            f"{requirement.name} {installed} does not match locked "
            f"constraint {requirement.specifier}"
        )
