from __future__ import annotations

from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


PROJECT_ROOT = Path(__file__).parents[1]
ROOT_REQUIREMENTS = (
    PROJECT_ROOT / "requirements-mujoco.txt",
    PROJECT_ROOT / "requirements-isaacsim.txt",
)


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
        destination = constraints if constraint_file else requirements
        if constraint_file:
            specifiers = list(parsed.specifier)
            assert len(specifiers) == 1 and specifiers[0].operator == "==", (
                f"{path}:{line_number}: constraint {parsed.name} must use "
                "one exact == pin"
            )
        normalized_name = canonicalize_name(parsed.name)
        previous = destination.setdefault(normalized_name, parsed)
        assert str(previous.specifier) == str(parsed.specifier), (
            f"{path}:{line_number}: conflicting pins for {parsed.name}"
        )


def _locked_requirements(root: Path) -> tuple[
    dict[str, Requirement],
    dict[str, Requirement],
]:
    requirements: dict[str, Requirement] = {}
    constraints: dict[str, Requirement] = {}
    _parse_requirement_tree(
        root,
        requirements,
        constraints,
    )
    return requirements, constraints


def test_requirement_profiles_are_reproducibly_locked() -> None:
    for root in ROOT_REQUIREMENTS:
        requirements, constraints = _locked_requirements(root)

        assert requirements, f"{root}: no requirements found"
        assert constraints, f"{root}: no constraints found"

        for name, requirement in requirements.items():
            constraint = constraints.get(name)
            effective = constraint or requirement
            specifiers = list(effective.specifier)
            assert (
                len(specifiers) == 1 and specifiers[0].operator == "=="
            ), (
                f"{root}: {requirement.name} must be pinned directly or by "
                "an exact constraint"
            )

            if constraint is not None:
                locked_version = specifiers[0].version
                assert requirement.specifier.contains(
                    locked_version,
                    prereleases=True,
                ), (
                    f"{root}: constraint {constraint} does not satisfy "
                    f"requirement {requirement}"
                )
