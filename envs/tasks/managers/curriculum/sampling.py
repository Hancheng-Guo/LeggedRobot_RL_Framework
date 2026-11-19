"""Resolve read-only curriculum dependencies for consumer managers."""
from collections.abc import Collection

from .types import CurriculumSampler, CurriculumTermProvider


def resolve_curriculum_samplers(
    names: Collection[str],
    provider: CurriculumTermProvider | None,
    owner: str,
) -> dict[str, CurriculumSampler]:
    if names and not isinstance(provider, CurriculumTermProvider):
        raise ValueError(f"{owner} requires a curriculum provider.")
    samplers = {}
    for name in names:
        assert provider is not None
        sampler = provider.get_term(name)
        if not isinstance(sampler, CurriculumSampler):
            raise TypeError(f"Curriculum term '{name}' must expose read-only get_bounds().")
        samplers[name] = sampler
    return samplers
