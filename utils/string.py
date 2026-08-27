import re


def camel_to_snake(name: str) -> str:
    return re.sub(r"([A-Z])", r"_\1", name).lstrip("_").lower()
