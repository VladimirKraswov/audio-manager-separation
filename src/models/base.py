from __future__ import annotations


class ModelUnavailableError(RuntimeError):
    """Raised when an optional external model adapter is not configured."""


def format_command(template: str, **values: object) -> list[str]:
    formatted = template.format(**values)
    return formatted.split()
