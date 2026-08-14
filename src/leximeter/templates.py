"""Prompt-template loading with the shuffle filters pre-registered."""

from __future__ import annotations

from jinja2 import Environment, PackageLoader, Template

from .utils.jinja import shuffled, shuffled_dict

_ENV = Environment(loader=PackageLoader("leximeter", "prompts"))
_ENV.filters["shuffled"] = shuffled
_ENV.filters["shuffled_dict"] = shuffled_dict


def load_prompt(filename: str) -> Template:
    """Return a compiled Jinja2 template from the packaged prompts directory."""
    return _ENV.get_template(filename)
