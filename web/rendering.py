"""Jinja2 environment helper for the v2 web shell.

Every v2 page renders through a single :class:`jinja2.Environment`
configured here so that the base template, partials, and per-tab
pages share one loader, one set of filters, and one autoescape
policy. Keeping this out of :mod:`api.app` means schema-only tests
do not need a filesystem templates directory on disk.

Why a dedicated module:

- **Single source of truth for paths.** The templates/static
  directories live next to this module; tests can call
  :func:`get_environment` with no args and get the same environment
  that the FastAPI app uses.
- **Autoescape on by default.** DESIGN.md's "evidence is a feature"
  principle means we surface raw user/event text inside Moment
  cards. Jinja autoescape on HTML/XML prevents inadvertent HTML
  injection from event payloads before the dedicated sanitizer
  lands with the Moment-card partial.
- **Deterministic trimming.** ``trim_blocks`` + ``lstrip_blocks``
  keep template output stable across refactors so snapshot-style
  tests don't drift on whitespace.

Notes
-----
The Jinja ``Environment`` is cached per (templates_dir) tuple for the
lifetime of the process — rebuilding an environment on every request
would defeat template compilation caching. Tests that need an isolated
environment pass an explicit ``templates_dir`` to bypass the cache.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

_WEB_DIR = Path(__file__).resolve().parent
DEFAULT_TEMPLATES_DIR: Path = _WEB_DIR / "templates"
DEFAULT_STATIC_DIR: Path = _WEB_DIR / "static"


@lru_cache(maxsize=4)
def get_environment(templates_dir: str | None = None) -> Environment:
    """Return the cached Jinja2 environment for the v2 web shell.

    Parameters
    ----------
    templates_dir:
        Absolute or relative path to the directory the ``FileSystemLoader``
        should serve. When ``None`` (the default) the built-in
        ``web/templates`` directory is used. Tests pass their own path
        (e.g. a ``tmp_path``) to exercise the loader without relying on
        the packaged templates.

    Returns
    -------
    jinja2.Environment
        A cached environment with autoescape enabled for HTML/XML,
        whitespace-trimmed blocks, and ``StrictUndefined`` disabled so
        the base template renders cleanly when optional context
        variables (``now_date``, ``now_time``, ``active_tab``) are
        absent.
    """

    resolved = Path(templates_dir) if templates_dir else DEFAULT_TEMPLATES_DIR
    env = Environment(
        loader=FileSystemLoader(str(resolved)),
        autoescape=select_autoescape(("html", "htm", "xml")),
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    return env


def render(template_name: str, context: dict | None = None, *, templates_dir: str | None = None) -> str:
    """Render ``template_name`` with the given context.

    Thin convenience wrapper around :func:`get_environment`; route
    handlers will call this directly so the import surface in each
    route module stays one line.
    """

    env = get_environment(templates_dir)
    template = env.get_template(template_name)
    return template.render(**(context or {}))


__all__ = [
    "DEFAULT_STATIC_DIR",
    "DEFAULT_TEMPLATES_DIR",
    "get_environment",
    "render",
]
