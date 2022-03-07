"""Dealing with models."""

import datetime
import posixpath
from typing import Any, Dict, Iterable, Optional, TypeVar

_T = TypeVar("_T")

Model = Dict[str, Any]


def create(*, path: str, **kwargs: Any) -> Model:
    """Create the most basic model."""
    # Originally from:
    # https://github.com/danielfrg/s3contents/blob/master/s3contents/genericmanager.py
    model = {
        "name": posixpath.basename(path),
        "path": path,
        "writable": True,
        "last_modified": None,
        "created": None,
        "content": None,
        "format": None,
        "mimetype": None,
    }
    model.update(**kwargs)
    return model


def to_utc(d: Optional[datetime.datetime]) -> Optional[datetime.datetime]:
    """Returns a version of d converted to UTC.

    If naive (no timezone), UTC will be added; if timezone-aware, the time
    will be converted.
    """
    if not d:
        return None
    if not d.tzinfo:
        return d.replace(tzinfo=datetime.timezone.utc)
    return d.astimezone(datetime.timezone.utc)


def max_present(stuff: Iterable[Optional[_T]]) -> Optional[_T]:
    """A version of ``max`` that filters out Nones.

    Returns ``None`` if the sequence is empty.
    """
    try:
        return max(  # type: ignore[type-var]
            thing for thing in stuff if thing is not None
        )
    except ValueError:
        # max raises this when given an empty sequence.
        return None


def fill_in_dates(m: Model) -> Model:
    """Fills in the ``created`` and ``last_modified`` fields if empty.

    Many parts of Jupyter throw a temper tantrum if either the ``created`` field
    or the ``last_modified`` field of a model is empty.  If either is missing,
    we replace it with now.

    Modifies and returns the passed-in object, so you can use it like:

        return fill_in_dates(build_model())
    """
    now = datetime.datetime.now(tz=datetime.timezone.utc)
    for prop in ("last_modified", "created"):
        if m.get(prop) is None:
            m[prop] = now
    return m
