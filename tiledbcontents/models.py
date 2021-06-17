"""Dealing with models."""

import datetime
from typing import Dict, Any
import posixpath

DUMMY_DATE = datetime.datetime.fromtimestamp(0, tz=datetime.timezone.utc)

Model = Dict[str, Any]


def create(*, path: str, **kwargs: Any) -> Model:
    """Create the most basic model."""
    # Originally from:
    # https://github.com/danielfrg/s3contents/blob/master/s3contents/genericmanager.py
    model = {
        "name": posixpath.basename(path),
        "path": path,
        "writable": True,
        "last_modified": DUMMY_DATE,
        "created": DUMMY_DATE,
        "content": None,
        "format": None,
        "mimetype": None,
    }
    model.update(**kwargs)
    return model


def to_utc(d: datetime.datetime) -> datetime.datetime:
    """Returns a version of d converted to UTC.

    If naive (no timezone), UTC will be added; if timezone-aware, the time
    will be converted.
    """
    if not d.tzinfo:
        return d.replace(tzinfo=datetime.timezone.utc)
    return d.astimezone(datetime.timezone.utc)
