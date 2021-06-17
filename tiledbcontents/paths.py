"""Tools for dealing with paths (and ancillary things like credentials)."""


import posixpath
import random
import string
from typing import List, Optional


NOTEBOOK_EXT = ".ipynb"
_NB_NO_DOT = NOTEBOOK_EXT[1:]


def tiledb_uri_from_path(path: str) -> str:
    """Builds a tiledb:// URI from a notebook cloud path.

    >>> tiledb_uri_from_path("cloud/owned/namespace/nbname")
    'tiledb://namespace/nbname'
    """

    parts = split(path)
    return f"tiledb://{parts[-2]}/{parts[-1]}"


_ID_CHARS = string.ascii_uppercase + string.digits


def generate_id(size: int = 6, chars: str = _ID_CHARS) -> str:
    """Generates a random-ish ID."""
    return "".join(random.choice(chars) for _ in range(size))


def increment_filename(filename: str, insert: str = "-") -> str:
    """Increments the number in a filename in the hopes of making it unique.

    >>> increment_filename("path-to-my.tar.gz")
    'path-to-my-1.tar.gz'
    >>> increment_filename("mr. number 9.ipynb", " ")
    'mr. number 10.ipynb'
    >>> increment_filename("no-extension-6")
    'no-extension-7'
    """
    basename, dot, ext = filename.rpartition(".")
    if ext != _NB_NO_DOT:
        # for non-ipynb files, assume that the first dot starts the suffix
        # (e.g. "something.tar.gz" -> "tar.gz"; not "gz")
        basename, dot, ext = filename.partition(".")

    before, sep, after = basename.rpartition(insert)
    if not sep:
        before, sep, after = after, insert, "0"
    try:
        after_int = int(after)
    except ValueError:
        # If the "after" is not an integer, we need to tack on our own int.
        before = before + sep + after
        after_int = 0
    return f"{before}{insert}{after_int + 1}{dot}{ext}"


def is_remote(path: str) -> bool:
    """Returns true if a path is remote; false if not.

    >>> is_remote("cloud/cuckooland")
    True
    >>> is_remote("cloud/clod/clone/clue/clip")
    True
    >>> is_remote("my/home/directory")
    False
    """
    return split(path)[0] == "cloud"


def is_remote_dir(path: str) -> bool:
    """Returns true if the path is a remote directory; false if not.

    >>> is_remote_dir("cloud/owned")
    True
    >>> is_remote_dir("cloud/disowned")
    False
    >>> is_remote_dir("cloud/public/something")
    True
    >>> is_remote_dir("cloud/shared/too/many/slashes")
    False
    """
    splits = split(path)
    if splits[0] != "cloud":
        return False
    if len(splits) == 1:
        return True
    if 3 < len(splits):
        return False
    return splits[1] in ("owned", "public", "shared")


def remove_prefix(path_prefix: str, path: str) -> str:
    """Removes a prefix and everything that comes before it from a path.

    >>> remove_prefix("and", "here and there")
    ' there'
    >>> remove_prefix("missing", "some other string")
    'some other string'
    """
    before, _, after = path.partition(path_prefix)
    return after or before


def extract_category(path: str) -> Optional[str]:
    """Pulls the category out of a Cloud path.

    >>> extract_category("cloud/owned/me/my-array")
    'owned'
    >>> extract_category("local/path/to/array")
    >>> extract_category("cloud")
    """
    parts = split(path)
    if parts[0] != "cloud":
        return None
    try:
        return parts[1]
    except IndexError:
        return None


def extract_namespace(path: str) -> Optional[str]:
    """Pulls the namespace out of a Cloud path.

    >>> extract_namespace("cloud/owned/me/my-array")
    'me'
    >>> extract_namespace("local/path/to/array")
    >>> extract_namespace("cloud/owned")
    """
    parts = split(path)
    if parts[0] != "cloud":
        return None
    try:
        return parts[2]
    except IndexError:
        return None


RESERVED_NAMES = frozenset(["cloud", "owned", "public", "shared"])


# Jupyter and URL-style path joining is literally just posixpath.join.
join = posixpath.join


def split(path: str) -> List[str]:
    """Splits a path into its component parts.

    >>> split("a/b/c")
    ['a', 'b', 'c']
    >>> split("")
    ['']
    """
    return path.split(posixpath.sep)


def strip(path: str) -> str:
    """Removes leading and trailing slashes from a path.

    Jupyter claims to do this already, but these are lies.

    https://jupyter-notebook.readthedocs.io/en/stable/extending/contents.html#apipaths

    >>> strip("/path/to/it")
    'path/to/it'
    >>> strip("////so many////")
    'so many'
    >>> strip("nothing")
    'nothing'
    """
    return path.strip(posixpath.sep)
