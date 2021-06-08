"""
Utilities for managing compat between notebook versions.

Taken from: https://github.com/danielfrg/s3contents/blob/master/s3contents/ipycompat.py
Which original took it from: https://github.com/quantopian/pgcontents/blob/master/pgcontents/utils/ipycompat.py
"""

from nbformat import from_dict
from nbformat import reads
from nbformat import writes
from nbformat.v4.nbbase import new_code_cell
from nbformat.v4.nbbase import new_markdown_cell
from nbformat.v4.nbbase import new_notebook
from nbformat.v4.nbbase import new_raw_cell
from nbformat.v4.rwbase import strip_transient
from notebook.services.contents.checkpoints import Checkpoints
from notebook.services.contents.checkpoints import GenericCheckpointsMixin
from notebook.services.contents.filecheckpoints import GenericFileCheckpoints
from notebook.services.contents.filemanager import FileContentsManager
from notebook.services.contents.manager import ContentsManager
from notebook.services.contents.tests.test_contents_api import APITest
from notebook.services.contents.tests.test_manager import TestContentsManager
from notebook.tests.launchnotebook import assert_http_error
from notebook.utils import to_os_path
from traitlets import Any
from traitlets import Bool
from traitlets import Dict
from traitlets import HasTraits
from traitlets import Instance
from traitlets import Integer
from traitlets import Unicode
from traitlets.config import Config

__all__ = [
    "APITest",
    "Any",
    "assert_http_error",
    "Bool",
    "Checkpoints",
    "Config",
    "ContentsManager",
    "Dict",
    "FileContentsManager",
    "GenericCheckpointsMixin",
    "GenericFileCheckpoints",
    "HasTraits",
    "Instance",
    "Integer",
    "TestContentsManager",
    "Unicode",
    "from_dict",
    "new_code_cell",
    "new_markdown_cell",
    "new_notebook",
    "new_raw_cell",
    "reads",
    "strip_transient",
    "to_os_path",
    "writes",
]
