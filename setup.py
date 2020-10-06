# coding: utf-8

"""
    TileDB JupyterLab Contents plugin
"""

from setuptools import setup, find_namespace_packages

NAME = "tiledbcontents"

# To install the library, run the following
#
# python setup.py install
#
# prerequisite: setuptools
# http://pypi.python.org/pypi/setuptools

REQUIRES = [
    "notebook",
    "ipykernel",
    "setuptools>=18.0",
    "setuptools-scm>=1.5.4",
    "setuptools-scm-git-archive",
    "tiledb>=0.6.6",
    "tiledb-cloud>=0.6.4",
]

setup(
    name=NAME,
    description="TileDB Contents Plugin for storing Jupyterlab Notebooks in TileDB Arrays",
    author_email="hello@tiledb.com",
    url="https://tiledb.com",
    keywords=["TileDB", "cloud", "jupyter", "notebook"],
    install_requires=REQUIRES,
    packages=find_namespace_packages(include=["tiledbcontents"]),
    include_package_data=True,
    zip_safe=False,
    use_scm_version={
        "version_scheme": "guess-next-dev",
        "local_scheme": "dirty-tag",
        "write_to": "tiledbcontents/version.py",
    },
    long_description="""\
    TileDB Contents Plugin for storing Jupyterlab Notebooks in TileDB Arrays
    """,
)
