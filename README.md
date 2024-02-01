# TileDB Cloud Jupyter Contents

This package contains a [juypterlab contents](https://jupyter-notebook.readthedocs.io/en/stable/extending/contents.html)
plugin to allow storing jupyter notebooks as TileDB arrays on TileDB Cloud. It also supports local filesystem access so
you can store notebooks locally as traditional json files or on the cloud as TileDB Arrays.


## Usage

To use the content manager, first you must install it:

```
pip install .
```

> Prefer installation via pip. The old way `python setup.up install` fetches dev versions of packages and may
break the build. Pip always fetches stable versions.

Next adjust your jupyter config usually `/etc/jupyter/jupyter_notebook_config.py` or you can create a
`jupyter_notebook_config.py` file in your current working directory.

Add the following line:
```
c.ServerApp.contents_manager_class = "tiledbcontents.AsyncTileDBCloudContentsManager"
```

## Local testing

The easiest way to test this locally is to work in a virtual environment.
You can use your preferred virtual environment management tool (e.g. Conda),
or to set up a fresh venv, run:

```
$ cd path/to/TileDB-Cloud-Jupyter-Contents
$ python3 -m venv venv
$ . ./venv/bin/activate
```

Install all the dependencies, and set yourself up for development:

```
$ pip install --editable .
[...]
$ pip install jupyterlab
[...]
```

Jupyter itself is not included in the `install_requires` because while it is
useful for development and testing, this code does not actually depend upon it.
Now you have all the prerequisites installed, and the `tiledbcontents` package
points to the files in your working directory.

If you have not already logged into TileDB Cloud, you will need to do so.
From a Python console, run [the `tiledb.cloud.login`
function](https://docs.tiledb.com/cloud/api-reference/utilities#login-sessions).
In this example we use [an API token that we
created](https://cloud.tiledb.com/settings/tokens), but you can also use a
username and password.

```
$ python
>>> import tiledb.cloud
>>> # This is a dummy value; replace it with your own token or user/pass.
>>> tiledb.cloud.login(token='aHR0cHM6Ly95b3V0dS5iZS9vSGc1U0pZUkhBMA==')
```

This will store a credential in your home directory that will be used by default
when accessing TileDB Cloud services through the API.

The only remaining step is to launch Jupyter and configure it to use the
`AsyncTileDBCloudContentsManager` class:

```
$ jupyter lab --ServerApp.contents_manager_class=tiledbcontents.AsyncTileDBCloudContentsManager
```

This will run a local test Jupyter lab server without changing your default
Jupyter configuration. You can make this your default by following the
instructions in "Usage" above.

Now the server is running, and you just need to follow Jupyter's directions to
connect! Certain actions (like file navigation) may take longer than you would
expect, because it has to communicate with the remote TileDB Cloud server.

## How it Works

The package works by storing the notebook in a dense array with certain metadata to indicate the current size
and type. We also set certain TileDB Cloud tags to associate the array as a Jupyter notebook.

### TileDB Array Schema

The array is created as a 1 dimensional dense array with a single attribute of contents, which is the bytes of the
notebook json format.

```
schema = tiledb.ArraySchema(
    domain=dom,
    sparse=True,
    attrs=[tiledb.Attr(name="contents", dtype=numpy.uint8, filters=tiledb.FilterList(tiledb.ZstdFilter()))],
    ctx=tiledb.cloud.Ctx(),
)
```

### Listing

The listing of cloud notebooks happens through the traditional array listings. We use the special tag of
`__jupyter-notebook`, to filter for arrays which are actually notebooks.

The listings show up under the "cloud" folder of the notebook file browser.
