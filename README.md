# TileDB Cloud Jupyter Contents

This package contains a [juypterlab contents](https://jupyter-notebook.readthedocs.io/en/stable/extending/contents.html)
plugin to allow storing jupyter notebooks as TileDB arrays on TileDB Cloud. It also supports local filesystem access so
you can store notebooks locally as traditional json files or on the cloud as TileDB Arrays.

## Usage

To use the content manager, first you must install it:

```
python setup.py install
```

Next adjust your jupyter config usually `/etc/jupyter/jupyter_notebook_config.py` or you can create a
`jupyter_notebook_config.py` file in your current working directory.

Add the following line:
```
c.NotebookApp.contents_manager_class = "tiledbcontents.TileDBCloudContentsManager"
```

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
