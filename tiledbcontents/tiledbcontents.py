import os
import json
import mimetypes
import datetime
import tiledb
import tiledb.cloud
import numpy

from tornado.web import HTTPError

from .ipycompat import ContentsManager
from .ipycompat import HasTraits, Unicode
from .ipycompat import reads, from_dict, GenericFileCheckpoints

DUMMY_CREATED_DATE = datetime.datetime.fromtimestamp(86400)
NBFORMAT_VERSION = 4


def base_model(path):
    """
    Taken from https://github.com/danielfrg/s3contents/blob/master/s3contents/genericmanager.py
    :return:
    """
    return {
        "name": path.rsplit("/", 1)[-1],
        "path": path,
        "writable": True,
        "last_modified": None,
        "created": None,
        "content": None,
        "format": None,
        "mimetype": None,
    }


def base_directory_model(path):
    """
    Taken from https://github.com/danielfrg/s3contents/blob/master/s3contents/genericmanager.py
    :return:
    """
    model = base_model(path)
    model.update(
        type="directory", last_modified=DUMMY_CREATED_DATE, created=DUMMY_CREATED_DATE,
    )
    return model


class TileDBContentsManager(ContentsManager, HasTraits):
    def __init__(self, **kwargs):
        self.vfs = tiledb.VFS()
        super(ContentsManager, self).__init__(**kwargs)

    def _save_notebook(self, model, uri):
        nb_contents = from_dict(model["content"])
        self.check_and_sign(nb_contents, uri)
        file_contents = json.dumps(model["content"]).encode("utf-8")

        self.__write_bytes_to_array(
            uri, file_contents, model.get("mimetype"), model.get("format")
        )

        self.validate_notebook_model(model)
        return model.get("message")

    def __create_array(self, uri):
        # The array will be be 1 dimensional with domain of 0 to max uint64. We use a tile extent of 1024 bytes
        dom = tiledb.Domain(
            tiledb.Dim(
                name="position",
                domain=(0, numpy.iinfo(numpy.uint64).max),
                tile=1024,
                dtype=numpy.uint64,
            )
        )

        # The array will be dense with a single attribute "a" so each (i,j) cell can store an integer.
        schema = tiledb.ArraySchema(
            domain=dom,
            sparse=False,
            attrs=[tiledb.Attr(name="contents", dtype=numpy.uint8)],
        )

        # Create the (empty) array on disk.
        tiledb.DenseArray.create(uri, schema)

    def __write_bytes_to_array(self, uri, contents, mimetype=None, format=None):

        if not self.vfs.is_dir(uri):
            self.__create_array(uri)

        with tiledb.open(uri, mode="w") as A:
            A[slice(0, len(contents) + 1)] = {"contents": contents}
            A.meta["file_size"] = len(contents)
            if mimetype is not None:
                A.meta["mimetype"] = mimetype
            if format is not None:
                A.meta["format"] = format

    def __save_file(self, model, uri):
        file_contents = model["content"]
        self.__write_bytes_to_array(
            uri, file_contents, model.get("mimetype"), model.get("format")
        )

    def __create_directory_and_group(self, path):
        """

        :param path:
        :return:
        """
        try:
            if not self.vfs.is_dir(path):
                self.vfs.create_dir(path)
            elif tiledb.object_type(path) == "group":
                return

            tiledb.group_create(path)
        except tiledb.TileDBError as e:
            raise HTTPError(500, e.message)

    def __notebook_from_array(self, uri, content=True):
        """
        Build a notebook model from database record.
        """
        model = base_model(uri)
        model["type"] = "notebook"
        # if self.fs.isfile(uri):
        #     model["last_modified"] = model["created"] = self.fs.lstat(path)["ST_MTIME"]
        # else:
        #     model["last_modified"] = model["created"] = DUMMY_CREATED_DATE
        if content:
            with tiledb.open(uri) as A:
                meta = A.meta
                file_content = A[slice(0, meta["file_size"] + 1)]
                nb_content = reads(
                    file_content["contents"].decode("utf-8"),
                    as_version=NBFORMAT_VERSION,
                )
                self.mark_trusted_cells(nb_content, uri)
                model["format"] = "json"
                model["content"] = nb_content
                self.validate_notebook_model(model)

        return model

    def __file_from_array(self, uri, content=True, format=None):
        """
        Build a notebook model from database record.
        """
        model = base_model(uri)

        # if self.fs.isfile(uri):
        #     model["last_modified"] = model["created"] = self.fs.lstat(path)["ST_MTIME"]
        # else:
        #     model["last_modified"] = model["created"] = DUMMY_CREATED_DATE
        if content:
            with tiledb.open(uri) as A:
                meta = A.meta
                model["mimetype"] = meta["mimetype"]
                if format in meta:
                    model["format"] = meta["format"]
                else:
                    model["format"] = format
                file_content = A[slice(0, meta["file_size"] + 1)]
                nb_content = file_content["contents"]
                model["content"] = nb_content
                self.validate_notebook_model(model)

        return model

    def __directory_model_from_path(self, path, content=False):
        self.log.debug(
            "S3contents.GenericManager._directory_model_from_path: path('%s') type(%s)",
            path,
            content,
        )
        model = base_directory_model(path)
        # if self.vfs.is_dir(path):
        #     lstat = self.fs.lstat(path)
        #     if "ST_MTIME" in lstat and lstat["ST_MTIME"]:
        #         model["last_modified"] = model["created"] = lstat["ST_MTIME"]
        if content:
            if not self.dir_exists(path):
                HTTPError(404, "{} does not exist".format(path))
            model["format"] = "json"
            model["content"] = self.__group_to_models(self.vfs.ls(path))
        return model

    def __group_to_models(self, paths):
        """
        Applies _notebook_model_from_s3_path or _file_model_from_s3_path to each entry of `paths`,
        depending on the result of `guess_type`.
        """
        ret = []
        for path in paths:
            # path = self.fs.remove_prefix(path, self.prefix)  # Remove bucket prefix from paths
            if os.path.basename(path) == self.fs.dir_keep_file:
                continue
            type_ = self.guess_type(path, allow_directory=True)
            if type_ == "notebook":
                ret.append(self.__notebook_from_array(path, False))
            elif type_ == "file":
                ret.append(self.__file_from_array(path, False, None))
            elif type_ == "directory":
                ret.append(self.__directory_model_from_path(path, False))
            else:
                HTTPError(500, "Unknown file type %s for file '%s'" % (type_, path))
        return ret

    def get(self, path, content=True, type=None, format=None):
        """Get a file or directory model."""

        if type is None:
            type = self.guess_type(path, allow_directory=True)
        if type == "notebook":
            return self.__notebook_from_array(path, False)
        elif type == "file":
            return self.__file_from_array(path, False, None)
        elif type == "directory":
            return self._directory_model_from_path(path, False)

    def save(self, model, path):
        """
        Save a file or directory model to path.
        Should return the saved model with no content.  Save implementations
        should call self.run_pre_save_hook(model=model, path=path) prior to
        writing any data.
        """
        self.run_pre_save_hook(model=model, path=path)

        path = path.strip("/")

        if "type" not in model:
            raise HTTPError(400, u"No file type provided")
        if "content" not in model and model["type"] != "directory":
            raise HTTPError(400, u"No file content provided")

        if model["type"] not in ("directory", "file", "notebook"):
            raise HTTPError(400, "Unhandled contents type: %s" % model["type"])

        try:
            if model["type"] == "notebook":
                validation_message = self._save_notebook(model, path)
            elif model["type"] == "file":
                validation_message = self.__save_file(model, path)
            else:
                validation_message = self.__create_directory_and_group(path)
        except Exception as e:
            self.log.error("Error while saving file: %s %s", path, e, exc_info=True)
            raise HTTPError(
                500, "Unexpected error while saving file: %s %s" % (path, e)
            )

        model = self.get(path, type=model["type"], content=False)
        if validation_message is not None:
            model["message"] = validation_message
        return model

    def delete_file(self, path):
        """Delete the file or directory at path."""
        if self.vfs.is_file(path):
            self.vfs.remove_file(path)
        else:
            self.vfs.remove_dir(path)

    def rename_file(self, old_path, new_path):
        """Rename a file or directory."""
        self.vfs.move_file(old_path, new_path)

    # ContentsManager API part 2: methods that have useable default
    # implementations, but can be overridden in subclasses.

    def dir_exists(self, path):
        """Does a directory exist at the given path?
        Like os.path.isdir
        Override this method in subclasses.
        Parameters
        ----------
        path : string
            The path to check
        Returns
        -------
        exists : bool
            Whether the path does indeed exist.
        """
        return self.vfs.is_dir(path) and tiledb.object_type(path) == "group"

    def is_hidden(self, path):
        """Is path a hidden directory or file?
        Parameters
        ----------
        path : string
            The path to check. This is an API path (`/` separated,
            relative to root dir).
        Returns
        -------
        hidden : bool
            Whether the path is hidden.
        """
        return False

    def file_exists(self, path=""):
        """Does a file exist at the given path?
        Like os.path.isfile
        Override this method in subclasses.
        Parameters
        ----------
        path : string
            The API path of a file to check for.
        Returns
        -------
        exists : bool
            Whether the file exists.
        """
        return self.vfs.is_file(path)

    def guess_type(self, path, allow_directory=True):
        """
        Guess the type of a file.

        Taken from https://github.com/danielfrg/s3contents/blob/master/s3contents/genericmanager.py

        If allow_directory is False, don't consider the possibility that the
        file is a directory.
        Parameters
        ----------
            obj: s3.Object or string
        """
        if path.endswith(".ipynb"):
            return "notebook"
        elif allow_directory and self.dir_exists(path):
            return "directory"
        else:
            return "file"
