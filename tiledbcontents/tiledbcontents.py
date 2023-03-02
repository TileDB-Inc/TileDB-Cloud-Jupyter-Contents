import base64
import json
import posixpath
from typing import List

import jupyter_server.files.handlers as jsfh
from jupyter_server.services.contents import filecheckpoints
from jupyter_server.services.contents import filemanager
import nbformat
import tiledb
import tiledb.cloud
import tornado.web
import traitlets

from . import arrays
from . import caching
from . import listings
from . import models
from . import paths

NBFORMAT_VERSION = 4

NOTEBOOK_MIME = "application/x-ipynb+json"


class AsyncTileDBCloudContentsManager(
    filemanager.AsyncFileContentsManager, traitlets.HasTraits
):
    # This makes the checkpoints get saved on this directory
    root_dir = traitlets.Unicode("./", config=True)

    # These are needed to reset the file downloader so that it actually works
    # rather than intercepting everything with a downloader that is only aware
    # of the filesystem.
    files_handler_class = traitlets.Type(jsfh.FilesHandler)
    files_handler_params = traitlets.Dict({})

    @traitlets.default("checkpoints_class")
    def _checkpoints_class_default(self):
        return AsyncTileDBCheckpoints

    async def _dir_model(self, path, content: bool = False):
        if not paths.is_remote(path) and not paths.is_remote_dir(path):
            return await super()._dir_model(path, content)

        if path == "cloud":
            cloud = models.create(path="cloud", type="directory")
            if content:
                cloud["format"] = "json"
                cloud["content"] = await caching.call(listings.all_notebooks)

                cloud["last_modified"] = models.max_present(
                    models.to_utc(cat.get("last_modified")) for cat in cloud["content"]
                )

            return cloud

        category, namespace = paths.category_namespace(path)
        if category:
            if namespace:
                return await caching.call(
                    listings.namespace, category, namespace, content=content
                )
            return await caching.call(listings.category, category, content=content)

        return models.create(
            path=path,
            type="directory",
            # last_modified=models.DUMMY_DATE,
            # created=models.DUMMY_DATE,
        )

    async def get(self, path, content=True, type=None, format=None):
        """Get a file or directory model."""
        path = paths.strip(path)
        try:
            if not paths.is_remote(path):
                # If this isn't a remote path, get the local file contents.
                model = await super().get(path, content, type, format)
                if path == "" and content and await caching.call(_is_cloud_enabled):
                    # If we're at the root, and there's an existing entry
                    # named "cloud", remove it from the list.
                    file_list: List[models.Model] = model["content"]
                    for idx in range(len(file_list)):
                        if file_list[idx]["path"] == "cloud":
                            file_list.pop(idx)
                            break

                    # Put our own "cloud" folder in.
                    cloud_content = await caching.call(listings.all_notebooks)
                    file_list.append(
                        models.create(
                            path="cloud",
                            type="directory",
                            format="json",
                            last_modified=models.max_present(
                                models.to_utc(cat.get("last_modified"))
                                for cat in cloud_content
                            ),
                        )
                    )

                return model

            if path.endswith(paths.NOTEBOOK_EXT):
                path = path[: -1 * len(paths.NOTEBOOK_EXT)]

            if type is None:
                if paths.is_remote_dir(path):
                    type = "directory"
                else:
                    type = await self.guess_type(path, allow_directory=True)

            if type == "notebook":
                return models.fill_in_dates(
                    await self._notebook_from_array(path, content=content)
                )
            if type == "file":
                return models.fill_in_dates(
                    await self._file_from_array(path, content=content, format=format)
                )
            if type == "directory":
                return models.fill_in_dates(
                    await self._dir_model(path, content=content)
                )
        except Exception as e:
            raise tornado.web.HTTPError(
                500, "Error opening notebook {}: {}".format(path, str(e))
            )

    async def save(self, model, path=""):
        """
        Save a file or directory model to path.
        Should return the saved model with no content.  Save implementations
        should call self.run_pre_save_hook(model=model, path=path) prior to
        writing any data.
        """
        path = paths.strip(path)
        try:
            model_type = model["type"]
        except KeyError:
            raise tornado.web.HTTPError(400, "No file type provided")

        if "content" not in model and model_type != "directory":
            raise tornado.web.HTTPError(400, "No file content provided")

        if model_type not in ("directory", "file", "notebook"):
            raise tornado.web.HTTPError(
                400, "Unhandled contents type: %s" % model["type"]
            )

        if not paths.is_remote(path):
            if model.get("tiledb:is_new"):
                # Since we don't try to increment the filename in self.new(),
                # do it here for newly-created files.
                dir, name = posixpath.split(path)
                incremented = await self.increment_filename(name, dir, insert="-")
                path = paths.join(dir, incremented)
            return await super().save(model, path)

        if path.endswith(paths.NOTEBOOK_EXT):
            path = path[: -len(paths.NOTEBOOK_EXT)]
            if model["type"] == "file":
                try:
                    _try_convert_file_to_notebook(model)
                except ValueError as ve:
                    raise tornado.web.HTTPError(
                        400, f"Cannot parse Jupyter notebook: {ve}"
                    )

        self.run_pre_save_hook(model=model, path=path)
        validation_message = None
        try:
            if model["type"] == "notebook":
                final_name, validation_message = await self._save_notebook_tiledb(
                    path, model
                )
                if final_name is not None:
                    parts = paths.split(path)
                    parts_length = len(parts)
                    parts[parts_length - 1] = final_name
                    path = paths.join(*parts)
            elif model["type"] == "file":
                raise tornado.web.HTTPError(
                    400, "Only .ipynb files may be created in the cloud."
                )
            else:
                if paths.is_remote(path):
                    raise tornado.web.HTTPError(
                        400,
                        "Trying to create unsupported type: %s in cloud"
                        % model["type"],
                    )
                # else:
                #     return super().save(model, path)
                # validation_message = self.__create_directory_and_group(path)
        except Exception as e:
            self.log.error("Error while saving file: %s %s", path, e, exc_info=True)
            raise

        model = await self.get(path, type=model["type"], content=False)
        if validation_message is not None:
            model["message"] = validation_message
        return model

    async def delete_file(self, path):
        """Delete the file or directory at path."""
        path = paths.strip(path)
        if paths.is_remote(path):

            if path.endswith(paths.NOTEBOOK_EXT):
                path = path[: -1 * len(paths.NOTEBOOK_EXT)]

            tiledb_uri = paths.tiledb_uri_from_path(path)
            try:
                caching.Array.purge(tiledb_uri)
                return await caching.call(tiledb.cloud.array.delete_array, tiledb_uri)
            except tiledb.cloud.tiledb_cloud_error.TileDBCloudError as e:
                raise tornado.web.HTTPError(
                    500, f"Error deregistering {tiledb_uri!r}: {e}"
                )
            except tiledb.TileDBError as e:
                raise tornado.web.HTTPError(
                    500,
                    str(e),
                )
        return await super().delete_file(path)

    async def rename_file(self, old_path, new_path):
        """Rename a file or directory."""
        old_path = paths.strip(old_path)
        new_path = paths.strip(new_path)
        if paths.is_remote(old_path):

            if old_path.endswith(paths.NOTEBOOK_EXT):
                old_path = old_path[: -1 * len(paths.NOTEBOOK_EXT)]

            tiledb_uri = paths.tiledb_uri_from_path(old_path)
            parts_new = paths.split(new_path)
            parts_new_length = len(parts_new)
            array_name_new = parts_new[parts_new_length - 1]

            try:
                caching.Array.purge(tiledb_uri)
                return await caching.call(
                    tiledb.cloud.notebook.rename_notebook, tiledb_uri, array_name_new
                )
            except tiledb.cloud.tiledb_cloud_error.TileDBCloudError as e:
                raise tornado.web.HTTPError(500, f"Error renaming {tiledb_uri!r}: {e}")
            except tiledb.TileDBError as e:
                raise tornado.web.HTTPError(
                    500,
                    str(e),
                )
        return await super().rename_file(old_path, new_path)

    # ContentsManager API part 2: methods that have usable default
    # implementations, but can be overridden in subclasses.

    async def dir_exists(self, path):
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
        path = paths.strip(path)
        if paths.is_remote_dir(path):
            return True

        return await super().dir_exists(path)

    async def is_hidden(self, path):
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
        path = paths.strip(path)
        if paths.is_remote(path):
            return False

        return await super().is_hidden(path)

    async def file_exists(self, path=""):
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
        path = paths.strip(path)
        if paths.is_remote(path):
            if path.endswith(paths.NOTEBOOK_EXT):
                path = path[: -1 * len(paths.NOTEBOOK_EXT)]
            return await caching.call(arrays.exists, path)
        return await super().file_exists(path)

    async def new_untitled(self, path="", type="", ext="", options=""):
        """Create a new untitled file or directory in path
        path must be a directory
        File extension can be specified.
        Use `new` to create files with a fully specified path (including filename).
        options is a json string passed by the TileDB Prompt User Contents Jupyterlab notebook extension for additional notebook creation options
        """
        path = paths.strip(path)
        if not self.dir_exists(path):
            raise tornado.web.HTTPError(404, "No such directory: %s" % path)

        model = {}
        if type:
            model["type"] = type

        if ext == ".ipynb":
            model.setdefault("type", "notebook")
        else:
            model.setdefault("type", "file")

        if options:
            try:
                options_json = json.loads(options)
                model["name"] = options_json["name"]
                model["tiledb:s3_prefix"] = options_json["s3_prefix"]
                try:
                    model["tiledb:s3_credentials"] = options_json["s3_credentials"]
                except KeyError:
                    # Enterprise users might use a backend where credentials
                    # are not required.
                    pass
            except Exception as e:
                raise tornado.web.HTTPError(
                    400, "Could not read TileDB user options: {}".format(e)
                )

        if model["type"] == "directory":
            prefix = self.untitled_directory
        elif model["type"] == "notebook":
            prefix = model.get("name", self.untitled_notebook)
            ext = ".ipynb"
        elif model["type"] == "file":
            prefix = self.untitled_file
        else:
            raise tornado.web.HTTPError(
                400, "Unexpected model type: %r" % model["type"]
            )

        # We don't do the "increment" step that the default ContentsManager does
        # because we generate a random suffix or increment the filename in
        # _save_notebook_tiledb.
        full_path = paths.join(path, prefix + ext)
        return await self.new(model, full_path)

    async def new(self, model=None, path=""):
        if model is None:
            model = {}
        model["tiledb:is_new"] = True
        return models.fill_in_dates(await super().new(model, path))

    async def copy(self, from_path, to_path=None):
        from_path = paths.strip(from_path)
        model = self.get(from_path)
        model.pop("path", None)
        if not to_path:
            # A missing to_path implies that we should create a duplicate
            # in the same location (with a new name).
            to_path = from_path
        else:
            to_path = paths.strip(to_path)
            if await self.dir_exists(to_path):
                # to_path may be a directory, in which case we copy
                # the model to an identically-named entry in that directory.
                from_parts = paths.split(from_path)
                from_filename = from_parts[-1]
                to_path = paths.join(to_path, from_filename)

        # As in new_untitled, we don't increment our filenames because they are
        # dedup'd in _save_notebook_tiledb.

        return await self.new(model, to_path)

    async def _save_notebook_tiledb(self, path: str, model: models.Model):
        """
        Save a notebook to tiledb array
        :param model: model notebook
        :param uri: URI of notebook
        :return: any messages
        """
        nb_contents = nbformat.from_dict(model["content"])
        self.check_and_sign(nb_contents, path)

        try:
            final_name = await arrays.write_data(
                path,
                json.dumps(model["content"]),
                mimetype=model.get("mimetype"),
                format=model.get("format"),
                type="notebook",
                s3_prefix=model.get("tiledb:s3_prefix", None),
                s3_credentials=model.get("tiledb:s3_credentials", None),
                is_user_defined_name="name" in model,
                is_new=model.get("tiledb:is_new", False),
            )
        except tiledb.TileDBError as tdbe:
            raise tornado.web.HTTPError(
                500, f"Error saving notebook to TileDB array: {tdbe}"
            ) from tdbe
        except Exception as ex:
            raise tornado.web.HTTPError(
                500, f"Unexpected error saving notebook: {ex}"
            ) from ex

        self.validate_notebook_model(model)
        return final_name, model.get("message")

    async def _notebook_from_array(self, path, content=True):
        """
        Build a notebook model from database record.
        """
        model = models.create(path=path, type="notebook")
        if content:
            tiledb_uri = paths.tiledb_uri_from_path(path)
            try:
                info = await caching.call(tiledb.cloud.array.info, tiledb_uri)
                model["last_modified"] = models.to_utc(info.last_accessed)
                if "write" not in info.allowed_actions:
                    model["writable"] = False

                arr = caching.Array.from_cache(tiledb_uri)

                nb_content = []
                file_content = await arr.read()
                if file_content is not None:
                    nb_content = nbformat.reads(
                        file_content,
                        as_version=NBFORMAT_VERSION,
                    )
                    self.mark_trusted_cells(nb_content, path)
                model["format"] = "json"
                model["content"] = nb_content
                self.validate_notebook_model(model)
            except tiledb.cloud.tiledb_cloud_error.TileDBCloudError as e:
                raise tornado.web.HTTPError(
                    400, "Error fetching notebook info: {}".format(str(e))
                )
            except tiledb.TileDBError as e:
                raise tornado.web.HTTPError(
                    400,
                    "Error fetching notebook: {}".format(str(e)),
                )
            except Exception as e:
                raise tornado.web.HTTPError(
                    400,
                    "Error fetching notebook: {}".format(str(e)),
                )

        return model

    async def _file_from_array(self, path, content=True, format=None):
        """
        Build a notebook model from database record.
        """
        model = models.create(path=path, type="file")

        if content:
            tiledb_uri = paths.tiledb_uri_from_path(path)
            try:
                info = await caching.call(tiledb.cloud.array.info, tiledb_uri)
                model["last_modified"] = models.to_utc(info.last_accessed)
                if "write" not in info.allowed_actions:
                    model["writable"] = False

                arr = caching.Array.from_cache(tiledb_uri)

                # Use cached meta, only file_size is ever updated
                meta = await arr.meta()
                # Get metadata information
                if "mimetype" in meta:
                    model["mimetype"] = meta["mimetype"]
                if "format" in meta:
                    model["format"] = meta["format"]
                else:
                    model["format"] = format

                if "type" in meta:
                    model["type"] = meta["type"]

                model["content"] = await arr.read()

                if meta.get("type") == "notebook":
                    nb_content = nbformat.reads(
                        model["content"],
                        as_version=NBFORMAT_VERSION,
                    )
                    self.mark_trusted_cells(nb_content, path)
                    model["format"] = "json"
                    model["content"] = nb_content
                    self.validate_notebook_model(model)
            except tiledb.cloud.tiledb_cloud_error.TileDBCloudError as e:
                raise tornado.web.HTTPError(
                    500, "Error fetching file info: {}".format(str(e))
                )
            except tiledb.TileDBError as e:
                raise tornado.web.HTTPError(
                    500,
                    "Error fetching file: {}".format(str(e)),
                )
            except Exception as e:
                raise tornado.web.HTTPError(
                    400,
                    "Error fetching file: {}".format(str(e)),
                )

        return model

    async def guess_type(self, path, allow_directory=True):
        """
        Guess the type of a file.

        Taken from https://github.com/danielfrg/s3contents/blob/master/s3contents/genericmanager.py

        If allow_directory is False, don't consider the possibility that the
        file is a directory.
        Parameters
        ----------
            obj: s3.Object or string
        """
        path = paths.strip(path)
        if paths.is_remote(path):
            if paths.is_remote_dir(path):
                return "directory"
            else:
                try:
                    tiledb_uri = paths.tiledb_uri_from_path(path)
                    return await arrays.fetch_type(tiledb_uri)
                except Exception:
                    return "directory"

        if path.endswith(".ipynb"):
            return "notebook"
        if allow_directory and await self.dir_exists(path):
            return "directory"
        return "file"


class AsyncTileDBCheckpoints(filecheckpoints.AsyncGenericFileCheckpoints):
    """
    A wrapper of a class which will in the future support checkpoints by time traveling.
    It inherits from AsyncGenericFileCheckpoints to checkpoint local files.
    """

    # Immutable version of the only model we return ourselves.
    _BASE_MODEL = (
        ("id", "checkpoints-not-supported"),
        ("last_modified", models._DUMMY_DATE),
    )

    async def create_file_checkpoint(self, content, format, path):
        """ -> checkpoint model"""
        if paths.is_remote(path):
            return dict(self._BASE_MODEL)
        return await super().create_file_checkpoint(content, format, path)

    async def create_notebook_checkpoint(self, nb, path):
        """ -> checkpoint model"""
        if paths.is_remote(path):
            return dict(self._BASE_MODEL)
        return await super().create_notebook_checkpoint(nb, path)

    async def get_file_checkpoint(self, checkpoint_id, path):
        """ -> {'type': 'file', 'content': <str>, 'format': {'text', 'base64'}}"""
        if paths.is_remote(path):
            self.no_such_checkpoint(path, checkpoint_id)
        return await super().get_file_checkpoint(checkpoint_id, path)

    async def get_notebook_checkpoint(self, checkpoint_id, path):
        """ -> {'type': 'notebook', 'content': <output of nbformat.read>}"""
        if paths.is_remote(path):
            self.no_such_checkpoint(path, checkpoint_id)
        return await super().get_notebook_checkpoint(checkpoint_id, path)

    async def delete_checkpoint(self, checkpoint_id, path):
        """deletes a checkpoint for a file"""
        if paths.is_remote(path):
            self.no_such_checkpoint(path, checkpoint_id)
        return await super().delete_checkpoint(checkpoint_id, path)

    async def list_checkpoints(self, path):
        """returns a list of checkpoint models for a given file,
        default just does one per file
        """
        if paths.is_remote(path):
            return []
        return await super().list_checkpoints(path)

    async def rename_checkpoint(self, checkpoint_id, old_path, new_path):
        """renames checkpoint from old path to new path"""
        if paths.is_remote(old_path):
            return
        return await super().rename_checkpoint(checkpoint_id, old_path, new_path)


def _try_convert_file_to_notebook(model):
    """Attempts to convert the passed ``model`` from a "file" to a "notebook".

    Modifies the passed-in model in-place. If the model cannot be converted
    to a notebook, the model is guaranteed to be unmodified.

    Raises a ValueError if there is any error converting the notebook.
    """

    try:
        fmt = model["format"]
        raw_content = model["content"]
    except KeyError as ke:
        raise ValueError(f"missing model key {ke.args[0]}")

    if fmt == "text":
        content = raw_content
    elif fmt == "base64":
        content = base64.b64decode(raw_content).decode("utf-8")
    else:
        raise ValueError(f"unknown content format {fmt!r}")

    nb = nbformat.reads(content, NBFORMAT_VERSION)

    model.update(
        format="json",
        mimetype=None,
        content=nb,
    )


def _is_cloud_enabled():
    """
    Check if a user is allowed to access notebook sharing
    """

    try:
        profile = tiledb.cloud.client.user_profile()
        if "notebook_sharing" in set(profile.enabled_features):
            return True

    except tiledb.cloud.tiledb_cloud_error.TileDBCloudError as e:
        raise tornado.web.HTTPError(
            400,
            "Error fetching user default s3 path for new notebooks {}".format(str(e)),
        )

    return False
