import base64
import datetime
import json
import posixpath
from typing import Any, Dict

import nbformat
import numpy
import tiledb
import tiledb.cloud
import tornado.web
import traitlets
from notebook.services.contents import checkpoints
from notebook.services.contents import filecheckpoints
from notebook.services.contents import filemanager
from notebook.services.contents import manager

from . import arrays
from . import caching
from . import paths


_DUMMY_DATE = datetime.datetime.fromtimestamp(0, tzinfo=datetime.timezone.utc)

NBFORMAT_VERSION = 4

NOTEBOOK_MIME = "application/x-ipynb+json"


def get_cloud_enabled():
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


class TileDBContents(manager.ContentsManager):
    """
    A general class for TileDB Contents, parent of the actual contents class and checkpoints
    """

    def _save_notebook_tiledb(self, path: str, model, *, is_new: bool):
        """
        Save a notebook to tiledb array
        :param model: model notebook
        :param uri: URI of notebook
        :return: any messages
        """
        nb_contents = nbformat.from_dict(model["content"])
        self.check_and_sign(nb_contents, path)
        file_contents = numpy.array(bytearray(json.dumps(model["content"]), "utf-8"))

        final_name = arrays.write_bytes(
            path,
            file_contents,
            mimetype=model.get("mimetype"),
            format=model.get("format"),
            type="notebook",
            s3_prefix=model.get("s3_prefix", None),
            s3_credentials=model.get("s3_credentials", None),
            is_user_defined_name="name" in model,
            is_new=is_new,
        )

        self.validate_notebook_model(model)
        return final_name, model.get("message")

    def _notebook_from_array(self, path, content=True):
        """
        Build a notebook model from database record.
        """
        model = _base_model(path=path, type="notebook")
        if content:
            tiledb_uri = paths.tiledb_uri_from_path(path)
            try:
                info = tiledb.cloud.array.info(tiledb_uri)
                model["last_modified"] = _to_utc(info.last_accessed)
                if "write" not in info.allowed_actions:
                    model["writable"] = False

                arr = caching.Array.from_cache(tiledb_uri)

                nb_content = []
                file_content = arr.read()
                if file_content is not None:
                    nb_content = nbformat.reads(
                        file_content["contents"].tostring().decode("utf-8", "backslashreplace"),
                        as_version=NBFORMAT_VERSION,
                    )
                    self.mark_trusted_cells(nb_content, path)
                model["format"] = "json"
                model["content"] = nb_content
                self.validate_notebook_model(model)
            except tiledb.cloud.tiledb_cloud_error.TileDBCloudError as e:
                raise tornado.web.HTTPError(400, "Error fetching notebook info: {}".format(str(e)))
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

    def _file_from_array(self, path, content=True, format=None):
        """
        Build a notebook model from database record.
        """
        model = _base_model(path=path, type="file")

        if content:
            tiledb_uri = paths.tiledb_uri_from_path(path)
            try:
                info = tiledb.cloud.array.info(tiledb_uri)
                model["last_modified"] = _to_utc(info.last_accessed)
                if "write" not in info.allowed_actions:
                    model["writable"] = False

                arr = caching.Array.from_cache(tiledb_uri)

                # Use cached meta, only file_size is ever updated
                meta = arr.cached_meta
                # Get metadata information
                if "mimetype" in meta:
                    model["mimetype"] = meta["mimetype"]
                if "format" in meta:
                    model["format"] = meta["format"]
                else:
                    model["format"] = format

                if "type" in meta:
                    model["type"] = meta["type"]

                file_content = arr.read()
                if file_content is not None:
                    nb_content = file_content["contents"]
                    model["content"] = nb_content
                else:
                    model["content"] = []

                if (
                    "type" in meta
                    and meta["type"] == "notebook"
                    and file_content is not None
                ):
                    nb_content = nbformat.reads(
                        file_content["contents"].tostring().decode("utf-8", "backslashreplace"),
                        as_version=NBFORMAT_VERSION,
                    )
                    self.mark_trusted_cells(nb_content, path)
                    model["format"] = "json"
                    model["content"] = nb_content
                    self.validate_notebook_model(model)
            except tiledb.cloud.tiledb_cloud_error.TileDBCloudError as e:
                raise tornado.web.HTTPError(500, "Error fetching file info: {}".format(str(e)))
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
        path = paths.strip(path)
        if paths.is_remote(path):
            if paths.is_remote_dir(path):
                return "directory"
            else:
                if path.endswith(paths.NOTEBOOK_EXT):
                    path = path[: -1 * len(paths.NOTEBOOK_EXT)]
                try:
                    tiledb_uri = paths.tiledb_uri_from_path(path)
                    return arrays.fetch_type(tiledb_uri)
                except Exception:
                    return "directory"

        if path.endswith(".ipynb"):
            return "notebook"
        elif allow_directory and self.dir_exists(path):
            return "directory"
        else:
            return "file"


class TileDBCheckpoints(filecheckpoints.GenericFileCheckpoints, TileDBContents, checkpoints.Checkpoints):
    """
    A wrapper of a class which will in the future support checkpoints by time traveling.
    It inherits from GenericFileCheckpoints for local notebooks
    """

    # Immutable version of the only model we return ourselves.
    _BASE_MODEL = (
        ("id", "checkpoints-not-supported"),
        ("last_modified", "models._DUMMY_DATE"),
    )

    def create_file_checkpoint(self, content, format, path):
        """ -> checkpoint model"""
        if not paths.is_remote(path):
            return super().create_file_checkpoint(content, format, path)

        return dict(self._BASE_MODEL)

    def create_notebook_checkpoint(self, nb, path):
        """ -> checkpoint model"""
        if not paths.is_remote(path):
            return super().create_notebook_checkpoint(nb, path)

        return dict(self._BASE_MODEL)

    def get_file_checkpoint(self, checkpoint_id, path):
        """ -> {'type': 'file', 'content': <str>, 'format': {'text', 'base64'}}"""
        if not paths.is_remote(path):
            return super().get_file_checkpoint(checkpoint_id, path)

    def get_notebook_checkpoint(self, checkpoint_id, path):
        """ -> {'type': 'notebook', 'content': <output of nbformat.read>}"""
        if not paths.is_remote(path):
            return super().get_notebook_checkpoint(checkpoint_id, path)

    def delete_checkpoint(self, checkpoint_id, path):
        """deletes a checkpoint for a file"""
        if not paths.is_remote(path):
            return super().delete_checkpoint(checkpoint_id, path)

    def list_checkpoints(self, path):
        """returns a list of checkpoint models for a given file,
        default just does one per file
        """
        path = paths.strip(path)
        if not paths.is_remote(path):
            return super().list_checkpoints(path)
        return []

    def rename_checkpoint(self, checkpoint_id, old_path, new_path):
        """renames checkpoint from old path to new path"""
        if not paths.is_remote(old_path):
            return super().rename_checkpoint(checkpoint_id, old_path, new_path)


class TileDBCloudContentsManager(TileDBContents, filemanager.FileContentsManager, traitlets.HasTraits):
    # This makes the checkpoints get saved on this directory
    root_dir = traitlets.Unicode("./", config=True)

    def __init__(self, **kwargs):
        super(filemanager.FileContentsManager, self).__init__(**kwargs)

    def _checkpoints_class_default(self):
        """
        Set checkpoint class to custom checkpoint class
        :return:
        """
        return TileDBCheckpoints

    def __list_namespace(self, category, namespace, content=False):
        """
        List all notebook arrays in a namespace, this is setup to mimic a "ls" of a directory
        :param category: category to list, shared, owned or public
        :param namespace: namespace to list
        :param content: should contents be included
        :return: model of namespace
        """
        arrays = []
        try:
            listing = caching.ArrayListing.from_cache(category, namespace)
            arrays = listing.arrays()
        except tiledb.cloud.tiledb_cloud_error.TileDBCloudError as e:
            raise tornado.web.HTTPError(
                500, "Error listing notebooks in {}: {}".format(namespace, str(e))
            )
        except tiledb.TileDBError as e:
            raise tornado.web.HTTPError(
                500,
                str(e),
            )
        except Exception as e:
            raise tornado.web.HTTPError(
                400,
                "Error listing notebooks in  {}: {}".format(namespace, str(e)),
            )

        model = _base_model(
            path=paths.join("cloud", category, namespace),
            type="directory",
        )
        if content:
            # Build model content if asked for
            model["format"] = "json"
            model["content"] = []
            if arrays is not None:
                for notebook in arrays:
                    nbmodel = _base_model(path=notebook.name)

                    # Add notebook extension to name, so jupyterlab will open with as a notebook
                    # It seems to check the extension even though we set the "type" parameter
                    nbmodel["path"] = "cloud/{}/{}/{}{}".format(
                        category, namespace, nbmodel["path"], paths.NOTEBOOK_EXT
                    )

                    nbmodel["last_modified"] = _to_utc(notebook.last_accessed)
                    # Update namespace directory based on last access notebook
                    _maybe_update_last_modified(model, notebook)

                    nbmodel["type"] = "notebook"

                    if (
                        notebook.allowed_actions is None
                        or "write" not in notebook.allowed_actions
                    ):
                        model["writable"] = False
                    model["content"].append(nbmodel)

        return model

    def __list_category(self, category, content=True):
        """
        This function should be switched to use sidebar data
        :param category:
        :param content:
        :return:
        """
        arrays = []
        try:
            arrays = caching.ArrayListing.from_cache(category).arrays()
        except tiledb.cloud.tiledb_cloud_error.TileDBCloudError as e:
            raise tornado.web.HTTPError(
                500, "Error listing notebooks in {}: {}".format(category, str(e))
            )
        except tiledb.TileDBError as e:
            raise tornado.web.HTTPError(
                500,
                str(e),
            )
        except Exception as e:
            raise tornado.web.HTTPError(
                400,
                "Error listing notebooks in  {}: {}".format(category, str(e)),
            )

        model = _base_model(path=paths.join("cloud", category), type="directory")
        if content:
            model["format"] = "json"
            model["content"] = []
            namespaces = {}
            if category == "owned":
                # For owned, we should always list the user and their
                # organizations. If there is actual notebooks they will be
                # listed below. This base listing is so users can create new
                # notebooks in any of the namespaces they are part of.
                try:
                    profile = tiledb.cloud.client.user_profile()
                    namespace_model = _base_model(
                        path=paths.join("cloud", category, profile.username),
                        type="directory",
                        format="json",
                    )
                    namespaces[profile.username] = namespace_model

                    for org in profile.organizations:
                        # Don't list public for owned
                        if org.organization_name == "public":
                            continue

                        namespace_model = _base_model(
                            path=paths.join("cloud", category, org.organization_name),
                            type="directory",
                            format="json",
                        )

                        namespaces[org.organization_name] = namespace_model

                except tiledb.cloud.tiledb_cloud_error.TileDBCloudError as e:
                    raise tornado.web.HTTPError(
                        500,
                        "Error listing notebooks in {}: {}".format(category, str(e)),
                    )
                except tiledb.TileDBError as e:
                    raise tornado.web.HTTPError(
                        500,
                        str(e),
                    )
                except Exception as e:
                    raise tornado.web.HTTPError(
                        400,
                        "Error listing notebooks in  {}: {}".format(category, str(e)),
                    )

            # If the arrays are non-empty list them out
            if arrays is not None:
                for notebook in arrays:
                    if notebook.namespace not in namespaces:
                        namespace_model = _base_model(
                            path=paths.join("cloud", category, notebook.namespace),
                            type="directory",
                            format="json",
                            writable=False,
                        )
                        namespaces[notebook.namespace] = namespace_model

                    # Update directory based on last access notebook
                    _maybe_update_last_modified(model, notebook)

                    # Update namespace directory based on last access notebook
                    _maybe_update_last_modified(namespaces[notebook.namespace], notebook)

            model["content"] = list(namespaces.values())

        return model

    def __build_cloud_notebook_lists(self):
        """
        Build a list of all notebooks across all categories
        :return:
        """

        ret = {
            "owned": _base_model(path="owned", type="directory"),
            "shared": _base_model(path="shared", type="directory"),
            "public": _base_model(path="public", type="directory"),
        }
        try:

            owned_listing = caching.ArrayListing.from_cache("owned")
            shared_listing = caching.ArrayListing.from_cache("shared")
            public_listing = caching.ArrayListing.from_cache("public")

            ret["owned"]["path"] = "cloud/owned"
            ret["public"]["path"] = "cloud/public"
            ret["shared"]["path"] = "cloud/shared"

            owned_notebooks = owned_listing.arrays()
            shared_notebooks = shared_listing.arrays()
            public_notebooks = public_listing.arrays()
            if owned_notebooks is not None:
                if len(owned_notebooks) > 0:
                    ret["owned"]["format"] = "json"
                    ret["owned"]["content"] = []
                    for notebook in owned_notebooks:
                        model = _base_model(
                            path=notebook.name,
                            type="notebook",
                            format="json",
                            last_modified=_to_utc(notebook.last_accessed),
                        )
                        # Add notebook extension to path, so jupyterlab will open with as a notebook
                        # It seems to check the extension even though we set the "type" parameter
                        model["path"] = "cloud/{}/{}{}".format(
                            "owned", model["path"], paths.NOTEBOOK_EXT
                        )
                        ret["owned"]["content"].append(model)

                        # Update category date
                        _maybe_update_last_modified(ret["owned"], notebook)

            if shared_notebooks is not None:
                if len(shared_notebooks) > 0:
                    ret["shared"]["format"] = "json"
                    ret["shared"]["content"] = []
                    for notebook in shared_notebooks:
                        model = _base_model(
                            path=notebook.name,
                            type="notebook",
                            format="json",
                            last_modified=_to_utc(notebook.last_accessed),
                        )
                        # Add notebook extension to path, so jupyterlab will open with as a notebook
                        # It seems to check the extension even though we set the "type" parameter
                        model["path"] = "cloud/{}/{}{}".format(
                            "shared", model["path"], paths.NOTEBOOK_EXT
                        )
                        ret["shared"]["content"].append(model)

                        # Update category date
                        _maybe_update_last_modified(ret["shared"], notebook)

            if public_notebooks is not None:
                if len(public_notebooks) > 0:
                    ret["public"]["format"] = "json"
                    ret["public"]["content"] = []
                    for notebook in public_notebooks:
                        model = _base_model(
                            path=notebook.name,
                            type="notebook",
                            format="json",
                            last_modified=_to_utc(notebook.last_accessed),
                        )
                        # Add notebook extension to path, so jupyterlab will open with as a notebook
                        # It seems to check the extension even though we set the "type" parameter
                        model["path"] = "cloud/{}/{}{}".format(
                            "public", model["path"], paths.NOTEBOOK_EXT
                        )
                        ret["public"]["content"].append(model)

                        # Update category date
                        _maybe_update_last_modified(ret["public"], notebook)

        except tiledb.cloud.tiledb_cloud_error.TileDBCloudError as e:
            raise tornado.web.HTTPError(
                500, "Error building cloud notebook info: {}".format(str(e))
            )
        except tiledb.TileDBError as e:
            raise tornado.web.HTTPError(
                500,
                str(e),
            )
        except Exception as e:
            raise tornado.web.HTTPError(
                500, "Error building cloud notebook info: {}".format(str(e))
            )

        return list(ret.values())

    def __directory_model_from_path(self, path, content=False):
        # if self.vfs.is_dir(path):
        #     lstat = self.fs.lstat(path)
        #     if "ST_MTIME" in lstat and lstat["ST_MTIME"]:
        model = _base_model(
            path=path,
            type="directory",
            last_modified=_DUMMY_DATE,
            created=_DUMMY_DATE,
        )
        if not paths.is_remote(path) and not paths.is_remote_dir(path):
            return super()._dir_model(path, content)

        if path == "cloud":
            cloud = _base_model(path="cloud", type="directory")
            if content:
                cloud["format"] = "json"
                cloud["content"] = self.__build_cloud_notebook_lists()

                cloud["last_modified"] = max(
                    _to_utc(cat["last_modified"]) for cat in cloud["content"])

            model = cloud
        else:
            category = paths.extract_category(path)
            namespace = paths.extract_namespace(path)

            if namespace is None:
                model = self.__list_category(category, content)
            elif category is not None and namespace is not None:
                model = self.__list_namespace(category, namespace, content)

        return model

    def get(self, path, content=True, type=None, format=None):
        """Get a file or directory model."""
        path = paths.strip(path)
        try:
            if not paths.is_remote(path):
                model = super().get(path, content, type, format)
                if path == "" and content and get_cloud_enabled():
                    cloud_content = self.__build_cloud_notebook_lists()
                    model["content"].append(
                        _base_model(
                            path="cloud",
                            type="directory",
                            content=content,
                            format="json",
                            last_modified=max(
                                _to_utc(cat["last_modified"]) for cat in cloud_content
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
                    type = self.guess_type(path, allow_directory=True)

            if type == "notebook":
                return self._notebook_from_array(path, content)
            elif type == "file":
                return self._file_from_array(path, content, format)
            elif type == "directory":
                return self.__directory_model_from_path(path, content)
                # if model is not None:
                #     model.
        except Exception as e:
            raise tornado.web.HTTPError(
                500, "Error opening notebook {}: {}".format(path, str(e))
            )

    def save(self, model, path=""):
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
            raise tornado.web.HTTPError(400, u"No file content provided")

        if model_type not in ("directory", "file", "notebook"):
            raise tornado.web.HTTPError(400, "Unhandled contents type: %s" % model["type"])

        if not paths.is_remote(path):
            return super().save(model, path)

        if path.endswith(paths.NOTEBOOK_EXT):
            path = path[:-len(paths.NOTEBOOK_EXT)]
            if model["type"] == "file":
                try:
                    _try_convert_file_to_notebook(model)
                except ValueError as ve:
                    raise tornado.web.HTTPError(400, f"Cannot parse Jupyter notebook: {ve}")

        is_new = True
        if (
            "content" in model
            and "metadata" in model["content"]
            and "language_info" in model["content"]["metadata"]
        ):
            is_new = False

        self.run_pre_save_hook(model=model, path=path)
        validation_message = None
        try:
            if model["type"] == "notebook":
                final_name, validation_message = self._save_notebook_tiledb(
                    path, model, is_new=is_new
                )
                if final_name is not None:
                    parts = paths.split(path)
                    parts_length = len(parts)
                    parts[parts_length - 1] = final_name
                    path = paths.join(*parts)
            elif model["type"] == "file":
                raise tornado.web.HTTPError(400, "Only .ipynb files may be created in the cloud.")
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

        model = self.get(path, type=model["type"], content=False)
        if validation_message is not None:
            model["message"] = validation_message
        return model

    def delete_file(self, path):
        """Delete the file or directory at path."""
        path = paths.strip(path)
        if paths.is_remote(path):

            if path.endswith(paths.NOTEBOOK_EXT):
                path = path[: -1 * len(paths.NOTEBOOK_EXT)]

            tiledb_uri = paths.tiledb_uri_from_path(path)
            try:
                caching.Array.purge(tiledb_uri)
                return tiledb.cloud.array.delete_array(
                    tiledb_uri, "application/x-ipynb+json"
                )
            except tiledb.cloud.tiledb_cloud_error.TileDBCloudError as e:
                raise tornado.web.HTTPError(
                    500, f"Error deregistering {tiledb_uri!r}: {e}"
                )
            except tiledb.TileDBError as e:
                raise tornado.web.HTTPError(
                    500,
                    str(e),
                )
        else:
            return super().delete_file(path)

    def rename_file(self, old_path, new_path):
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
                return tiledb.cloud.notebook.rename_notebook(
                    uri=tiledb_uri, notebook_name=array_name_new
                )
            except tiledb.cloud.tiledb_cloud_error.TileDBCloudError as e:
                raise tornado.web.HTTPError(500, f"Error renaming {tiledb_uri!r}: {e}")
            except tiledb.TileDBError as e:
                raise tornado.web.HTTPError(
                    500,
                    str(e),
                )
        else:
            return super().rename_file(old_path, new_path)

    # ContentsManager API part 2: methods that have usable default
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
        path = paths.strip(path)
        if paths.is_remote_dir(path):
            return True

        return super().dir_exists(path)

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
        path = paths.strip(path)
        if paths.is_remote(path):
            return False

        return super().is_hidden(path)

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
        path = paths.strip(path)
        if paths.is_remote(path):
            if path.endswith(paths.NOTEBOOK_EXT):
                path = path[: -1 * len(paths.NOTEBOOK_EXT)]
            return arrays.exists(path)
        return super().file_exists(path)

    def new_untitled(self, path="", type="", ext="", options=""):
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
                model["s3_prefix"] = options_json["s3_prefix"]
                model["s3_credentials"] = options_json["s3_credentials"]
            except Exception as e:
                raise tornado.web.HTTPError(
                    400, u"Could not read TileDB user options: {}".format(e)
                )

        if model["type"] == "notebook" and "name" in model:
            path = u"{0}/{1}".format(path, model["name"] + ".ipynb")
            return self.new(model, path)

        insert = ""
        if model["type"] == "directory":
            untitled = self.untitled_directory
            insert = " "
        elif model["type"] == "notebook":
            untitled = self.untitled_notebook
            ext = ".ipynb"
        elif model["type"] == "file":
            untitled = self.untitled_file
        else:
            raise tornado.web.HTTPError(400, "Unexpected model type: %r" % model["type"])

        name = self.increment_filename(untitled + ext, path, insert=insert)
        path = u"{0}/{1}".format(path, name)
        return self.new(model, path)


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


def _to_utc(d: datetime.datetime) -> datetime.datetime:
    """Returns a version of d converted to UTC.

    If naive (no timezone), UTC will be added; if timezone-aware, the time
    will be converted.
    """
    if not d.tzinfo:
        return d.replace(tzinfo=datetime.timezone.utc)
    return d.astimezone(datetime.timezone.utc)


def _maybe_update_last_modified(model: Dict[str, Any], notebook: Any) -> None:
    nb_last_acc = _to_utc(notebook.last_accessed)
    md_last_mod = _to_utc(model["last_modified"])
    model["last_modified"] = max(nb_last_acc, md_last_mod)


def _base_model(*, path: str, **kwargs) -> Dict[str, Any]:
    """Create the most basic model."""
    # Originally from:
    # https://github.com/danielfrg/s3contents/blob/master/s3contents/genericmanager.py
    model = {
        "name": posixpath.basename(path),
        "path": path,
        "writable": True,
        "last_modified": _DUMMY_DATE,
        "created": _DUMMY_DATE,
        "content": None,
        "format": None,
        "mimetype": None,
    }
    model.update(**kwargs)
