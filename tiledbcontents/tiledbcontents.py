import base64
import datetime
import json
import os
import random
import string
import time
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

from . import caching


DUMMY_CREATED_DATE = datetime.datetime.fromtimestamp(86400)
NBFORMAT_VERSION = 4

NOTEBOOK_MIME = "application/x-ipynb+json"

NOTEBOOK_EXT = ".ipynb"

JUPYTER_IMAGE_NAME_ENV = "JUPYTER_IMAGE_NAME"
JUPYTER_IMAGE_SIZE_ENV = "JUPYTER_IMAGE_SIZE"


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


def get_s3_prefix(namespace):
    """
    Get S3 path from the user profile or organization profile
    :return: s3 path or error
    """
    try:
        profile = tiledb.cloud.client.user_profile()

        if namespace == profile.username:
            if profile.default_s3_path is not None:
                return os.path.join(profile.default_s3_path, "notebooks")
        else:
            organization = tiledb.cloud.client.organization(namespace)
            if organization.default_s3_path is not None:
                return os.path.join(organization.default_s3_path, "notebooks")
    except tiledb.cloud.tiledb_cloud_error.TileDBCloudError as e:
        raise tornado.web.HTTPError(
            400,
            "Error fetching user default s3 path for new notebooks {}".format(str(e)),
        )

    return None


def get_s3_credentials(namespace):
    """
    Get credentials for default S3 path from the user profile or organization profile
    :return: s3 credentials or error
    """
    try:
        profile = tiledb.cloud.client.user_profile()

        if namespace == profile.username:
            if profile.default_s3_path_credentials_name is not None:
                return profile.default_s3_path_credentials_name
        else:
            organization = tiledb.cloud.client.organization(namespace)
            if organization.default_s3_path_credentials_name is not None:
                return organization.default_s3_path_credentials_name
    except tiledb.cloud.tiledb_cloud_error.TileDBCloudError as e:
        raise tornado.web.HTTPError(
            400,
            "Error fetching default credentials for {} default s3 path for new notebooks {}".format(
                namespace, str(e)
            ),
        )

    return None


def base_model(path):
    """
    Taken from https://github.com/danielfrg/s3contents/blob/master/s3contents/genericmanager.py
    :return:
    """
    return {
        "name": path.rsplit("/", 1)[-1],
        "path": path,
        "writable": True,
        "last_modified": DUMMY_CREATED_DATE,
        "created": DUMMY_CREATED_DATE,
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
        type="directory",
        last_modified=DUMMY_CREATED_DATE,
        created=DUMMY_CREATED_DATE,
    )
    return model


def remove_path_prefix(path_prefix, path):
    """
    Remove a prefix
    :param path_prefix:
    :param path:
    :return:
    """
    ret = path.split(path_prefix, 1)
    if len(ret) > 1:
        return ret[1]
    return ret


class TileDBContents(manager.ContentsManager):
    """
    A general class for TileDB Contents, parent of the actual contents class and checkpoints
    """

    def _save_notebook_tiledb(self, model, uri):
        """
        Save a notebook to tiledb array
        :param model: model notebook
        :param uri: URI of notebook
        :return: any messages
        """
        nb_contents = nbformat.from_dict(model["content"])
        self.check_and_sign(nb_contents, uri)
        file_contents = numpy.array(bytearray(json.dumps(model["content"]), "utf-8"))

        final_name = self._write_bytes_to_array(
            uri,
            file_contents,
            model.get("mimetype"),
            model.get("format"),
            "notebook",
            model["s3_prefix"] if "s3_prefix" in model else None,
            model["s3_credentials"] if "s3_credentials" in model else None,
            "name" in model,
        )

        self.validate_notebook_model(model)
        return final_name, model.get("message")

    def _increment_filename(self, filename, insert="-"):
        """Increment a filename until it is unique.

        Parameters
        ----------
        filename : unicode
            The name of a file, including extension
        path : unicode
            The API path of the target's directory
        insert: unicode
            The characters to insert after the base filename

        Returns
        -------
        name : unicode
            A filename that is unique, based on the input filename.
        """
        # Extract the full suffix from the filename (e.g. .tar.gz)
        basename, dot, ext = filename.rpartition(".")
        if ext != "ipynb":
            basename, dot, ext = filename.partition(".")

        suffix = dot + ext

        parts = basename.split(insert)
        start = 0
        if len(parts) > 1:
            start_str = parts[len(parts) - 1]
            if start_str.isdigit():
                start = int(start_str)

            basename = insert.join(parts[0 : len(parts) - 1])

        start += 1
        if start:
            insert_i = "{}{}".format(insert, start)
        else:
            insert_i = ""
        name = u"{basename}{insert}{suffix}".format(
            basename=basename, insert=insert_i, suffix=suffix
        )
        return name

    def _increment_notebook(self, filename, insert="-"):
        """Increment a notebook filename until it is unique.

        Parameters
        ----------
        filename : unicode
            The name of a file, including extension
        path : unicode
            The API path of the target's directory
        insert: unicode
            The characters to insert after the base filename

        Returns
        -------
        name : unicode
            A filename that is unique, based on the input filename.
        """
        # Extract the full suffix from the filename (e.g. .tar.gz)
        basename, dot, ext = filename.rpartition(".")
        if ext != "ipynb":
            basename, dot, ext = filename.partition(".")

        suffix = dot + ext

        parts = basename.split(insert)
        start = 0
        if len(parts) > 1:
            start_str = parts[len(parts) - 1]
            if start_str.isdigit():
                start = int(start_str)

            basename = insert.join(parts[0 : len(parts) - 1])

        start += 1
        if start:
            insert_i = "{}{}".format(insert, start)
        else:
            insert_i = ""
        name = u"{basename}{insert}{suffix}".format(
            basename=basename, insert=insert_i, suffix=suffix
        )
        return name

    def id_generator(self, size=6, chars=string.ascii_uppercase + string.digits):
        return "".join(random.choice(chars) for _ in range(size))

    def _create_array(
        self,
        uri,
        retry=0,
        s3_prefix=None,
        s3_credentials=None,
        is_user_defined_name=False,
    ):
        """
        Create a new array for storing notebook file
        :param uri: location to create array
        :param name: name to register under
        :param retry: number of times to retry request
        :param s3_prefix: S3 path to write to
        :param s3_credentials: S3 credentials associated with the S3 prefix as labelled on TileDB Cloud
        :param is_user_defined_name: boolean indicating whether the user provided their own filename
        :return:
        """
        try:
            parts = uri.split("/")
            parts_len = len(parts)
            namespace = parts[parts_len - 2]
            array_name = parts[parts_len - 1] + "_" + self.id_generator()

            # Use the general cloud context by default.
            tiledb_create_context = caching.CLOUD_CONTEXT

            if s3_credentials is None:
                s3_credentials = get_s3_credentials(namespace)
                # Retrieving credentials is optional
                # If None, default credentials will be used

            if s3_credentials is not None:
                cfg_dict = {}
                cfg_dict["rest.creation_access_credentials_name"] = s3_credentials
                # update context with config having header set
                tiledb_create_context = tiledb.cloud.Ctx(cfg_dict)

            # The array will be be 1 dimensional with domain of 0 to max uint64. We use a tile extent of 1024 bytes
            dom = tiledb.Domain(
                tiledb.Dim(
                    name="position",
                    domain=(0, numpy.iinfo(numpy.uint64).max - 1025),
                    tile=1024,
                    dtype=numpy.uint64,
                    ctx=tiledb_create_context,
                ),
                ctx=tiledb_create_context,
            )

            schema = tiledb.ArraySchema(
                domain=dom,
                sparse=False,
                attrs=[
                    tiledb.Attr(
                        name="contents",
                        dtype=numpy.uint8,
                        filters=tiledb.FilterList([tiledb.ZstdFilter()]),
                    )
                ],
                ctx=tiledb_create_context,
            )

            if is_user_defined_name:
                array_name = parts[parts_len - 1]
            else:
                array_name = parts[parts_len - 1] + "_" + self.id_generator()

            if namespace is not None and (
                namespace == "cloud"
                or namespace == "owned"
                or namespace == "public"
                or namespace == "shared"
            ):
                raise tornado.web.HTTPError(
                    403,
                    "`{}` is not a valid folder to create notebooks, please select a proper namespace (username or organization name)".format(
                        namespace
                    ),
                )

            if namespace is not None and (
                namespace == "cloud"
                or namespace == "owned"
                or namespace == "public"
                or namespace == "shared"
            ):
                raise tornado.web.HTTPError(
                    403,
                    "`{}` is not a valid folder to create notebooks, please select a proper namespace (username or organization name)".format(
                        namespace
                    ),
                )

            if s3_prefix is None:
                s3_prefix = get_s3_prefix(namespace)
                if s3_prefix is None:
                    raise tornado.web.HTTPError(
                        403,
                        "You must set the default s3 prefix path for notebooks in {} profile settings".format(
                            namespace
                        ),
                    )

            tiledb_uri_s3 = "tiledb://{}/{}".format(
                namespace, os.path.join(s3_prefix, array_name)
            )

            # Create the (empty) array on disk.
            tiledb.DenseArray.create(tiledb_uri_s3, schema)

            tiledb_uri = "tiledb://{}/{}".format(namespace, array_name)
            time.sleep(0.25)

            file_properties = {}
            # Get image name from env if exists
            # This is stored as a tag for TileDB Cloud for searching, filtering and launching
            image_name = os.getenv(JUPYTER_IMAGE_NAME_ENV)
            if image_name is not None:
                file_properties[
                    tiledb.cloud.rest_api.models.FilePropertyName.IMAGE
                ] = image_name

            # Get image size from env if exists
            # This is stored as a tag for TileDB Cloud for searching, filtering and launching
            image_size = os.getenv(JUPYTER_IMAGE_SIZE_ENV)
            if image_size is not None:
                file_properties[
                    tiledb.cloud.rest_api.models.FilePropertyName.SIZE
                ] = image_size

            tiledb.cloud.array.update_info(uri=tiledb_uri, array_name=array_name)

            if len(file_properties) == 0:
                file_properties = None

            tiledb.cloud.array.update_file_properties(
                uri=tiledb_uri,
                file_type=tiledb.cloud.rest_api.models.FileType.NOTEBOOK,
                file_properties=file_properties,
            )

            return tiledb_uri, array_name
        except tiledb.TileDBError as e:
            if "Error while listing with prefix" in str(e):
                # It is possible to land here if user sets wrong default s3 credentials with respect to default s3 path
                raise tornado.web.HTTPError(
                    400, "Error creating file, %s Are your credentials valid?" % str(e)
                )

            if "already exists" in str(e):
                parts = uri.split("/")
                parts_length = len(parts)
                array_name = parts[parts_length - 1]

                array_name = self._increment_filename(array_name)

                parts[parts_length - 1] = array_name
                uri = "/".join(parts)

                return self._create_array(
                    uri, retry, s3_prefix, s3_credentials, is_user_defined_name
                )
            elif retry:
                retry -= 1
                return self._create_array(
                    uri, retry, s3_prefix, s3_credentials, is_user_defined_name
                )
        except tornado.web.HTTPError as e:
            raise e
        except Exception as e:
            raise tornado.web.HTTPError(400, "Error creating file %s " % str(e))

        return None

    def _array_exists(self, path):
        """
        Check if an array exists in TileDB Cloud
        :param path:
        :return:
        """
        tiledb_uri = self.tiledb_uri_from_path(path)
        try:
            tiledb.cloud.array.info(tiledb_uri)
            return True
        except tiledb.cloud.tiledb_cloud_error.TileDBCloudError as e:
            if str(e) == "Array or Namespace Not found":
                return False

        return False

    def _write_bytes_to_array(
        self,
        uri,
        contents,
        mimetype=None,
        format=None,
        type=None,
        s3_prefix=None,
        s3_credentials=None,
        is_user_defined_name=False,
    ):
        """
        Write given bytes to the array. Will create the array if it does not exist
        :param uri: array to write to
        :param contents: bytes to write
        :param mimetype: mimetype to set in metadata
        :param format: format to set in metadata
        :param type: type to set in metadata
        :param s3_prefix: S3 path to write to
        :param s3_credentials: S3 credentials associated with the S3 prefix as labelled on TileDB Cloud
        :param is_user_defined_name: boolean indicating whether the user provided their own filename
        :return:
        """
        tiledb_uri = self.tiledb_uri_from_path(uri)
        final_array_name = None
        if self._is_new:
            # if not self._array_exists(uri):
            tiledb_uri, final_array_name = self._create_array(
                tiledb_uri, 5, s3_prefix, s3_credentials, is_user_defined_name
            )

        with tiledb.open(tiledb_uri, mode="w", ctx=caching.CLOUD_CONTEXT) as A:
            A[0 : len(contents)] = {"contents": contents}
            A.meta["file_size"] = len(contents)
            if mimetype is not None:
                A.meta["mimetype"] = mimetype
            if format is not None:
                A.meta["format"] = format
            if type is not None:
                A.meta["type"] = type

        caching.Array.from_cache(tiledb_uri, {"contents": contents})

        return final_array_name

    def tiledb_uri_from_path(self, path):
        """
        Build a tiledb:// URI from a notebook cloud path
        :param path:
        :return: tiledb uri
        """

        parts = path.split(os.sep)
        if len(parts) == 1:
            parts = path.split("/")

        length = len(parts)
        return "tiledb://{}/{}".format(parts[length - 2], parts[length - 1])

    def _notebook_from_array(self, uri, content=True):
        """
        Build a notebook model from database record.
        """
        model = base_model(uri)

        model["type"] = "notebook"
        if content:
            tiledb_uri = self.tiledb_uri_from_path(uri)
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
                    self.mark_trusted_cells(nb_content, uri)
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

    def _file_from_array(self, uri, content=True, format=None):
        """
        Build a notebook model from database record.
        """
        model = base_model(uri)
        model["type"] = "file"

        if content:
            tiledb_uri = self.tiledb_uri_from_path(uri)
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
                    self.mark_trusted_cells(nb_content, uri)
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

    def _is_remote_path(self, path):
        """
        Checks if a path is remote or not
        :param path:
        :return:
        """
        if path.split(os.sep)[0] == "cloud" or path.split("/")[0] == "cloud":
            return True
        return False

    def _is_remote_dir(self, path):
        """
        Checks if a path is a remote dir or not
        :param path:
        :return:
        """
        for sep in [os.sep, "/"]:
            splits = path.split(sep)
            if len(splits) == 1 and splits[0] == "cloud":
                return True
            if (
                (len(splits) == 2 or len(splits) == 3)
                and splits[0] == "cloud"
                and (
                    splits[1] == "owned"
                    or splits[1] == "public"
                    or splits[1] == "shared"
                )
            ):
                return True

        return False

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
        path_fixed = path.strip("/")
        if self._is_remote_path(path_fixed):
            if self._is_remote_dir(path_fixed):
                return "directory"
            else:
                if path_fixed.endswith(NOTEBOOK_EXT):
                    path_fixed = path_fixed[: -1 * len(NOTEBOOK_EXT)]
                try:
                    tiledb_uri = self.tiledb_uri_from_path(path_fixed)
                    return self._get_type(tiledb_uri)
                except Exception:
                    return "directory"

        if path.endswith(".ipynb"):
            return "notebook"
        elif allow_directory and self.dir_exists(path):
            return "directory"
        else:
            return "file"

    def _get_mimetype(self, uri):
        """
        Fetch mimetype from array metadata
        :param uri: of array
        :return:
        """
        try:
            arr = caching.Array.from_cache(uri)
            meta = arr.cached_meta
            if "mimetype" in meta:
                return meta["mimetype"]
        except tiledb.cloud.tiledb_cloud_error.TileDBCloudError as e:
            raise tornado.web.HTTPError(500, "Error getting mimetype: {}".format(str(e)))
        except tiledb.TileDBError as e:
            raise tornado.web.HTTPError(
                500,
                str(e),
            )
        except Exception as e:
            raise tornado.web.HTTPError(
                400,
                "Error getting file MIME: {}".format(str(e)),
            )

        return None

    def _get_type(self, uri):
        """
        Fetch type from array metadata
        :param uri: of array
        :return:
        """
        try:
            arr = caching.Array.from_cache(uri)
            meta = arr.cached_meta
            if "type" in meta:
                return meta["type"]
        except tiledb.cloud.tiledb_cloud_error.TileDBCloudError as e:
            raise tornado.web.HTTPError(500, "Error getting type: {}".format(str(e)))
        except tiledb.TileDBError as e:
            raise tornado.web.HTTPError(
                500,
                str(e),
            )
        except Exception as e:
            raise tornado.web.HTTPError(
                400,
                "Error getting file type: {}".format(str(e)),
            )

        return None


class TileDBCheckpoints(filecheckpoints.GenericFileCheckpoints, TileDBContents, checkpoints.Checkpoints):
    """
    A wrapper of a class which will in the future support checkpoints by time traveling.
    It inherits from GenericFileCheckpoints for local notebooks
    """

    def _tiledb_checkpoint_model(self):
        return dict(
            id="checkpoints-not-supported",
            last_modified=DUMMY_CREATED_DATE,
        )

    def create_file_checkpoint(self, content, format, path):
        """ -> checkpoint model"""
        path_fixed = path.strip("/")
        if not self._is_remote_path(path_fixed):
            return super().create_file_checkpoint(content, format, path)

        return self._tiledb_checkpoint_model()

    def create_notebook_checkpoint(self, nb, path):
        """ -> checkpoint model"""
        path_fixed = path.strip("/")
        if not self._is_remote_path(path_fixed):
            return super().create_notebook_checkpoint(nb, path)

        return self._tiledb_checkpoint_model()

    def get_file_checkpoint(self, checkpoint_id, path):
        """ -> {'type': 'file', 'content': <str>, 'format': {'text', 'base64'}}"""
        path_fixed = path.strip("/")
        if not self._is_remote_path(path_fixed):
            return super().get_file_checkpoint(checkpoint_id, path)

    def get_notebook_checkpoint(self, checkpoint_id, path):
        """ -> {'type': 'notebook', 'content': <output of nbformat.read>}"""
        path_fixed = path.strip("/")
        if not self._is_remote_path(path_fixed):
            return super().get_notebook_checkpoint(checkpoint_id, path)

    def delete_checkpoint(self, checkpoint_id, path):
        """deletes a checkpoint for a file"""
        path_fixed = path.strip("/")
        if not self._is_remote_path(path_fixed):
            return super().delete_checkpoint(checkpoint_id, path)

    def list_checkpoints(self, path):
        """returns a list of checkpoint models for a given file,
        default just does one per file
        """
        path_fixed = path.strip("/")
        if not self._is_remote_path(path_fixed):
            return super().list_checkpoints(path)
        return []

    def rename_checkpoint(self, checkpoint_id, old_path, new_path):
        """renames checkpoint from old path to new path"""
        path_fixed = old_path.strip("/")
        if not self._is_remote_path(path_fixed):
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

        model = base_directory_model(namespace)
        model["path"] = "cloud/{}/{}".format(category, namespace)
        if content:
            # Build model content if asked for
            model["format"] = "json"
            model["content"] = []
            if arrays is not None:
                for notebook in arrays:
                    nbmodel = base_model(notebook.name)

                    # Add notebook extension to name, so jupyterlab will open with as a notebook
                    # It seems to check the extension even though we set the "type" parameter
                    nbmodel["path"] = "cloud/{}/{}/{}{}".format(
                        category, namespace, nbmodel["path"], NOTEBOOK_EXT
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

        model = base_directory_model(category)
        model["path"] = "cloud/{}".format(category)
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
                    namespace_model = base_directory_model(profile.username)
                    namespace_model["format"] = "json"
                    namespace_model["path"] = "cloud/{}/{}".format(
                        category, profile.username
                    )
                    namespaces[profile.username] = namespace_model

                    for org in profile.organizations:
                        # Don't list public for owned
                        if org.organization_name == "public":
                            continue

                        namespace_model = base_directory_model(org.organization_name)
                        namespace_model["format"] = "json"
                        namespace_model["path"] = "cloud/{}/{}".format(
                            category, org.organization_name
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
                        namespace_model = base_directory_model(notebook.namespace)
                        namespace_model["format"] = "json"
                        namespace_model["writable"] = False
                        namespace_model["path"] = "cloud/{}/{}".format(
                            category, notebook.namespace
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
            "owned": base_directory_model("owned"),
            "shared": base_directory_model("shared"),
            "public": base_directory_model("public"),
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
                        model = base_model(notebook.name)
                        model["type"] = "notebook"
                        model["format"] = "json"
                        model["last_modified"] = _to_utc(notebook.last_accessed)
                        # Add notebook extension to path, so jupyterlab will open with as a notebook
                        # It seems to check the extension even though we set the "type" parameter
                        model["path"] = "cloud/{}/{}{}".format(
                            "owned", model["path"], NOTEBOOK_EXT
                        )
                        ret["owned"]["content"].append(model)

                        # Update category date
                        _maybe_update_last_modified(ret["owned"], notebook)

            if shared_notebooks is not None:
                if len(shared_notebooks) > 0:
                    ret["shared"]["format"] = "json"
                    ret["shared"]["content"] = []
                    for notebook in shared_notebooks:
                        model = base_model(notebook.name)
                        model["type"] = "notebook"
                        model["format"] = "json"
                        model["last_modified"] = _to_utc(notebook.last_accessed)
                        # Add notebook extension to path, so jupyterlab will open with as a notebook
                        # It seems to check the extension even though we set the "type" parameter
                        model["path"] = "cloud/{}/{}{}".format(
                            "shared", model["path"], NOTEBOOK_EXT
                        )
                        ret["shared"]["content"].append(model)

                        # Update category date
                        _maybe_update_last_modified(ret["shared"], notebook)

            if public_notebooks is not None:
                if len(public_notebooks) > 0:
                    ret["public"]["format"] = "json"
                    ret["public"]["content"] = []
                    for notebook in public_notebooks:
                        model = base_model(notebook.name)
                        model["type"] = "notebook"
                        model["format"] = "json"
                        model["last_modified"] = _to_utc(notebook.last_accessed)
                        # Add notebook extension to path, so jupyterlab will open with as a notebook
                        # It seems to check the extension even though we set the "type" parameter
                        model["path"] = "cloud/{}/{}{}".format(
                            "public", model["path"], NOTEBOOK_EXT
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

    def __category_from_path(self, path):
        """
        Get the category from a cloud path
        :param path:
        :return:
        """
        parts = path.split(os.sep)
        if len(parts) == 0:
            parts = path.split("/")

        if parts[0] == "cloud" and len(parts) > 1:
            return parts[1]

        return None

    def __namespace_from_path(self, path):
        """
        Get the namespace from a cloud path
        :param path:
        :return:
        """
        parts = path.split(os.sep)
        if len(parts) == 1:
            parts = path.split("/")

        if parts[0] == "cloud" and len(parts) > 2:
            return parts[2]

        return None

    def __directory_model_from_path(self, path, content=False):
        # if self.vfs.is_dir(path):
        #     lstat = self.fs.lstat(path)
        #     if "ST_MTIME" in lstat and lstat["ST_MTIME"]:
        model = base_directory_model(path)
        model["last_modified"] = model["created"] = DUMMY_CREATED_DATE
        if not self._is_remote_path(path) and not self._is_remote_dir(path):
            return super()._dir_model(path, content)

        if path == "cloud":
            cloud = base_directory_model("cloud")
            if content:
                cloud["format"] = "json"
                cloud["content"] = self.__build_cloud_notebook_lists()

                cloud["last_modified"] = max(
                    _to_utc(cat["last_modified"]) for cat in cloud["content"])

            model = cloud
        else:
            category = self.__category_from_path(path)
            namespace = self.__namespace_from_path(path)

            if namespace is None:
                model = self.__list_category(category, content)
            elif category is not None and namespace is not None:
                model = self.__list_namespace(category, namespace, content)

        return model

    def get(self, path, content=True, type=None, format=None):
        """Get a file or directory model."""
        path_fixed = path.strip("/")

        if path_fixed is None or path_fixed == "":
            path_fixed = "."

        try:
            if not self._is_remote_path(path_fixed):
                model = super().get(path, content, type, format)
                if path_fixed == "." and content and get_cloud_enabled():
                    cloud = base_directory_model("cloud")
                    cloud["content"] = self.__build_cloud_notebook_lists()
                    cloud["format"] = "json"
                    cloud["last_modified"] = max(
                        _to_utc(cat["last_modified"]) for cat in cloud["content"])
                    model["content"].append(cloud)

                return model

            if path_fixed.endswith(NOTEBOOK_EXT):
                path_fixed = path_fixed[: -1 * len(NOTEBOOK_EXT)]

            if type is None:
                if self._is_remote_dir(path_fixed):
                    type = "directory"
                else:
                    type = self.guess_type(path, allow_directory=True)

            if type == "notebook":
                return self._notebook_from_array(path_fixed, content)
            elif type == "file":
                return self._file_from_array(path_fixed, content, format)
            elif type == "directory":
                return self.__directory_model_from_path(path_fixed, content)
                # if model is not None:
                #     model.
        except Exception as e:
            raise tornado.web.HTTPError(
                500, "Error opening notebook {}: {}".format(path_fixed, str(e))
            )

    def save(self, model, path=""):
        """
        Save a file or directory model to path.
        Should return the saved model with no content.  Save implementations
        should call self.run_pre_save_hook(model=model, path=path) prior to
        writing any data.
        """
        path_fixed = path.strip("/") or "."

        try:
            model_type = model["type"]
        except KeyError:
            raise tornado.web.HTTPError(400, "No file type provided")

        if "content" not in model and model_type != "directory":
            raise tornado.web.HTTPError(400, u"No file content provided")

        if model_type not in ("directory", "file", "notebook"):
            raise tornado.web.HTTPError(400, "Unhandled contents type: %s" % model["type"])

        if not self._is_remote_path(path_fixed):
            return super().save(model, path)

        if path_fixed.endswith(NOTEBOOK_EXT):
            path_fixed = path_fixed[:-len(NOTEBOOK_EXT)]
            if model["type"] == "file":
                try:
                    _try_convert_file_to_notebook(model)
                except ValueError as ve:
                    raise tornado.web.HTTPError(400, f"Cannot parse Jupyter notebook: {ve}")

        self._is_new = True
        if (
            "content" in model
            and "metadata" in model["content"]
            and "language_info" in model["content"]["metadata"]
        ):
            self._is_new = False

        self.run_pre_save_hook(model=model, path=path)
        validation_message = None
        try:
            if model["type"] == "notebook":
                final_name, validation_message = self._save_notebook_tiledb(
                    model, path_fixed
                )
                if final_name is not None:
                    parts = path.split("/")
                    parts_length = len(parts)
                    parts[parts_length - 1] = final_name
                    path = "/".join(parts)
            elif model["type"] == "file":
                raise tornado.web.HTTPError(400, "Only .ipynb files may be created in the cloud.")
            else:
                if self._is_remote_path(path_fixed):
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
        path_fixed = path.strip("/")
        if self._is_remote_path(path_fixed):

            if path_fixed.endswith(NOTEBOOK_EXT):
                path_fixed = path_fixed[: -1 * len(NOTEBOOK_EXT)]

            tiledb_uri = self.tiledb_uri_from_path(path_fixed)
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
        old_path_fixed = old_path.strip("/")
        if self._is_remote_path(old_path_fixed):

            if old_path_fixed.endswith(NOTEBOOK_EXT):
                old_path_fixed = old_path_fixed[: -1 * len(NOTEBOOK_EXT)]

            tiledb_uri = self.tiledb_uri_from_path(old_path_fixed)
            parts_new = new_path.split("/")
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
        if path is None or path == "":
            path = "."

        path_fixed = path.strip("/")
        if self._is_remote_dir(path_fixed):
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
        path_fixed = path.strip("/")
        if self._is_remote_path(path_fixed):
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
        # if path is None or path == "":
        #     path = "."

        path_fixed = path.strip("/")
        if self._is_remote_path(path_fixed):
            if path_fixed.endswith(NOTEBOOK_EXT):
                path_fixed = path_fixed[: -1 * len(NOTEBOOK_EXT)]
            return self._array_exists(path_fixed)
        return super().file_exists(path)

    def new_untitled(self, path="", type="", ext="", options=""):
        """Create a new untitled file or directory in path
        path must be a directory
        File extension can be specified.
        Use `new` to create files with a fully specified path (including filename).
        options is a json string passed by the TileDB Prompt User Contents Jupyterlab notebook extension for additional notebook creation options
        """
        path = path.strip("/")
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
