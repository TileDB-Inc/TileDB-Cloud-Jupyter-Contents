import itertools
import numbers
import os
import json
import mimetypes
import datetime
import time
import tiledb
import tiledb.cloud
import numpy
from notebook.services.contents.checkpoints import GenericCheckpointsMixin, Checkpoints
from notebook.services.contents.filemanager import FileContentsManager

from tornado.web import HTTPError

from .ipycompat import ContentsManager
from .ipycompat import HasTraits, Unicode
from .ipycompat import reads, from_dict, GenericFileCheckpoints

DUMMY_CREATED_DATE = datetime.datetime.fromtimestamp(86400)
NBFORMAT_VERSION = 4

NOTEBOOK_MIME = "application/x-ipynb+json"

TAG_JUPYTER_NOTEBOOK = "__jupyter-notebook"


def http_error(code, message):
    return HTTPError(code=code, message=message, reason=message)


def get_s3_prefix(namespace):
    """
    Get S3 path from user profile
    :return:
    """
    try:
        profile = tiledb.cloud.client.user_profile()

        if namespace == profile.username:
            if (
                profile.notebook_settings is not None
                and profile.notebook_settings.default_s3_path is not None
            ):
                return profile.notebook_settings.default_s3_path
        else:
            organization = tiledb.cloud.client.organization(namespace)
            if (
                organization.notebook_settings is not None
                and organization.notebook_settings.default_s3_path is not None
            ):
                return organization.notebook_settings.default_s3_path
    except tiledb.cloud.tiledb_cloud_error.TileDBCloudError as e:
        raise http_error(
            400,
            "Error fetching user default s3 path for new notebooks {}".format(str(e)),
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
        format="json",
    )
    return model


def remove_path_prefix(path_prefix, path):
    ret = path.split(path_prefix, 1)
    if len(ret) > 1:
        return ret[1]
    return ret


class TileDBContents(ContentsManager):
    def _save_notebook(self, model, uri):
        nb_contents = from_dict(model["content"])
        self.check_and_sign(nb_contents, uri)
        file_contents = numpy.array(bytearray(json.dumps(model["content"]), "utf-8"))

        final_name = self._write_bytes_to_array(
            uri, file_contents, model.get("mimetype"), model.get("format"), "notebook"
        )

        self.validate_notebook_model(model)
        return model.get("message")

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

    def _create_array(self, uri, name, retry=0):
        """

        :param uri:
        :param name:
        :param retry: number of times to retry request
        :return:
        """
        try:
            # The array will be be 1 dimensional with domain of 0 to max uint64. We use a tile extent of 1024 bytes
            dom = tiledb.Domain(
                tiledb.Dim(
                    name="position",
                    domain=(0, numpy.iinfo(numpy.uint64).max - 1025),
                    tile=1024,
                    dtype=numpy.uint64,
                    ctx=tiledb.cloud.Ctx(),
                ),
                ctx=tiledb.cloud.Ctx(),
            )

            schema = tiledb.ArraySchema(
                domain=dom,
                sparse=True,
                attrs=[tiledb.Attr(name="contents", dtype=numpy.uint8)],
                ctx=tiledb.cloud.Ctx(),
            )

            parts = uri.split("/")
            parts_len = len(parts)
            namespace = parts[parts_len - 2]
            array_name = parts[parts_len - 1]

            s3_prefix = get_s3_prefix(namespace)
            if s3_prefix is None:
                raise http_error(
                    400,
                    "You must set the default s3 prefix path for notebooks in {} profile settings".format(
                        namespace
                    ),
                )

            tiledb_uri_s3 = "tiledb://{}/{}".format(namespace, s3_prefix + array_name)

            # Create the (empty) array on disk.
            tiledb.SparseArray.create(tiledb_uri_s3, schema)

            tiledb_uri = "tiledb://{}/{}".format(namespace, array_name)
            time.sleep(0.25)
            tiledb.cloud.array.update_info(
                uri=tiledb_uri, array_name=name, tags=[TAG_JUPYTER_NOTEBOOK]
            )

            return array_name
        except tiledb.TileDBError as e:
            if "already exists" in str(e):
                parts = uri.split("/")
                parts_length = len(parts)
                array_name = parts[parts_length - 1]
                #
                # name_parts = array_name.split("-")
                # name_parts_length = len(name_parts)
                # if name_parts_length > 1:
                #     intVal = name_parts[name_parts_length - 1]
                #     intVal = int(intVal) + 1
                #     name_parts[name_parts_length - 1] = str(intVal)
                #     array_name = "-".join(name_parts)
                # else:
                #     array_name += "-1"

                # path

                array_name = self._increment_filename(array_name)

                parts[parts_length - 1] = array_name
                uri = "/".join(parts)

                return self._create_array(uri, name, retry)
            elif retry:
                retry -= 1
                return self._create_array(uri, name, retry)
        except HTTPError as e:
            raise e
        except Exception as e:
            raise http_error(400, "Error creating file %s " % str(e))

        return None

    def _array_exists(self, path):
        tiledb_uri = self.tiledb_uri_from_path(path)
        try:
            tiledb.cloud.array.info(tiledb_uri)
            return True
        except tiledb.cloud.tiledb_cloud_error.TileDBCloudError as e:
            if str(e) == "Array or Namespace Not found":
                return False

        return False

    def _write_bytes_to_array(
        self, uri, contents, mimetype=None, format=None, type=None
    ):
        tiledb_uri = self.tiledb_uri_from_path(uri)
        # if not self.vfs.is_dir(uri):
        #     self.__create_array(uri)
        final_array_name = None
        if not self._array_exists(uri):
            name = tiledb_uri.split("/")
            name = name[len(name) - 1]
            final_array_name = self._create_array(tiledb_uri, name, 5)

        with tiledb.open(tiledb_uri, mode="w", ctx=tiledb.cloud.Ctx()) as A:
            A[range(len(contents))] = {"contents": contents}
            A.meta["file_size"] = len(contents)
            if mimetype is not None:
                A.meta["mimetype"] = mimetype
            if format is not None:
                A.meta["format"] = format
            if type is not None:
                A.meta["type"] = type

        return final_array_name

    def _save_file(self, model, uri):
        file_contents = model["content"]
        return self._write_bytes_to_array(
            uri, file_contents, model.get("mimetype"), model.get("format"), "file"
        )

    # def __create_directory_and_group(self, path):
    #     """
    #
    #     :param path:
    #     :return:
    #     """
    #     try:
    #         if not self.vfs.is_dir(path):
    #             self.vfs.create_dir(path)
    #         elif tiledb.object_type(path) == "group":
    #             return
    #
    #         tiledb.group_create(path)
    #     except tiledb.TileDBError as e:
    #         raise http_error(500, e.message)

    def tiledb_uri_from_path(self, path):

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
                model["last_modified"] = info.last_accessed
                if "write" not in info.allowed_actions:
                    model["writable"] = False
                with tiledb.open(tiledb_uri, ctx=tiledb.cloud.Ctx()) as A:
                    meta = A.meta
                    file_content = A[slice(0, meta["file_size"])]
                    nb_content = reads(
                        file_content["contents"].tostring().decode("utf-8"),
                        as_version=NBFORMAT_VERSION,
                    )
                    self.mark_trusted_cells(nb_content, uri)
                    model["format"] = "json"
                    model["content"] = nb_content
                    self.validate_notebook_model(model)
            except tiledb.cloud.tiledb_cloud_error.TileDBCloudError as e:
                raise http_error(400, "Error fetching notebook info: {}".format(str(e)))
            except tiledb.TileDBError as e:
                raise http_error(
                    500, str(e),
                )

        return model

    def _file_from_array(self, uri, content=True, format=None):
        """
        Build a notebook model from database record.
        """
        model = base_model(uri)
        model["type"] = "file"

        # if self.fs.isfile(uri):
        #     model["last_modified"] = model["created"] = self.fs.lstat(path)["ST_MTIME"]
        # else:
        #     model["last_modified"] = model["created"] = DUMMY_CREATED_DATE
        if content:
            tiledb_uri = self.tiledb_uri_from_path(uri)
            try:
                info = tiledb.cloud.array.info(tiledb_uri)
                model["last_modified"] = info.last_accessed
                if "write" not in info.allowed_actions:
                    model["writable"] = False
                with tiledb.open(tiledb_uri, ctx=tiledb.cloud.Ctx()) as A:
                    meta = A.meta
                    if "mimetype" in meta:
                        model["mimetype"] = meta["mimetype"]
                    if "format" in meta:
                        model["format"] = meta["format"]
                    else:
                        model["format"] = format

                    if "type" in meta:
                        model["type"] = meta["type"]
                    file_content = A[slice(0, meta["file_size"])]
                    nb_content = file_content["contents"]
                    model["content"] = nb_content
                    self.validate_notebook_model(model)
            except tiledb.cloud.tiledb_cloud_error.TileDBCloudError as e:
                raise http_error(500, "Error fetching file info: {}".format(str(e)))
            except tiledb.TileDBError as e:
                raise http_error(
                    500, str(e),
                )

        return model

    def _is_remote_path(self, path):
        """
        Checks if a path is remote or not
        :param path:
        :return:
        """
        # if path.startswith("cloud/public") or path.startswith("cloud/shared") or path.startswith("cloud/mine"):
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
        pathFixed = path.strip("/")
        if self._is_remote_path(pathFixed):
            if self._is_remote_dir(pathFixed):
                return "directory"
            else:
                try:
                    tiledb_uri = self.tiledb_uri_from_path(pathFixed)
                    return self._get_type(tiledb_uri)
                    # if self.__get_mimetype(tiledb_uri) == NOTEBOOK_MIME:
                    #     return "notebook"
                    # else:
                    #     return "file"
                except Exception as e:
                    return "directory"
            # self.log.error("Error while saving file: %s %s", path, e, exc_info=True)
            # raise http_error(
            #     500, "Unexpected error while saving file: %s %s" % (path, e)
            # )
            return "file"

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
            with tiledb.open(uri, ctx=tiledb.cloud.Ctx()) as A:
                meta = A.meta
                if "mimetype" in meta:
                    return meta["mimetype"]
        except tiledb.cloud.tiledb_cloud_error.TileDBCloudError as e:
            raise http_error(500, "Error getting mimetype: {}".format(str(e)))
        except tiledb.TileDBError as e:
            raise http_error(
                500, str(e),
            )

        return None

    def _get_type(self, uri):
        """
        Fetch type from array metadata
        :param uri: of array
        :return:
        """
        try:
            with tiledb.open(uri, ctx=tiledb.cloud.Ctx()) as A:
                meta = A.meta
                if "type" in meta:
                    return meta["type"]
        except tiledb.cloud.tiledb_cloud_error.TileDBCloudError as e:
            raise http_error(500, "Error getting type: {}".format(str(e)))
        except tiledb.TileDBError as e:
            raise http_error(
                500, str(e),
            )

        return None


class TileDBCheckpoints(GenericFileCheckpoints, TileDBContents, Checkpoints):
    """requires the following methods:"""

    def _tiledb_checkpoint_model(self):
        return dict(id="checkpoints-not-supported", last_modified=DUMMY_CREATED_DATE,)

    def create_file_checkpoint(self, content, format, path):
        """ -> checkpoint model"""
        pathFixed = path.strip("/")
        if not self._is_remote_path(pathFixed):
            return super().create_file_checkpoint(content, format, path)

        return self._tiledb_checkpoint_model()

    def create_notebook_checkpoint(self, nb, path):
        """ -> checkpoint model"""
        pathFixed = path.strip("/")
        if not self._is_remote_path(pathFixed):
            return super().create_notebook_checkpoint(nb, path)

        return self._tiledb_checkpoint_model()

    def get_file_checkpoint(self, checkpoint_id, path):
        """ -> {'type': 'file', 'content': <str>, 'format': {'text', 'base64'}}"""
        pathFixed = path.strip("/")
        if not self._is_remote_path(pathFixed):
            return super().get_file_checkpoint(checkpoint_id, path)

    def get_notebook_checkpoint(self, checkpoint_id, path):
        """ -> {'type': 'notebook', 'content': <output of nbformat.read>}"""
        pathFixed = path.strip("/")
        if not self._is_remote_path(pathFixed):
            return super().get_notebook_checkpoint(checkpoint_id, path)

    def delete_checkpoint(self, checkpoint_id, path):
        """deletes a checkpoint for a file"""
        pathFixed = path.strip("/")
        if not self._is_remote_path(pathFixed):
            return super().delete_checkpoint(checkpoint_id, path)

    def list_checkpoints(self, path):
        """returns a list of checkpoint models for a given file,
        default just does one per file
        """
        pathFixed = path.strip("/")
        if not self._is_remote_path(pathFixed):
            return super().list_checkpoints(path)
        return []

    def rename_checkpoint(self, checkpoint_id, old_path, new_path):
        """renames checkpoint from old path to new path"""
        pathFixed = old_path.strip("/")
        if not self._is_remote_path(pathFixed):
            return super().rename_checkpoint(checkpoint_id, old_path, new_path)


class TileDBCloudContentsManager(TileDBContents, FileContentsManager, HasTraits):
    # This makes the checkpoints get saved on this directory
    root_dir = Unicode("./", config=True)

    def __init__(self, **kwargs):
        super(FileContentsManager, self).__init__(**kwargs)

    def _checkpoints_class_default(self):
        # return NoOpCheckpoints
        return TileDBCheckpoints

    def __list_namespace(self, category, namespace, content=False):
        arrays = []
        try:
            if category == "owned":
                arrays = tiledb.cloud.client.list_arrays(
                    tag=TAG_JUPYTER_NOTEBOOK, namespace=namespace
                )
            elif category == "shared":
                arrays = tiledb.cloud.client.list_shared_arrays(
                    tag=TAG_JUPYTER_NOTEBOOK, namespace=namespace
                )
            elif category == "public":
                arrays = tiledb.cloud.client.list_public_arrays(
                    tag=TAG_JUPYTER_NOTEBOOK, namespace=namespace
                )
        except tiledb.cloud.tiledb_cloud_error.TileDBCloudError as e:
            raise http_error(
                500, "Error listing notebooks in {}: ".format(namespace, str(e))
            )
        except tiledb.TileDBError as e:
            raise http_error(
                500, str(e),
            )

        model = base_directory_model(namespace)
        model["path"] = "cloud/{}/{}".format(category, namespace)
        if content:
            model["format"] = "json"
            model["content"] = []
            for notebook in arrays:
                nbmodel = base_model(notebook.name)
                nbmodel["path"] = "cloud/{}/{}/{}".format(
                    category, namespace, nbmodel["path"]
                )
                nbmodel["last_modified"] = notebook.last_accessed
                nbmodel["type"] = "notebook"
                if "write" not in notebook.allowed_actions:
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
            # if category == "owned":
            #     arrays = tiledb.cloud.client.list_arrays(tag=TAG_JUPYTER_NOTEBOOK)
            if category == "shared":
                arrays = tiledb.cloud.client.list_shared_arrays(
                    tag=TAG_JUPYTER_NOTEBOOK
                )
            elif category == "public":
                arrays = tiledb.cloud.client.list_public_arrays(
                    tag=TAG_JUPYTER_NOTEBOOK
                )
        except tiledb.cloud.tiledb_cloud_error.TileDBCloudError as e:
            raise http_error(
                500, "Error listing notebooks in {}: {}".format(category, str(e))
            )
        except tiledb.TileDBError as e:
            raise http_error(
                500, str(e),
            )

        model = base_directory_model(category)
        model["path"] = "cloud/{}".format(category)
        if content:
            model["format"] = "json"
            model["content"] = []
            namespaces = {}
            if (arrays is None or len(arrays) == 0) and category == "owned":
                # If the arrays are empty, and the category is for owned, we should list the user and their
                # organizations so they can create new notebooks
                try:
                    profile = tiledb.cloud.client.user_profile()
                    namespace_model = base_directory_model(profile.username)
                    namespace_model["path"] = "cloud/{}/{}".format(
                        category, profile.username
                    )
                    namespaces[profile.username] = namespace_model

                    for org in profile.organizations:
                        # Don't list public for owned
                        if org.organization_name == "public":
                            continue

                        namespace_model = base_directory_model(org.organization_name)
                        namespace_model["path"] = "cloud/{}/{}".format(
                            category, org.organization_name
                        )
                        namespaces[org.organization_name] = namespace_model
                except tiledb.cloud.tiledb_cloud_error.TileDBCloudError as e:
                    raise http_error(
                        500,
                        "Error listing notebooks in {}: {}".format(category, str(e)),
                    )
                except tiledb.TileDBError as e:
                    raise http_error(
                        500, str(e),
                    )

            else:
                for notebook in arrays:
                    namespace_model = base_directory_model(notebook.namespace)
                    namespace_model["writable"] = False
                    namespace_model["path"] = "cloud/{}/{}".format(
                        category, notebook.namespace
                    )
                    namespaces[notebook.namespace] = namespace_model

            model["content"] = list(namespaces.values())

        return model

    def __build_cloud_notebook_lists(self):

        ret = {
            "owned": base_directory_model("owned"),
            "public": base_directory_model("public"),
            "shared": base_directory_model("shared"),
        }
        try:
            owned_notebooks = tiledb.cloud.client.list_arrays(tag=TAG_JUPYTER_NOTEBOOK)
            shared_notebooks = tiledb.cloud.client.list_shared_arrays(
                tag=TAG_JUPYTER_NOTEBOOK
            )
            public_notebooks = tiledb.cloud.client.list_public_arrays(
                tag=TAG_JUPYTER_NOTEBOOK
            )

            ret["owned"]["path"] = "cloud/owned"
            ret["public"]["path"] = "cloud/public"
            ret["shared"]["path"] = "cloud/shared"

            if len(owned_notebooks) > 0:
                ret["owned"]["format"] = "json"
                ret["owned"]["content"] = []
                for notebook in owned_notebooks:
                    model = base_model(notebook.name)
                    model["type"] = "notebook"
                    model["last_modified"] = notebook.last_accessed
                    model["path"] = "cloud/{}/{}".format("owned", model["path"])
                    ret["owned"]["content"].append(model)

            if len(shared_notebooks) > 0:
                ret["shared"]["format"] = "json"
                ret["shared"]["content"] = []
                for notebook in owned_notebooks:
                    model = base_model(notebook.name)
                    model["type"] = "notebook"
                    model["last_modified"] = notebook.last_accessed
                    model["path"] = "cloud/{}/{}".format("shared", model["path"])
                    ret["shared"]["content"].append(model)

            if len(public_notebooks) > 0:
                ret["public"]["format"] = "json"
                ret["public"]["content"] = []
                for notebook in owned_notebooks:
                    model = base_model(notebook.name)
                    model["type"] = "notebook"
                    model["last_modified"] = notebook.last_accessed
                    model["path"] = "cloud/{}/{}".format("public", model["path"])
                    ret["public"]["content"].append(model)
        except tiledb.cloud.tiledb_cloud_error.TileDBCloudError as e:
            raise http_error(
                500, "Error building cloud notebook info: {}".format(str(e))
            )
        except tiledb.TileDBError as e:
            raise http_error(
                500, str(e),
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
            cloud["format"] = "json"
            cloud["content"] = self.__build_cloud_notebook_lists()
            # model["content"] = [cloud]
            model = cloud
        else:
            category = self.__category_from_path(path)
            namespace = self.__namespace_from_path(path)

            if namespace is None:
                model = self.__list_category(category, content)
            elif category is not None and namespace is not None:
                model = self.__list_namespace(category, namespace, content)

        return model

    def __group_to_models(self, path_prefix, paths):
        """
        Applies _notebook_model_from_s3_path or _file_model_from_s3_path to each entry of `paths`,
        depending on the result of `guess_type`.
        """
        ret = []
        for path in paths:
            # path = remove_path_prefix("file://" + os.getcwd() + "/", path)
            # path_after = remove_path_prefix(path_prefix, path)
            # path = path_after
            # if os.path.basename(path) == self.dir_keep_file:
            #      continue
            type_ = self.guess_type(path, allow_directory=True)
            if type_ == "notebook":
                ret.append(self._notebook_from_array(path, False))
            elif type_ == "file":
                ret.append(self._file_from_array(path, False, None))
            elif type_ == "directory":
                ret.append(self.__directory_model_from_path(path, False))
            else:
                HTTPError(500, "Unknown file type %s for file '%s'" % (type_, path))
        return ret

    def get(self, path, content=True, type=None, format=None):
        """Get a file or directory model."""
        pathFixed = path.strip("/")

        if pathFixed == "" or pathFixed is None:
            pathFixed = "."

        if not self._is_remote_path(pathFixed):
            model = super().get(path, content, type, format)
            if pathFixed == "." and content:
                cloud = base_directory_model("cloud")
                cloud["content"] = self.__build_cloud_notebook_lists()
                model["content"].append(cloud)

            return model

        if type is None:
            if self._is_remote_dir(pathFixed):
                type = "directory"
            else:
                type = self.guess_type(path, allow_directory=True)

        if type == "notebook":
            return self._notebook_from_array(pathFixed, content)
        elif type == "file":
            return self._file_from_array(pathFixed, content, format)
        elif type == "directory":
            return self.__directory_model_from_path(pathFixed, content)
            # if model is not None:
            #     model.

    def save(self, model, path=""):
        """
        Save a file or directory model to path.
        Should return the saved model with no content.  Save implementations
        should call self.run_pre_save_hook(model=model, path=path) prior to
        writing any data.
        """
        self.run_pre_save_hook(model=model, path=path)
        pathFixed = path.strip("/")

        if pathFixed == "" or pathFixed is None:
            pathFixed = "."

        if "type" not in model:
            raise http_error(400, u"No file type provided")
        if "content" not in model and model["type"] != "directory":
            raise http_error(400, u"No file content provided")

        if model["type"] not in ("directory", "file", "notebook"):
            raise http_error(400, "Unhandled contents type: %s" % model["type"])

        if not self._is_remote_path(pathFixed):
            return super().save(model, path)

        validation_message = None
        try:
            if model["type"] == "notebook":
                validation_message = self._save_notebook(model, pathFixed)
            elif model["type"] == "file":
                validation_message = self._save_file(model, pathFixed)
            else:
                if self._is_remote_path(pathFixed):
                    raise http_error(
                        400,
                        "Trying to create unsupported type: %s in cloud"
                        % model["type"],
                    )
                # else:
                #     return super().save(model, path)
                # validation_message = self.__create_directory_and_group(path)
        except Exception as e:
            self.log.error("Error while saving file: %s %s", path, e, exc_info=True)
            raise e

        model = self.get(path, type=model["type"], content=False)
        if validation_message is not None:
            model["message"] = validation_message
        return model

    def delete_file(self, path):
        """Delete the file or directory at path."""
        # if self.vfs.is_file(path):
        #     self.vfs.remove_file(path)
        # else:
        #     self.vfs.remove_dir(path)
        if self._is_remote_path(path):
            tiledb_uri = self.tiledb_uri_from_path(path)
            try:
                return tiledb.cloud.array.deregister_array(tiledb_uri)
            except tiledb.cloud.tiledb_cloud_error.TileDBCloudError as e:
                raise http_error(
                    500, "Error deregistering {}: ".format(tiledb_uri, str(e))
                )
            except tiledb.TileDBError as e:
                raise http_error(
                    500, str(e),
                )
        else:
            return super().delete_file(path)

    def rename_file(self, old_path, new_path):
        """Rename a file or directory."""
        # self.vfs.move_file(old_path, new_path)
        return super().rename(old_path, new_path)

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
        if path == "" or path is None:
            path = "."

        pathFixed = path.strip("/")
        if self._is_remote_dir(pathFixed):
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
        pathFixed = path.strip("/")
        if self._is_remote_path(pathFixed):
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
        # if path == "" or path is None:
        #     path = "."

        pathFixed = path.strip("/")
        if self._is_remote_path(pathFixed):
            return self._array_exists(pathFixed)
        return super().file_exists(path)
