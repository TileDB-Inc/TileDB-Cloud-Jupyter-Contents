"""Functions which handle accessing arrays, local and remote."""

import os
import time
from typing import Optional, Tuple

import numpy
import tiledb.cloud
import tornado.web

from . import caching
from . import paths

JUPYTER_IMAGE_NAME_ENV = "JUPYTER_IMAGE_NAME"
JUPYTER_IMAGE_SIZE_ENV = "JUPYTER_IMAGE_SIZE"


def exists(path: str) -> bool:
    """Checks if an array exists in TileDB Cloud."""
    tdb_uri = paths.tiledb_uri_from_path(path)
    try:
        tiledb.cloud.array.info(tdb_uri)
        return True
    except tiledb.cloud.TileDBCloudError:
        pass
    return False


def fetch_type(uri: str) -> Optional[str]:
    try:
        arr = caching.Array.from_cache(uri)
    except tiledb.cloud.tiledb_cloud_error.TileDBCloudError as e:
        raise tornado.web.HTTPError(500, f"Error getting type: {e}") from e
    except tiledb.TileDBError as e:
        raise tornado.web.HTTPError(500, str(e)) from e
    except Exception as e:
        raise tornado.web.HTTPError(400, f"Error getting file type: {e}") from e
    try:
        return arr.cached_meta["type"]
    except KeyError:
        return None


def current_milli_time() -> int:
    return round(time.time() * 1000)


def write_bytes(
    path: str,
    contents: numpy.ndarray,
    *,
    mimetype: Optional[str] = None,
    format: Optional[str] = None,
    type: Optional[str] = None,
    s3_prefix: Optional[str] = None,
    s3_credentials: Optional[str] = None,
    is_user_defined_name: bool = False,
    is_new: bool = False,
) -> Optional[str]:
    """Writes the given bytes to the array. Will create the array if it doesn't exist.

    :param uri: The URI of the array to write to.
    :param contents: The bytes to write.
    :param mimetype: The MIME type to set in the metadata.
    :param format: The format to set in the metadata.
    :param s3_prefix: The S3 path to write to.
    :param s3_credentials: The name within TileDB Cloud of the s3 credentials
        to use when writing to the ``s3_prefix``.
    :param is_user_defined_name: True to indicate that the user provided
        the given filename. False to indicate that it was generated.
    :param is_new: True if we are creating a new array, false if we are writing
        to an existing array.
    :return: If creating a new array, the name of that newly-created array.
        If writing to an existing array, None.
    """
    tiledb_uri = paths.tiledb_uri_from_path(path)
    final_array_name = None
    if is_new:
        # if not arrays.exists(uri):
        tiledb_uri, final_array_name = create(
            tiledb_uri,
            retry=5,
            s3_prefix=s3_prefix,
            s3_credentials=s3_credentials,
            is_user_defined_name=is_user_defined_name,
        )

    # TODO: remove calling tiledb.open with timestamp when related TileDB Core bug is fixed
    # Calling tiledb.open with timestamp should not be needed because TileDB-Py always send the current timestamp
    # Due to bug in TileDB Core the timestamp at which write is done is always 0
    with tiledb.open(tiledb_uri, mode="w", timestamp=current_milli_time(), ctx=caching.CLOUD_CONTEXT) as arr:
        arr[0:len(contents)] = {"contents": contents}
        arr.meta["file_size"] = len(contents)
        if mimetype is not None:
            arr.meta["mimetype"] = mimetype
        if format is not None:
            arr.meta["format"] = format
        if type is not None:
            arr.meta["type"] = type

    caching.Array.force_create(tiledb_uri, {"contents": contents})

    return final_array_name


def create(
    path: str,
    *,
    retry: int = 0,
    s3_prefix: Optional[str] = None,
    s3_credentials: Optional[str] = None,
    is_user_defined_name: bool = False,
) -> Tuple[str, str]:
    """Creates a new array for storing a notebook file.

    :param uri: The location to create the array.
    :param retry: The number of times to retry the request if it fails.
    :param s3_prefix: The S3 path to write to.
    :param s3_credentials: The name within TileDB Cloud of the S3 credentials
        to use when writing to the ``s3_prefix``.
    :param is_user_defined_name: True to indicate that the user provided
        the given filename. False to indicate that it was generated.
    :return: A tuple of (the TileDB URI, the name of the array)
    """
    while True:
        try:
            parts = paths.split(path)
            namespace = parts[-2]

            # Use the general cloud context by default.
            tiledb_create_context = caching.CLOUD_CONTEXT

            # Retrieving credentials is optional
            # If None, default credentials will be used
            s3_credentials = s3_credentials or _namespace_s3_credentials(namespace)

            if s3_credentials is not None:
                # update context with config having header set
                tiledb_create_context = tiledb.cloud.Ctx({
                    "rest.creation_access_credentials_name": s3_credentials,
                })
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
                array_name = parts[-1]
            else:
                array_name = parts[-1] + "_" + paths.generate_id()

            if namespace in paths.RESERVED_NAMES:
                raise tornado.web.HTTPError(
                    403,
                    f"{namespace!r} is not a valid folder to create notebooks. "
                    "Please select a proper namespace (username or organization name).",
                )

            s3_prefix = s3_prefix or _namespace_s3_prefix(namespace)
            if s3_prefix is None:
                raise tornado.web.HTTPError(
                    403,
                    "You must set the default s3 prefix path for notebooks "
                    f"in {namespace} profile settings",
                )

            tiledb_uri_s3 = "tiledb://" + paths.join(namespace, s3_prefix, array_name)

            # Create the (empty) array on disk.
            tiledb.DenseArray.create(tiledb_uri_s3, schema)

            tiledb_uri = "tiledb://" + paths.join(namespace, array_name)
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

            tiledb.cloud.array.update_file_properties(
                uri=tiledb_uri,
                file_type=tiledb.cloud.rest_api.models.FileType.NOTEBOOK,
                # If file_properties is empty, don't send anything at all.
                file_properties=file_properties or None,
            )

            return tiledb_uri, array_name
        except tiledb.TileDBError as e:
            if "Error while listing with prefix" in str(e):
                # It is possible to land here if user sets wrong default s3 credentials with respect to default s3 path
                raise tornado.web.HTTPError(
                    400, f"Error creating file: {e}. Are your credentials valid?"
                ) from e

            if "already exists" in str(e):
                # OK, let's try incrementing the filename.
                parts = paths.split(path)
                array_name = parts[-1]

                array_name = paths.increment_filename(array_name)

                parts[-1] = array_name
                path = paths.join(*parts)
                # This doesn't count as a retry.
            elif retry:
                retry -= 1
            else:
                # We're out of retries.
                raise
        except tornado.web.HTTPError:
            # HTTPErrors are all errors that we raise ourselves,
            # so just reraise them.
            raise
        except Exception as e:
            raise tornado.web.HTTPError(400, f"Error creating file: {e}") from e


def _namespace_s3_prefix(namespace: str) -> Optional[str]:
    """Fetches the default S3 path prefix from the user or organization profile."""
    try:
        profile = tiledb.cloud.client.user_profile()

        if namespace == profile.username:
            if profile.default_s3_path is not None:
                return paths.join(profile.default_s3_path, "notebooks")
            return None
        organization = tiledb.cloud.client.organization(namespace)
        if organization.default_s3_path is not None:
            return paths.join(organization.default_s3_path, "notebooks")
        return None
    except tiledb.cloud.tiledb_cloud_error.TileDBCloudError as e:
        raise tornado.web.HTTPError(
            400,
            f"Error fetching user default s3 path for new notebooks {e}"
        ) from e


def _namespace_s3_credentials(namespace: str) -> Optional[str]:
    """Fetches the default S3 credentials from the user or organization profile."""
    try:
        profile = tiledb.cloud.client.user_profile()

        if namespace == profile.username:
            return profile.default_s3_path_credentials_name
        organization = tiledb.cloud.client.organization(namespace)
        return organization.default_s3_path_credentials_name
    except tiledb.cloud.tiledb_cloud_error.TileDBCloudError as e:
        raise tornado.web.HTTPError(
            400,
            f"Error fetching default credentials for {namespace!r} default "
            f"s3 path for new notebooks: {e}",
        ) from e
