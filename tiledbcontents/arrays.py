"""Functions which handle accessing arrays, local and remote."""

import datetime
import os
import time
from typing import Any, Dict, Optional, Sequence, Tuple
from typing_extensions import Self

import numpy as np
import tornado.web
from tiledb import cloud

from . import async_tools
from . import models
from . import paths

JUPYTER_IMAGE_NAME_ENV = "JUPYTER_IMAGE_NAME"
JUPYTER_IMAGE_SIZE_ENV = "JUPYTER_IMAGE_SIZE"


class ArrayHandle:
    """Provides methods to easily read and write arrays."""

    @classmethod
    def from_path(cls, path: str) -> Self:
        return cls(paths.tiledb_uri_from_path(path))

    def __init__(self, uri: str) -> None:
        self.uri = uri
        """The TileDB URI of the array."""

    async def cloud_info(self) -> Any:
        """Calls ``tiledb.cloud.array.info`` on this Array."""
        return await async_tools.call(cloud.array.info, self.uri)

    async def model_updates(self) -> models.Model:
        """Returns a dictionary of updates to apply to the returned model."""
        to_update = {}
        info_ft = self.cloud_info()
        ts_ft = self.timestamps()
        info = await info_ft
        to_update["last_modified"] = models.to_utc(info.last_accessed)
        if "write" not in info.allowed_actions:
            to_update["writable"] = False
        ts = await ts_ft
        if ts:
            to_update["last_modified"] = ts[-1]
        return to_update

    async def exists(self) -> bool:
        try:
            await self.cloud_info()
            return True
        except cloud.TileDBCloudError:
            return False

    async def fetch_type(self) -> Optional[str]:
        try:
            _, meta = await self.read_at(metadata_only=True)
        except Exception as e:
            raise tornado.web.HTTPError(
                500, f"Error reading {self.uri!r} file type: {e}"
            )
        return meta.get("type")

    async def read_at(
        self,
        timestamp: Optional[datetime.datetime] = None,
        *,
        include_content: bool = True,
    ) -> Tuple[Optional[bytes], Dict[str, Any]]:
        """Reads the array at the given timestamp, or now if not provided."""
        return await async_tools.call_external(
            _read_sync, self.uri, timestamp, include_content=include_content
        )

    async def write(self, contents: str, new_meta: Dict[str, Any]) -> None:
        contents_bytes = contents.encode("utf-8")
        new_meta = {k: v for k, v in new_meta.items() if v is not None}
        await async_tools.call_external(
            _write_sync,
            self.uri,
            contents_bytes,
            new_meta,
        )

    async def timestamps(self) -> Sequence[datetime.datetime]:
        return tuple(
            _from_millis(dt)
            for dt in await async_tools.call_external(_timestamps_sync, self.uri)
        )


async def write_or_create(
    path: str,
    contents: str,
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
        tiledb_uri, final_array_name = await async_tools.call_external(
            create,
            tiledb_uri,
            retry=5,
            s3_prefix=s3_prefix,
            s3_credentials=s3_credentials,
            is_user_defined_name=is_user_defined_name,
        )

    array = ArrayHandle(tiledb_uri)
    await array.write(contents, dict(mimetype=mimetype, format=format, type=type))
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
    import tiledb

    parts = paths.split(path)
    namespace = parts[-2]
    array_name = parts[-1]
    if not is_user_defined_name:
        array_name += "_" + paths.generate_id()

    if namespace in paths.RESERVED_NAMES:
        raise tornado.web.HTTPError(
            403,
            f"{namespace!r} is not a valid folder to create notebooks. "
            "Please select a proper namespace (username or organization name).",
        )

    # Retrieving credentials is optional
    # If None, default credentials will be used
    s3_credentials = s3_credentials or _namespace_s3_credentials(namespace)

    if s3_credentials is None:
        # Use the general cloud context by default.
        tiledb_create_context = cloud.Ctx()
    else:
        # update context with config having header set
        tiledb_create_context = cloud.Ctx(
            {
                "rest.creation_access_credentials_name": s3_credentials,
            }
        )

    # The array will be be 1 dimensional with domain of 0 to max uint64.
    # We use a tile extent of 1024 bytes.
    dom = tiledb.Domain(
        tiledb.Dim(
            name="position",
            domain=(0, np.iinfo(np.uint64).max - 1025),
            tile=1024,
            dtype=np.uint64,
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
                dtype=np.uint8,
                filters=tiledb.FilterList([tiledb.ZstdFilter()]),
            )
        ],
        ctx=tiledb_create_context,
    )

    s3_prefix = s3_prefix or _namespace_s3_prefix(namespace)
    if s3_prefix is None:
        raise tornado.web.HTTPError(
            403,
            "You must set the default storage prefix path for notebooks "
            f"in {namespace} profile settings",
        )

    while True:
        try:
            tiledb_uri = "tiledb://" + paths.join(namespace, array_name)

            # Because TileDB Cloud now allows creating arrays with duplicate
            # names, we can't rely on DenseArray.create to error out if we try
            # to create an array that shadows an existing name.
            if tiledb.object_type(tiledb_uri):
                # OK, let's try incrementing the filename.
                array_name = paths.increment_filename(array_name)
                # Start from the top. This doesn't count as a retry.
                continue

            # Create the (empty) array on disk.
            tiledb_uri_s3 = "tiledb://" + paths.join(namespace, s3_prefix, array_name)
            tiledb.DenseArray.create(tiledb_uri_s3, schema)
            break  # Success! Leave the retry loop.
        except tiledb.TileDBError as e:
            if "Error while listing with prefix" in str(e):
                # It is possible to land here if user sets the wrong
                # default credentials with respect to the default storage path.
                raise tornado.web.HTTPError(
                    400, f"Error creating file: {e}. Are your credentials valid?"
                ) from e

            if retry <= 0:
                # We're out of retries.
                raise tornado.web.HTTPError(500, str(e)) from e
            retry -= 1
        except tornado.web.HTTPError:
            # HTTPErrors are all errors that we raise ourselves,
            # so just reraise them.
            raise
        except Exception as e:
            raise tornado.web.HTTPError(400, f"Error creating file: {e}") from e
    time.sleep(0.25)

    file_properties = {}
    # Get image name from env if exists
    # This is stored as a tag for TileDB Cloud for searching, filtering,
    # and launching.
    image_name = os.getenv(JUPYTER_IMAGE_NAME_ENV)
    if image_name is not None:
        file_properties[cloud.rest_api.models.FilePropertyName.IMAGE] = image_name

    # Get image size from env if exists
    # This is stored as a tag for TileDB Cloud for searching, filtering,
    # and launching.
    image_size = os.getenv(JUPYTER_IMAGE_SIZE_ENV)
    if image_size is not None:
        file_properties[cloud.rest_api.models.FilePropertyName.SIZE] = image_size

    cloud.array.update_info(uri=tiledb_uri, array_name=array_name)

    cloud.array.update_file_properties(
        uri=tiledb_uri,
        file_type=cloud.rest_api.models.FileType.NOTEBOOK,
        # If file_properties is empty, don't send anything at all.
        file_properties=file_properties or None,
    )

    return tiledb_uri, array_name


def _namespace_s3_prefix(namespace: str) -> Optional[str]:
    """Fetches the default S3 path prefix from the user or organization profile."""
    try:
        profile = cloud.client.user_profile()

        if namespace == profile.username:
            if (
                profile.asset_locations.notebooks
                and profile.asset_locations.notebooks.path
            ):
                return profile.asset_locations.notebooks.path
            if profile.default_s3_path is not None:
                return paths.join(profile.default_s3_path, "notebooks")
            return None
        organization = cloud.client.organization(namespace)
        if (
            organization.asset_locations.notebooks
            and organization.asset_locations.notebooks.path
        ):
            return organization.asset_locations.notebooks.path
        if organization.default_s3_path is not None:
            return paths.join(organization.default_s3_path, "notebooks")
        return None
    except cloud.TileDBCloudError as e:
        raise tornado.web.HTTPError(
            400, f"Error fetching user default s3 path for new notebooks {e}"
        ) from e


def _namespace_s3_credentials(namespace: str) -> Optional[str]:
    """Fetches the default S3 credentials from the user or organization profile."""
    try:
        profile = cloud.client.user_profile()

        if namespace == profile.username:
            if (
                profile.asset_locations.notebooks
                and profile.asset_locations.notebooks.credentials_name
            ):
                return profile.asset_locations.notebooks.credentials_name
            return profile.default_s3_path_credentials_name
        organization = cloud.client.organization(namespace)
        if (
            organization.asset_locations.notebooks
            and organization.asset_locations.notebooks.credentials_name
        ):
            return organization.asset_locations.notebooks.credentials_name
        return organization.default_s3_path_credentials_name
    except cloud.TileDBCloudError as e:
        raise tornado.web.HTTPError(
            400,
            f"Error fetching default credentials for {namespace!r} default "
            f"s3 path for new notebooks: {e}",
        ) from e


def _read_sync(
    uri: str,
    timestamp: Optional[datetime.datetime] = None,
    *,
    include_content: bool = True,
) -> Tuple[Optional[bytes], Dict[str, Any]]:
    import tiledb

    with tiledb.open(
        uri,
        "r",
        timestamp=_to_millis(timestamp),
        ctx=cloud.Ctx(),
    ) as arr:
        meta = dict(arr.meta.items())
        try:
            file_size = meta["file_size"]
        except KeyError:
            raise Exception(
                f"file_size metadata entry not present in {uri}"
                f" (existing keys: {set(meta)})"
            )
        np_contents: Optional[np.ndarray] = None
        if include_content:
            np_contents = arr[0:file_size]["contents"]
    contents: Optional[bytes] = None
    if np_contents is not None:
        contents = np_contents.tobytes()
    return contents, meta


def _write_sync(
    uri: str,
    contents_bytes: bytes,
    new_meta: Dict[str, Optional[str]],
) -> None:
    import tiledb

    contents_arr = np.frombuffer(contents_bytes, dtype=np.uint8)
    with tiledb.open(
        uri,
        mode="w",
        ctx=cloud.Ctx(),
        timestamp=int(time.time() * 1000),
    ) as arr:
        arr[: len(contents_bytes)] = {"contents": contents_arr}
        arr.meta["file_size"] = len(contents_bytes)
        arr.meta.update(new_meta)


def _timestamps_sync(uri: str) -> Sequence[int]:
    """Loads the end timestamps of the given array."""
    import tiledb

    frags = tiledb.array_fragments(uri)
    # Timestamps are stored as (start, end), and we want the end.
    return tuple(f.timestamp_range[1] for f in frags)


def _to_millis(dt: Optional[datetime.datetime]) -> Optional[int]:
    if dt:
        return round(1000 * dt.timestamp())
    return None


def _from_millis(millis: int) -> datetime.datetime:
    return datetime.datetime.fromtimestamp(millis / 1000, tz=datetime.timezone.utc)
