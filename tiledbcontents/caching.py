"""Tools to handle caching of TileDB arrays and listings."""

import datetime
import time
from typing import Any, Dict, Optional, Sequence, Tuple, Type

import numpy as np
from tiledb import cloud
from tiledb.cloud import client

from . import async_tools
from . import models


class Array:
    """An Array wrapper that will cache metadata and the last-written data."""

    def __init__(self, uri: str):
        self._uri = uri
        """The URI of the array."""
        self._meta: Optional[Dict[str, Any]] = None
        """The metadata of the array as of the last open."""

    _cache: Dict[str, "Array"] = {}
    """The cache."""

    @classmethod
    def from_cache(cls, uri: str) -> "Array":
        try:
            return cls._cache[uri]
        except KeyError:
            pass
        ret = cls._cache[uri] = cls(uri)

        return ret

    @classmethod
    def purge(cls, uri: str) -> None:
        cls._cache.pop(uri, None)

    async def read(self) -> str:
        contents, self._meta = await async_tools.call_external(_read_sync, self._uri)
        return contents

    async def read_at(self, timestamp: datetime.datetime) -> str:
        """Reads the array at the given timestamp.

        This method will NOT update the contents cache, since this is an old
        version of the array.
        """
        contents, _ = await async_tools.call_external(_read_sync, self._uri, timestamp)
        return contents

    async def meta(self) -> models.Model:
        if self._meta is None:
            # We need to load array metadata for the first time.
            self._meta = await async_tools.call_external(_load_meta_sync, self._uri)
            assert self._meta is not None  # to shut up mypy
        return dict(self._meta)  # Make a copy to avoid spooky action at a distance.

    async def write_data(
        self,
        contents: str,
        new_meta: Dict[str, Optional[str]],
    ) -> None:
        contents_bytes = contents.encode("utf-8")
        new_meta = {k: v for k, v in new_meta.items() if v}
        await async_tools.call_external(
            _write_sync,
            self._uri,
            contents_bytes,
            new_meta,
        )
        # Ensure that we've already loaded metadata...
        await self.meta()
        assert self._meta is not None
        self._meta.update(new_meta)

    async def timestamps(self) -> Sequence[datetime.datetime]:
        return tuple(
            _from_millis(dt)
            for dt in await async_tools.call_external(_timestamps_sync, self._uri)
        )


def _read_sync(
    uri: str, timestamp: Optional[datetime.datetime] = None
) -> Tuple[str, Dict[str, Any]]:
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
        np_contents: np.ndarray = arr[0:file_size]["contents"]
    contents_bytes = np_contents.tobytes()
    contents = contents_bytes.decode("utf-8")
    return contents, meta


def _load_meta_sync(uri: str) -> Dict[str, Any]:
    """Synchronously loads metadata."""
    import tiledb

    with tiledb.open(uri, ctx=cloud.Ctx()) as arr:
        return dict(arr.meta.items())


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


_CATEGORY_LOADERS = {
    "owned": client.list_arrays,
    "shared": client.list_shared_arrays,
    "public": client.list_public_arrays,
}

CATEGORIES = frozenset(_CATEGORY_LOADERS)

_CACHE_SECS = 4


class ArrayListing:
    """An ArrayListing which will cache results for a specified time."""

    def __init__(self, category: str, namespace: Optional[str] = None):
        """
        Create an ArrayListing which will cache results for specified time
        :param category: category to list
        :param namespace: namespace to filter to
        :param cache_secs: cache time, defaults to 4 seconds
        """
        self.category = category
        self.namespace = namespace
        self._array_listing_future = None
        self.last_fetched: Optional[float] = None

    # A global cache of ArrayListings by category.
    _cache: Dict[Tuple[str, Optional[str]], "ArrayListing"] = {}

    @classmethod
    def from_cache(
        cls: Type["ArrayListing"],
        category: str,
        namespace: Optional[str] = None,
    ) -> "ArrayListing":
        """Fetches an ArrayListing from the cache, or constructs a new one if absent."""
        try:
            return cls._cache[category, namespace]
        except KeyError:
            pass
        cls._cache[category, namespace] = cls(category, namespace)
        return cls._cache[category, namespace]

    def _should_fetch(self):
        return (
            self.last_fetched is None
            or self._array_listing_future is None
            or self.last_fetched + _CACHE_SECS < time.time()
        )

    def _fetch(self, page: int = 1, per_page: int = 100):
        if self._should_fetch():
            try:
                loader = _CATEGORY_LOADERS[self.category]
            except KeyError:
                raise ValueError(
                    f"Invalid category name {self.category!r}; "
                    f"must be one of {set(_CATEGORY_LOADERS.keys())}"
                ) from None
            self._array_listing_future = loader(
                file_type=[cloud.rest_api.models.FileType.NOTEBOOK],
                namespace=self.namespace,
                async_req=True,
                per_page=per_page,
                page=page,
            )
            self.last_fetched = time.time()

        return self._array_listing_future

    def arrays(self):
        page = 1
        result = self._fetch(page=page).get()
        arrays = []
        if result:
            arrays = result.arrays
            total_pages = result.pagination_metadata.total_pages
            if total_pages > 1:
                for i in range(2, total_pages+1):
                    result = self._fetch(page=i)
                    if result and result.arrays:
                        arrays + result.arrays

        return arrays


_R = TypeVar("_R")


async def call(__fn: Callable[..., _R], *args: Any, **kwargs: Any) -> _R:
    """Calls ``__fn(*args, **kwargs)`` on an executor as to not block."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, functools.partial(__fn, *args, **kwargs))
