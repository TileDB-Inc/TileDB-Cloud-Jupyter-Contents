"""Tools to handle caching of TileDB arrays and listings."""

import time
from typing import Any, Dict, Optional, Tuple, Type

import tiledb
import tiledb.cloud
import tiledb.cloud.client
import tornado.web

CLOUD_CONTEXT = tiledb.cloud.Ctx()


class Array:
    """Caching wrapper around a TileDB Array."""

    def __init__(self, uri: str, contents: Optional[Dict[str, Any]] = None):
        """
        Create an Array wrapping a TileDB Array class
        :param uri:
        """
        self.uri = uri
        try:
            self.array: tiledb.Array = tiledb.open(uri, ctx=CLOUD_CONTEXT)
        except Exception as e:
            raise tornado.web.HTTPError(400, f"Error in Array init: {e}") from e
        self.contents_fetched = False
        self.cached_meta: Dict[str, Any] = {}
        self.cache_metadata()
        # Cache contents if exist during first array write
        self.cached_contents: Optional[Dict[str, Any]] = contents

    # A global cache of Arrays by URI.
    _cache: Dict[str, "Array"] = {}

    @classmethod
    def from_cache(cls: Type["Array"], uri: str, *args, **kwargs) -> "Array":
        """Fetches an Array from the cache, or constructs a new one if not present."""
        try:
            return cls._cache[uri]
        except KeyError:
            pass
        cls._cache[uri] = cls(uri, *args, **kwargs)
        return cls._cache[uri]

    @classmethod
    def purge(cls: Type["Array"], uri: str) -> None:
        cls._cache.pop(uri, None)

    def read(self) -> Optional[Dict[str, Any]]:
        """
        Fetch all contents of the array based on file_size metadata field
        :return: raw bytes of content
        """

        try:
            if self.cached_contents is not None:
                contents = self.cached_contents
                # Invalidate cached contents after first read
                # Used only to speed up first read after creation, avoiding the server roundtrip
                # since contents are already available
                self.cached_contents = None
                return contents

            if self.contents_fetched:
                self.reopen()

            self.contents_fetched = True
            meta = self.array.meta
            if "file_size" in meta:
                return self.array[slice(0, meta["file_size"])]
        except Exception as e:
            raise tornado.web.HTTPError(400, f"Error in Array::read: {e}") from e

        return None

    def reopen(self):
        """
        Reopen an array at the current timestamp
        :return:
        """
        try:
            if self.array is not None:
                self.array.close()

            self.array = tiledb.open(self.uri, ctx=CLOUD_CONTEXT)
        except Exception as e:
            raise tornado.web.HTTPError(400, f"Error in Array::reopen: {e}") from e

    def cache_metadata(self):
        try:
            self.cached_meta = dict(self.array.meta.items())
        except Exception as e:
            raise tornado.web.HTTPError(
                400, f"Error in Array::cache_metadata: {e}"
            ) from e


_CATEGORY_LOADERS = {
    "owned": tiledb.cloud.client.list_arrays,
    "shared": tiledb.cloud.client.list_shared_arrays,
    "public": tiledb.cloud.client.list_public_arrays,
}

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
        self.array_listing_future = None
        self.last_fetched: Optional[float] = None

    # A global cache of ArrayListings by category.
    _cache: Dict[Tuple[str, Optional[str]], "ArrayListing"] = {}

    @classmethod
    def from_cache(
        cls: Type["ArrayListing"], category: str, namespace: Optional[str] = None,
    ) -> "ArrayListing":
        """Fetches an ArrayListing from the cache, or constructs a new one if not present."""
        try:
            return cls._cache[category, namespace]
        except KeyError:
            pass
        cls._cache[category, namespace] = cls(category, namespace)
        return cls._cache[category, namespace]

    def _should_fetch(self):
        return (
            self.last_fetched is None
            or self.array_listing_future is None
            or self.last_fetched + _CACHE_SECS < time.time()
        )

    def fetch(self):
        if self._should_fetch():
            try:
                loader = _CATEGORY_LOADERS[self.category]
            except KeyError:
                raise ValueError(
                    f"Invalid category name {self.category!r}; "
                    f"must be one of {set(_CATEGORY_LOADERS.keys())}"
                ) from None
            self.array_listing_future = loader(
                file_type=[tiledb.cloud.rest_api.models.FileType.NOTEBOOK],
                namespace=self.namespace,
                async_req=True,
            )
            self.last_fetched = time.time()

        return self

    def get(self):
        if self.array_listing_future is None:
            self.fetch()

        return self.array_listing_future.get()

    def arrays(self):
        ret = self.get()
        return ret and ret.arrays
