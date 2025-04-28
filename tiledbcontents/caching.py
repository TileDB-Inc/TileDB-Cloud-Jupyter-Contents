"""Tools to handle caching of TileDB arrays and listings."""

from concurrent import futures
import functools
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from tiledb import cloud
from tiledb.cloud import client



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
        self._executor = futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="array-cache-"
        )

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

    def _maybe_fetch(self):
        if self._should_fetch():
            try:
                loader = _CATEGORY_LOADERS[self.category]
            except KeyError:
                raise ValueError(
                    f"Invalid category name {self.category!r}; "
                    f"must be one of {set(_CATEGORY_LOADERS.keys())}"
                ) from None
            partial = functools.partial(
                loader,
                file_type=[cloud.rest_api.models.FileType.NOTEBOOK],
                namespace=self.namespace,
            )
            self._array_listing_future = self._executor.submit(_load_paginated, partial)

            self.last_fetched = time.time()

        assert self._array_listing_future
        return self._array_listing_future

    def arrays(self):
        return self._maybe_fetch().result()


_PER_PAGE = 100


def _load_paginated(partial: Callable[..., Any]) -> List[object]:
    first_result = partial(page=1, per_page=_PER_PAGE)
    everything = list(first_result.arrays or ())
    total_pages = int(first_result.pagination_metadata.total_pages)
    for subsequent in range(1, total_pages):
        next_result = partial(page=subsequent + 1, per_page=_PER_PAGE)
        everything.extend(next_result.arrays or ())
    return everything
