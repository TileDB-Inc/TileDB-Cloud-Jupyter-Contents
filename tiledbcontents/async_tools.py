"""Stuff used to run TileDB core code asynchronously in another process.

Also includes some miscellaneous utilities.
"""


import asyncio
import functools
import os
import pathlib
import sys
import tempfile
from asyncio import subprocess
from typing import Callable, Optional, TypeVar

import cloudpickle
from typing_extensions import ParamSpec, Self

_P = ParamSpec("_P")
_R = TypeVar("_R")


async def call(__fn: Callable[_P, _R], *args: _P.args, **kwargs: _P.kwargs) -> _R:
    """Calls ``__fn(*args, **kwargs)`` on an executor as to not block.

    If you are making a call which eventually uses the core TileDB library,
    **do not use this function**! Instead, use ``call_external``, to ensure
    that your operations are fork-safe.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, functools.partial(__fn, *args, **kwargs))


_stub_lock = asyncio.Lock()
"""Lock guarding the creation of the lazy stub."""
_lazy_stub: Optional["_CallClient"] = None
"""A lazily-created stub used to make external calls.

Will be initialized the first time ``call_external`` is run.
"""


async def call_external(
    __fn: Callable[_P, _R], *args: _P.args, **kwargs: _P.kwargs
) -> _R:
    """Calls ``__fn(*args, **kwargs)`` in an external process without blocking.

    All calls which go to TileDB Core libraries need to be done in an external
    process since TileDB Core is not fork-safe, and JupyterLab uses forking
    to launch terminals.

    All calls sent to this function are handled in separate threads within the
    same external process.
    """
    global _lazy_stub
    async with _stub_lock:
        if _lazy_stub is None:
            _lazy_stub = await _CallClient.create()
    return await _lazy_stub.run(__fn, *args, **kwargs)


class _CallClient:
    """Stub to run callables in an external process."""

    def __init__(self, proc: subprocess.Process, addr: pathlib.Path) -> None:
        """Basic initializer. Use ``.create()`` to build a usable instance."""
        self._proc = proc
        """The async subprocess that is handling calls."""
        self._addr = addr
        """The socket address to write to."""

    async def run(
        self, __fn: Callable[_P, _R], *args: _P.args, **kwargs: _P.kwargs
    ) -> _R:
        """Calls ``__fn(*args, **kwargs)`` in an external process."""
        reader, writer = await asyncio.open_unix_connection(self._addr)
        call = functools.partial(__fn, *args, **kwargs)
        cloudpickle.dump(call, writer)
        await writer.drain()
        writer.write_eof()
        resp_bytes = await reader.read()
        response = cloudpickle.loads(resp_bytes)
        return response.result()

    async def close(self) -> None:
        """Asks the external process handling calls to exit gracefully."""
        _, writer = await asyncio.open_unix_connection(self._addr)
        cloudpickle.dump(None, writer)
        await writer.drain()
        writer.write_eof()
        self._addr.unlink()
        self._addr.parent.rmdir()
        await self._proc.wait()

    @classmethod
    async def create(cls) -> Self:
        """Starts a new external call handler and connects to it."""
        dir = tempfile.mkdtemp("tiledb-contents")
        sock_path = pathlib.Path(dir) / "sock"
        child = await asyncio.create_subprocess_exec(
            sys.executable,  # Use the same Python interpreter as this process.
            "-m",
            "tiledbcontents.call_server",
            "--parent-pid",
            str(os.getpid()),
            str(sock_path),
        )
        # Wait for the child process to create the socket.
        while not sock_path.exists():
            try:
                await asyncio.wait_for(child.wait(), 0.01)
            except asyncio.TimeoutError:
                # This is fine; it means the child process is running
                # but hasn't yet created the socket.
                await asyncio.sleep(0.01)
            else:
                # If we get here, that means that the wait_for call succeeded,
                # which means that the child process exited (i.e., crashed).
                raise OSError("Failed to start TileDB communication process")
        return cls(child, sock_path)
