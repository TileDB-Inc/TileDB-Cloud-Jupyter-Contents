"""Extremely simple server process which makes calls and returns results.

We use this to separate out calls which use TileDB core libraries from
the process which runs the rest of JupyterLab.
"""

import argparse
import os
import socketserver
import threading
import time
from typing import Generic, NoReturn, TypeVar

import attrs
import cloudpickle

_T_co = TypeVar("_T_co", covariant=True)


@attrs.define(frozen=True, slots=True)
class Success(Generic[_T_co]):
    """Wrapper for a successful return."""

    value: _T_co

    def result(self) -> _T_co:
        return self.value


@attrs.define(frozen=True, slots=True)
class Failure:
    """Wrapper for a failed return."""

    exception: Exception

    def result(self) -> NoReturn:
        raise self.exception


class Executor(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        result: object
        try:
            req = cloudpickle.load(self.rfile)
        except Exception as e:
            result = Failure(e)
        else:
            if req is None:
                # This is the last request we should handle.
                self.server.shutdown()
                return
            try:
                result = Success(req())
            except Exception as e:
                result = Failure(e)
        cloudpickle.dump(result, self.wfile)


def parent_watchdog(ppid: int, server: socketserver.BaseServer) -> None:
    """Watchdog function to reap this process if the parent exits."""
    while True:
        # When we're started, we are created as a child process of some parent,
        # which has been passed in as the `--parent-pid` argument.
        # If that parent process shuts down, we are adopted by some other
        # process (usually PID 1), and our parent PID changes. Since that means
        # no more requests, we should shut down.
        if os.getppid() != ppid:
            server.shutdown()
            return
        time.sleep(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parent-pid",
        type=int,
        default=None,
        help="""
            PID of the parent process starting this program.
            Used to automatically end this process if the parent exits.
        """,
    )
    parser.add_argument("socket", help="Unix socket path to listen on.")

    parsed = parser.parse_args()

    srv = socketserver.ThreadingUnixStreamServer(parsed.socket, Executor)
    if parsed.parent_pid:
        threading.Thread(
            target=parent_watchdog,
            args=(parsed.parent_pid, srv),
            daemon=True,
            name="parent-watchdog",
        ).start()
    srv.serve_forever()


if __name__ == "__main__":
    main()
