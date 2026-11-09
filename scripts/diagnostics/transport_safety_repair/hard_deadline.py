"""Hard wall-clock deadlines for isolated, single-attempt transports.

The callable runs in a forked worker so the parent can terminate a stuck
request. Fork is intentional here: callers may be closures around an SDK
client and therefore are not generally pickleable. This module fails closed
on platforms without fork; it never falls back to an unbounded thread.
"""
from __future__ import annotations

import multiprocessing as mp
from multiprocessing.connection import wait
import time
from typing import Any, Callable


class HardDeadlineExceeded(TimeoutError):
    """The transport worker exceeded its wall-clock allowance and was killed."""


class HardDeadlineUnavailable(RuntimeError):
    """A safely terminable worker could not be started on this platform."""


class RemoteTransportError(RuntimeError):
    """The isolated transport raised; details are intentionally not propagated."""


def _worker(send_conn: Any, transport: Callable[[], Any]) -> None:
    try:
        try:
            value = transport()
        except BaseException as exc:
            # Do not serialize arbitrary exception text: API/client exceptions
            # can contain request metadata or other sensitive values.
            send_conn.send(("error", type(exc).__name__))
        else:
            send_conn.send(("ok", value))
    finally:
        send_conn.close()


def call_with_hard_deadline(transport: Callable[[], Any], timeout_seconds: float,
                            *, terminate_grace_seconds: float = 0.25) -> Any:
    """Run exactly one callable and kill its worker at a hard elapsed limit.

    The deadline starts before worker creation. A timed-out call may already
    have reached the remote service, so callers must count it as uncertain
    usage and must not retry it.
    """
    if not callable(transport):
        raise TypeError("transport must be callable")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if terminate_grace_seconds < 0:
        raise ValueError("terminate_grace_seconds cannot be negative")
    if "fork" not in mp.get_all_start_methods():
        raise HardDeadlineUnavailable("fork worker support is unavailable; refusing an unbounded call")

    context = mp.get_context("fork")
    receive_conn, send_conn = context.Pipe(duplex=False)
    process = context.Process(target=_worker, args=(send_conn, transport), daemon=True)
    deadline = time.monotonic() + float(timeout_seconds)
    started = False
    try:
        process.start()
        started = True
        send_conn.close()
        remaining = max(0.0, deadline - time.monotonic())
        ready = wait([receive_conn, process.sentinel], timeout=remaining)

        # Prefer a completed response if both the pipe and process exit are
        # ready at the same time.
        if receive_conn in ready or receive_conn.poll(0):
            try:
                status, payload = receive_conn.recv()
            except EOFError as exc:
                raise RemoteTransportError("worker exited without a result") from exc
            process.join(timeout=terminate_grace_seconds)
            if process.is_alive():
                process.terminate()
                process.join(timeout=terminate_grace_seconds)
            if status == "error":
                raise RemoteTransportError(f"transport worker raised {payload}")
            if status != "ok":
                raise RemoteTransportError("transport worker returned an invalid status")
            return payload

        if process.sentinel in ready:
            process.join()
            if receive_conn.poll(0):
                try:
                    status, payload = receive_conn.recv()
                except EOFError as exc:
                    raise RemoteTransportError("worker exited without a result") from exc
                if status == "ok":
                    return payload
                raise RemoteTransportError(f"transport worker raised {payload}")
            raise RemoteTransportError(f"transport worker exited with code {process.exitcode}")

        process.terminate()
        process.join(timeout=terminate_grace_seconds)
        if process.is_alive():
            process.kill()
            process.join()
        raise HardDeadlineExceeded(
            f"transport exceeded hard deadline of {float(timeout_seconds):g} seconds; worker terminated"
        )
    except BaseException:
        if started and process.is_alive():
            process.terminate()
            process.join(timeout=terminate_grace_seconds)
            if process.is_alive():
                process.kill()
                process.join()
        raise
    finally:
        receive_conn.close()
        if not started:
            send_conn.close()

