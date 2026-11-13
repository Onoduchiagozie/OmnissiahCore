"""Bounded bridge from blocking generation to an asynchronous HTTP stream."""
import asyncio
import logging
import threading
from concurrent.futures import TimeoutError

from Api.services.stream_contract import validated_sse_frames

logger = logging.getLogger(__name__)


async def stream_events(factory, buffer_size=16):
    """Close the upstream on failure/disconnect and stop queued work.

    A blocking provider read cannot be interrupted by Python; it remains
    governed by the provider timeout. No new tokens/work are requested after
    cancellation is observed.
    """
    loop = asyncio.get_running_loop()
    queue = asyncio.Queue(maxsize=buffer_size)
    stopped = threading.Event()

    def send(frame):
        if stopped.is_set():
            return False
        pending = asyncio.run_coroutine_threadsafe(queue.put(frame), loop)
        while not stopped.is_set():
            try:
                pending.result(timeout=0.1)
                return True
            except TimeoutError:
                continue
        pending.cancel()
        return False

    def produce():
        upstream = None
        validated = None
        try:
            if stopped.is_set():
                return
            upstream = factory(stopped)
            validated = validated_sse_frames(upstream)
            for frame in validated:
                if not send(frame):
                    break
        except Exception:
            logger.exception("Query stream failed")
            if not stopped.is_set():
                send("data: [ERROR] The service could not finish your request.\n\n")
        finally:
            for iterator in (validated, upstream):
                close = getattr(iterator, "close", None)
                if close:
                    try:
                        close()
                    except Exception:
                        logger.exception("Could not close query iterator")
            if not stopped.is_set():
                send(None)

    threading.Thread(target=produce, daemon=True, name="query-stream").start()
    try:
        while True:
            frame = await queue.get()
            if frame is None:
                break
            yield frame
    finally:
        stopped.set()
