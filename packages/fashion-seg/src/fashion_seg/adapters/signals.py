"""Process signals as a ``fashion_seg.ports.StopSignal``: Ctrl+C or SIGTERM stop the training.

The training then saves a checkpoint at the end of its current step and returns, instead of
being killed mid-step.
"""

import logging
import signal
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from types import FrameType

logger = logging.getLogger(__name__)


@contextmanager
def stop_on_signal() -> Iterator[threading.Event]:
    """Turn SIGINT and SIGTERM into a stop request for the duration of a ``with`` block.

    Yields
    ------
    threading.Event
        Set when a stop is requested (a ``StopSignal``).
    """
    requested = threading.Event()

    # pylint: disable-next=unused-argument  # the signal handler signature
    def request_stop(signum: int, frame: FrameType | None) -> None:
        """Record the stop request.

        Parameters
        ----------
        signum : int
            The signal.
        frame : FrameType | None
            The interrupted frame.
        """
        requested.set()
        logger.warning("Stop requested (signal %d): saving at the end of the current step", signum)

    previous = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield requested
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
