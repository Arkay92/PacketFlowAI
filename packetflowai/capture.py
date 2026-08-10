"""Lifecycle-managed live capture and PCAP replay services."""

import logging
from collections.abc import Callable
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Thread
from typing import Any

from .config import RuntimeConfig
from .inference import PacketInferenceService

PacketHandler = Callable[[Any], None]


def available_interfaces() -> list[str]:
    try:
        from scapy.interfaces import get_if_list
    except ImportError as error:
        raise RuntimeError("listing interfaces requires Scapy") from error
    return list(get_if_list())


class CaptureService:
    def __init__(self, interface: str, pipeline: PacketInferenceService,
                 config: RuntimeConfig, handler: PacketHandler | None = None):
        self.interface = interface
        self.pipeline = pipeline
        self.config = config
        self.handler = handler or self._default_handler
        self.stop_event = Event()
        self.queue: Queue[Any] = Queue(maxsize=config.queue_size)
        self.capture_thread: Thread | None = None
        self.processing_thread: Thread | None = None
        self.dropped_packets = 0
        self.last_error: BaseException | None = None

    def _default_handler(self, packet: Any) -> None:
        assessment, decisions = self.pipeline.process(packet)
        logging.info(
            "Packet %s classified as %s confidence %.4f risk %.2f",
            assessment.event_id,
            assessment.local_prediction.label,
            assessment.local_prediction.confidence,
            assessment.risk_score,
        )
        for decision in decisions:
            logging.warning("Response action %s for packet %s", decision.action, assessment.event_id)

    def _enqueue(self, packet: Any) -> None:
        try:
            self.queue.put_nowait(packet)
        except Full:
            self.dropped_packets += 1

    def _capture_loop(self) -> None:
        try:
            from scapy.sendrecv import sniff
            while not self.stop_event.is_set():
                sniff(
                    iface=self.interface,
                    prn=self._enqueue,
                    store=False,
                    timeout=self.config.capture_poll_seconds,
                )
        except BaseException as error:
            self.last_error = error
            self.stop_event.set()

    def _processing_loop(self) -> None:
        while not self.stop_event.is_set() or not self.queue.empty():
            try:
                packet = self.queue.get(timeout=0.2)
            except Empty:
                continue
            try:
                self.handler(packet)
            except Exception:
                logging.exception("Packet processing failed")
            finally:
                self.queue.task_done()

    def start(self) -> None:
        if self.capture_thread and self.capture_thread.is_alive():
            raise RuntimeError("capture service is already running")
        self.stop_event.clear()
        self.capture_thread = Thread(target=self._capture_loop, name="packet-capture", daemon=True)
        self.processing_thread = Thread(target=self._processing_loop, name="packet-processing", daemon=True)
        self.capture_thread.start()
        self.processing_thread.start()

    def stop(self) -> None:
        self.stop_event.set()

    def join(self, timeout: float | None = None) -> None:
        if self.capture_thread:
            self.capture_thread.join(timeout)
        if self.processing_thread:
            self.processing_thread.join(timeout)
        if self.last_error:
            raise RuntimeError("capture service failed") from self.last_error


class ReplayService:
    def __init__(self, pipeline: PacketInferenceService, handler: PacketHandler | None = None):
        self.pipeline = pipeline
        self.handler = handler or self._default_handler

    def _default_handler(self, packet: Any) -> None:
        assessment, decisions = self.pipeline.process(packet)
        logging.info(
            "Replay packet %s label=%s confidence=%.4f actions=%s",
            assessment.event_id,
            assessment.local_prediction.label,
            assessment.local_prediction.confidence,
            [decision.action for decision in decisions],
        )

    def run(self, path: Path, limit: int | None = None) -> int:
        try:
            from scapy.utils import PcapReader
        except ImportError as error:
            raise RuntimeError("PCAP replay requires Scapy") from error
        processed = 0
        with PcapReader(str(path)) as packets:
            for packet in packets:
                try:
                    self.handler(packet)
                    processed += 1
                except ValueError as error:
                    logging.debug("Skipping unsupported replay packet: %s", error)
                if limit is not None and processed >= limit:
                    break
        return processed
