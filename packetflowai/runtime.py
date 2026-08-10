"""Pluggable capture backends and bounded flow-oriented runtime."""

import os
import time
import tracemalloc
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Thread
from typing import Any, Protocol

import numpy as np

from .domain import FlowFeatures
from .flows import FlowEngine
from .protocols import observation_from_scapy

PacketCallback = Callable[[Any], None]
FlowCallback = Callable[[FlowFeatures], Any]


class CaptureBackend(Protocol):
    def run(self, callback: PacketCallback, stop_event: Event) -> None: ...


class ScapyCaptureBackend:
    def __init__(self, interface: str, capture_filter: str | None = None, poll_seconds: float = 1.0):
        self.interface = interface
        self.capture_filter = capture_filter
        self.poll_seconds = poll_seconds

    def run(self, callback: PacketCallback, stop_event: Event) -> None:
        from scapy.sendrecv import sniff
        while not stop_event.is_set():
            sniff(
                iface=self.interface,
                filter=self.capture_filter,
                prn=callback,
                store=False,
                timeout=self.poll_seconds,
            )


class PcapReplayBackend:
    def __init__(self, path: Path, realtime: bool = False, speed: float = 1.0,
                 limit: int | None = None):
        if speed <= 0:
            raise ValueError("replay speed must be positive")
        self.path = path
        self.realtime = realtime
        self.speed = speed
        self.limit = limit

    def run(self, callback: PacketCallback, stop_event: Event) -> None:
        from scapy.utils import PcapReader
        previous_timestamp: float | None = None
        with PcapReader(str(self.path)) as packets:
            for index, packet in enumerate(packets):
                if stop_event.is_set() or (self.limit is not None and index >= self.limit):
                    break
                timestamp = float(getattr(packet, "time", 0.0))
                if self.realtime and previous_timestamp is not None:
                    delay = max(0.0, timestamp - previous_timestamp) / self.speed
                    stop_event.wait(delay)
                callback(packet)
                previous_timestamp = timestamp


@dataclass(frozen=True)
class RuntimeSnapshot:
    packets: int
    flows: int
    dropped_packets: int
    queue_depth: int
    packets_per_second: float
    flows_per_second: float
    feature_latency_p50_ms: float
    feature_latency_p95_ms: float
    feature_latency_p99_ms: float
    inference_latency_p50_ms: float
    inference_latency_p95_ms: float
    inference_latency_p99_ms: float
    process_cpu_seconds: float
    traced_memory_bytes: int


class RuntimeMetrics:
    def __init__(self, latency_samples: int = 10_000):
        self.started_at = time.monotonic()
        self.packets = 0
        self.flows = 0
        self.dropped_packets = 0
        self.feature_latencies: deque[float] = deque(maxlen=latency_samples)
        self.inference_latencies: deque[float] = deque(maxlen=latency_samples)
        if not tracemalloc.is_tracing():
            tracemalloc.start()

    @staticmethod
    def _percentiles(samples: Iterable[float]) -> tuple[float, float, float]:
        values = list(samples)
        if not values:
            return 0.0, 0.0, 0.0
        result = np.percentile(values, [50, 95, 99])
        return float(result[0]), float(result[1]), float(result[2])

    def snapshot(self, queue_depth: int = 0) -> RuntimeSnapshot:
        elapsed = max(time.monotonic() - self.started_at, 1e-9)
        feature = self._percentiles(self.feature_latencies)
        inference = self._percentiles(self.inference_latencies)
        current_memory, _ = tracemalloc.get_traced_memory()
        return RuntimeSnapshot(
            packets=self.packets,
            flows=self.flows,
            dropped_packets=self.dropped_packets,
            queue_depth=queue_depth,
            packets_per_second=self.packets / elapsed,
            flows_per_second=self.flows / elapsed,
            feature_latency_p50_ms=feature[0],
            feature_latency_p95_ms=feature[1],
            feature_latency_p99_ms=feature[2],
            inference_latency_p50_ms=inference[0],
            inference_latency_p95_ms=inference[1],
            inference_latency_p99_ms=inference[2],
            process_cpu_seconds=sum(os.times()[:2]),
            traced_memory_bytes=current_memory,
        )


class FlowRuntime:
    """The same flow pipeline is used by live and replay backends."""

    def __init__(self, engine: FlowEngine, flow_handler: FlowCallback,
                 include_payload_metadata: bool = False):
        self.engine = engine
        self.flow_handler = flow_handler
        self.include_payload_metadata = include_payload_metadata
        self.metrics = RuntimeMetrics()

    def process_packet(self, packet: Any) -> None:
        started = time.perf_counter()
        observation = observation_from_scapy(packet, self.include_payload_metadata)
        state = self.engine.update(observation)
        self.metrics.packets += 1
        self.metrics.feature_latencies.append((time.perf_counter() - started) * 1000)
        if state.state in {"RESET", "CLOSED"}:
            flow = self.engine.close(state.key, "CLOSED")
            if flow:
                self._emit(flow)
        for flow in self.engine.expire(observation.timestamp):
            self._emit(flow)

    def _emit(self, flow: FlowFeatures) -> None:
        started = time.perf_counter()
        self.flow_handler(flow)
        self.metrics.inference_latencies.append((time.perf_counter() - started) * 1000)
        self.metrics.flows += 1

    def flush(self) -> None:
        for state in self.engine.active():
            flow = self.engine.close(state.key, "FLUSHED")
            if flow:
                self._emit(flow)


class RuntimeService:
    def __init__(self, backend: CaptureBackend, pipeline: FlowRuntime, queue_size: int = 1_000,
                 overflow_policy: str = "drop_newest", batch_size: int = 32):
        if overflow_policy not in {"drop_newest", "drop_oldest", "block"}:
            raise ValueError("unsupported overflow policy")
        self.backend = backend
        self.pipeline = pipeline
        self.overflow_policy = overflow_policy
        self.batch_size = batch_size
        self.queue: Queue[Any] = Queue(maxsize=queue_size)
        self.stop_event = Event()
        self.backend_thread: Thread | None = None
        self.worker_thread: Thread | None = None
        self.error: BaseException | None = None

    def _enqueue(self, packet: Any) -> None:
        if self.overflow_policy == "block":
            self.queue.put(packet)
            return
        try:
            self.queue.put_nowait(packet)
        except Full:
            self.pipeline.metrics.dropped_packets += 1
            if self.overflow_policy == "drop_oldest":
                try:
                    self.queue.get_nowait()
                    self.queue.task_done()
                    self.queue.put_nowait(packet)
                except (Empty, Full):
                    pass

    def _run_backend(self) -> None:
        try:
            self.backend.run(self._enqueue, self.stop_event)
        except BaseException as error:
            self.error = error
        finally:
            self.stop_event.set()

    def _run_worker(self) -> None:
        while not self.stop_event.is_set() or not self.queue.empty():
            batch = []
            try:
                batch.append(self.queue.get(timeout=0.2))
            except Empty:
                continue
            while len(batch) < self.batch_size:
                try:
                    batch.append(self.queue.get_nowait())
                except Empty:
                    break
            for packet in batch:
                try:
                    self.pipeline.process_packet(packet)
                except ValueError:
                    pass
                finally:
                    self.queue.task_done()
        self.pipeline.flush()

    def start(self) -> None:
        self.stop_event.clear()
        self.backend_thread = Thread(target=self._run_backend, name="capture-backend", daemon=True)
        self.worker_thread = Thread(target=self._run_worker, name="flow-worker", daemon=True)
        self.backend_thread.start()
        self.worker_thread.start()

    def stop(self, drain: bool = True) -> None:
        self.stop_event.set()
        if drain:
            self.queue.join()

    def join(self, timeout: float | None = None) -> None:
        if self.backend_thread:
            self.backend_thread.join(timeout)
        if self.worker_thread:
            self.worker_thread.join(timeout)
        if self.error:
            raise RuntimeError("capture backend failed") from self.error

    def metrics(self) -> dict[str, Any]:
        return asdict(self.pipeline.metrics.snapshot(self.queue.qsize()))
