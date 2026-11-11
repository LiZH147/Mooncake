"""Pinned CPU mirrors with bounded batch pipelines for vLLM KV transfers."""

import logging
import os
import threading
import time

import torch

from mooncake.staging_pipeline import StagingPipeline, capture_events, producer_events

logger = logging.getLogger(__name__)

RANGE_LOCK_BYTES = 64 * 1024
DEFAULT_MAX_INFLIGHT = 4


def _trace_enabled() -> bool:
    """Return whether per-transfer CPU staging tracing is enabled."""
    return os.getenv("MOONCAKE_STAGING_TRACE", "").lower() in {"1", "true", "yes", "on"}


class CPUStagingMirror:
    """Keep registered CPU buffers alive until all local copies/transfers stop.

    Block indices and byte offsets match the contiguous GPU cache regions.
    Only the mirror addresses are advertised to peers. vLLM must retain the
    corresponding blocks until the connector reports completion.
    """

    def __init__(
        self,
        engine,
        caches: list[torch.Tensor],
        *,
        trace_logger=None,
        protocol="unspecified",
    ):
        if not caches or any(
            not cache.is_cuda or not cache.is_contiguous() or cache.numel() == 0
            for cache in caches
        ):
            raise ValueError("CPU staging requires nonempty contiguous CUDA caches")
        self.engine = engine
        self.trace_enabled = _trace_enabled()
        self._logger = trace_logger if trace_logger is not None else logger
        self._protocol = protocol
        self.caches = caches
        self.buffers: list[torch.Tensor] = []
        self._registered: list[int] = []
        self._state_lock = threading.RLock()
        self._active_cv = threading.Condition(self._state_lock)
        self._active_operations = 0
        self._range_lock_guard = threading.Lock()
        self._range_locks = {}
        self._max_inflight = max(
            1, int(os.getenv("MOONCAKE_STAGING_MAX_INFLIGHT", DEFAULT_MAX_INFLIGHT))
        )
        self._pipeline = StagingPipeline(self._max_inflight)
        self._closing = False
        self._closed = False
        self._buffer_ids: dict[int, int] = {}
        try:
            for cache in caches:
                buffer = torch.empty(
                    cache.shape, dtype=cache.dtype, device="cpu", pin_memory=True
                )
                self.buffers.append(buffer)
                self._buffer_ids[id(buffer)] = len(self.buffers) - 1
                # Wildcard location resolves the pinned host allocation's NUMA
                # node instead of assuming every worker is on cpu:0.
                ret = engine.register_memory(buffer.data_ptr(), buffer.nbytes, "*")
                if ret != 0:
                    raise RuntimeError(f"CPU staging memory registration failed: {ret}")
                self._registered.append(buffer.data_ptr())
            if self.trace_enabled:
                self._logger.info(
                    "[CPU-STAGING] registered buffers=%d bytes=%d pinned=true protocol=%s",
                    len(self.buffers),
                    sum(buffer.nbytes for buffer in self.buffers),
                    self._protocol,
                )
        except Exception:
            # Report the original registration failure, not a secondary
            # cleanup error; the cleanup error is logged instead.
            try:
                self.close()
            except Exception:
                self._logger.exception("CPU staging cleanup after failure also failed")
            raise

    @property
    def addresses(self) -> list[int]:
        return [buffer.data_ptr() for buffer in self.buffers]

    def _check_open(self):
        if self._closed or self._closing:
            raise RuntimeError("CPU staging is closed")

    def _begin_operation(self):
        with self._state_lock:
            self._check_open()
            self._active_operations += 1

    def _end_operation(self):
        with self._state_lock:
            self._active_operations -= 1
            self._active_cv.notify_all()

    def _acquire_ranges(self, ranges):
        keys = set()
        for _, buffer, offset, count in ranges:
            byte_offset = offset * buffer.element_size()
            byte_size = count * buffer.element_size()
            first = byte_offset // RANGE_LOCK_BYTES
            last = (byte_offset + byte_size - 1) // RANGE_LOCK_BYTES
            buffer_id = self._buffer_ids[id(buffer)]
            keys.update((buffer_id, stripe) for stripe in range(first, last + 1))
        with self._range_lock_guard:
            locks = [
                self._range_locks.setdefault(key, threading.Lock())
                for key in sorted(keys)
            ]
        for lock in locks:
            lock.acquire()
        return locks

    @staticmethod
    def _release_ranges(locks):
        for lock in reversed(locks):
            lock.release()

    def _ranges(self, addresses: list[int], lengths: list[int]):
        if len(addresses) != len(lengths):
            raise ValueError("Transfer address and length counts differ")
        ranges = []
        for address, length in zip(addresses, lengths):
            for cache, buffer in zip(self.caches, self.buffers):
                offset = address - buffer.data_ptr()
                if 0 <= offset < buffer.nbytes and 0 < length <= buffer.nbytes - offset:
                    size = cache.element_size()
                    if offset % size or length % size:
                        raise ValueError("Transfer range is not aligned to cache dtype")
                    ranges.append((cache, buffer, offset // size, length // size))
                    break
            else:
                raise ValueError("Transfer range lies outside CPU staging buffers")
        return ranges

    def write(
        self, remote_session, addresses, remote_addresses, lengths, *, ready_events=None
    ) -> int:
        """Drain every batch before the caller sends TRANS_DONE or frees blocks."""
        self._begin_operation()
        try:
            if len(remote_addresses) != len(addresses):
                raise ValueError("Local and remote transfer address counts differ")
            ranges = self._ranges(addresses, lengths)  # Validate before any I/O.
            events = producer_events(
                {cache.device for cache, _, _, _ in ranges}, ready_events
            )
            batches, batch, size = [], [], 0
            limit = self._pipeline.batch_bytes
            for source, target, length, row in zip(
                addresses, remote_addresses, lengths, ranges
            ):
                alignment = row[0].element_size()
                if limit < alignment:
                    raise ValueError("Staging batch size is smaller than cache dtype")
                offset = 0
                while offset < length:
                    count = min(length - offset, limit - size)
                    count -= count % alignment
                    if count == 0:
                        batches.append(batch)
                        batch, size = [], 0
                        continue
                    batch.append((source + offset, target + offset, count))
                    offset += count
                    size += count
                    if size == limit:
                        batches.append(batch)
                        batch, size = [], 0
            if batch:
                batches.append(batch)

            def transfer(batch):
                sources, targets, sizes = map(list, zip(*batch))
                rows = self._ranges(sources, sizes)
                queued = time.perf_counter()
                locks = self._acquire_ranges(rows)
                try:
                    started = time.perf_counter()
                    copies = self._pipeline.copy(
                        rows, to_host=True, ready_events=events
                    )
                    copied = time.perf_counter()
                    result = self.engine.batch_transfer_sync_write(
                        remote_session, sources, targets, sizes
                    )
                    if self.trace_enabled:
                        self._logger.info(
                            "[CPU-STAGING] send session=%s protocol=%s bytes=%d ranges=%d "
                            "d2h_ms=%.3f transfer_ms=%.3f result=%d "
                            "range_wait_ms=%.3f copies=%d inflight_cap=%d",
                            remote_session,
                            self._protocol,
                            sum(sizes),
                            len(rows),
                            (copied - started) * 1000,
                            (time.perf_counter() - copied) * 1000,
                            result,
                            (started - queued) * 1000,
                            copies,
                            self._max_inflight,
                        )
                    return result
                finally:
                    self._release_ranges(locks)

            results = self._pipeline.run(transfer, batches)
            return next((result for result in results if result != 0), 0)
        finally:
            self._end_operation()

    def load_blocks(self, block_ids: list[int], *, ready_events=None):
        """Copy acknowledged blocks on background copy streams, then complete."""
        self._begin_operation()
        try:
            blocks = sorted(set(block_ids))
            if not blocks:
                return
            if blocks[0] < 0 or any(blocks[-1] >= c.shape[0] for c in self.caches):
                raise ValueError("Received block index lies outside KV cache")
            if ready_events is None:
                ready_events = capture_events({cache.device for cache in self.caches})
            # Coalesce contiguous block IDs, without gathering into temporary
            # tensors or copying the unrequested gaps between blocks.
            runs = []
            start = previous = blocks[0]
            for block in blocks[1:]:
                if block != previous + 1:
                    runs.append((start, previous + 1))
                    start = block
                previous = block
            runs.append((start, previous + 1))
            rows = []
            for cache, buffer in zip(self.caches, self.buffers):
                elements = cache[0].numel()
                limit = max(1, self._pipeline.batch_bytes // cache.element_size())
                for first, end in runs:
                    offset, stop = first * elements, end * elements
                    while offset < stop:
                        count = min(limit, stop - offset)
                        rows.append((cache, buffer, offset, count))
                        offset += count

            # A sparse request can produce thousands of small ranges. Submit
            # bounded groups so each worker waits for one CUDA event per batch,
            # instead of creating a task and synchronizing for every range.
            batches, batch, batch_bytes = [], [], 0
            for row in rows:
                row_bytes = row[3] * row[0].element_size()
                if batch and batch_bytes + row_bytes > self._pipeline.batch_bytes:
                    batches.append(batch)
                    batch, batch_bytes = [], 0
                batch.append(row)
                batch_bytes += row_bytes
            if batch:
                batches.append(batch)

            def copy(batch):
                locks = self._acquire_ranges(batch)
                try:
                    return self._pipeline.copy(
                        batch, to_host=False, ready_events=ready_events
                    )
                finally:
                    self._release_ranges(locks)

            started = time.perf_counter()
            copies = sum(self._pipeline.run(copy, batches))
            if self.trace_enabled:
                self._logger.info(
                    "[CPU-STAGING] recv blocks=%d bytes=%d h2d_ms=%.3f "
                    "inflight_cap=%d copies=%d batches=%d",
                    len(blocks),
                    len(blocks) * sum(c[0].nbytes for c in self.caches),
                    (time.perf_counter() - started) * 1000,
                    self._max_inflight,
                    copies,
                    len(batches),
                )
        finally:
            self._end_operation()

    def close(self):
        with self._state_lock:
            self._closing = True
            while self._active_operations:
                self._active_cv.wait()
            self._pipeline.close()
            # Attempt every unregistration so one failure does not leave the
            # remaining buffers registered. Buffers that failed to unregister
            # stay alive, and a later close() retries them.
            failure = None
            for address in self._registered[:]:
                ret = self.engine.unregister_memory(address)
                if ret != 0:
                    if failure is None:
                        failure = RuntimeError(
                            f"CPU staging memory unregistration failed: {ret}"
                        )
                    continue
                self._registered.remove(address)
            if failure is not None:
                raise failure
            self._closed = True
            self.buffers.clear()
            self._buffer_ids.clear()
            self._range_locks.clear()
