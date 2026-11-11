"""Bounded background transfers and event-fenced copies for CPU KV staging."""

import os
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import torch


def capture_events(devices):
    """Call on the thread/streams producing the KV, before handing it off."""
    events = []
    for device in devices:
        with torch.cuda.device(device):
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(device))
            events.append(event)
    return tuple(events)


def producer_events(devices, events):
    if events is not None:
        return tuple(events)
    # Compatibility for callers with no producer-stream contract (including
    # old vLLM patches). Never infer readiness from a background thread's stream.
    for device in devices:
        torch.cuda.synchronize(device)
    return ()


def coalesce_ranges(ranges):
    """Merge adjacent/overlapping copies within the same GPU/host mirror.

    Offsets and lengths are in tensor elements. Network descriptors and Store
    key boundaries are deliberately unchanged.
    """
    groups = {}
    for gpu, host, offset, count in ranges:
        key = (id(gpu), id(host))
        groups.setdefault(key, (gpu, host, []))[2].append((offset, offset + count))
    merged = []
    for gpu, host, intervals in groups.values():
        start, end = sorted(intervals)[0]
        for left, right in sorted(intervals)[1:]:
            if left <= end:
                end = max(end, right)
            else:
                merged.append((gpu, host, start, end - start))
                start, end = left, right
        merged.append((gpu, host, start, end - start))
    return merged


class StagingPipeline:
    """Run a bounded window of batches; drain every submitted task on error.

    Native Store/TransferEngine methods retain their synchronous contract, but
    run in this pool, never on a vLLM scheduler or communication event loop.
    Each pool thread owns copy streams; no device-wide synchronization is
    needed when callers supply the producer's readiness events.
    """

    def __init__(self, workers):
        self.workers = workers
        self.batch_bytes = int(os.getenv("MOONCAKE_STAGING_BATCH_BYTES", 32 << 20))
        if self.batch_bytes <= 0:
            raise ValueError("MOONCAKE_STAGING_BATCH_BYTES must be positive")
        self._local = threading.local()
        self._pool = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="mooncake-staging"
        )

    def run(self, function, batches):
        iterator = iter(enumerate(batches))
        pending, results = {}, {}
        try:
            while True:
                while len(pending) < self.workers:
                    item = next(iterator, None)
                    if item is None:
                        break
                    index, batch = item
                    pending[self._pool.submit(function, batch)] = index
                if not pending:
                    return [results[i] for i in range(len(results))]
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    index = pending.pop(future)
                    results[index] = future.result()
        finally:
            # An exception must not let the caller free/reuse GPU or host
            # blocks while another batch is still copying or doing RDMA.
            for future in pending:
                future.cancel()
            if pending:
                wait(pending)

    def copy(self, ranges, *, to_host, ready_events=()):
        ranges = coalesce_ranges(ranges)
        by_device = {}
        for row in ranges:
            by_device.setdefault(row[0].device, []).append(row)
        streams = getattr(self._local, "streams", None)
        if streams is None:
            streams = self._local.streams = {}
        for device, rows in by_device.items():
            with torch.cuda.device(device):
                if device not in streams:
                    streams[device] = torch.cuda.Stream(device=device)
                stream = streams[device]
                with torch.cuda.stream(stream):
                    try:
                        for event in ready_events:
                            stream.wait_event(event)
                        for gpu, host, offset, count in rows:
                            gpu_view = gpu.view(-1).narrow(0, offset, count)
                            host_view = host.view(-1).narrow(0, offset, count)
                            dst, src = (
                                (host_view, gpu_view)
                                if to_host
                                else (gpu_view, host_view)
                            )
                            dst.copy_(src, non_blocking=True)
                    finally:
                        # Wait only for this batch's copies, on a background
                        # thread. Also drain already-enqueued copies on error.
                        complete = torch.cuda.Event()
                        complete.record(stream)
                        complete.synchronize()
        return len(ranges)

    def close(self):
        self._pool.shutdown(wait=True)
