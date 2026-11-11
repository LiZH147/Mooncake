"""Concurrency, drain-on-error and CUDA readiness tests for staging."""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from mooncake.staging_pipeline import StagingPipeline, capture_events


def test_window_is_bounded_and_results_keep_input_order():
    pipeline = StagingPipeline(2)
    barrier = threading.Barrier(2)
    active = peak = 0
    lock = threading.Lock()

    def task(index):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        barrier.wait(timeout=5)
        with lock:
            active -= 1
        return index * 2

    try:
        assert pipeline.run(task, range(8)) == list(range(0, 16, 2))
        assert peak == 2
    finally:
        pipeline.close()


def test_error_drains_other_inflight_batches():
    pipeline = StagingPipeline(2)
    entered, release, failed = (threading.Event() for _ in range(3))

    def task(index):
        if index == 0:
            assert entered.wait(5)
            failed.set()
            raise RuntimeError("native transfer failed")
        entered.set()
        assert release.wait(5)

    try:
        with ThreadPoolExecutor(max_workers=1) as caller:
            future = caller.submit(pipeline.run, task, [0, 1])
            try:
                assert failed.wait(5)
                assert not future.done(), "must retain buffers while batch 1 is active"
            finally:
                release.set()
            with pytest.raises(RuntimeError, match="native transfer failed"):
                future.result(timeout=5)
    finally:
        pipeline.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_copy_fences_producer_stream_and_coalesces_without_device_sync(monkeypatch):
    pipeline = StagingPipeline(2)
    gpu = torch.zeros(1024, device="cuda", dtype=torch.uint8)
    host = torch.full((1024,), 99, pin_memory=True, dtype=torch.uint8)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        torch.cuda._sleep(1_000_000)
        gpu.fill_(37)
        events = capture_events({gpu.device})

    def forbidden(*args, **kwargs):
        raise AssertionError("device-wide synchronization is forbidden")

    monkeypatch.setattr(torch.cuda, "synchronize", forbidden)
    try:
        rows = [(gpu, host, 0, 256), (gpu, host, 256, 256), (gpu, host, 768, 256)]

        def copy(batch):
            return pipeline.copy(batch, to_host=True, ready_events=events)

        assert pipeline.run(copy, [rows]) == [2]
        assert torch.all(host[:512] == 37)
        assert torch.all(host[512:768] == 99)
        assert torch.all(host[768:] == 37)
    finally:
        pipeline.close()
