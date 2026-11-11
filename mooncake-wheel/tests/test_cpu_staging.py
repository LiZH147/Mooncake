import logging
import ctypes
import threading
from unittest.mock import Mock

import pytest
import torch

from mooncake.cpu_staging import CPUStagingMirror, _trace_enabled
from mooncake.staging_pipeline import capture_events


def test_staging_trace_environment(monkeypatch):
    monkeypatch.delenv("MOONCAKE_STAGING_TRACE", raising=False)
    assert not _trace_enabled()
    for value in ("1", "true", "YES", "on"):
        monkeypatch.setenv("MOONCAKE_STAGING_TRACE", value)
        assert _trace_enabled()
    monkeypatch.setenv("MOONCAKE_STAGING_TRACE", "0")
    assert not _trace_enabled()


class FakeEngine:
    def __init__(self):
        self.registered = set()
        self.payloads = []

    def register_memory(self, address, capacity, location):
        assert location == "*"
        self.registered.add(address)
        return 0

    def unregister_memory(self, address):
        self.registered.remove(address)
        return 0

    def batch_transfer_sync_write(self, session, sources, targets, lengths):
        self.payloads.append((session, list(sources), list(targets), list(lengths)))
        return 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("trace", [False, True])
def test_cpu_staging_copies_selected_blocks_and_releases_memory(
    monkeypatch, caplog, trace
):
    monkeypatch.setenv("MOONCAKE_STAGING_TRACE", "1" if trace else "0")
    trace_logger = logging.getLogger("test_staging_trace")
    caplog.set_level(logging.INFO, logger=trace_logger.name)
    cache = torch.zeros((4, 8), dtype=torch.float32, device="cuda")
    cache[1].fill_(3)
    cache[3].fill_(7)
    engine = FakeEngine()
    mirror = CPUStagingMirror(
        engine, [cache], trace_logger=trace_logger, protocol="tcp"
    )
    block_bytes = cache[0].nbytes

    source = mirror.addresses[0] + block_bytes
    assert mirror.write("peer", [source], [1234], [block_bytes]) == 0
    assert torch.equal(mirror.buffers[0][1], cache[1].cpu())
    assert engine.payloads[-1][1] == [source]

    cache[1].zero_()
    mirror.buffers[0][1].fill_(11)
    mirror.load_blocks([1])
    assert torch.all(cache[1] == 11)

    if trace:
        assert "registered buffers=1 bytes=128 pinned=true protocol=tcp" in caplog.text
        assert "send session=peer protocol=tcp bytes=32 ranges=1" in caplog.text
        assert "d2h_ms=" in caplog.text
        assert "transfer_ms=" in caplog.text
        assert "result=0" in caplog.text
        assert "recv blocks=1 bytes=32 h2d_ms=" in caplog.text
        assert "rdma_ms" not in caplog.text
    else:
        assert "[CPU-STAGING]" not in caplog.text

    mirror.close()
    assert not engine.registered
    mirror.close()
    with pytest.raises(RuntimeError, match="closed"):
        mirror.load_blocks([0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_invalid_ranges_and_registration_rollback():
    cache = torch.zeros((4, 8), device="cuda")
    engine = FakeEngine()
    mirror = CPUStagingMirror(engine, [cache])
    for address, length in [
        (mirror.addresses[0] - 1, 4),
        (mirror.addresses[0], cache.nbytes + 1),
        (mirror.addresses[0] + 1, 4),
    ]:
        with pytest.raises(ValueError):
            mirror.write("peer", [address], [1234], [length])
    with pytest.raises(ValueError):
        mirror.load_blocks([4])
    assert not engine.payloads
    mirror.close()

    register = engine.register_memory

    def fail_second(address, capacity, location):
        return -1 if engine.registered else register(address, capacity, location)

    engine.register_memory = fail_second
    with pytest.raises(RuntimeError, match="registration failed"):
        CPUStagingMirror(engine, [cache, cache.clone()])
    assert not engine.registered


def test_reject_cpu_cache():
    with pytest.raises(ValueError, match="CUDA caches"):
        CPUStagingMirror(FakeEngine(), [torch.zeros(4)])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_sparse_h2d_ranges_use_bounded_batches(monkeypatch):
    monkeypatch.setenv("MOONCAKE_STAGING_BATCH_BYTES", "64")
    monkeypatch.setenv("MOONCAKE_STAGING_MAX_INFLIGHT", "2")
    cache = torch.zeros((8, 8), dtype=torch.float32, device="cuda")
    mirror = CPUStagingMirror(FakeEngine(), [cache])
    mirror.buffers[0].fill_(9)
    mirror._pipeline.copy = Mock(wraps=mirror._pipeline.copy)
    try:
        mirror.load_blocks([0, 2, 4, 6])
        assert mirror._pipeline.copy.call_count == 2
        assert torch.all(cache[[0, 2, 4, 6]] == 9)
        assert torch.all(cache[[1, 3, 5, 7]] == 0)
    finally:
        mirror.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_large_direct_range_splits_and_overlaps_without_early_completion(monkeypatch):
    monkeypatch.setenv("MOONCAKE_STAGING_MAX_INFLIGHT", "2")
    monkeypatch.setenv("MOONCAKE_STAGING_BATCH_BYTES", "65536")
    barrier = threading.Barrier(2)

    class CopyEngine(FakeEngine):
        def batch_transfer_sync_write(self, session, sources, targets, lengths):
            barrier.wait(timeout=10)
            for source, target, size in zip(sources, targets, lengths):
                ctypes.memmove(target, source, size)
            return 0

    cache = torch.full((4, 65536), 19, device="cuda", dtype=torch.uint8)
    target = torch.zeros_like(cache, device="cpu")
    mirror = CPUStagingMirror(CopyEngine(), [cache])
    events = capture_events({cache.device})

    def forbidden(*args, **kwargs):
        raise AssertionError("unexpected device synchronize")

    monkeypatch.setattr(torch.cuda, "synchronize", forbidden)
    try:
        assert (
            mirror.write(
                "peer",
                mirror.addresses,
                [target.data_ptr()],
                [target.nbytes],
                ready_events=events,
            )
            == 0
        )
        assert torch.all(target == 19)
        mirror.buffers[0].fill_(27)
        cache.zero_()
        mirror.load_blocks([0, 1, 3])
        assert torch.all(cache[:2] == 27)
        assert torch.all(cache[2] == 0)
        assert torch.all(cache[3] == 27)
    finally:
        mirror.close()


class _FlakyUnregisterEngine(FakeEngine):
    def __init__(self, failing_calls):
        super().__init__()
        self.failing_calls = set(failing_calls)
        self.unregister_calls = 0

    def unregister_memory(self, address):
        self.unregister_calls += 1
        if self.unregister_calls in self.failing_calls:
            return -1
        return super().unregister_memory(address)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_close_retries_failed_unregistration_without_leaking_others():
    caches = [torch.zeros((2, 8), dtype=torch.float32, device="cuda") for _ in range(2)]
    engine = _FlakyUnregisterEngine(failing_calls={1})
    mirror = CPUStagingMirror(engine, caches)
    with pytest.raises(RuntimeError, match="unregistration failed"):
        mirror.close()
    # The second buffer was still unregistered despite the first failing.
    assert len(engine.registered) == 1
    assert not mirror._closed
    mirror.close()  # Retry succeeds and releases the remaining registration.
    assert engine.registered == set()
    assert mirror._closed


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_registration_failure_reports_original_error_when_cleanup_fails():
    caches = [torch.zeros((2, 8), dtype=torch.float32, device="cuda") for _ in range(2)]

    class FailingRegisterEngine(_FlakyUnregisterEngine):
        def register_memory(self, address, capacity, location):
            if len(self.registered) == 1:
                return -2
            return super().register_memory(address, capacity, location)

    engine = FailingRegisterEngine(failing_calls={1})
    with pytest.raises(RuntimeError, match="registration failed: -2"):
        CPUStagingMirror(engine, caches)
