"""Opt-in, two-process CUDA -> pinned host -> RDMA -> host -> CUDA test.

Set MOONCAKE_STAGING_RDMA_TEST=1, MOONCAKE_TEST_DEVICE and
MOONCAKE_TEST_HOST. MOONCAKE_TEST_ENGINE can name a freshly built engine.so.
"""

import importlib.util
import logging
import multiprocessing
import os

import pytest
import torch

from mooncake.cpu_staging import CPUStagingMirror


def make_engine():
    logging.basicConfig(level=logging.INFO)
    # Set before the first C++ Environ singleton access, including in children.
    os.environ["MOONCAKE_RDMA_HOST_ONLY"] = "1"
    os.environ["WITH_NVIDIA_PEERMEM"] = "0"
    os.environ["MC_FORCE_HCA"] = "1"
    os.environ["MC_INTRA_NVLINK"] = "0"
    path = os.getenv("MOONCAKE_TEST_ENGINE")
    if path:
        spec = importlib.util.spec_from_file_location("engine", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        engine = module.TransferEngine()
    else:
        from mooncake.engine import TransferEngine

        engine = TransferEngine()
    host = os.environ["MOONCAKE_TEST_HOST"]
    assert (
        engine.initialize(
            host, "P2PHANDSHAKE", "rdma", os.environ["MOONCAKE_TEST_DEVICE"]
        )
        == 0
    )
    return engine, f"{host}:{engine.get_rpc_port()}"


def receiver(pipe, elements):
    mirror = None
    try:
        engine, session = make_engine()
        cache = torch.zeros((8, elements), dtype=torch.uint8, device="cuda")
        mirror = CPUStagingMirror(engine, [cache], protocol="rdma")
        pipe.send((session, mirror.addresses))
        for _ in range(3):
            assert pipe.poll(60), "Timed out waiting for RDMA completion"
            value = pipe.recv()
            mirror.load_blocks([1, 3, 4])
            expected = torch.zeros_like(cache)
            expected[1].fill_(value)
            expected[3:5].fill_(value)
            assert torch.equal(cache, expected), "RDMA/H2D payload mismatch"
            pipe.send("verified")
        mirror.close()
        mirror = None
        pipe.send("closed")
    except Exception as exc:
        pipe.send(("error", repr(exc)))
        raise
    finally:
        if mirror is not None:
            mirror.close()
        pipe.close()


@pytest.mark.skipif(
    os.getenv("MOONCAKE_STAGING_RDMA_TEST") != "1",
    reason="Set MOONCAKE_STAGING_RDMA_TEST=1 for hardware test",
)
@pytest.mark.parametrize("elements", [4096, 1024 * 1024, 64 * 1024 * 1024])
def test_two_process_rdma(elements):
    ctx = multiprocessing.get_context("spawn")
    parent, child = ctx.Pipe()
    process = ctx.Process(target=receiver, args=(child, elements))
    process.start()
    child.close()
    mirror = None
    try:
        assert parent.poll(60), "Receiver initialization timed out"
        session, remote_addresses = parent.recv()
        assert session != "error", remote_addresses
        engine, _ = make_engine()
        cache = torch.empty((8, elements), dtype=torch.uint8, device="cuda")
        assert engine.register_memory(cache.data_ptr(), cache.nbytes) != 0
        mirror = CPUStagingMirror(engine, [cache], protocol="rdma")
        for value in [17, 93, 201]:
            # Use a non-default compute stream to exercise the D2H dependency.
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                cache.fill_(value)
            assert (
                mirror.write(
                    session,
                    [mirror.addresses[0] + i * elements for i in [1, 3]],
                    [remote_addresses[0] + i * elements for i in [1, 3]],
                    [elements, 2 * elements],
                )
                == 0
            )
            parent.send(value)
            assert parent.poll(60), "Receiver verification timed out"
            assert parent.recv() == "verified"
        assert parent.poll(60)
        assert parent.recv() == "closed"
        process.join(30)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join(10)
        if mirror is not None:
            mirror.close()
        parent.close()
