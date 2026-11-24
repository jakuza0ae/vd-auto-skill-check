import mmap
from pathlib import Path
import struct
import time

import numpy as np

from vd.capture import DEFAULT_ROI
from vd.vulkan_capture import VulkanCapture


HEADER_FORMAT = "=IIIIIIQQQfIIIIIII64sIIIIIII4xQQQQQQ80s"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
MAGIC = 0x56444C59
VERSION = 3


def write_snapshot(path: Path, *, vk_format=44, capture_count=7, capture_ts_ns=None,
                   swapchain=(1600, 878), roi=(667, 341, 4, 3),
                   pixel=(51, 153, 26, 255), raw=None, epoch=None):
    capture_ts_ns = capture_ts_ns or time.monotonic_ns()
    x, y, w, h = roi
    stride = w * 4
    if raw is None:
        raw = bytes(pixel) * (w * h)
    assert len(raw) == stride * h
    values = (
        MAGIC, VERSION, HEADER_SIZE, 0b1111, 3, 2,
        epoch if epoch is not None else capture_ts_ns - 1_000_000, capture_ts_ns, capture_count, 60.0,
        0, swapchain[0], swapchain[1], vk_format, 1, 4, 0,
        b"Main\0" + b"\0" * 59,
        x, y, w, h, stride, HEADER_SIZE, len(raw),
        capture_count, capture_ts_ns, capture_count, 1, capture_ts_ns+100, capture_ts_ns+200, b"\0" * 80,
    )
    path.write_bytes(struct.pack(HEADER_FORMAT, *values) + raw)


def test_vulkan_capture_reads_bgra_and_normalizes_to_solver_frame(tmp_path):
    shm = tmp_path / "vd_layer_shm.dat"
    write_snapshot(shm)

    with VulkanCapture(shm_path=shm) as capture:
        frame = capture.next(timeout=0.05)

    assert frame is not None
    assert frame.sequence == 7
    assert frame.image.shape == (240, 320, 3)
    assert frame.image.dtype == np.uint8
    assert frame.image.flags.writeable is False
    assert tuple(frame.image[120, 160]) == (51, 153, 26)  # BGR from BGRA
    assert frame.source_size == (1600, 878)
    assert capture.roi == DEFAULT_ROI
    assert frame.timestamp_kind == "vulkan_monotonic"


def test_vulkan_capture_center_crops_and_pads_without_resizing(tmp_path):
    shm = tmp_path / "vd_layer_shm.dat"
    w = h = 256
    pixels = np.zeros((h, w, 4), dtype=np.uint8)
    pixels[:, :, 3] = 255
    pixels[:8, :, :3] = (0, 0, 255)
    pixels[8:248, :, :3] = (0, 255, 0)
    pixels[248:, :, :3] = (255, 0, 0)
    pixels[128, 128, :3] = (17, 33, 99)
    write_snapshot(shm, roi=(672, 311, w, h), raw=pixels.tobytes())

    with VulkanCapture(shm_path=shm) as capture:
        frame = capture.next(timeout=0.05)

    assert frame is not None
    assert frame.image.shape == (240, 320, 3)
    # 256x256 -> crop 8 rows from top/bottom, then pad 32 columns per side.
    assert tuple(frame.image[0, 160]) == (0, 255, 0)
    assert tuple(frame.image[-1, 160]) == (0, 255, 0)
    assert tuple(frame.image[120, 0]) == (0, 0, 0)
    assert tuple(frame.image[120, 31]) == (0, 0, 0)
    assert tuple(frame.image[120, 160]) == (17, 33, 99)


def test_vulkan_capture_preserves_raw_ui_scale(tmp_path):
    shm = tmp_path / "vd_layer_shm.dat"
    write_snapshot(shm, roi=(672, 311, 256, 256))

    with VulkanCapture(shm_path=shm) as capture:
        frame = capture.next(timeout=0.05)

    assert frame is not None
    assert frame.image.shape == (240, 320, 3)
    assert capture.ui_scale == 1.0


def test_vulkan_capture_converts_rgba_to_bgr(tmp_path):
    shm = tmp_path / "vd_layer_shm.dat"
    write_snapshot(shm, vk_format=37, pixel=(26, 153, 51, 255))

    with VulkanCapture(shm_path=shm) as capture:
        frame = capture.next(timeout=0.05)

    assert frame is not None
    assert tuple(frame.image[120, 160]) == (51, 153, 26)


def test_vulkan_capture_waits_for_new_capture_count(tmp_path):
    shm = tmp_path / "vd_layer_shm.dat"
    write_snapshot(shm, capture_count=4)

    with VulkanCapture(shm_path=shm) as capture:
        first = capture.next(timeout=0.05)
        assert first is not None
        assert capture.next(after=first.sequence, timeout=0.02) is None


def test_vulkan_capture_rejects_missing_pixel_ready_flag(tmp_path):
    shm = tmp_path / "vd_layer_shm.dat"
    write_snapshot(shm)
    raw = bytearray(shm.read_bytes())
    struct.pack_into("=I", raw, 12, 0b0111)
    shm.write_bytes(raw)

    with VulkanCapture(shm_path=shm) as capture:
        assert capture.next(timeout=0.02) is None


def test_vulkan_capture_implements_dispatch_capture_contract(tmp_path):
    shm = tmp_path / "vd_layer_shm.dat"
    write_snapshot(shm)

    capture = VulkanCapture(shm_path=shm)
    frame = capture.next(timeout=0.05)

    assert frame is not None
    assert capture.latest is frame
    assert capture.error is None
    assert capture.proc.poll() is None
    with capture.cv:
        assert capture.latest.sequence == frame.sequence

    capture.close()
    assert capture.stopping is True
    assert capture.proc.poll() is not None


def test_consumer_lease_has_fixed_size_expires_and_excludes_second_writer(tmp_path):
    import pytest
    shm=tmp_path/'vd_layer_shm.dat'
    write_snapshot(shm)
    capture=VulkanCapture(shm_path=shm)
    with capture:
        lease=shm.with_name('vd_layer_control.dat')
        data=lease.read_bytes()
        assert len(data)==16
        magic,version,expires=struct.unpack('=IIQ',data)
        assert magic==MAGIC and version==VERSION
        assert 0<expires-time.monotonic_ns()<=2_000_000_000
        other=VulkanCapture(shm_path=shm)
        with pytest.raises(BlockingIOError):
            other.__enter__()
        time.sleep(.280)
        refreshed=struct.unpack('=IIQ',lease.read_bytes())[2]
        assert refreshed>expires
        assert lease.stat().st_size==16
    assert struct.unpack('=IIQ',lease.read_bytes())[2]==0
    with VulkanCapture(shm_path=shm) as next_capture:
        assert next_capture.next(timeout=.02) is not None


def test_v3_timing_fields_match_pixels_and_sequence(tmp_path):
    shm=tmp_path/'vd_layer_shm.dat'
    write_snapshot(shm)
    with VulkanCapture(shm_path=shm) as capture:
        frame=capture.next(timeout=.02)
        assert frame.source_present_id==7
        assert frame.source_generation==1
        assert frame.media_time<=frame.copy_completed_time<=frame.published_time
        assert not capture.has_newer(7)
        assert capture.has_newer(6)


def test_old_live_layer_is_rejected_with_manual_restart_instruction(tmp_path):
    import pytest
    shm=tmp_path/'vd_layer_shm.dat'
    write_snapshot(shm)
    data=bytearray(shm.read_bytes())
    struct.pack_into('=I',data,4,2)
    shm.write_bytes(data)
    with VulkanCapture(shm_path=shm) as capture:
        with pytest.raises(RuntimeError,match='Close Sober manually'):
            capture.next(timeout=.02)


def test_native_telemetry_is_saved_off_thread_and_stops_on_close(tmp_path):
    import json
    import threading
    shm=tmp_path/'vd_layer_shm.dat'
    write_snapshot(shm)
    status=dict(status_time_ns=time.monotonic_ns(),present_timing={'samples':5})
    (tmp_path/'vd_layer_status.json').write_text(json.dumps(status))
    received=threading.Event()
    events=[]
    def callback(event):
        assert threading.current_thread().name=='VD-native-telemetry'
        events.append(event)
        received.set()
    with VulkanCapture(shm_path=shm) as capture:
        capture.start_telemetry(callback)
        assert received.wait(1)
    assert events[0]['kind']=='NATIVE_PRESENT_TIMING'
    assert events[0]['status']['present_timing']['samples']==5
    assert not capture._status_thread.is_alive()


def test_reader_immediately_accepts_counter_reset_from_new_producer(tmp_path):
    shm=tmp_path/'vd_layer_shm.dat'
    write_snapshot(shm,capture_count=700,epoch=100)
    with VulkanCapture(shm_path=shm) as capture:
        first=capture.next(timeout=.01)
        assert first.sequence==700
        assert first.source_epoch==(3,100)
        assert not capture.has_newer(first.sequence)
        write_snapshot(shm,capture_count=1,epoch=200)
        assert capture.has_newer(first.sequence)
        second=capture.next(after=first.sequence,timeout=.01)
        assert second.sequence==701
        assert second.source_epoch==(3,200)
        assert not capture.has_newer(second.sequence)
        write_snapshot(shm,capture_count=4,epoch=200)
        assert capture.has_newer(second.sequence)
        third=capture.next(after=second.sequence,timeout=.01)
        assert third.sequence==704
        assert third.source_present_id==4


def test_destroyed_swapchain_invalidates_even_without_new_capture_count(tmp_path):
    shm=tmp_path/'vd_layer_shm.dat'
    write_snapshot(shm,epoch=100)
    with VulkanCapture(shm_path=shm) as capture:
        frame=capture.next(timeout=.01)
        assert not capture.has_newer(frame.sequence)
        data=bytearray(shm.read_bytes())
        struct.pack_into('=I',data,12,0b0111)
        shm.write_bytes(data)
        assert capture.has_newer(frame.sequence)
        assert capture.next(frame.sequence,timeout=.001) is None


def test_diagnostic_snapshot_on_restart_keeps_reader_sequence_monotonic(tmp_path):
    shm=tmp_path/'vd_layer_shm.dat'
    write_snapshot(shm,capture_count=500,epoch=100)
    with VulkanCapture(shm_path=shm) as capture:
        first=capture.next(timeout=.01)
        write_snapshot(shm,capture_count=1,epoch=200)
        assert capture._snapshot()['count']==501
        assert capture.has_newer(first.sequence)
        fresh=capture.next(first.sequence,timeout=.01)
        assert fresh.sequence==501
        assert fresh.source_epoch==(3,200)
