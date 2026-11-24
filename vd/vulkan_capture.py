from __future__ import annotations

import fcntl
import json
import mmap
import os
from pathlib import Path
import struct
import threading
import time

import cv2
import numpy as np

from vd.capture import DEFAULT_ROI, Frame


SHM_MAGIC = 0x56444C59
SHM_VERSION = 3
STATUS_CAPTURE_ENABLED = 1 << 2
STATUS_PIXELS_READY = 1 << 3
HEADER_FORMAT = "=IIIIIIQQQfIIIIIII64sIIIIIII4xQQQQQQ80s"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
BGRA_FORMATS = {44, 50}  # VK_FORMAT_B8G8R8A8_UNORM/SRGB
RGBA_FORMATS = {37, 43}  # VK_FORMAT_R8G8R8A8_UNORM/SRGB
DEFAULT_SHM = Path.home() / ".var" / "app" / "org.vinegarhq.Sober" / "data" / "vulkan" / "vd_layer_shm.dat"


class _CaptureProcessState:
    """Small compatibility surface used by vd.dispatch for capture liveness."""

    def __init__(self, owner):
        self.owner = owner

    def poll(self):
        return 0 if self.owner.stopping else None


class VulkanCapture:
    """Read the latest swapchain ROI published by the VD Vulkan layer.

    The layer publishes a physical-resolution center ROI. This reader converts the
    swapchain's 4-byte pixel format to BGR and places those pixels into the solver's
    canonical 320x240 frame using center crop/padding only. It never rescales the
    game UI, so the detector keeps the physical UI scale reported by the layer.

    It also exposes the same publication/liveness surface as PortalCapture
    (cv/latest/error/proc/stopping), because vd.dispatch intentionally verifies the
    exact frame being acted on while holding the capture publication lock.
    """

    def __init__(self, roi=None, *, synthetic=False, fps=60, priority=5,
                 source="vulkan", normalize_window_scale=False, shm_path=None):
        if source != "vulkan":
            raise ValueError("VulkanCapture requires source='vulkan'")
        if synthetic:
            raise ValueError("Vulkan capture cannot be synthetic")
        if normalize_window_scale:
            raise ValueError("Vulkan capture is normalized automatically")
        if type(fps) is not int or not 1 <= fps <= 240:
            raise ValueError("FPS must be an integer in [1, 240]")
        if type(priority) is not int or not 0 <= priority <= 19:
            raise ValueError("Capture priority must be an integer in [0, 19]")
        if roi is not None and tuple(roi) != DEFAULT_ROI:
            raise ValueError("Vulkan capture currently uses the canonical solver ROI")

        self.roi = DEFAULT_ROI
        self.ui_scale = 1.0
        self.shm_path = Path(shm_path) if shm_path is not None else DEFAULT_SHM
        self.cv = threading.Condition()
        self.error = None
        self.latest = None
        self.stopping = False
        self.proc = _CaptureProcessState(self)
        self._file = None
        self._mapping = None
        self._lease_file = None
        self._lease_stop = threading.Event()
        self._lease_thread = None
        self._status_stop = threading.Event()
        self._status_thread = None
        self._source_epoch = None
        self._sequence_offset = 0
        self._last_snapshot_sequence = -1
        self.dispatch_deadline = None

    def _ensure_lease(self):
        if self._lease_file is not None:
            return
        path=self.shm_path.with_name('vd_layer_control.dat')
        f=os.fdopen(os.open(path,os.O_RDWR|os.O_CREAT,0o600),'r+b')
        try:
            fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            os.ftruncate(f.fileno(),16)
            os.pwrite(f.fileno(),struct.pack('=IIQ',SHM_MAGIC,SHM_VERSION,0),0)
            self._lease_file=f
            self._heartbeat()
            self._lease_thread=threading.Thread(target=self._lease_loop,
                name='VD-vulkan-consumer',daemon=True)
            self._lease_thread.start()
        except BaseException:
            f.close()
            self._lease_file=None
            raise

    def _heartbeat(self):
        os.pwrite(self._lease_file.fileno(),
            struct.pack('=Q',time.monotonic_ns()+2_000_000_000),8)

    def _lease_loop(self):
        while not self._lease_stop.wait(.250):
            try:
                self._heartbeat()
            except OSError as exc:
                self.error='Consumer heartbeat failed: '+str(exc)
                return

    def start_telemetry(self, callback):
        """Sample native counters off the decision and heartbeat threads."""
        if self._status_thread is not None:
            raise RuntimeError('Native telemetry already started')
        path=self.shm_path.with_name('vd_layer_status.json')
        def sample():
            previous=None
            while not self._status_stop.is_set():
                try:
                    with path.open() as source:
                        status=json.loads(source.read(65536))
                    stamp=status.get('status_time_ns')
                    now=time.monotonic()
                    if (isinstance(stamp,int) and stamp!=previous
                            and 0<=now-stamp/1e9<=2
                            and status.get('present_timing')):
                        callback(dict(kind='NATIVE_PRESENT_TIMING',at=now,status=status))
                        previous=stamp
                except (OSError,ValueError):
                    pass
                if self._status_stop.wait(.250):
                    break
        self._status_thread=threading.Thread(target=sample,
            name='VD-native-telemetry',daemon=True)
        self._status_thread.start()

    def has_newer(self, sequence):
        mm=self._mapping
        if mm is None:
            return True
        try:
            first=struct.unpack_from('=I',mm,20)[0]
            flags=struct.unpack_from('=I',mm,12)[0]
            count=struct.unpack_from('=Q',mm,176)[0]
            epoch=(struct.unpack_from('=I',mm,16)[0],struct.unpack_from('=Q',mm,24)[0])
            second=struct.unpack_from('=I',mm,20)[0]
            return bool(not flags & STATUS_PIXELS_READY
                        or first&1 or first!=second
                        or epoch!=getattr(self,'_source_epoch',None)
                        or count+getattr(self,'_sequence_offset',0)!=sequence)
        except (ValueError,struct.error):
            return True

    def prediction_current(self, frame):
        mm=self._mapping
        if mm is None:
            return False
        try:
            first=struct.unpack_from('=I',mm,20)[0]
            flags=struct.unpack_from('=I',mm,12)[0]
            epoch=(struct.unpack_from('=I',mm,16)[0],struct.unpack_from('=Q',mm,24)[0])
            generation=struct.unpack_from('=Q',mm,200)[0]
            second=struct.unpack_from('=I',mm,20)[0]
            return bool(not first&1 and first==second and flags & STATUS_PIXELS_READY
                and epoch==frame.source_epoch and generation==frame.source_generation)
        except (ValueError,struct.error):
            return False

    def prediction_busy(self):
        """Pixel publication is transient; it does not require another CV pass."""
        try:
            return bool(self._mapping is not None
                        and struct.unpack_from('=I',self._mapping,20)[0]&1)
        except (ValueError,struct.error):
            return False

    def _close_mapping(self):
        if self._mapping is not None:
            self._mapping.close()
            self._mapping = None
        if self._file is not None:
            self._file.close()
            self._file = None

    def _ensure_mapping(self):
        if self._mapping is not None:
            return True
        try:
            f = self.shm_path.open("rb")
            if self.shm_path.stat().st_size < HEADER_SIZE:
                f.close()
                return False
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        except (FileNotFoundError, OSError, ValueError):
            return False
        self._file = f
        self._mapping = mm
        return True

    def _snapshot(self, after=-1, deadline=None):
        if not self._ensure_mapping():
            return None
        mm = self._mapping
        if mm is None or len(mm) < HEADER_SIZE:
            self._close_mapping()
            return None

        for attempt in range(4):
            if attempt and deadline is not None and time.monotonic()>=deadline:
                return None
            try:
                header = bytes(mm[:HEADER_SIZE])
                values = struct.unpack(HEADER_FORMAT, header)
            except (ValueError, struct.error):
                self._close_mapping()
                return None

            magic, version, struct_size, flags, _pid, seq1 = values[:6]
            if magic != SHM_MAGIC:
                raise RuntimeError("VD Vulkan shared-memory magic not found")
            if version != SHM_VERSION or struct_size != HEADER_SIZE:
                raise RuntimeError(
                    f"VD Vulkan layer v{SHM_VERSION} required (found v{version}, size={struct_size}). "
                    "Close Sober manually and start it again using ./start-vulkan-sober.sh."
                )
            if seq1 & 1:
                continue
            if not (flags & STATUS_CAPTURE_ENABLED) or not (flags & STATUS_PIXELS_READY):
                return None

            sw_width, sw_height, sw_format = values[11:14]
            roi_x, roi_y, roi_w, roi_h, roi_stride, roi_offset, roi_size = values[18:25]
            capture_count, capture_ts = values[25:27]
            epoch=(_pid,values[6])
            offset=getattr(self,'_sequence_offset',0)
            if getattr(self,'_source_epoch',None) not in (None,epoch):
                offset=max(after,getattr(self,'_last_snapshot_sequence',-1))+1-int(capture_count)
            sequence=int(capture_count)+offset
            if sequence<=after:
                return None
            source_present_id,generation,completed,published=values[27:31]
            if roi_w <= 0 or roi_h <= 0 or roi_stride != roi_w * 4:
                return None
            if roi_size != roi_stride * roi_h:
                raise RuntimeError("Unexpected VD Vulkan ROI layout")
            end = roi_offset + roi_size
            if roi_offset < HEADER_SIZE or end > len(mm):
                raise RuntimeError("VD Vulkan ROI points outside shared-memory file")
            raw = bytes(mm[roi_offset:end])
            seq2 = struct.unpack_from("=I", mm, 20)[0]
            if seq1 == seq2 and not (seq2 & 1):
                self._source_epoch=epoch
                self._sequence_offset=offset
                self._last_snapshot_sequence=sequence
                return dict(
                    count=sequence, native_count=int(capture_count), source_epoch=epoch,
                    timestamp_ns=int(capture_ts),
                    source_present_id=int(source_present_id),generation=int(generation),
                    completed_ns=int(completed),published_ns=int(published),
                    source_size=(int(sw_width), int(sw_height)), format=int(sw_format),
                    roi=(int(roi_x), int(roi_y), int(roi_w), int(roi_h)), raw=raw,
                )
        return None

    @staticmethod
    def _center_crop_or_pad(image, target_w, target_h):
        height, width = image.shape[:2]

        if width > target_w:
            x = (width - target_w) // 2
            image = image[:, x:x + target_w]
        if height > target_h:
            y = (height - target_h) // 2
            image = image[y:y + target_h, :]

        height, width = image.shape[:2]
        if (width, height) == (target_w, target_h):
            return np.array(image, copy=True)

        output = np.zeros((target_h, target_w, image.shape[2]), dtype=image.dtype)
        x = max(0, (target_w - width) // 2)
        y = max(0, (target_h - height) // 2)
        output[y:y + height, x:x + width] = image
        return output

    def _to_bgr(self, snapshot):
        _x, _y, width, height = snapshot["roi"]
        pixels = np.frombuffer(snapshot["raw"], np.uint8).reshape(height, width, 4)
        vk_format = snapshot["format"]
        if vk_format in BGRA_FORMATS:
            bgr = cv2.cvtColor(pixels,cv2.COLOR_BGRA2BGR)
        elif vk_format in RGBA_FORMATS:
            bgr = cv2.cvtColor(pixels,cv2.COLOR_RGBA2BGR)
        else:
            raise RuntimeError(f"Unsupported Vulkan swapchain format: {vk_format}")

        target_w, target_h = DEFAULT_ROI[2:]
        bgr = self._center_crop_or_pad(bgr, target_w, target_h)
        self.ui_scale = 1.0
        bgr.flags.writeable = False
        return bgr

    def set_dispatch_deadline(self, deadline):
        self.dispatch_deadline=deadline

    def next(self, after=-1, timeout=1.0):
        if hasattr(self,'_lease_file'):
            self._ensure_lease()
        deadline = time.monotonic() + timeout
        fine_window=(.001 if timeout<=.001 or getattr(self,'dispatch_deadline',None) is not None else 0)
        while not self.stopping:
            snapshot = self._snapshot(after=after,deadline=deadline)
            if snapshot is not None and snapshot["count"] > after:
                received = time.monotonic()
                image = self._to_bgr(snapshot)
                timestamp_ns = snapshot["timestamp_ns"]
                frame = Frame(
                    sequence=snapshot["count"],
                    image=image,
                    media_time=timestamp_ns / 1e9 if timestamp_ns else None,
                    received_time=received,
                    consumed_time=time.monotonic(),
                    timestamp_kind="vulkan_monotonic",
                    synthetic=False,
                    pts_ns=timestamp_ns or None,
                    negotiated_caps=f"vulkan-format={snapshot['format']};raw-roi={snapshot['roi'][2]}x{snapshot['roi'][3]}",
                    worker_cpu_time_ns=None,
                    source_size=snapshot["source_size"],
                    source_present_id=snapshot['source_present_id'],
                    source_generation=snapshot['generation'],
                    source_epoch=snapshot['source_epoch'],
                    copy_completed_time=snapshot['completed_ns']/1e9,
                    published_time=snapshot['published_ns']/1e9,
                )
                with self.cv:
                    self.latest = frame
                    self.cv.notify_all()
                return frame
            now = time.monotonic()
            if now >= deadline:
                return None
            # Leave the final millisecond to bounded polling: a submillisecond
            # sleep can wake several milliseconds late under scheduler load.
            # Longer capture waits still sleep to keep idle CPU use low.
            remaining=deadline-now
            if remaining>fine_window:
                time.sleep(min(.002,remaining-fine_window))
        return None

    def close(self, *, fast=False):
        with self.cv:
            self.stopping = True
            self.cv.notify_all()
        if getattr(self,'_status_thread',None) is not None:
            self._status_stop.set()
            self._status_thread.join(timeout=1)
        if getattr(self,'_lease_thread',None) is not None:
            self._lease_stop.set()
            self._lease_thread.join(timeout=.5)
        if getattr(self,'_lease_file',None) is not None:
            try:
                os.pwrite(self._lease_file.fileno(),struct.pack('=Q',0),8)
            finally:
                self._lease_file.close()
                self._lease_file=None
        self._close_mapping()

    def __enter__(self):
        self._ensure_lease()
        return self

    def __exit__(self, exc_type, *exc):
        self.close(fast=exc_type is not None and issubclass(exc_type, KeyboardInterrupt))
