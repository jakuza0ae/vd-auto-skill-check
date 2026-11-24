import pytest

from vd.vulkan_capture import VulkanCapture


class FakeClock:
    def __init__(self, start=10.0):
        self.now = start
        self.sleeps = []

    def monotonic(self):
        self.now+=.000001
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def test_vulkan_capture_timeout_does_not_oversleep_short_dispatch_deadline(monkeypatch):
    capture = VulkanCapture.__new__(VulkanCapture)
    capture.stopping = False
    capture._snapshot = lambda **_: None

    clock = FakeClock()
    monkeypatch.setattr('vd.vulkan_capture.time.monotonic', clock.monotonic)
    monkeypatch.setattr('vd.vulkan_capture.time.sleep', clock.sleep)

    started = clock.monotonic()
    assert capture.next(after=1, timeout=.0007) is None
    elapsed = clock.monotonic() - started

    assert .0007<=elapsed<=.00071
    assert not clock.sleeps


def test_real_snapshot_odd_seqlock_respects_submillisecond_deadline(monkeypatch,tmp_path):
    import struct
    from test_vulkan_capture_source import write_snapshot
    shm=tmp_path/'vd_layer_shm.dat'
    write_snapshot(shm)
    data=bytearray(shm.read_bytes())
    struct.pack_into('=I',data,20,3)
    shm.write_bytes(data)
    capture=VulkanCapture(shm_path=shm)
    assert capture._ensure_mapping()
    clock=FakeClock()
    monkeypatch.setattr('vd.vulkan_capture.time.monotonic',clock.monotonic)
    monkeypatch.setattr('vd.vulkan_capture.time.sleep',clock.sleep)
    try:
        assert capture.next(after=1,timeout=.0007) is None
        assert .0007<=clock.now-10<=.00071
        assert not clock.sleeps
    finally:
        capture.close()


def test_unchanged_sequence_does_not_copy_roi_bytes(tmp_path):
    from test_vulkan_capture_source import write_snapshot,HEADER_SIZE
    shm=tmp_path/'vd_layer_shm.dat'
    write_snapshot(shm,capture_count=7)

    class TrackedBytes(bytearray):
        reads=0
        def __getitem__(self,key):
            if isinstance(key,slice) and (key.start or 0)>=HEADER_SIZE:
                self.reads+=1
            return super().__getitem__(key)
        def close(self):
            pass

    capture=VulkanCapture(shm_path=shm)
    data=TrackedBytes(shm.read_bytes())
    capture._mapping=data
    capture._ensure_mapping=lambda:True
    try:
        assert capture._snapshot(after=7) is None
        assert data.reads==0
        assert capture._snapshot(after=6)['count']==7
        assert data.reads==1
    finally:
        capture.close()


@pytest.mark.parametrize('planned',[False,True])
def test_fine_polling_is_reserved_for_pending_dispatch(monkeypatch,planned):
    capture=VulkanCapture.__new__(VulkanCapture)
    capture.stopping=False
    reads=[]
    capture._snapshot=lambda **_:reads.append(1)
    clock=FakeClock()
    capture.set_dispatch_deadline(clock.now+.02 if planned else None)
    monkeypatch.setattr('vd.vulkan_capture.time.monotonic',clock.monotonic)
    monkeypatch.setattr('vd.vulkan_capture.time.sleep',clock.sleep)
    assert capture.next(after=1,timeout=.020) is None
    assert clock.sleeps
    if planned:
        assert len(reads)>100
    else:
        assert len(reads)<=12
    assert max(clock.sleeps)<=.002


def test_busy_publication_keeps_timer_but_never_bypasses_source_identity(tmp_path):
    import struct
    from vd.dispatch import dispatch
    from vd.engine import Engine
    from vd.motion import Motion
    from vd.vision import Arc
    from test_vulkan_capture_source import write_snapshot
    shm=tmp_path/'vd_layer_shm.dat'
    write_snapshot(shm,capture_ts_ns=1_000_000_000,epoch=10)
    with VulkanCapture(shm_path=shm) as capture:
        frame=capture.next(timeout=.02)
        engine=Engine(lead_seconds=0)
        engine.planner.begin(Arc(40,10),45)
        plan=engine.planner.update(Motion(1,45,900,.1,2,1,10),frame_at=1,now=1)
        data=bytearray(shm.read_bytes())
        struct.pack_into('=I',data,20,3)
        shm.write_bytes(data)
        assert capture.prediction_busy()
        assert dispatch(engine,capture,frame.sequence,clock=lambda:1.) is None
        assert engine.reason=='CAPTURE_WRITING'
        assert engine.planner.current is plan
        write_snapshot(shm,capture_ts_ns=1_000_000_000,epoch=20)
        assert not capture.prediction_busy()
        assert dispatch(engine,capture,frame.sequence,clock=lambda:1.) is None
        assert engine.reason=='NEW_FRAME_PENDING'
        assert not engine.planner.fired
