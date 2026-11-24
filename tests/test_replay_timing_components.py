import json

import numpy as np
import pytest

from vd import runtime


@pytest.mark.parametrize('additive',[False,True])
def test_replay_reads_legacy_and_separate_timing_components(tmp_path,monkeypatch,additive):
    manifest=dict(lead_seconds=.035,lead_uncertainty=.020,learn_lead=True)
    settings=dict(lead_seconds=.040,lead_uncertainty=.023)
    if additive:
        manifest['timing_model']='response_plus_dispatch_v1'
        settings['timing_components']=dict(response_delay=.035,response_uncertainty=.020,
            dispatch_delay=.005,dispatch_uncertainty=.003)
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    row=dict(sequence=1,decision_time=1.,media_time=1.,held=True,state=settings)
    monkeypatch.setattr(runtime,'read_recording',lambda _:iter([(row,np.zeros((240,320,3),np.uint8))]))
    result=runtime.replay(tmp_path)
    assert result['frames']==1
    assert result['claims']==0
    assert not result['game_outcome_measured']


def test_replay_rejects_mismatched_component_sum(tmp_path,monkeypatch):
    manifest=dict(lead_seconds=.035,lead_uncertainty=.020,learn_lead=True,
        timing_model='response_plus_dispatch_v1')
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    settings=dict(lead_seconds=.060,lead_uncertainty=.023,
        timing_components=dict(response_delay=.035,response_uncertainty=.020,
            dispatch_delay=.005,dispatch_uncertainty=.003))
    row=dict(sequence=1,decision_time=1.,media_time=1.,held=True,state=settings)
    monkeypatch.setattr(runtime,'read_recording',lambda _:iter([(row,np.zeros((240,320,3),np.uint8))]))
    with pytest.raises(ValueError,match='timing components'):
        runtime.replay(tmp_path)


def test_replay_recreates_engine_when_source_epoch_changes_without_generation_change(tmp_path,monkeypatch):
    from vd.vision import Arc,Measurement,Needle
    (tmp_path/'manifest.json').write_text(json.dumps(dict(lead_seconds=0,lead_uncertainty=0)))
    class Detector:
        roi_offset=(0,0)
        def __init__(self,**_):
            pass
        def measure(self,image,timestamp,**_):
            phase=30+300*(timestamp-1)
            return Measurement(timestamp,(160,162.5),.99,Arc(130,10),Arc(141,42),
                (Needle(phase,80,.9,30,2),),'OK')
    rows=[]
    for index in range(4):
        row=dict(sequence=index,decision_time=1+index*.02,media_time=1+index*.02,
            held=True,source_generation=1,source_epoch=[3,100 if index<3 else 200])
        rows.append((row,np.zeros((240,320,3),np.uint8)))
    monkeypatch.setattr(runtime,'Detector',Detector)
    monkeypatch.setattr(runtime,'read_recording',lambda _:iter(rows))
    result=runtime.replay(tmp_path)
    assert any(event['kind']=='END' and event['reason']=='SOURCE_RECREATED' for event in result['events'])
    assert result['claims']==0
