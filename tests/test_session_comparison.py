import json

import pytest

from tools.compare_solver_sessions import present_intervals,summarize,write_outcome_template


def native(generation,counts,total_ns):
    histogram=[0]*256
    for bucket,count in counts.items():
        histogram[bucket]=count
    return dict(kind='NATIVE_PRESENT_TIMING',status=dict(pid=7,init_timestamp_ns=1,
        present_timing=dict(generation=generation,histogram=histogram,
            bucket_width_ns=1_000_000,samples=sum(histogram),sum_ns=total_ns)))


def test_present_comparison_excludes_pre_recording_and_swapchain_history():
    result=present_intervals([
        native(1,{10:100},1_000_000_000),
        native(1,{10:102,40:1},1_060_000_000),
        native(2,{255:7},3_000_000_000),
        native(2,{255:8},3_300_000_000)])
    assert result['samples']==4
    assert result['mean_ms']==90
    assert result['p50_lower_ms']==10
    assert result['p99_lower_ms']==255
    assert result['p99_upper_ms'] is None
    assert not result['display_or_gpu_frame_time_measured']


def test_counter_reversal_is_not_reported_as_improvement():
    with pytest.raises(ValueError,match='Non-monotonic'):
        present_intervals([native(1,{10:10},100_000_000),native(1,{10:9},90_000_000)])


def test_cv_prediction_never_becomes_confirmed_game_verdict(tmp_path):
    events=[dict(kind='BEGIN',generation=1),dict(kind='BEGIN',generation=2),
        dict(kind='KEYDOWN',generation=1,plan=dict(target_grade='GREAT')),
        dict(kind='CV_LANDING',generation=1,landing=dict(label='GREAT'))]
    (tmp_path/'events.jsonl').write_text('\n'.join(json.dumps(e) for e in events))
    result=summarize(tmp_path)
    assert result['cv_predictions']=={'GREAT':1}
    assert result['game_outcomes']==dict(confirmed={},unknown=2,great_rate=None)
    csv_path=tmp_path/'verdicts.csv'
    write_outcome_template(tmp_path,csv_path)
    csv_path.write_text('generation,outcome,evidence\n1,GOOD,video 00:13 HUD verdict\n2,UNKNOWN,\n')
    result=summarize(tmp_path,csv_path)
    assert result['game_outcomes']==dict(confirmed={'GOOD':1},unknown=1,great_rate=0)
    csv_path.write_text('generation,outcome,evidence\n1,GREAT,\n')
    with pytest.raises(ValueError,match='evidence'):
        summarize(tmp_path,csv_path)
