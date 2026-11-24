from pathlib import Path
import subprocess


def test_async_capture_contract(tmp_path):
    root=Path(__file__).resolve().parents[1]
    binary=tmp_path/'capture-contract'
    build=subprocess.run(['g++','-std=c++17','-O1','-pthread',
        str(root/'tests/native_capture_contract.cpp'),'-o',str(binary)],
        capture_output=True,text=True,timeout=60)
    assert build.returncode==0,build.stderr
    result=subprocess.run([str(binary)],capture_output=True,text=True,timeout=10)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'async capture contract: PASS' in result.stdout
