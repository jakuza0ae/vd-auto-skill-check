from pathlib import Path

from tools.generate_script_index import main,render


def test_index_tracks_add_delete_and_rename_without_reading_sources(tmp_path,monkeypatch):
    source=tmp_path/'vd'
    source.mkdir()
    first=source/'first.py'
    first.write_text('raise RuntimeError("Do not execute or read source for indexing")')
    original=Path.read_text
    def guarded_read(path,*args,**kwargs):
        assert path.suffix not in ('.py','.cmd','.lua','.luau'), 'Indexing must use metadata only'
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',guarded_read)
    root=['--root',str(tmp_path)]
    assert main(root)==0
    assert main(root+['--check'])==0
    second=source/'second.cmd'
    second.write_text('')
    assert main(root+['--check'])==1
    assert main(root)==0
    second.rename(source/'renamed.cmd')
    assert main(root+['--check'])==1
    assert main(root)==0
    first.unlink()
    assert main(root+['--check'])==1
    assert main(root)==0
    assert main(root+['--check'])==0
    assert 'renamed.cmd | Windows batch' in render(tmp_path)
    assert 'first.py' not in render(tmp_path)


def test_index_excludes_recordings_and_marks_new_luau_and_rojo_unverified(tmp_path):
    (tmp_path/'recordings').mkdir()
    (tmp_path/'recordings'/'irrelevant.py').write_text('')
    (tmp_path/'game.client.luau').write_text('')
    (tmp_path/'default.project.json').write_text('{}')
    index=render(tmp_path)
    assert 'irrelevant.py' not in index
    assert 'LocalScript (unverified filename inference)' in index
    assert 'Mapping is unverified' in index


def test_index_is_stable_across_checkouts_and_line_endings(tmp_path):
    first = tmp_path / 'first'
    second = tmp_path / 'second'
    first.mkdir()
    second.mkdir()
    (first / 'run.py').write_bytes(b'print(1)\n')
    (second / 'run.py').write_bytes(b'print(1)\r\n')
    assert render(first) == render(second)


def test_repository_index_is_current():
    assert main(['--check'])==0
