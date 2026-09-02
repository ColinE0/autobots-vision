"""SessionLog: every bench tool leaves an append-only, attributable record."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.sessionlog import SessionLog


def test_sessions_append_with_headers(tmp_path):
    log = SessionLog('bench_demo', 'k=v', root=tmp_path)
    log.line('first', echo=False)
    log.close()
    log2 = SessionLog('bench_demo', root=tmp_path)   # second run appends
    log2.line('second', echo=False)
    log2.close()

    text = (tmp_path / 'bench_demo.log').read_text()
    assert text.count('--- session') == 2
    assert 'k=v' in text
    assert 'first' in text and 'second' in text


def test_lines_are_timestamped(tmp_path):
    log = SessionLog('t', root=tmp_path)
    log.line('hello', echo=False)
    log.close()
    for ln in (tmp_path / 't.log').read_text().splitlines():
        if not ln.startswith('--- session'):
            assert ln[2] == ':' and ln[5] == ':', ln   # HH:MM:SS prefix


def test_echo_off_prints_nothing(capsys, tmp_path):
    log = SessionLog('t', root=tmp_path)
    stamped = log.line('quiet', echo=False)
    log.close()
    assert stamped.endswith('  quiet')
    assert capsys.readouterr().out == ''


def test_header_survives_no_git(tmp_path):
    # tmp_path is not a git repo; the header must still write (rev=unknown
    # or a parent repo's rev, never an exception)
    log = SessionLog('t', root=tmp_path)
    log.close()
    assert (tmp_path / 't.log').read_text().startswith('--- session')
