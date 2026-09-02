"""
Append-only session logs for the bench and bring-up tools.

Every hardware tool opens one of these so a bench run leaves evidence
(the integration-testing lesson: terminal scrollback is not a record).
One log per tool, in the repo root, gitignored, append-only:

    from tools.sessionlog import SessionLog
    log = SessionLog('test_motors', 'test_duty=0.245')
    log.line('front_left FORWARD')       # prints AND logs, timestamped
    log.close()

The session header carries the date, the git revision, and whatever
key=value context the tool passes, so a number found in a log months later
is attributable to a code state. test_camera.py and detect_preview.py
predate this helper and keep their own equivalent inline logging.
"""
import pathlib
import subprocess
import time

_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _git_rev(root):
    try:
        out = subprocess.run(['git', '-C', str(root), 'describe', '--always', '--dirty'],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or 'unknown'
    except Exception:
        return 'unknown'


class SessionLog:
    """Timestamped append-only log, one file per tool, in the repo root."""

    def __init__(self, name, context='', root=None):
        root = pathlib.Path(root) if root else _ROOT
        self.path = root / f"{name}.log"
        self._f = open(self.path, 'a')
        header = time.strftime('--- session %Y-%m-%d %H:%M:%S')
        header += f"  rev={_git_rev(root)}"
        if context:
            header += f"  {context}"
        self._f.write(header + '\n')
        self._f.flush()

    def line(self, text, echo=True):
        """Log a timestamped line; echo=True also prints it as written."""
        stamped = time.strftime('%H:%M:%S') + f"  {text}"
        self._f.write(stamped + '\n')
        self._f.flush()
        if echo:
            print(text, flush=True)
        return stamped

    def close(self):
        self._f.close()
