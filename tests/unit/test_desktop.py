"""Desktop launcher helpers (no window or server is started here)."""

from __future__ import annotations

import socket
from pathlib import Path

import pytest
from PIL import Image

from cocoasense import desktop


def test_free_port_skips_a_port_in_use() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        busy = s.getsockname()[1]
        assert not desktop.port_is_free(busy)
        assert desktop.free_port(busy, tries=20) != busy
        assert not desktop.is_up(busy)  # listening, but not a Streamlit server


def test_free_port_gives_up(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(desktop, "port_is_free", lambda port: False)
    with pytest.raises(RuntimeError, match="no free port"):
        desktop.free_port(9000, tries=3)


def test_commands() -> None:
    cmd = desktop.server_command(8555)
    assert cmd[1:4] == ["-m", "streamlit", "run"]
    assert cmd[4].endswith("Overview.py") and Path(cmd[4]).exists()
    assert cmd[cmd.index("--server.port") + 1] == "8555"
    assert cmd[cmd.index("--server.address") + 1] == "127.0.0.1"
    win = desktop.app_window_command(Path("edge.exe"), "http://127.0.0.1:8555", Path("prof"))
    assert "--app=http://127.0.0.1:8555" in win
    assert f"--user-data-dir={Path('prof')}" in win


def test_find_browser(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("PROGRAMFILES(X86)", "PROGRAMFILES", "LOCALAPPDATA"):
        monkeypatch.setenv(key, str(tmp_path / "none"))
    assert desktop.find_browser() is None
    chrome = tmp_path / "pf" / "Google" / "Chrome" / "Application" / "chrome.exe"
    chrome.parent.mkdir(parents=True)
    chrome.touch()
    monkeypatch.setenv("PROGRAMFILES", str(tmp_path / "pf"))
    assert desktop.find_browser() == chrome


def test_running_port_reads_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    assert desktop.app_dir() == tmp_path / "CocoaSense"
    assert desktop._running_port() is None  # no state file
    desktop._state_file().write_text('{"port": 8777}', encoding="utf-8")
    monkeypatch.setattr(desktop, "is_up", lambda port: port == 8777)
    assert desktop._running_port() == 8777
    desktop._state_file().write_text("not json", encoding="utf-8")
    assert desktop._running_port() is None


def test_icon_has_all_sizes(tmp_path: Path) -> None:
    path = desktop.make_icon(tmp_path / "icons" / "app.ico")
    with Image.open(path) as ico:
        assert {(16, 16), (32, 32), (256, 256)} <= set(ico.info["sizes"])


def test_shortcut_script_quotes_paths() -> None:
    script = desktop.shortcut_script(
        Path("C:/Users/o'neil/Desktop/CocoaSense AI.lnk"),
        Path("C:/py/pythonw.exe"),
        "-m cocoasense.desktop",
        Path("C:/repo"),
        Path("C:/icons/app.ico"),
    )
    assert "o''neil" in script  # single quotes doubled for PowerShell
    assert "$s.Arguments = '-m cocoasense.desktop'" in script
    assert script.rstrip().endswith("$s.Save()")
    assert "app.ico,0'" in script


class _FakeProc:
    """Stands in for the browser process (SYNTHETIC)."""

    def __init__(self, exits_after: int | None = None) -> None:
        self.pid, self.polls, self.terminated, self.exits_after = 4242, 0, False, exits_after

    def poll(self) -> int | None:
        self.polls += 1
        if self.terminated or (self.exits_after is not None and self.polls > self.exits_after):
            return 0
        return None

    def terminate(self) -> None:
        self.terminated = True


def test_watch_window_stops_background_browser_after_window_closes() -> None:
    proc = _FakeProc()
    shown = iter([False, True, True] + [False] * 1000)  # opens, then closes, browser lingers
    desktop.watch_window(proc, lambda pid: next(shown), poll=0.001, grace=0.01)  # type: ignore[arg-type]
    assert proc.terminated


def test_watch_window_returns_when_browser_exits() -> None:
    proc = _FakeProc(exits_after=3)
    desktop.watch_window(proc, lambda pid: True, poll=0.001, grace=0.01)  # type: ignore[arg-type]
    assert not proc.terminated


def test_watch_window_gives_up_if_no_window_ever_appears() -> None:
    proc = _FakeProc()
    desktop.watch_window(proc, lambda pid: False, poll=0.001, grace=0.01, startup=0.01)  # type: ignore[arg-type]
    assert proc.terminated
