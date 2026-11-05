"""Desktop launcher: open the dashboard in its own app window.

``python -m cocoasense.desktop`` (what the desktop shortcut runs) starts the Streamlit server on a
free local port, opens it in a chromeless Edge or Chrome window (``--app`` mode, separate browser
profile) and stops the server when that window is closed. A second launch while the app is open
reuses the running server. Logs, the icon and the browser profile live in
``%LOCALAPPDATA%\\CocoaSense``.

``cocoasense install-shortcut`` puts a CocoaSense shortcut on the Windows desktop.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser
from collections.abc import Callable
from pathlib import Path

from cocoasense.config import PROJECT_ROOT

APP_NAME = "CocoaSense AI"
PORT_START = 8501
DASHBOARD = PROJECT_ROOT / "src" / "cocoasense" / "dashboard" / "Overview.py"
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def app_dir() -> Path:
    """Per-user folder for the log, icon, browser profile and launcher state."""
    base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / ".local" / "share")
    path = base / "CocoaSense"
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_browser() -> Path | None:
    """Microsoft Edge or Google Chrome, whichever is installed first."""
    roots = [os.environ.get(k) for k in ("PROGRAMFILES(X86)", "PROGRAMFILES", "LOCALAPPDATA")]
    rel = [
        Path("Microsoft/Edge/Application/msedge.exe"),
        Path("Google/Chrome/Application/chrome.exe"),
    ]
    for r in rel:
        for root in roots:
            if root and (Path(root) / r).exists():
                return Path(root) / r
    return None


def port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def free_port(start: int = PORT_START, tries: int = 50) -> int:
    """First local port from ``start`` that nothing is listening on."""
    for port in range(start, start + tries):
        if port_is_free(port):
            return port
    raise RuntimeError(f"no free port in {start}..{start + tries - 1}")


def is_up(port: int) -> bool:
    """True if a Streamlit server answers its health check on ``port``."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=2) as r:
            return bool(r.status == 200)
    except OSError:
        return False


def server_command(port: int) -> list[str]:
    python = Path(sys.executable)
    if python.name.lower() == "pythonw.exe":  # the server is happier with a normal interpreter
        python = python.with_name("python.exe")
    return [
        str(python),
        "-m",
        "streamlit",
        "run",
        str(DASHBOARD),
        "--server.headless",
        "true",
        "--server.port",
        str(port),
        "--server.address",
        "127.0.0.1",
        "--browser.gatherUsageStats",
        "false",
        "--client.toolbarMode",
        "viewer",
    ]


def app_window_command(browser: Path, url: str, profile: Path) -> list[str]:
    """Chromeless window with its own profile, so the call blocks until the window closes."""
    return [
        str(browser),
        f"--app={url}",
        f"--user-data-dir={profile}",
        "--no-first-run",
        "--no-default-browser-check",
        "--window-size=1440,900",
    ]


def _state_file() -> Path:
    return app_dir() / "launcher.json"


def _running_port() -> int | None:
    """Port of a server started by another launcher that is still answering."""
    try:
        port = int(json.loads(_state_file().read_text(encoding="utf-8"))["port"])
    except (OSError, ValueError, KeyError):
        return None
    return port if is_up(port) else None


def _visible_window_pids() -> set[int]:  # pragma: no cover - Windows API
    """Process ids that own a visible top-level window."""
    if sys.platform != "win32":
        return set()
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    pids: set[int] = set()

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def collect(hwnd: int, _: int) -> bool:
        if user32.IsWindowVisible(hwnd):
            pid = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            pids.add(pid.value)
        return True

    user32.EnumWindows(collect, 0)
    return pids


def watch_window(
    proc: subprocess.Popen[bytes],
    has_window: Callable[[int], bool],
    poll: float = 1.0,
    grace: float = 5.0,
    startup: float = 60.0,
) -> None:
    """Return once the app window is gone.

    Either the browser exits, or its window has been absent for ``grace`` seconds after first
    appearing (or ``startup`` seconds pass without one). Edge can keep running in the background
    after its last window closes, so in that case the browser process (dedicated profile) is
    stopped as well.
    """
    seen, missing_since, start = False, None, time.monotonic()
    while proc.poll() is None:
        now = time.monotonic()
        if has_window(proc.pid):
            seen, missing_since = True, None
        elif seen or now - start > startup:
            missing_since = missing_since or now
            if now - missing_since >= grace:
                proc.terminate()
                return
        time.sleep(poll)


def launch(timeout: float = 90.0) -> None:  # pragma: no cover - starts processes and a window
    """Start the server (or reuse one), open the app window, stop the server when it closes."""
    browser = find_browser()
    profile = app_dir() / "browser-profile"
    port = _running_port()
    if port is not None:  # already open: just bring up another window
        url = f"http://127.0.0.1:{port}"
        if browser:
            subprocess.Popen(app_window_command(browser, url, profile))
        else:
            webbrowser.open(url)
        return

    port = free_port()
    url = f"http://127.0.0.1:{port}"
    with open(app_dir() / "dashboard.log", "w", encoding="utf-8") as log:
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        server = subprocess.Popen(
            server_command(port),
            cwd=PROJECT_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            creationflags=NO_WINDOW,
        )
        try:
            deadline = time.monotonic() + timeout
            while not is_up(port):
                if server.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError(
                        f"The dashboard server did not start. See {app_dir() / 'dashboard.log'}."
                    )
                time.sleep(0.5)
            _state_file().write_text(json.dumps({"port": port}), encoding="utf-8")
            if browser:
                window = subprocess.Popen(app_window_command(browser, url, profile))
                if sys.platform == "win32":
                    watch_window(window, lambda pid: pid in _visible_window_pids())
                else:
                    window.wait()
            else:
                webbrowser.open(url)
                server.wait()  # no app window to watch; runs until the server is stopped
        finally:
            server.terminate()
            _state_file().unlink(missing_ok=True)


# ------------------------------------------------------------------- shortcut


def make_icon(path: Path) -> Path:
    """Draw the app icon (a cocoa pod over a rising price line) as a multi-size .ico."""
    from PIL import Image, ImageDraw

    size = 256
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((8, 8, 248, 248), radius=48, fill=(74, 44, 24, 255))
    d.ellipse((70, 34, 186, 222), fill=(201, 121, 43, 255))
    for x in (100, 128, 156):  # pod ridges
        d.line((x, 48, x, 208), fill=(150, 82, 26, 255), width=6)
    d.line(
        [(30, 200), (80, 170), (120, 182), (170, 118), (226, 70)],
        fill=(255, 236, 179, 255),
        width=14,
        joint="curve",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    return path


def _ps_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def shortcut_script(link: Path, target: Path, args: str, workdir: Path, icon: Path) -> str:
    """PowerShell that writes a Windows .lnk shortcut."""
    return "; ".join(
        [
            f"$s = (New-Object -ComObject WScript.Shell).CreateShortcut({_ps_quote(str(link))})",
            f"$s.TargetPath = {_ps_quote(str(target))}",
            f"$s.Arguments = {_ps_quote(args)}",
            f"$s.WorkingDirectory = {_ps_quote(str(workdir))}",
            f"$s.IconLocation = {_ps_quote(str(icon) + ',0')}",
            f"$s.Description = {_ps_quote(APP_NAME + ' dashboard')}",
            "$s.Save()",
        ]
    )


def desktop_folder() -> Path:  # pragma: no cover - asks Windows
    """The user's Desktop (follows OneDrive redirection)."""
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command", "[Environment]::GetFolderPath('Desktop')"],
        capture_output=True,
        text=True,
        check=True,
        creationflags=NO_WINDOW,
    )
    return Path(out.stdout.strip())


def create_shortcut(folder: Path | None = None) -> Path:  # pragma: no cover - Windows only
    """Write ``CocoaSense AI.lnk`` to the desktop (or ``folder``); return its path."""
    if sys.platform != "win32":
        raise RuntimeError("desktop shortcuts are created on Windows only")
    folder = folder or desktop_folder()
    link = folder / f"{APP_NAME}.lnk"
    icon = make_icon(app_dir() / "cocoasense.ico")
    pythonw = Path(sys.executable).with_name("pythonw.exe")
    script = shortcut_script(link, pythonw, "-m cocoasense.desktop", PROJECT_ROOT, icon)
    subprocess.run(["powershell", "-NoProfile", "-Command", script], check=True)
    return link


def main() -> None:  # pragma: no cover - entry point of the shortcut
    try:
        launch()
    except Exception as exc:  # no console under pythonw: show the error in a dialog
        if sys.platform == "win32":
            import ctypes

            ctypes.windll.user32.MessageBoxW(None, str(exc), APP_NAME, 0x10)
        raise


if __name__ == "__main__":  # pragma: no cover
    main()


# ------------------------------------------------------------ automatic updates

TASK_NAME = "CocoaSense auto-update"


def schedule_script(pythonw: Path, workdir: Path, at: str = "09:00") -> str:
    """PowerShell that registers the daily + at-logon update task (only when online)."""
    action = (
        f"New-ScheduledTaskAction -Execute {_ps_quote(str(pythonw))} "
        f"-Argument '-m cocoasense.autoupdate' -WorkingDirectory {_ps_quote(str(workdir))}"
    )
    return "; ".join(
        [
            f"$a = {action}",
            f"$t1 = New-ScheduledTaskTrigger -Daily -At {_ps_quote(at)}",
            "$t2 = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME",
            "$s = New-ScheduledTaskSettingsSet -StartWhenAvailable -RunOnlyIfNetworkAvailable "
            "-AllowStartIfOnBatteries -DontStopIfGoingOnBatteries "
            "-ExecutionTimeLimit (New-TimeSpan -Hours 1)",
            f"Register-ScheduledTask -TaskName {_ps_quote(TASK_NAME)} -Action $a "
            "-Trigger $t1,$t2 -Settings $s -Force "
            f"-Description {_ps_quote('Checks for new cocoa prices and retrains CocoaSense')}"
            " | Out-Null",
        ]
    )


def register_schedule(at: str = "09:00") -> str:  # pragma: no cover - Windows only
    """Create (or replace) the Windows scheduled task; return a description."""
    if sys.platform != "win32":
        raise RuntimeError("automatic updates are scheduled on Windows only")
    pythonw = Path(sys.executable).with_name("pythonw.exe")
    script = schedule_script(pythonw, PROJECT_ROOT, at)
    subprocess.run(["powershell", "-NoProfile", "-Command", script], check=True)
    return f"'{TASK_NAME}' runs daily at {at} and at every log-on, whenever you are online"


def remove_schedule() -> None:  # pragma: no cover - Windows only
    """Delete the scheduled task if it exists."""
    script = (
        f"Unregister-ScheduledTask -TaskName {_ps_quote(TASK_NAME)} -Confirm:$false "
        "-ErrorAction SilentlyContinue"
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", script], check=True)


# ---------------------------------------------------------------- notifications

POWERSHELL_APP_ID = r"{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\WindowsPowerShell\v1.0\powershell.exe"


def _xml_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def toast_script(title: str, body: str) -> str:
    """PowerShell that shows a Windows toast notification (no extra modules needed)."""
    xml = (
        "<toast><visual><binding template='ToastGeneric'>"
        f"<text>{_xml_escape(title)}</text><text>{_xml_escape(body)}</text>"
        "</binding></visual></toast>"
    )
    return "; ".join(
        [
            "[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, "
            "ContentType=WindowsRuntime] | Out-Null",
            "[Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, "
            "ContentType=WindowsRuntime] | Out-Null",
            "$x = New-Object Windows.Data.Xml.Dom.XmlDocument",
            f"$x.LoadXml({_ps_quote(xml)})",
            "$t = [Windows.UI.Notifications.ToastNotification]::new($x)",
            "[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("
            f"{_ps_quote(POWERSHELL_APP_ID)}).Show($t)",
        ]
    )


def notify(title: str, body: str) -> None:  # pragma: no cover - Windows UI
    """Show a desktop notification; failures are ignored (notifications are optional)."""
    if sys.platform != "win32":
        return
    subprocess.run(
        ["powershell", "-NoProfile", "-Command", toast_script(title, body)],
        check=False,
        capture_output=True,
        creationflags=NO_WINDOW,
    )
