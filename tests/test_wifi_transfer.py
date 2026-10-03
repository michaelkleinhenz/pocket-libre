"""WiFi transfer: the firmware 1.8 socket protocol, against a fake device.

FakeDevice models what was observed on a real recorder (firmware 1.8, WiFi
firmware V9; see PROTOCOL.md):

  * APP&WIFIO raises the AP; WIFIS goes 3 -> 2, and 1 once a client joins.
  * The transfer socket listens while the AP is up, for at most two
    connections per AP session; after that it refuses until APP&WIFIC +
    APP&WIFIO restart the AP.
  * APP&U&<date>&<ts> answers MCU&U&<size>; APP&U&WIFI then answers
    MCU&U&WIFI and MCU&U&<size>, sends the file and the 10-byte end marker on
    the open connection, and reports MCU&OFF.

What these tests cannot show is that a real device still behaves this way;
they pin the client to the behaviour that was measured.
"""

import asyncio
import socket
from types import SimpleNamespace

import pytest

from pocket_libre.commands import PocketCommander, Recording, split_messages
from pocket_libre.hostwifi import parse_netsh_interfaces, split_terse, windows_profile
from pocket_libre.protocol import (
    END_MARKER,
    FILES_PER_AP_SESSION,
    TRANSFER_PORT,
    WIFI_STATUS_CLIENT_JOINED,
    WIFI_STATUS_STARTING,
    WIFI_STATUS_WAITING_FOR_CLIENT,
)
from pocket_libre.wifi import WifiSession, WifiTransferError, receive_file


def mp3(n: int) -> bytes:
    frame = b"\xff\xf3\x48\xc4" + bytes(range(140))
    return (frame * (n // len(frame) + 1))[:n]


FILES = {
    "20261003142550": mp3(885_788),
    "20261003141332": mp3(722_348)[::-1],
    "20261003160116": mp3(92_062),
}


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class FakeDevice:
    def __init__(self, port: int, files: dict[str, bytes] = FILES, send_marker: bool = True):
        self.port = port
        self.files = files
        self.send_marker = send_marker
        self.status = 0
        self.server: asyncio.AbstractServer | None = None
        self.accepted = 0
        self.ap_starts = 0
        self.writer: asyncio.StreamWriter | None = None
        self.staged: bytes | None = None
        self.violations: list[str] = []
        self.sent: list[str] = []
        self.reply = None  # set by FakeCommander

    async def _serve(self, reader, writer):
        self.accepted += 1
        self.writer = writer
        if self.accepted >= FILES_PER_AP_SESSION:
            self.server.close()  # stops listening: further connects are refused
        try:
            await reader.read()  # until the client closes
        finally:
            if self.writer is writer:
                self.writer = None
            writer.close()

    async def handle(self, command: str) -> None:
        self.sent.append(command)
        if command == "WIFIO":
            self.status = WIFI_STATUS_STARTING
            self.accepted = 0
            self.ap_starts += 1
            self.server = await asyncio.start_server(self._serve, "127.0.0.1", self.port,
                                                     reuse_address=True)
            self.reply("MCU&WIFIO")
            self.status = WIFI_STATUS_WAITING_FOR_CLIENT
        elif command == "WIFI":
            self.reply("MCU&WIFI&PKT01_GREY_TEST&abcd1234")
        elif command == "WIFIS":
            self.reply(f"MCU&WIFIS&{self.status}")
        elif command == "WIFIC":
            if self.server:
                self.server.close()
            self.status = 0
            self.reply("MCU&WIFIC")
        elif command == "WPING":
            self.reply("MCU&WPING")
        elif command == "U&WIFI":
            if self.writer is None:
                self.violations.append("U&WIFI without an open connection")
                return
            self.reply("MCU&U&WIFIMCU&U&" + str(len(self.staged)))  # two in one notification
            self.writer.write(self.staged + (END_MARKER if self.send_marker else b""))
            await self.writer.drain()
            self.reply("MCU&OFF")
        elif command.startswith("U&"):
            _, _, ts = command.split("&")
            self.staged = self.files[ts]
            self.reply(f"MCU&U&{len(self.staged)}")


class FakeCommander(PocketCommander):
    """The real commander's message handling, with the device faked behind _write."""

    def __init__(self, device: FakeDevice):
        super().__init__("AA:BB:CC:DD:EE:FF")
        self.client = SimpleNamespace(is_connected=True)
        self.device = device
        device.reply = lambda text: self._on_response(0, bytearray(text.encode()))

    async def _write(self, command: str) -> None:
        await self.device.handle(command)

    async def start_audio_sink(self) -> None:
        pass

    async def stop_audio_sink(self) -> None:
        pass


class FakeHostWifi:
    def __init__(self, device: FakeDevice):
        self.device = device
        self.calls: list[str] = []

    async def setup(self):
        self.calls.append("setup")

    async def prepare(self, ssid, password):
        self.calls.append(f"prepare {ssid} {password}")

    async def join(self, ssid, deadline):
        self.calls.append("join")
        await asyncio.sleep(0.05)
        self.device.status = WIFI_STATUS_CLIENT_JOINED
        return True

    async def leave(self):
        self.calls.append("leave")

    async def restore(self):
        self.calls.append("restore")


def session_for(device, host_wifi, **kw):
    return WifiSession(FakeCommander(device), host_wifi, host="127.0.0.1", port=device.port,
                       heartbeat=0.05, status_interval=0.02, switch_delay=0.01,
                       first_byte_timeout=2, idle_timeout=2, **kw)


@pytest.fixture(autouse=True)
def fast_cycle(monkeypatch):
    """WifiSession.cycle waits 2 s for the AP to go down; not needed here."""
    real_sleep = asyncio.sleep

    async def sleep(seconds, *args, **kwargs):
        return await real_sleep(min(seconds, 0.05), *args, **kwargs)

    monkeypatch.setattr("pocket_libre.wifi.asyncio.sleep", sleep)


# ── The session ─────────────────────────────────


@pytest.mark.asyncio
async def test_three_files_restart_the_ap_after_two(tmp_path):
    device = FakeDevice(free_port())
    host_wifi = FakeHostWifi(device)
    names = list(FILES)
    async with session_for(device, host_wifi) as session:
        results = [await session.download(Recording("2026-10-03", ts, 0), tmp_path / f"{ts}.mp3")
                   for ts in names]

    for ts, result in zip(names, results, strict=True):
        assert (tmp_path / f"{ts}.mp3").read_bytes() == FILES[ts]  # marker stripped
        assert result.marker_ok and result.size == len(FILES[ts])
    assert device.ap_starts == 2
    assert device.violations == []
    assert host_wifi.calls == ["setup", "prepare PKT01_GREY_TEST abcd1234", "join",
                               "leave", "join", "restore"]
    assert device.sent[-1] == "WIFIC"  # the AP is lowered on the way out
    assert not list(tmp_path.glob("*.part"))


@pytest.mark.asyncio
async def test_switch_follows_the_file_request(tmp_path):
    """The app's order: U&<file> first, then U&WIFI — never the other way."""
    device = FakeDevice(free_port())
    async with session_for(device, FakeHostWifi(device)) as session:
        await session.download(Recording("2026-10-03", "20261003160116", 0), tmp_path / "a.mp3")
    commands = [c for c in device.sent if c not in ("WIFIS", "WPING")]
    assert commands == ["WIFIO", "WIFI", "U&2026-10-03&20261003160116", "U&WIFI", "WIFIC"]


@pytest.mark.asyncio
async def test_failure_still_lowers_the_ap_and_restores_wifi(tmp_path):
    device = FakeDevice(free_port())
    handle = device.handle

    async def silent_switch(command):  # the device never sends the file
        if command == "U&WIFI":
            device.sent.append(command)
            return
        await handle(command)

    device.handle = silent_switch
    host_wifi = FakeHostWifi(device)
    with pytest.raises(WifiTransferError):
        async with session_for(device, host_wifi) as session:
            session.first_byte_timeout = 0.2
            await session.download(Recording("2026-10-03", "20261003160116", 0), tmp_path / "a.mp3")
    assert "WIFIC" in device.sent
    assert host_wifi.calls[-1] == "restore"
    assert not (tmp_path / "a.mp3").exists() and not list(tmp_path.glob("*.part"))


@pytest.mark.asyncio
async def test_missing_marker_is_reported_but_keeps_the_file(tmp_path):
    device = FakeDevice(free_port(), send_marker=False)
    async with session_for(device, FakeHostWifi(device)) as session:
        result = await session.download(Recording("2026-10-03", "20261003160116", 0),
                                        tmp_path / "a.mp3")
    assert not result.marker_ok
    assert (tmp_path / "a.mp3").read_bytes() == FILES["20261003160116"]


# ── receive_file ────────────────────────────────


def _reader(data: bytes, eof: bool = True) -> asyncio.StreamReader:
    reader = asyncio.StreamReader()
    reader.feed_data(data)
    if eof:
        reader.feed_eof()
    return reader


@pytest.mark.asyncio
async def test_receive_file_splits_the_marker_off(tmp_path):
    body = mp3(5000)
    ok = await receive_file(_reader(body + END_MARKER), len(body), tmp_path / "f.mp3")
    assert ok and (tmp_path / "f.mp3").read_bytes() == body


@pytest.mark.asyncio
async def test_receive_file_rejects_a_short_transfer(tmp_path):
    with pytest.raises(WifiTransferError, match="closed the connection"):
        await receive_file(_reader(mp3(100)), 5000, tmp_path / "f.mp3")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_receive_file_times_out_without_data(tmp_path):
    with pytest.raises(WifiTransferError, match="no data"):
        await receive_file(_reader(b"", eof=False), 10, tmp_path / "f.mp3", first_byte_timeout=0.1)
    assert list(tmp_path.iterdir()) == []


# ── Protocol details ────────────────────────────


def test_split_messages_handles_concatenated_replies():
    assert split_messages("MCU&WIFIOMCU&OFF") == ["MCU&WIFIO", "MCU&OFF"]
    assert split_messages("MCU&U&123\0\0") == ["MCU&U&123"]


@pytest.mark.asyncio
async def test_wait_for_message_sees_replies_after_the_mark():
    device = FakeDevice(free_port())
    cmd = FakeCommander(device)
    device.reply("MCU&OFF")  # before the mark: ignored
    since = cmd.mark()
    asyncio.get_running_loop().call_later(0.05, device.reply, "MCU&U&WIFIMCU&OFF")
    assert await cmd.wait_for_message("OFF", since, timeout=1) == ""
    assert await cmd.wait_for_message("U", since, timeout=0.1, accept=str.isdigit) is None


def test_status_codes_match_observed_order():
    """3 right after WIFIO, 2 waiting for a client, 1 once a client joined."""
    assert (WIFI_STATUS_STARTING, WIFI_STATUS_WAITING_FOR_CLIENT, WIFI_STATUS_CLIENT_JOINED) == (3, 2, 1)
    assert TRANSFER_PORT == 8475
    assert END_MARKER == bytes.fromhex("ba5a028f04ba5a028f04")


# ── Host WiFi helpers ───────────────────────────


def test_split_terse_unescapes_colons():
    assert split_terse(r"Home\:Net:1234:802-11-wireless:wlan0") == \
        ["Home:Net", "1234", "802-11-wireless", "wlan0"]


def test_netsh_parsing_english_and_german():
    en = "    Name                   : Wi-Fi\n    State                  : connected\n" \
         "    SSID                   : Home\n    Profile                : Home\n"
    de = "    Name                   : WLAN\n    Status                 : Verbunden\n" \
         "    SSID                   : Zuhause\n    Profil                 : Zuhause\n"
    assert parse_netsh_interfaces(en) == [{"name": "Wi-Fi", "state": "connected",
                                           "ssid": "Home", "profile": "Home"}]
    assert parse_netsh_interfaces(de)[0]["profile"] == "Zuhause"


def test_windows_profile_is_hidden_and_escaped():
    xml = windows_profile("A&B", "p<w>")
    assert "<name>A&amp;B</name>" in xml and "p&lt;w&gt;" in xml
    assert "<nonBroadcast>true</nonBroadcast>" in xml
