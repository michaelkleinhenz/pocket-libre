"""BLE download: disconnects mid-transfer must retry, never save a short file."""

import asyncio

import pytest
from bleak.exc import BleakError
from click.testing import CliRunner

from pocket_libre import commands
from pocket_libre.commands import PocketCommander, Recording, download_with_retry
from pocket_libre.protocol import MP3_SYNC_WORD

REC = Recording("2026-10-03", "20261003090502", 0)


class _FakeClient:
    """Feeds audio chunks, optionally dropping the link part-way through.

    After a disconnect, bleak raises from stop_notify because the services
    are gone — the real failure seen on firmware 1.7.
    """

    def __init__(self, cmd, chunks, disconnect=False):
        self.cmd = cmd
        self.chunks = chunks
        self.disconnect = disconnect

    async def start_notify(self, char, callback):
        def deliver():
            for chunk in self.chunks:
                callback(None, bytearray(chunk))
            if self.disconnect:
                self.cmd._on_disconnect(self)

        # Arrive once download_ble is waiting, as notifications do.
        asyncio.get_running_loop().call_soon(deliver)

    async def stop_notify(self, char):
        if self.cmd._disconnected:
            raise BleakError("Service Discovery has not been performed yet")


def _commander(chunks, expected, disconnect=False):
    cmd = PocketCommander("AA:BB:CC:DD:EE:FF")
    cmd.client = _FakeClient(cmd, chunks, disconnect)

    async def fake_send(command, verbose=False):
        return [f"MCU&U&{expected}"]

    cmd._send = fake_send
    return cmd


# ── download_ble ────────────────────────────────


@pytest.mark.asyncio
async def test_download_ble_survives_disconnect():
    cmd = _commander([MP3_SYNC_WORD + b"\x00" * 98], expected=1000, disconnect=True)
    data = await cmd.download_ble(REC)
    assert len(data) == 100
    assert cmd.last_expected_size == 1000


@pytest.mark.asyncio
async def test_download_ble_records_expected_size_on_success():
    payload = MP3_SYNC_WORD + b"\x00" * 98
    cmd = _commander([payload], expected=len(payload))
    assert await cmd.download_ble(REC) == payload
    assert cmd.last_expected_size == len(payload)


# ── download_with_retry ─────────────────────────


class _FakeCommander:
    """Stands in for a fresh BLE connection per attempt."""

    attempts: list = []

    def __init__(self, address):
        self._disconnected = False
        self.last_expected_size = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def authenticate(self, key):
        return True

    async def download_ble(self, recording, progress_callback=None):
        data, expected, disconnected = _FakeCommander.attempts.pop(0)
        self.last_expected_size = expected
        self._disconnected = disconnected
        return data


@pytest.fixture
def fake_commander(monkeypatch):
    monkeypatch.setattr(commands, "PocketCommander", _FakeCommander)
    return _FakeCommander


FULL = MP3_SYNC_WORD + b"\x00" * 998


@pytest.mark.asyncio
async def test_retry_after_disconnect(fake_commander):
    fake_commander.attempts = [(FULL[:400], 1000, True), (FULL, 1000, False)]
    data = await download_with_retry("addr", "key", REC, retry_delay=0)
    assert data == FULL
    assert fake_commander.attempts == []


@pytest.mark.asyncio
async def test_retry_when_short_of_announced_size(fake_commander):
    """A stalled transfer ends without a disconnect; the device's announced
    size is what reveals it — the duration estimate is 0 for `download`."""
    fake_commander.attempts = [(FULL[:900], 1000, False), (FULL, 1000, False)]
    assert await download_with_retry("addr", "key", REC, retry_delay=0) == FULL


@pytest.mark.asyncio
async def test_gives_up_with_nothing_rather_than_partial(fake_commander):
    fake_commander.attempts = [(FULL[:400], 1000, True)] * 3
    assert await download_with_retry("addr", "key", REC, retry_delay=0) == b""


# ── download-all ────────────────────────────────


def test_download_all_never_writes_a_failed_download(tmp_path, monkeypatch):
    from pocket_libre import cli

    class _Lister(_FakeCommander):
        async def list_all_recordings(self):
            return [REC]

    async def failing_download(*args, **kwargs):
        return b""

    monkeypatch.setattr(cli, "PocketCommander", _Lister)
    monkeypatch.setattr(commands, "download_with_retry", failing_download)
    monkeypatch.setattr(cli, "load_config", lambda: {})

    result = CliRunner().invoke(cli.cli, [
        "download-all", "--address", "addr", "--key", "k" * 16,
        "--output-dir", str(tmp_path),
    ])

    assert result.exit_code == 0, result.output
    assert not (tmp_path / REC.date / REC.filename).exists()
