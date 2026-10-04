"""CLI behaviour for the usb command, with the BLE commander faked out."""

import pytest
from click.testing import CliRunner

from pocket_libre import cli as cli_module


class FakeCommander:
    """Stands in for PocketCommander; records which USB calls were made."""

    def __init__(self, auth_ok=True, set_state=None, get_state=None):
        self.auth_ok = auth_ok
        self.set_state = set_state
        self.get_state = get_state
        self.calls = []

    def __call__(self, address):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def authenticate(self, session_key):
        return self.auth_ok

    async def set_usb(self, enabled):
        self.calls.append(("set", enabled))
        return self.set_state

    async def get_usb(self):
        self.calls.append(("get",))
        return self.get_state


@pytest.fixture
def run_usb(monkeypatch):
    monkeypatch.setattr(cli_module, "load_config", lambda: {})

    def _run(fake, *args):
        monkeypatch.setattr(cli_module, "PocketCommander", fake)
        return CliRunner().invoke(
            cli_module.cli,
            ["usb", *args, "--address", "AA:BB:CC:DD:EE:FF", "--key", "00" * 16],
        )

    return _run


def test_usb_auth_failure_exits_nonzero(run_usb):
    fake = FakeCommander(auth_ok=False)
    result = run_usb(fake, "on")
    assert result.exit_code == 1
    assert "Authentication failed" in result.output
    assert fake.calls == []


def test_usb_on_uses_state_from_set_reply(run_usb):
    fake = FakeCommander(set_state=True)
    result = run_usb(fake, "on")
    assert result.exit_code == 0
    assert fake.calls == [("set", True)]
    assert "USB mass storage: on" in result.output


def test_usb_set_falls_back_to_get_when_reply_has_no_state(run_usb):
    fake = FakeCommander(set_state=None, get_state=False)
    result = run_usb(fake, "off")
    assert result.exit_code == 0
    assert fake.calls == [("set", False), ("get",)]
    assert "USB mass storage: off" in result.output


def test_usb_set_reports_mismatch(run_usb):
    fake = FakeCommander(set_state=False)
    result = run_usb(fake, "on")
    assert result.exit_code == 1
    assert "did not switch USB on" in result.output


def test_usb_status_only_queries(run_usb):
    fake = FakeCommander(get_state=True)
    result = run_usb(fake)
    assert result.exit_code == 0
    assert fake.calls == [("get",)]


def test_usb_status_unknown_exits_nonzero(run_usb):
    fake = FakeCommander(get_state=None)
    result = run_usb(fake, "status")
    assert result.exit_code == 1
    assert "did not report a USB state" in result.output
