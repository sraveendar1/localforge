"""
Cover the front-end-agnostic trust decision API shared by the terminal prompt and the desktop UI.
"""

from pathlib import Path
from unittest.mock import MagicMock
import pytest
from localforge import trust


@pytest.fixture(autouse=True)
def _clean_trust(tmp_path):
    """Never leave the shared trusted_folders.json dirty."""
    yield
    trust.untrust(tmp_path)


@pytest.mark.untrusted
def test_apply_choice_yes(tmp_path):
    assert trust.apply_choice(tmp_path, "yes")
    assert trust.is_trusted(tmp_path)


@pytest.mark.untrusted
def test_apply_choice_no(tmp_path):
    assert not trust.apply_choice(tmp_path, "no")
    assert not trust.is_trusted(tmp_path)


@pytest.mark.untrusted
def test_apply_choice_unknown_choice(tmp_path):
    with pytest.raises(ValueError):
        trust.apply_choice(tmp_path, "unknown")


@pytest.mark.untrusted
def test_decide_already_trusted(tmp_path):
    trust.trust(tmp_path)
    ask_mock = MagicMock()
    ask_mock.return_value = "yes"
    assert trust.decide(tmp_path, ask_mock)
    ask_mock.assert_not_called()


@pytest.mark.untrusted
def test_decide_asks_and_trusts(tmp_path):
    ask_mock = MagicMock()
    ask_mock.return_value = "yes"
    assert trust.decide(tmp_path, ask_mock)
    assert trust.is_trusted(tmp_path)


@pytest.mark.untrusted
def test_decide_asks_and_untrusts(tmp_path):
    ask_mock = MagicMock()
    ask_mock.return_value = "no"
    assert not trust.decide(tmp_path, ask_mock)
    assert not trust.is_trusted(tmp_path)


@pytest.mark.untrusted
def test_looks_like_home(tmp_path):
    assert trust.looks_like_home(Path.home())
    assert trust.looks_like_home(Path("/"))
    assert not trust.looks_like_home(tmp_path)
