"""Versions are yyyymmdd.N, the source is the single truth, and the release tag is v<version>."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

import localforge
from localforge import cli

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("set_version", ROOT / "scripts" / "set_version.py")
sv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sv)


def test_the_source_version_is_a_calendar_version():
    assert sv.valid(localforge.__version__)


@pytest.mark.parametrize("good", ["20261013.1", "20261231.12"])
def test_valid_versions(good):
    assert sv.valid(good)


@pytest.mark.parametrize("bad", ["0.1.0", "20261013", "20261013.0", "20261313.1", "2026-10-13.1", "v20261013.1", "20261013.1.0"])
def test_invalid_versions(bad):
    assert not sv.valid(bad)


def test_next_version_counts_up_within_a_day_and_restarts_on_a_new_day():
    import datetime

    today = datetime.date(2026, 10, 13)
    assert sv.next_version(today, now="20261012.4") == "20261013.1"
    assert sv.next_version(today, now="20261013.1") == "20261013.2"
    assert sv.next_version(today, now="20261013.9") == "20261013.10"  # numbers, not text: .10 is after .9


def test_numbers_compare_as_numbers():
    assert sv._key("20261013.10") > sv._key("20261013.9")
    assert sv._key("20261014.1") > sv._key("20261013.99")


def test_every_file_carrying_the_version_agrees():
    assert sv.verify() == []


def test_the_semver_files_get_a_trailing_zero():
    version = localforge.__version__
    assert json.loads((ROOT / "desktop/src-tauri/tauri.conf.json").read_text())["version"] == version + ".0"
    assert json.loads((ROOT / "desktop/package.json").read_text())["version"] == version + ".0"


def test_the_package_version_comes_from_the_source():
    pyproject = (ROOT / "pyproject.toml").read_text()
    assert 'dynamic = ["version"]' in pyproject and 'path = "src/localforge/__init__.py"' in pyproject


def test_dash_dash_version_prints_it():
    result = CliRunner().invoke(cli.app, ["--version"])
    assert result.exit_code == 0 and result.output.strip() == f"localforge {localforge.__version__}"


def test_a_push_must_raise_the_version(tmp_path):
    repo = tmp_path / "r"
    (repo / "src/localforge").mkdir(parents=True)
    (repo / "src/localforge/__init__.py").write_text(f'__version__ = "{localforge.__version__}"\n')
    run = lambda *a: subprocess.run(["git", *a], cwd=repo, check=True, capture_output=True)
    run("init", "-q")
    run("-c", "user.email=a@b", "-c", "user.name=t", "add", "-A")
    run("-c", "user.email=a@b", "-c", "user.name=t", "commit", "-qm", "x")
    # same version as the ref: refused; a higher one passes
    original = sv.ROOT
    sv.ROOT = repo
    try:
        assert "already at" in sv.newer_than("HEAD")
        (repo / "src/localforge/__init__.py").write_text('__version__ = "99990101.1"\n')
        sv.INIT = repo / "src/localforge/__init__.py"
        assert sv.newer_than("HEAD") is None
    finally:
        sv.ROOT = original
        sv.INIT = original / "src/localforge/__init__.py"


def test_the_release_is_named_after_the_version():
    release = (ROOT / ".github/workflows/release.yml").read_text()
    assert "tagName: ${{ steps.version.outputs.tag }}" in release and "inputs.tag" not in release
    assert (ROOT / ".github/workflows/version-check.yml").exists()


def test_the_backend_reports_its_version_when_ready(tmp_path):
    from localforge import serve

    assert serve.__version__ == localforge.__version__
