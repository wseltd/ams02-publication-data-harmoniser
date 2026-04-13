"""Tests for the dataset validation runner module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ams02wb.schema.dataset_validator import (
    validate_dataset_dir,
    validate_dataset_file,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_VALID_RECORD = {
    "x_centre": 10.0,
    "x_min": 5.0,
    "x_max": 15.0,
    "y_value": 1.23,
    "stat_err": 0.01,
}

_INVALID_RECORD = {
    "x_centre": -1.0,
    "x_min": 5.0,
    "x_max": 15.0,
    "y_value": 1.23,
    "stat_err": 0.01,
}


def _write_json(path: Path, data: object) -> Path:
    path.write_text(json.dumps(data))
    return path


# ---------------------------------------------------------------------------
# validate_dataset_file
# ---------------------------------------------------------------------------


class TestValidateDatasetFile:
    """Tests for validate_dataset_file."""

    def test_valid_file_returns_pass(self, tmp_path: Path) -> None:
        fpath = _write_json(tmp_path / "ok.json", [_VALID_RECORD])
        result = validate_dataset_file(fpath)
        assert result["status"] == "PASS"
        assert result["findings"] == []
        assert result["file_path"] == str(fpath)

    def test_invalid_record_returns_fail(self, tmp_path: Path) -> None:
        fpath = _write_json(tmp_path / "bad.json", [_INVALID_RECORD])
        result = validate_dataset_file(fpath)
        assert result["status"] == "FAIL"
        assert len(result["findings"]) > 0

    def test_mixed_records_returns_fail(self, tmp_path: Path) -> None:
        fpath = _write_json(tmp_path / "mix.json", [_VALID_RECORD, _INVALID_RECORD])
        result = validate_dataset_file(fpath)
        assert result["status"] == "FAIL"

    def test_empty_list_returns_pass(self, tmp_path: Path) -> None:
        fpath = _write_json(tmp_path / "empty.json", [])
        result = validate_dataset_file(fpath)
        assert result["status"] == "PASS"
        assert result["findings"] == []

    def test_non_list_json_raises_type_error(self, tmp_path: Path) -> None:
        fpath = _write_json(tmp_path / "obj.json", {"not": "a list"})
        with pytest.raises(TypeError, match="Expected a JSON array") as exc_info:
            validate_dataset_file(fpath)
        assert "dict" in str(exc_info.value)

    def test_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
        missing = tmp_path / "nope.json"
        with pytest.raises(FileNotFoundError) as exc_info:
            validate_dataset_file(missing)
        assert str(missing) in str(exc_info.value)


# ---------------------------------------------------------------------------
# validate_dataset_dir
# ---------------------------------------------------------------------------


class TestValidateDatasetDir:
    """Tests for validate_dataset_dir."""

    def test_nonexistent_dir_raises(self, tmp_path: Path) -> None:
        missing_dir = tmp_path / "missing"
        with pytest.raises(FileNotFoundError, match="does not exist") as exc_info:
            validate_dataset_dir(missing_dir)
        assert str(missing_dir) in str(exc_info.value)

    def test_empty_dir_returns_empty_list(self, tmp_path: Path) -> None:
        assert validate_dataset_dir(tmp_path) == []

    def test_multiple_files_sorted_by_path(self, tmp_path: Path) -> None:
        _write_json(tmp_path / "b.json", [_VALID_RECORD])
        _write_json(tmp_path / "a.json", [_VALID_RECORD])
        results = validate_dataset_dir(tmp_path)
        assert len(results) == 2
        assert results[0]["file_path"] < results[1]["file_path"]

    def test_mixed_pass_and_fail(self, tmp_path: Path) -> None:
        _write_json(tmp_path / "good.json", [_VALID_RECORD])
        _write_json(tmp_path / "bad.json", [_INVALID_RECORD])
        results = validate_dataset_dir(tmp_path)
        statuses = {r["file_path"].split("/")[-1]: r["status"] for r in results}
        assert statuses["good.json"] == "PASS"
        assert statuses["bad.json"] == "FAIL"

    def test_non_json_files_ignored(self, tmp_path: Path) -> None:
        _write_json(tmp_path / "data.json", [_VALID_RECORD])
        (tmp_path / "readme.txt").write_text("not json")
        results = validate_dataset_dir(tmp_path)
        assert len(results) == 1
