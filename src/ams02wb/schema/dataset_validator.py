"""Dataset validation runner for AMS-02 harmonised JSON files.

Reads JSON dataset files (each containing a list of records), validates
each record using the schema validators, and reports per-file results.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Literal, TypedDict

from ams02wb.schema.validators import ValidationFinding, validate_record


class FileResult(TypedDict):  # noqa: F401, E501 — TypedDict; bracket access in validate_dataset_dir prevents dataclass conversion
    """Validation result for a single dataset file."""

    file_path: str
    status: Literal["PASS", "FAIL"]
    findings: List[ValidationFinding]

    def __repr__(self) -> str:
        return (
            f"FileResult(file_path={self.get('file_path')!r}, "
            f"status={self.get('status')!r})"
        )


def validate_dataset_file(path: Path) -> FileResult:
    """Validate a single JSON dataset file.

    The file must contain a JSON array of record dicts. Each record is
    validated with validate_record. Status is FAIL if any record has
    findings, PASS otherwise.

    Args:
        path: Path to a JSON file containing a list of records.

    Returns:
        A FileResult dict with file_path, status, and all findings.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
        TypeError: If the top-level JSON value is not a list.
    """
    with open(path) as fh:
        data: Any = json.load(fh)

    if not isinstance(data, list):
        raise TypeError(
            f"Expected a JSON array of records, got {type(data).__name__}"
        )

    all_findings: List[ValidationFinding] = []
    for record in data:
        all_findings.extend(validate_record(record))

    return FileResult(
        file_path=str(path),
        status="FAIL" if all_findings else "PASS",
        findings=all_findings,
    )


def validate_dataset_dir(input_dir: Path) -> List[FileResult]:
    """Validate all JSON files in a directory.

    Globs *.json non-recursively, validates each file, and returns
    results sorted by file_path.

    Args:
        input_dir: Directory containing JSON dataset files.

    Returns:
        List of FileResult dicts, sorted by file_path.
        Empty list if no JSON files are found.

    Raises:
        FileNotFoundError: If input_dir does not exist.
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"Directory does not exist: {input_dir}")

    json_files = sorted(input_dir.glob("*.json"))

    if not json_files:
        return []

    return sorted(
        [validate_dataset_file(f) for f in json_files],
        key=lambda r: r["file_path"],
    )
