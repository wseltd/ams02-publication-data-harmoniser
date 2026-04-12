"""JSON exporter for AMS-02 measurement data.

Provides AMS02Encoder for numpy/datetime serialisation and export_json
for writing harmonised datasets to JSON files with full float64 precision.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any

import numpy as np


class AMS02Encoder(json.JSONEncoder):
    """JSON encoder that handles numpy arrays, numpy scalars, and datetimes.

    Converts numpy ndarrays to nested Python lists (preserving float64 precision),
    numpy integer/floating scalars to native Python types, and datetime objects
    to ISO 8601 strings.  Delegates all other types to the base encoder, which
    raises TypeError for unhandled types.
    """

    def default(self, obj: Any) -> Any:
        """Serialise numpy and datetime objects to JSON-compatible types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()
        if isinstance(obj, datetime.date):
            return obj.isoformat()
        return super().default(obj)


def export_json(
    data: dict[str, Any],
    path: Path,
    *,
    indent: int | None = 2,
) -> Path:
    """Write a data dictionary to a JSON file using AMS02Encoder.

    Args:
        data: Dictionary to serialise.  May contain numpy arrays, numpy
              scalars, and datetime objects.
        path: Destination file path.  Parent directory must exist.
        indent: JSON indentation level.  Pass None for compact output.

    Returns:
        Resolved Path to the written JSON file.
    """
    resolved = Path(path).resolve()
    with open(resolved, "w", encoding="utf-8") as fh:
        json.dump(data, fh, cls=AMS02Encoder, indent=indent)
    return resolved
