"""Tests for ams02wb.likelihood.fitready — fit-ready dataset builder."""

from __future__ import annotations

import numpy as np
import pytest

from ams02wb.likelihood.fitready import (
    build_fit_dataset,
    build_fit_dataset_from_measurement,
)
from ams02wb.schema.models import Measurement


# -- Helpers ------------------------------------------------------------------

def _make_dataset(n: int = 3, **overrides) -> dict:
    """Build a minimal valid dataset, allowing per-call overrides."""
    defaults = {
        "y": np.arange(n, dtype=np.float64),
        "x": np.linspace(1.0, 10.0, n),
        "covariance": np.eye(n),
        "uncertainty_label": "published",
        "mode": "diagonal",
        "provenance": {"paper": "test-doi"},
    }
    defaults.update(overrides)
    return build_fit_dataset(**defaults)


# -- Key presence and structure -----------------------------------------------

class TestFitDatasetStructure:

    def test_fit_dataset_keys_present(self):
        """All required keys must appear in the returned dict."""
        result = _make_dataset()
        expected_keys = {
            "y", "x", "covariance", "uncertainty_label", "mode",
            "provenance", "species", "x_axis_type", "y_unit", "n_points",
        }
        assert set(result.keys()) == expected_keys

    def test_fit_dataset_n_points_matches(self):
        """n_points must equal len(y)."""
        for n in (1, 5, 20):
            result = _make_dataset(n=n)
            assert result["n_points"] == n
            assert result["n_points"] == len(result["y"])

    def test_fit_dataset_provenance_preserved(self):
        """Provenance dict must be returned as-is, not copied or stripped."""
        prov = {"paper": "doi:10.1234", "table": "T3", "extra": [1, 2]}
        result = _make_dataset(provenance=prov)
        assert result["provenance"] is prov

    def test_optional_metadata_defaults_to_empty_string(self):
        """species, x_axis_type, y_unit default to '' when not supplied."""
        result = _make_dataset()
        assert result["species"] == ""
        assert result["x_axis_type"] == ""
        assert result["y_unit"] == ""

    def test_optional_metadata_passed_through(self):
        """Explicit species/x_axis_type/y_unit are preserved."""
        result = _make_dataset(
            species="proton", x_axis_type="rigidity", y_unit="m^-2 sr^-1 s^-1 GV^-1"
        )
        assert result["species"] == "proton"
        assert result["x_axis_type"] == "rigidity"
        assert result["y_unit"] == "m^-2 sr^-1 s^-1 GV^-1"


# -- Shape validation ---------------------------------------------------------

class TestFitDatasetShapeValidation:

    def test_fit_dataset_shape_mismatch_raises(self):
        """Mismatched y/x lengths must raise ValueError."""
        with pytest.raises(ValueError, match="len\\(y\\)=3 but len\\(x\\)=2"):
            build_fit_dataset(
                y=np.array([1.0, 2.0, 3.0]),
                x=np.array([10.0, 20.0]),
                covariance=np.eye(3),
                uncertainty_label="published",
                mode="diagonal",
                provenance={},
            )

    def test_covariance_shape_mismatch_raises(self):
        """Covariance matrix with wrong dimensions must raise ValueError."""
        with pytest.raises(ValueError, match="covariance is \\(2, 2\\)"):
            build_fit_dataset(
                y=np.array([1.0, 2.0, 3.0]),
                x=np.array([1.0, 2.0, 3.0]),
                covariance=np.eye(2),
                uncertainty_label="published",
                mode="diagonal",
                provenance={},
            )

    def test_single_point_dataset(self):
        """A single-point dataset is valid and should not raise."""
        result = _make_dataset(n=1)
        assert result["n_points"] == 1
        assert result["covariance"].shape == (1, 1)

    def test_zero_length_arrays(self):
        """Zero-length arrays are shape-consistent and should not raise."""
        result = build_fit_dataset(
            y=np.array([]),
            x=np.array([]),
            covariance=np.empty((0, 0)),
            uncertainty_label="published",
            mode="diagonal",
            provenance={},
        )
        assert result["n_points"] == 0


# -- Label validation ---------------------------------------------------------

class TestFitDatasetLabelValidation:

    def test_fit_dataset_invalid_label_raises(self):
        """An unrecognised uncertainty_label must raise ValueError."""
        with pytest.raises(ValueError, match="Invalid uncertainty_label"):
            _make_dataset(uncertainty_label="guessed")

    def test_all_valid_labels_accepted(self):
        """Each canonical label string must be accepted without error."""
        for label in ("published", "derived", "assumed"):
            result = _make_dataset(uncertainty_label=label)
            assert result["uncertainty_label"] == label

    def test_label_case_sensitivity(self):
        """Labels are case-sensitive — 'Published' is not valid."""
        with pytest.raises(ValueError, match="Invalid uncertainty_label"):
            _make_dataset(uncertainty_label="Published")


# -- build_fit_dataset_from_measurement ---------------------------------------

class TestFitDatasetFromMeasurement:

    def test_fit_dataset_from_measurement(self):
        """Constructs a valid dataset from Measurement objects."""
        measurements = [
            Measurement(value=1.5, energy_mid=10.0, species="PROTON"),
            Measurement(value=2.5, energy_mid=20.0, species="PROTON"),
            Measurement(value=3.5, energy_mid=30.0, species="PROTON"),
        ]
        cov = np.eye(3)
        prov = {"paper": "doi:test"}

        result = build_fit_dataset_from_measurement(
            measurements=measurements,
            covariance=cov,
            uncertainty_label="published",
            mode="diagonal",
            provenance=prov,
        )

        np.testing.assert_array_equal(result["y"], [1.5, 2.5, 3.5])
        np.testing.assert_array_equal(result["x"], [10.0, 20.0, 30.0])
        assert result["n_points"] == 3
        assert result["species"] == "PROTON"
        assert result["provenance"] is prov

    def test_from_measurement_covariance_mismatch_raises(self):
        """Covariance mismatch still caught when building from measurements."""
        measurements = [
            Measurement(value=1.0, energy_mid=5.0),
            Measurement(value=2.0, energy_mid=10.0),
        ]
        with pytest.raises(ValueError, match="covariance"):
            build_fit_dataset_from_measurement(
                measurements=measurements,
                covariance=np.eye(3),
                uncertainty_label="published",
                mode="diagonal",
                provenance={},
            )
