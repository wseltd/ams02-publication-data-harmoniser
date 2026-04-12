"""Canonical data models for AMS-02 publication data."""

from __future__ import annotations

import enum
from datetime import datetime

from pydantic import BaseModel


class UncertaintyLabel(enum.Enum):
    """How an uncertainty value was obtained.

    PUBLISHED — taken directly from the source table.
    DERIVED   — result of a transformation (symmetrisation, heuristic split).
    ASSUMED   — filled from a default because the source did not provide it.
    """

    PUBLISHED = "published"
    DERIVED = "derived"
    ASSUMED = "assumed"

    def __repr__(self) -> str:
        return f"UncertaintyLabel.{self.name}"


class Species(BaseModel):
    """Physical species properties needed for axis conversions.

    Stores charge, mass, and mass number so that rigidity-to-kinetic-energy
    conversions can be performed without hard-coding particle properties
    at the call site.
    """

    name: str
    charge_z: int
    mass_gev: float
    mass_number_a: int

    def __repr__(self) -> str:
        return (
            f"Species(name={self.name!r}, Z={self.charge_z}, "
            f"mass={self.mass_gev}, A={self.mass_number_a})"
        )


class Measurement(BaseModel):
    """A single data point from an AMS-02 publication table.

    Immutable by convention — harmonisation produces new instances
    rather than mutating existing ones.
    """

    energy_low: float = 0.0
    energy_high: float = 0.0
    energy_mid: float = 0.0
    value: float = 0.0
    unit: str = "GeV"
    axis_type: str = "kinetic_energy_per_nucleon"
    species: str = "PROTON"
    stat_error_low: float | None = None
    stat_error_high: float | None = None
    sys_error_low: float | None = None
    sys_error_high: float | None = None
    stat_err_pos: float | None = None
    stat_err_neg: float | None = None
    sys_err_pos: float | None = None
    sys_err_neg: float | None = None
    stat_err_label: UncertaintyLabel | None = None
    sys_err_label: UncertaintyLabel | None = None
    time_start: str | int | float | None = None
    time_end: str | int | float | None = None
    time_start_utc: datetime | None = None
    time_end_utc: datetime | None = None

    def __repr__(self) -> str:
        return (
            f"Measurement(energy=[{self.energy_low}, {self.energy_high}], "
            f"value={self.value}, unit={self.unit!r}, "
            f"axis={self.axis_type!r}, species={self.species!r})"
        )


class ProvenanceRecord(BaseModel):
    """Tracks the origin of an ingested dataset.

    Each field records one axis of provenance so that any dataset can be
    traced back to its source paper, table, and file.
    """

    paper_doi: str
    table_id: str
    file_url: str
    ingested_at: str
    source_type: str

    def __repr__(self) -> str:
        return (
            f"ProvenanceRecord(doi={self.paper_doi!r}, "
            f"table={self.table_id!r}, source={self.source_type!r})"
        )
