"""Canonical data models for AMS-02 publication data."""

from __future__ import annotations

from pydantic import BaseModel


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

    energy_low: float
    energy_high: float
    energy_mid: float
    value: float
    unit: str = "GeV"
    axis_type: str = "kinetic_energy_per_nucleon"
    species: str = "PROTON"
    stat_error_low: float | None = None
    stat_error_high: float | None = None
    sys_error_low: float | None = None
    sys_error_high: float | None = None

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
