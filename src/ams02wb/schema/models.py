"""Canonical data models for AMS-02 publication data."""

from pydantic import BaseModel


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
