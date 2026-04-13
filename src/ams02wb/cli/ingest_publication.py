"""CLI command: ingest-publication — fetch, parse, and store a single publication."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import click
import requests  # type: ignore[import-untyped]

from ams02wb.crawler import CrawlerError, INDEX_URL
from ams02wb.crawler.attachment_resolver import resolve_attachments
from ams02wb.crawler.index_loader import PublicationEntry, load_publication_index
from ams02wb.crawler.page_fetcher import fetch_publication_page

logger = logging.getLogger(__name__)


def _parse_csv_to_measurements(
    csv_text: str,
    paper_id: str,
) -> list[dict]:
    """Parse AMS-02 CSV text into a list of Measurement-compatible dicts.

    Handles the common AMS-02 daily flux CSV format with columns like:
    date, rigidity_min, rigidity_max, flux, stat_error, sys_error_td, sys_error_total

    Also handles static spectrum CSVs with rigidity bins + flux + errors.
    """
    lines = csv_text.strip().split("\n")
    if not lines:
        return []

    # Find header line (first non-empty, non-comment line)
    header_line = ""
    header_idx = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            header_line = stripped
            header_idx = i
            break

    if not header_line:
        return []

    # Parse header — normalise to lowercase
    reader = csv.reader([header_line])
    raw_headers = next(reader)
    headers = [h.strip().lower() for h in raw_headers]

    # Detect column roles by keyword matching
    col_date = None
    col_rig_min = None
    col_rig_max = None
    col_flux = None
    col_stat_err = None
    col_sys_err = None

    for i, h in enumerate(headers):
        if "date" in h:
            col_date = i
        elif "rigidity_min" in h or "rig_min" in h or h in ("rigidity_min gv", "r_min"):
            col_rig_min = i
        elif "rigidity_max" in h or "rig_max" in h or h in ("rigidity_max gv", "r_max"):
            col_rig_max = i
        elif "flux" in h and "error" not in h:
            if col_flux is None:  # take first flux column
                col_flux = i
        elif "statistical" in h or "stat" in h:
            if col_stat_err is None:
                col_stat_err = i
        elif ("systematic" in h or "sys" in h) and "total" in h:
            col_sys_err = i
        elif ("systematic" in h or "sys" in h) and col_sys_err is None:
            col_sys_err = i

    # Also try positional detection for the common AMS format:
    # date, rig_min, rig_max, flux, stat_err, sys_err_td, sys_err_total
    if col_flux is None and len(headers) >= 4:
        # Heuristic: if first column has "date" or looks like a date,
        # assume columns are: date, rig_min, rig_max, flux, [errors...]
        if col_date is not None:
            offset = 1
        else:
            offset = 0
        if col_rig_min is None and offset + 1 < len(headers):
            col_rig_min = offset
        if col_rig_max is None and offset + 2 < len(headers):
            col_rig_max = offset + 1
        if col_flux is None and offset + 3 < len(headers):
            col_flux = offset + 2
        if col_stat_err is None and offset + 4 < len(headers):
            col_stat_err = offset + 3
        if col_sys_err is None:
            # Take last error column as sys_err_total
            if len(headers) >= offset + 6:
                col_sys_err = len(headers) - 1
            elif offset + 5 < len(headers):
                col_sys_err = offset + 4

    if col_flux is None:
        logger.warning("Could not identify flux column in CSV for paper %s", paper_id)
        return []

    # Determine species from header text or paper_id
    species = "proton"  # default
    header_joined = " ".join(headers)
    if "helium" in header_joined or "he_flux" in header_joined:
        species = "helium"
    elif "electron" in header_joined or "e-_flux" in header_joined:
        species = "electron"
    elif "positron" in header_joined or "e+_flux" in header_joined:
        species = "positron"
    elif "antiproton" in header_joined or "pbar" in header_joined:
        species = "antiproton"

    # Parse data rows
    measurements: list[dict] = []
    data_text = "\n".join(lines[header_idx + 1 :])
    data_reader = csv.reader(StringIO(data_text))

    for row in data_reader:
        if not row or not any(cell.strip() for cell in row):
            continue

        try:
            flux_val = float(row[col_flux].strip()) if col_flux is not None else 0.0
        except (ValueError, IndexError):
            continue

        try:
            rig_min = float(row[col_rig_min].strip()) if col_rig_min is not None else 0.0
        except (ValueError, IndexError):
            rig_min = 0.0

        try:
            rig_max = float(row[col_rig_max].strip()) if col_rig_max is not None else 0.0
        except (ValueError, IndexError):
            rig_max = 0.0

        try:
            stat_err = float(row[col_stat_err].strip()) if col_stat_err is not None else None
        except (ValueError, IndexError):
            stat_err = None

        try:
            sys_err = float(row[col_sys_err].strip()) if col_sys_err is not None else None
        except (ValueError, IndexError):
            sys_err = None

        date_str = None
        if col_date is not None:
            try:
                date_str = row[col_date].strip()
            except IndexError:
                pass

        m = {
            "energy_low": rig_min,
            "energy_high": rig_max,
            "energy_mid": (rig_min + rig_max) / 2.0 if rig_min and rig_max else 0.0,
            "value": flux_val,
            "unit": "m^-2 sr^-1 s^-1 GV^-1",
            "axis_type": "rigidity",
            "species": species.upper(),
        }

        if stat_err is not None:
            m["stat_error_low"] = stat_err
            m["stat_error_high"] = stat_err
        if sys_err is not None:
            m["sys_error_low"] = sys_err
            m["sys_error_high"] = sys_err
        if date_str:
            m["time_start"] = date_str
            m["time_end"] = date_str

        measurements.append(m)

    return measurements


def run_single_ingest(
    session: requests.Session,
    entry: PublicationEntry,
    output_dir: Path,
) -> dict:
    """Ingest one publication: fetch page, resolve attachments, download and parse.

    Downloads all attachments, parses CSV files into Measurement records,
    and writes a JSON dataset file containing both metadata and parsed data.
    """
    page = fetch_publication_page(session, entry.url)
    attachments = resolve_attachments(page.html, page.base_url)

    attachment_records: list[dict[str, str]] = []
    content_parts: list[bytes] = []
    all_measurements: list[dict] = []

    raw_dir = output_dir / "raw" / entry.paper_id
    raw_dir.mkdir(parents=True, exist_ok=True)

    for att in attachments:
        resp = session.get(att.url)
        resp.raise_for_status()
        raw = resp.content
        content_parts.append(raw)

        # Save raw file
        raw_file = raw_dir / att.filename
        raw_file.write_bytes(raw)

        attachment_records.append({
            "filename": att.filename,
            "url": att.url,
            "content_type": att.content_type,
            "size_bytes": str(len(raw)),
            "local_path": str(raw_file),
        })

        # Parse CSV attachments into measurements
        if att.content_type == "csv":
            try:
                csv_text = raw.decode("utf-8")
                parsed = _parse_csv_to_measurements(csv_text, entry.paper_id)
                all_measurements.extend(parsed)
                logger.info(
                    "Parsed %d measurements from %s", len(parsed), att.filename
                )
            except Exception as exc:
                logger.warning("Failed to parse CSV %s: %s", att.filename, exc)

    hasher = hashlib.sha256()
    for part in content_parts:
        hasher.update(part)
    if not content_parts:
        hasher.update(page.html.encode("utf-8"))

    dataset = {
        "publication_id": entry.paper_id,
        "title": entry.title,
        "source_url": entry.url,
        "attachments": attachment_records,
        "measurements": all_measurements,
        "provenance": {
            "source_url": entry.url,
            "content_hash": f"sha256:{hasher.hexdigest()}",
            "parse_method": "csv_table_extraction" if all_measurements else "metadata_only",
            "ingested_at": datetime.now(timezone.utc).isoformat(),
            "record_count": len(all_measurements),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / f"{entry.paper_id}.json"
    dest.write_text(json.dumps(dataset, indent=2), encoding="utf-8")

    logger.info(
        "Ingested publication %s -> %s (%d measurements)",
        entry.paper_id, dest, len(all_measurements),
    )
    return dataset


@click.command("ingest-publication")
@click.option(
    "--publication-id",
    required=True,
    help="Numeric paper ID to ingest (e.g. '123').",
)
@click.option(
    "--output-dir",
    default="./ams02wb-data/",
    type=click.Path(),
    help="Directory to write ingested dataset JSON into.",
)
def ingest_publication(publication_id: str, output_dir: str) -> None:
    """Ingest a single AMS-02 publication by ID."""
    session = requests.Session()

    try:
        entries = load_publication_index(session, INDEX_URL)
    except CrawlerError as exc:
        logger.error("Network error fetching index: %s", exc)
        sys.exit(1)

    matching = [e for e in entries if e.paper_id == publication_id]
    if not matching:
        logger.error(
            "Publication ID %r not found in index (%d entries checked)",
            publication_id,
            len(entries),
        )
        sys.exit(1)

    entry = matching[0]

    try:
        run_single_ingest(session, entry, Path(output_dir))
    except (CrawlerError, requests.RequestException) as exc:
        logger.error("Ingestion failed for %s: %s", publication_id, exc)
        sys.exit(1)
