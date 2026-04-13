# AMS-02 Publication Data Harmonizer and Likelihood Workbench

An open-source Python CLI tool that sits above the AMS-02 public data layer and solves the missing problems researchers face when combining results across multiple AMS-02 publications.

## The Problem

AMS-02 on the International Space Station has published 30+ peer-reviewed measurements of cosmic-ray fluxes — protons, helium, electrons, positrons, antiprotons, boron-to-carbon ratios, lithium isotopes, and more. These results are the most precise cosmic-ray data ever collected.

But using them together is painful:

- **Heterogeneous table formats** — some papers publish CSV files, others embed data in 2,825-page PDF supplements
- **Inconsistent axes** — proton flux is given vs rigidity (GV), isotope results use kinetic energy per nucleon (GeV/n), antiproton results use absolute rigidity
- **Different time windows** — daily proton fluxes cover 2,824 days, helium covers 2,824 days with slightly different gaps, electrons span 3,300 days, positrons span 3,268 days
- **Heterogeneous uncertainty reporting** — some papers give stat+sys totals, others decompose into trigger/acceptance/unfolding/scale components, time-series add "time-dependent systematic" categories
- **No published covariance matrices** for most datasets — yet bin-to-bin correlations are decisive at AMS-02 precision levels, as shown in the [B/C cookbook paper](https://arxiv.org/abs/1904.08210) and [antiproton correlation studies](https://link.aps.org/doi/10.1103/PhysRevResearch.2.043017)

Every researcher doing dark matter searches, solar modulation studies, or cosmic-ray propagation fits has to write one-off parsers, unit converters, and alignment scripts from scratch. This tool eliminates that repeated work.

## What This Tool Does

```
AMS publication pages → Ingestion → Canonical schema → Time-series alignment → Fit-ready likelihood
                           ↓              ↓                    ↓                      ↓
                      raw files +    harmonised         proton+helium+e-+e+       covariance matrix
                      provenance     25-field schema    on common dates           + uncertainty labels
```

1. **Publication compiler** — indexes all 30 AMS-02 publications from [ams02.space/publications](https://ams02.space/publications), downloads CSV and PDF attachments, parses data tables, and stores raw files with full provenance
2. **Schema harmonisation** — normalises all measurements into one 25-field canonical schema with rigidity-to-energy conversion and species name normalisation
3. **Time-series alignment** — aligns daily flux series across species on a common time grid (intersection or union join) with diagnostics showing which dates were dropped
4. **Uncertainty-aware likelihood** — builds fit-ready covariance matrices with three modes, each clearly labelled as "published", "derived", or "assumed"
5. **Solar/heliospheric context hooks** — attach external time-series (modulation potentials, sunspot numbers) to aligned data by time key
6. **Multi-format export** — CSV, Parquet, JSON, and USINE-compatible output for downstream fitting frameworks

## Installation

```bash
# From source
git clone <repo-url>
cd ams02-publication-data-harmonizer
pip install -e .

# Verify
ams02wb --version
```

Requires Python 3.11+.

## Quick Start

```bash
# 1. See what AMS-02 papers are available
ams02wb index-publications --output-dir ./data

# 2. Ingest the daily proton flux paper
ams02wb ingest-publication --publication-id 202105 --output-dir ./data

# 3. Harmonise into canonical schema
ams02wb harmonise --input-dir ./data --output-dir ./data/harmonised

# 4. Export to CSV
ams02wb export-dataset --dataset ./data/harmonised/202105.json --format csv --output proton_daily.csv
```

## Full Workflow Examples

### Scenario 1: Multi-species time-series alignment

A researcher studying solar modulation wants proton and helium daily fluxes on a common time grid:

```bash
# Ingest both species
ams02wb ingest-publication --publication-id 202105 --output-dir ./data  # proton daily
ams02wb ingest-publication --publication-id 202201 --output-dir ./data  # helium daily

# Harmonise (converts rigidity → kinetic energy, normalises species names)
ams02wb harmonise --input-dir ./data --output-dir ./data/harmonised

# Align on common dates (intersection = only dates both species have)
ams02wb align-time-series \
  --input-dir ./data/harmonised \
  --species proton --species helium \
  --join intersection \
  --output aligned_proton_helium.parquet
```

This produces a wide-form Parquet file with proton and helium measurements side-by-side on 2,824 common dates (May 2011 – October 2019), with provenance tracking which original tables contributed.

### Scenario 2: Building a likelihood for dark matter fits

A researcher fitting antiproton data needs a covariance matrix. AMS-02 does not publish official covariance matrices for most datasets:

```bash
# Ingest and harmonise
ams02wb ingest-publication --publication-id 201601 --output-dir ./data
ams02wb harmonise --input-dir ./data --output-dir ./data/harmonised
ams02wb export-dataset --dataset ./data/harmonised/201601.json --format parquet --output antiproton.parquet

# Diagonal likelihood (stat errors only, labelled "published")
ams02wb build-likelihood --dataset antiproton.parquet --mode diag --output fit_diag.parquet

# Kernel-correlation likelihood (assumed systematic correlations)
ams02wb build-likelihood --dataset antiproton.parquet --mode kernel_corr --corr-length 0.25 --output fit_kernel.parquet
```

The kernel-correlation mode builds: `C_sys_ij = sigma_sys_i * sigma_sys_j * exp(-|log10(x_i) - log10(x_j)| / L)` where L is the user-set correlation length. The JSON sidecar will say `"uncertainty_label": "assumed"` — never claiming this is an official AMS covariance.

### Scenario 3: Per-day likelihood for time-dependent analysis

For time-series data, building one covariance for all 83,757 rows is impractical. Use `--time-bin` or `--per-day`:

```bash
# Single day (30 rigidity bins → 30×30 matrix)
ams02wb build-likelihood --dataset proton.parquet --mode diag \
  --time-bin 2015-06-15 --output fit_single_day.parquet

# One likelihood per day for a week
ams02wb build-likelihood --dataset proton.parquet --mode diag \
  --time-bin 2015-06-15:2015-06-21 --per-day --output ./per_day/

# One likelihood per day for the entire dataset
ams02wb build-likelihood --dataset proton.parquet --mode diag \
  --per-day --output ./all_days/
```

### Scenario 4: Attaching solar modulation context

```python
from ams02wb.alignment.context import ContextSeries, attach_context, load_context_csv
import pandas as pd

# Load harmonised data
df = pd.read_parquet("aligned_proton_helium.parquet")

# Load modulation potential from a CSV file
ctx = load_context_csv(
    "phi_modulation.csv",            # CSV with 'time' and 'value' columns
    name="phi_MV",
    description="Force-field modulation potential",
    source="Usoskin et al. (2017)",
)

# Attach by nearest time match
df_with_phi = attach_context(df, ctx, method="nearest")
# Now df_with_phi has columns: phi_MV, phi_MV_source
```

Methods: `"nearest"` (closest time match), `"interpolate"` (linear interpolation), `"exact"` (exact date match, NaN for missing).

## CLI Reference

| Command | Description |
|---------|-------------|
| `ams02wb index-publications` | Fetch the AMS-02 publication index (30 papers) |
| `ams02wb ingest-publication --publication-id ID` | Download and parse one publication's tables |
| `ams02wb ingest-all` | Ingest all 30 publications |
| `ams02wb harmonise --input-dir DIR` | Normalise all ingested data into canonical schema |
| `ams02wb validate --input-dir DIR` | Validate harmonised JSON files against schema |
| `ams02wb align-time-series --species A --species B` | Align daily series across species |
| `ams02wb build-likelihood --dataset FILE --mode MODE` | Build fit-ready covariance matrix |
| `ams02wb export-dataset --dataset FILE --format FMT` | Export to csv, parquet, json, or usine format |

## Supported Species

Proton, Helium, Electron, Positron, Antiproton, Carbon, Boron, Oxygen, Nitrogen, Lithium, Beryllium, Deuteron, Neon, Magnesium, Silicon, Iron, Sulfur, Sodium, Aluminum, Fluorine.

## Supported Publications

All 30 AMS-02 publications from [ams02.space/publications](https://ams02.space/publications) are indexed. Papers with CSV attachments are fully parsed; papers with only PDF supplements use text-based extraction. Verified ingested papers:

| Paper ID | Title | Records | Species |
|----------|-------|---------|---------|
| 202105 | Periodicities in the Daily Proton Fluxes (2011-2019) | 83,757 | Proton |
| 202201 | Properties of Daily Helium Fluxes | 72,889 | Helium |
| 201601 | Antiproton Flux and Properties of Elementary Particle Fluxes | 140 | Antiproton, Electron |
| 201602 | Boron to Carbon Flux Ratio | 134 | Boron, Carbon |
| 201501 | Precision Measurement of the Proton Flux (1 GV - 1.8 TV) | 72 | Proton |

## Canonical Schema

Every harmonised record contains these 25 fields:

| Field | Description |
|-------|-------------|
| `dataset_id` | Unique dataset identifier |
| `publication_id` | Source paper ID (e.g. "202105") |
| `publication_title` | Full paper title |
| `publication_url` | URL to the AMS publication page |
| `table_id` | Source table identifier within the paper |
| `species_num` | Numerator species (e.g. "PROTON", "HELIUM") |
| `species_den` | Denominator species for ratios (e.g. "CARBON" for B/C) |
| `measurement_type` | Type: flux, ratio, time_series, isotope_flux, isotope_ratio |
| `x_axis_type` | Axis: rigidity, kinetic_energy, kinetic_energy_per_nucleon |
| `x_axis_unit` | Unit of x-axis (e.g. "GV", "GeV") |
| `x_min` | Lower bin edge |
| `x_max` | Upper bin edge |
| `x_centre` | Bin centre |
| `y_value` | Measured value (flux, ratio) |
| `y_unit` | Unit of measurement |
| `time_start` | Start of measurement period (ISO 8601) |
| `time_stop` | End of measurement period (ISO 8601) |
| `time_label` | Human-readable time label |
| `stat_err` | Statistical uncertainty |
| `sys_err_total` | Total systematic uncertainty |
| `sys_err_components` | Decomposed systematic components (JSON) |
| `scale_err` | Energy/rigidity scale uncertainty |
| `upper_limit_flag` | Whether the value is an upper limit |
| `metadata_json` | Additional metadata (JSON) |
| `provenance_json` | Full provenance chain back to source paper/table/URL |

## Likelihood Modes

| Mode | Label | Description |
|------|-------|-------------|
| `diag` | published | Diagonal covariance from published stat errors: `C_ii = sigma_stat_i^2` |
| `grouped_sys` | derived | Stat (diagonal) + fully-correlated systematic: `C = diag(sigma_stat^2) + sigma_sys * sigma_sys^T` |
| `kernel_corr` | assumed | Systematic correlation kernel: `C_sys_ij = sigma_sys_i * sigma_sys_j * exp(-|log10(x_i/x_j)| / L)` where L is user-set |

Every likelihood output includes a JSON sidecar stating whether the uncertainty model is **published** (from the paper), **derived** (computed from published components), or **assumed** (user-specified model).

## Provenance

Every record tracks its origin:

- `source_url` — the AMS publication page it came from
- `content_hash` — SHA-256 of the downloaded source file
- `parse_method` — how the data was extracted (csv_table_extraction, pdf_text_extraction)
- `ingested_at` — when the data was ingested
- `harmonisation_metadata` — which normalisation steps were applied

Raw source files (CSV, PDF) are stored in `raw/<paper_id>/` alongside parsed output.

## Data Sources

This tool ingests data from:

- **[ams02.space/publications](https://ams02.space/publications)** — AMS-02 publication index with downloadable tables and supplemental files
- **[ams02.web.cern.ch](https://ams02.web.cern.ch/)** — Official AMS-02 site at CERN

Related tools and databases (complementary, not replaced):

- **[CRDB](https://lpsc.in2p3.fr/crdb/)** — Cosmic-Ray Database at LPSC/IN2P3/CNRS ([Python package](https://pypi.org/project/crdb/))
- **[USINE](https://dmaurin.gitlab.io/USINE/input_cr_data.html)** — Cosmic-ray propagation and fitting framework
- **[pbarlike](https://ams02antiprotonlikelihood.readthedocs.io/)** — AMS-02 antiproton likelihood calculator
- **[HelMod](https://helmod.org/)** — Solar modulation model calibrated against AMS-02

Key references:

- [Fitting B/C cosmic-ray data in the AMS-02 era: A cookbook](https://arxiv.org/abs/1904.08210) — documents the need for covariance matrices
- [Dark matter or correlated errors: Systematics of the AMS-02 antiproton excess](https://link.aps.org/doi/10.1103/PhysRevResearch.2.043017) — shows impact of systematic correlations
- [CRDB v4.1 paper](https://link.springer.com/article/10.1140/epjc/s10052-023-12092-8) — describes the cosmic-ray database ecosystem

## Trade-offs

- Targets AMS-02 publications specifically rather than general cosmic-ray data. Supporting other experiments would require new parsers.
- Schema validation uses physical bounds (energy 0.1-5000 GeV) tuned to AMS-02 measurement ranges.
- PDF table extraction uses text-line parsing as pdfplumber's gridline detection does not work on AMS's space-delimited table format. Extraction confidence should be verified for new papers.

## Limitations

- PDF extraction is best-effort for AMS's space-delimited table format. CSV-attached papers (the majority of recent publications) are fully parsed.
- Covariance matrices are only used when published by AMS. When they are not published (which is most datasets), the tool provides user-selectable correlation models clearly labelled as "assumed".
- Time-series alignment performs a Cartesian join of rigidity bins across species. For 30 proton bins x 26 helium bins x 2,824 days, this produces ~2.2M rows.
- Full-dataset likelihood building requires `--time-bin` or `--per-day` for time-series data to avoid memory issues.

## Non-goals

- **Not a replacement for CRDB.** CRDB excels at cross-experiment data discovery and retrieval. This tool adds AMS-specific schema harmonisation and likelihood construction on top.
- **Not a fitting framework.** Use USINE, GAMBIT, or your own MCMC pipeline downstream. This tool produces the fit-ready inputs.
- **Not a raw event-data platform.** AMS-02 does not publicly release event-level data. This tool works with the published measurement tables.
- **No invented physics.** Every assumption is labelled. No covariance matrix is claimed as official unless AMS published it.

## License

MIT
