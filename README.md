# AMS-02 Publication Data Harmonizer

Python tool for ingesting, harmonising, and analysing AMS-02 cosmic-ray publication data. Normalises heterogeneous table formats into a canonical schema, constructs uncertainty-aware likelihoods, and exports fit-ready datasets.

## Installation

```bash
pip install -e .
```

## Usage

```bash
ams02wb --help
```

## Trade-offs

- Targets AMS-02 publications specifically rather than general cosmic-ray data formats. Supporting other experiments would require new parsers.
- Schema validation uses physical bounds (energy 0.1–5000 GeV) tuned to AMS-02 measurement ranges. Data outside these ranges is flagged, not silently accepted.

## Limitations

- PDF table extraction is best-effort and may require manual correction for complex layouts.
- Covariance matrices are only used when published; no invented covariance is applied.

## Non-goals

- Replacing CRDB. This tool is complementary — it adds provenance tracking and likelihood construction, not a general database.
- Real-time data ingestion. This is a batch processing tool for published results.
