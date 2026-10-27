# ml-equity-xs

Code, stored results and paper source for a Financial Engineering course paper on machine-learning
cross-sectional return models (ridge, gradient-boosted trees, small neural networks, a cross-sectional
transformer, graph models) on a free US equity panel. The paper asks whether such models, evaluated
with pre-registered portfolio rules, per-name costs and null distributions, reach a net Sharpe ratio of 1.5.
It reports that they do not, and quantifies what would be required.

## Data (all free, no accounts)

* Nasdaq Trader symbol directories (current listings)
* yfinance daily bars for current common stocks (survivor-only; split-adjusted close and adjusted close)
* SEC EDGAR: data.sec.gov submissions and XBRL companyfacts, plus the bulk financial-statement notes
  data sets (customer-supplier disclosures), 8-K item codes and filing acceptance times
* Kenneth French daily factors; FRED DGS3MO (risk-free rate)
* BEA input-output use tables with SIC -> NAICS -> BEA crosswalks (E5 graph edges)

SEC requests carry a declared-contact User-Agent (`SEC_USER_AGENT` in `src/eqxs/equities_xs.py`). Set `EQXS_SEC_USER_AGENT` to `<project> (<name>; <e-mail>)` before running any ingest, as SEC's fair-access policy requires.
## Data zones

`config/data_zones_equities.yaml` defines three zones: TRAIN 2004-01-01 to 2023-01-01, a walk-forward
stage 2023-01-01 to 2024-07-01, and a sealed zone from 2024-07-01. Every loader reads through
`eqxs.data_zones.safe_load_parquet`, which drops and refuses rows at or after the sealed start (or an
earlier explicit bound) unless an unlock token is supplied; `assert_sealed_clean()` proves at the end of a
run that no sealed read was granted.

## Layout

    src/eqxs/        library: data loaders and ingest parsers, zone tripwire, event-study and ML machinery,
                     window/horizon grid, economic-link graph (edge tables, point-in-time resolver, nulls)
    scripts/         ingest and panel/edge builders
    experiments/     the numbered experiments (P9 to P12c) and experiments/gnn (P13 graph models)
    results/         stored outputs the paper is built from (JSON, RESULTS.md, STATE.json)
    docs/            pre-registration documents
    paper/           LaTeX source, table and figure builders
    tests/           unit tests on synthetic inputs (no data needed)

## Reproducing

    export PYTHONPATH=src
    python scripts/ingest_equities_xs.py nasdaq        # then yfinance, sec-universe, sec-fetch, sec-build, ff, fred, sec-bulk-fetch, ...
    python scripts/ingest_bea_io.py fetch && python scripts/ingest_bea_io.py build
    python scripts/build_eightk_items.py
    python scripts/build_edges.py                      # E1 / E4 / E5 edge tables
    python scripts/equity_ml_xs_train.py --build-only  # monthly feature panel
    python scripts/equity_ml_window_grid_train.py --cells all
    python experiments/p9_ml_capacity.py e1            # then p9_*, p12_*, p12c_* in numeric order
    python experiments/gnn/build_graphs.py --horizon all --relation all

Data go to `data_cache/` (git-ignored), experiment outputs to `results/`; each script's header documents its
arguments. The earnings-calendar ingest keeps only the per-accession override tables (pre-announcements).
The data are not part of this repository: the unit tests and the paper build run from a fresh clone, the experiments do not.

## Tests and paper

    PYTHONPATH=src python -m pytest tests -q
    PYTHON=$(which python) bash paper/build.sh && bash paper/check.sh

The paper build reads only `results/`, regenerates every table and figure from it and compiles
`paper/main.tex` (needs latexmk). `check.sh` enforces the 20-page cap, resolved references and number macros.
