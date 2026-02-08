# nfl-data-pipeline

[![CI](https://github.com/thadhutch/nfl-data-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/thadhutch/nfl-data-pipeline/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/nfl-data-pipeline)](https://pypi.org/project/nfl-data-pipeline/)
[![PyPI downloads](https://img.shields.io/pypi/dt/nfl-data-pipeline)](https://pypi.org/project/nfl-data-pipeline/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end data pipeline that combines [PFF](https://www.pff.com/) team grades with [Pro Football Reference](https://www.pro-football-reference.com/) game and betting data, then produces analysis-ready datasets with rolling averages, rankings, and over/under features.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Prerequisites](#prerequisites)
- [Configuration](#configuration)
- [Usage](#usage)
  - [CLI](#cli)
  - [Python API](#python-api)
  - [Make Targets](#make-targets)
- [Pipeline Architecture](#pipeline-architecture)
- [Project Structure](#project-structure)
- [Development](#development)
- [Contributing](#contributing)
- [Known Limitations](#known-limitations)
- [License](#license)

---

## Features

- **PFF Scraping** &mdash; Selenium-based scraper for PFF team grades (requires PFF Premium)
- **PFR Scraping** &mdash; Proxy-rotated scraper for Pro Football Reference boxscores
- **Data Normalization** &mdash; Standardizes dates and team names across sources
- **Dataset Merging** &mdash; Inner join on date + team columns
- **Rolling Averages** &mdash; Pre-game cumulative stat averages per team per season
- **Games Played Tracking** &mdash; Cumulative games played before each matchup
- **Feature Rankings** &mdash; Per-date rankings across all teams
- **CLI + Python API** &mdash; Run the full pipeline or any individual step

## Installation

Install from [PyPI](https://pypi.org/project/nfl-data-pipeline/):

```bash
pip install nfl-data-pipeline
```

Or install from source with [Poetry](https://python-poetry.org/):

```bash
git clone https://github.com/thadhutch/nfl-data-pipeline.git
cd nfl-data-pipeline
poetry install
```

## Prerequisites

| Requirement | Why |
|---|---|
| Python 3.12+ | Runtime |
| Google Chrome + [ChromeDriver](https://chromedriver.chromium.org/) | PFF scraper uses Selenium to render client-side data |
| [PFF Premium](https://www.pff.com/) subscription | Authenticates access to PFF team grades |
| Rotating proxies (CSV) | PFR rate-limits aggressively; proxies prevent blocks |

## Configuration

Create a `.env` file (see [`.env.example`](.env.example)) or export environment variables directly:

```bash
cp .env.example .env   # then fill in your credentials
```

| Variable | Default | Description |
|---|---|---|
| `PFF_EMAIL` | &mdash; | PFF account email **(required for PFF scraping)** |
| `PFF_PASSWORD` | &mdash; | PFF account password **(required for PFF scraping)** |
| `NFL_SEASONS` | `2024` | Comma-separated seasons to scrape from PFF |
| `NFL_START_YEAR` | `2024` | First year for PFR boxscore URL collection |
| `NFL_END_YEAR` | `2024` | Last year for PFR boxscore URL collection |
| `NFL_MAX_WEEK` | `18` | Final week to scrape in the last season |
| `NFL_DATA_DIR` | `data` | Base directory for all output files |
| `NFL_PROXY_FILE` | `proxies/proxies.csv` | Path to proxy list (`address:port:user:password` per line) |

## Usage

### CLI

The `nfl-pipeline` command is available after installation:

```bash
# Full end-to-end pipeline
nfl-pipeline pipeline

# Scrape from a single source
nfl-pipeline scrape pff          # PFF grades (scrape + date parsing + name normalization)
nfl-pipeline scrape pfr          # PFR game data (URLs + scrape + date/name normalization)

# Run all post-processing steps
nfl-pipeline process all

# Run individual processing steps
nfl-pipeline process merge
nfl-pipeline process over-under
nfl-pipeline process averages
nfl-pipeline process games-played
nfl-pipeline process rankings

# Check installed version
nfl-pipeline --version
```

### Python API

Every pipeline step is importable:

```python
import nfl_data_pipeline

# Scraping
nfl_data_pipeline.scrape_pff_data()
nfl_data_pipeline.collect_boxscore_urls()
nfl_data_pipeline.scrape_all_game_info()

# Processing
nfl_data_pipeline.merge_datasets()
nfl_data_pipeline.process_over_under()
nfl_data_pipeline.compute_rolling_averages()
nfl_data_pipeline.add_games_played()
nfl_data_pipeline.compute_rankings()
```

Or run an entire pipeline at once:

```python
from nfl_data_pipeline.pipeline import (
    run_full_pipeline,
    run_pff_pipeline,
    run_pfr_pipeline,
    run_processing_pipeline,
)

run_full_pipeline()        # end-to-end
run_pff_pipeline()         # PFF scraping chain only
run_pfr_pipeline()         # PFR scraping chain only
run_processing_pipeline()  # post-processing only
```

### Make Targets

A `Makefile` is included for common development workflows:

```bash
make all            # Full pipeline end-to-end
make pff            # PFF scraping + processing chain
make pfr            # PFR scraping + processing chain
make merge          # Merge PFF and PFR data (runs both chains first)
make rankings       # Full postprocessing through rankings
make test           # Run the test suite
make clean          # Remove all generated data files
make dirs           # Create data directory structure
```

## Pipeline Architecture

```
PFF Scrape              PFR Scrape
    |                       |
    v                       v
Extract Dates          Normalize Dates
    |                       |
    v                       v
Normalize Names        Normalize Names
    |                       |
    +----------+   +--------+
               |   |
               v   v
              Merge
                |
                v
           Over/Under
                |
                v
         Rolling Averages
                |
                v
          Games Played
                |
                v
            Rankings
```

**Output files** are written to `NFL_DATA_DIR` (default: `data/`):

| Stage | Output |
|---|---|
| PFF scrape | `data/pff/raw_team_data.csv` |
| PFF normalized | `data/pff/normalized_team_data.csv` |
| PFR URLs | `data/pfr/boxscores_urls.txt` |
| PFR normalized | `data/pfr/final_pfr_odds.csv` |
| Merged | `data/pff_and_pfr_data.csv` |
| Final dataset | `data/over-under/v1-dataset-gp-ranked.csv` |

## Project Structure

```
nfl-data-pipeline/
├── src/nfl_data_pipeline/
│   ├── __init__.py           # Public API re-exports
│   ├── _config.py            # Paths, env vars, logging
│   ├── cli.py                # Click CLI entry point
│   ├── pipeline.py           # Pipeline orchestrators
│   ├── teams.py              # Team name/abbreviation mappings
│   ├── scrapers/
│   │   ├── pff.py            # PFF grades scraper (Selenium)
│   │   ├── pfr.py            # PFR game data scraper
│   │   ├── pfr_urls.py       # PFR boxscore URL collector
│   │   ├── auth.py           # PFF authentication
│   │   └── proxies.py        # Proxy loading utilities
│   ├── parsers/
│   │   ├── pff_dates.py      # PFF date extraction
│   │   ├── pff_teams.py      # PFF team name normalization
│   │   ├── pfr_dates.py      # PFR date normalization
│   │   └── pfr_teams.py      # PFR team name extraction
│   └── processing/
│       ├── merge.py          # Merge PFF + PFR datasets
│       ├── over_under.py     # O/U betting line extraction
│       ├── rolling_averages.py
│       ├── games_played.py
│       └── rankings.py
├── tests/
├── pyproject.toml
├── Makefile
├── LICENSE
└── README.md
```

## Development

```bash
# Clone and install with dev dependencies
git clone https://github.com/thadhutch/nfl-data-pipeline.git
cd nfl-data-pipeline
poetry install

# Run the test suite
poetry run pytest -v

# Run a specific test file
poetry run pytest tests/test_rolling_averages.py -v
```

CI runs automatically on every push to `master` and on pull requests via [GitHub Actions](https://github.com/thadhutch/nfl-data-pipeline/actions). Releases are published to PyPI through [Trusted Publishers](https://docs.pypi.org/trusted-publishers/).

## Contributing

Contributions are welcome! To get started:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes and add tests where appropriate
4. Run the test suite (`poetry run pytest -v`)
5. Commit your changes (`git commit -m "Add my feature"`)
6. Push to your fork (`git push origin feature/my-feature`)
7. Open a Pull Request

Please make sure all existing tests pass before submitting a PR.

## Known Limitations

- **PFF scraping is DOM-dependent.** The scraper relies on XPath selectors tied to PFF's frontend. If PFF changes their page structure, the selectors in `scrapers/pff.py` will need updating.
- **PFR scraping requires rotating proxies.** Without them, requests will be rate-limited and blocked.
- **The PFF scraper is slow by design.** It drives a real Chrome browser via Selenium because PFF renders data client-side.
- **Data files are not tracked in git.** Run the pipeline to generate them, or bring your own data in the expected CSV format.

## License

This project is licensed under the [MIT License](LICENSE).

---

<p align="center">
  <a href="https://pypi.org/project/nfl-data-pipeline/">PyPI</a> &middot;
  <a href="https://github.com/thadhutch/nfl-data-pipeline/issues">Issues</a> &middot;
  <a href="https://github.com/thadhutch/nfl-data-pipeline/actions">CI Status</a>
</p>
