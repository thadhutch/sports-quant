# nfl-data-pipeline

[![CI](https://github.com/thadhutcheson/nfl-data-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/thadhutcheson/nfl-data-pipeline/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A pip-installable data pipeline that scrapes NFL team grades from [PFF](https://www.pff.com/) (Pro Football Focus) and game/betting data from [Pro Football Reference](https://www.pro-football-reference.com/), merges the datasets, and runs postprocessing (rolling averages, rankings) to produce a dataset for over/under analysis.

## Quick Start

```bash
pip install nfl-data-pipeline
```

Or install from source with Poetry:

```bash
git clone https://github.com/thadhutcheson/nfl-data-pipeline.git
cd nfl-data-pipeline
poetry install
```

## Features

- **PFF Scraping** -- Selenium-based scraper for PFF team grades (requires PFF Premium)
- **PFR Scraping** -- Proxy-rotated scraper for Pro Football Reference boxscores
- **Date & Team Normalization** -- Standardizes dates and team names across sources
- **Dataset Merging** -- Inner join on date + team columns
- **Rolling Averages** -- Pre-game cumulative stat averages per team per season
- **Games Played Tracking** -- Cumulative games played before each matchup
- **Feature Rankings** -- Per-date rankings across all teams
- **CLI Interface** -- `nfl-pipeline` command with `scrape`, `process`, and `pipeline` subcommands
- **Python API** -- Import and call any step programmatically

## Prerequisites

- Python 3.12+
- Google Chrome + [ChromeDriver](https://chromedriver.chromium.org/)
- A [PFF Premium](https://www.pff.com/) subscription (for PFF scraping)
- Rotating proxies in CSV format (for PFR scraping)

## Setup

```bash
# Install dependencies
poetry install

# Copy and fill in credentials
cp .env.example .env

# Add your proxies
mkdir -p proxies
# Place your proxies.csv in proxies/ (format: address:port:user:password per line)
```

## Configuration

Override defaults with environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `NFL_SEASONS` | `2024` | Comma-separated list of seasons for PFF scraping |
| `NFL_START_YEAR` | `2024` | Start year for PFR URL scraping |
| `NFL_END_YEAR` | `2024` | End year for PFR URL scraping |
| `NFL_MAX_WEEK` | `18` | Last week to scrape in the final year |
| `NFL_DATA_DIR` | `data` | Base directory for all data output |
| `NFL_PROXY_FILE` | `proxies/proxies.csv` | Path to proxy CSV file |
| `PFF_EMAIL` | - | PFF account email |
| `PFF_PASSWORD` | - | PFF account password |

## CLI Usage

```bash
# Run the full pipeline end-to-end
nfl-pipeline pipeline

# Scrape only PFF data (scrape + parse dates + normalize names)
nfl-pipeline scrape pff

# Scrape only PFR data (URLs + game data + parse dates + normalize names)
nfl-pipeline scrape pfr

# Run all post-processing steps
nfl-pipeline process all

# Run individual processing steps
nfl-pipeline process merge
nfl-pipeline process over-under
nfl-pipeline process averages
nfl-pipeline process games-played
nfl-pipeline process rankings

# Show version
nfl-pipeline --version
```

## Python API

```python
import nfl_data_pipeline

# Run individual steps
nfl_data_pipeline.scrape_pff_data()
nfl_data_pipeline.collect_boxscore_urls()
nfl_data_pipeline.scrape_all_game_info()
nfl_data_pipeline.merge_datasets()
nfl_data_pipeline.process_over_under()
nfl_data_pipeline.compute_rolling_averages()
nfl_data_pipeline.add_games_played()
nfl_data_pipeline.compute_rankings()

# Or run full pipelines
from nfl_data_pipeline.pipeline import run_full_pipeline, run_pff_pipeline, run_pfr_pipeline
run_full_pipeline()
```

## Pipeline

```
PFF Scrape          PFR Scrape
    |                   |
    v                   v
Extract Dates      Normalize Dates
    |                   |
    v                   v
Normalize Names    Normalize Names
    |                   |
    +-------+   +-------+
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

## Project Structure

```
nfl-data-pipeline/
├── src/
│   └── nfl_data_pipeline/
│       ├── __init__.py              # __version__, top-level re-exports
│       ├── _config.py               # Paths, env vars, logging setup
│       ├── teams.py                 # Team name/abbreviation mappings
│       ├── cli.py                   # Click CLI entry point
│       ├── pipeline.py              # Full pipeline orchestrator
│       ├── scrapers/
│       │   ├── pff.py               # PFF grades scraper
│       │   ├── pfr.py               # PFR game data scraper
│       │   ├── pfr_urls.py          # PFR boxscore URL collector
│       │   ├── auth.py              # PFF authentication
│       │   └── proxies.py           # Shared proxy loading
│       ├── parsers/
│       │   ├── pff_dates.py         # PFF date extraction
│       │   ├── pff_teams.py         # PFF team name normalization
│       │   ├── pfr_dates.py         # PFR date normalization
│       │   └── pfr_teams.py         # PFR team name extraction
│       └── processing/
│           ├── merge.py             # Merge PFF + PFR datasets
│           ├── over_under.py        # O/U betting line extraction
│           ├── rolling_averages.py  # Rolling stat averages
│           ├── games_played.py      # Cumulative games played
│           └── rankings.py          # Feature rankings
├── tests/
├── pyproject.toml
├── Makefile
├── LICENSE
└── README.md
```

## Make Commands

```bash
make all            # Run the full pipeline end-to-end
make pff            # Run only the PFF scraping + processing chain
make pfr            # Run only the PFR scraping + processing chain
make merge          # Merge PFF and PFR data (runs both chains first)
make rankings       # Run full postprocessing through rankings
make test           # Run the test suite
make clean          # Remove all generated data files
make dirs           # Create data directory structure
```

## Notes

- **PFF scraping is fragile.** It relies on XPath selectors tied to PFF's DOM structure. If PFF changes their frontend, the selectors in `scrapers/pff.py` will need updating.
- **PFR scraping requires proxies.** Pro Football Reference rate-limits aggressively. Without rotating proxies, requests will be blocked.
- **The PFF scraper uses a real browser.** It opens Chrome via Selenium, logs in with your credentials, and navigates page by page. This is slow but necessary since PFF renders data client-side.
- **Data files are not tracked in git.** Run the pipeline to generate them, or bring your own data in the expected format.
