CLI = poetry run sports-quant

.PHONY: dirs pff pfr merge over-under averages games-played rankings all pipeline model-train model-backtest test clean

dirs:
	mkdir -p data/pff data/pfr data/over-under

# --- PFF Pipeline ---

pff: dirs
	$(CLI) scrape pff

# --- PFR Pipeline ---

pfr: dirs
	$(CLI) scrape pfr

# --- Merge + Postprocessing ---

merge: pff pfr
	$(CLI) process merge

over-under: merge
	$(CLI) process over-under

averages: over-under
	$(CLI) process averages

games-played: averages
	$(CLI) process games-played

rankings: games-played
	$(CLI) process rankings

all: rankings

pipeline: dirs
	$(CLI) pipeline

# --- Modeling ---

model-train:
	$(CLI) model train

model-backtest:
	$(CLI) model backtest

test:
	poetry run pytest -v

clean:
	rm -rf data/pff/* data/pfr/* data/over-under/*
