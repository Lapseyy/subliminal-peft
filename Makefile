# subliminal-peft: experiments -> results -> figures -> paper PDF
#
#   make test        run the test suite (CPU, ~1 min)
#   make followups   run the queued experiments (pilot 5: new dataset, pilot 6: penguin control)
#   make controls    non-animal teacher controls (pilot 7: neutral prompt, pilot 8: coffee)
#   make collect     read runs/ into paper/generated/results.json (pure Python)
#   make figures     results.json -> figures, tables and number macros (needs matplotlib, torch)
#   make paper       build paper/main.pdf (needs pdflatex + bibtex)
#   make all         collect + figures + paper
#   make clean       delete LaTeX build files and generated outputs
#
# Windows: install GNU make (winget install GnuWin32.Make, or choco install make) and a TeX
# distribution (MiKTeX). Run from the project root with the venv active. Override the Python
# executable with: make PYTHON=python3 all

PYTHON ?= python
LATEX  ?= pdflatex -interaction=nonstopmode -halt-on-error
BIBTEX ?= bibtex

GEN := paper/generated

.PHONY: all test followups controls pilot5 pilot6 pilot7 pilot8 collect figures paper clean

all: paper

test:
	$(PYTHON) -m pytest -q

pilot5:
	$(PYTHON) -m sublim.sweep configs/pilot5.json

pilot6:
	$(PYTHON) -m sublim.sweep configs/pilot6.json

followups: pilot5 pilot6

pilot7:
	$(PYTHON) -m sublim.sweep configs/pilot7.json

pilot8:
	$(PYTHON) -m sublim.sweep configs/pilot8.json

controls: pilot7 pilot8

collect:
	$(PYTHON) paper/scripts/collect_results.py

$(GEN)/bf16_sim.json: paper/scripts/bf16_sim.py sublim/optim.py
	$(PYTHON) paper/scripts/bf16_sim.py

figures: collect $(GEN)/bf16_sim.json
	$(PYTHON) paper/scripts/make_figures.py

paper: figures
	cd paper && $(LATEX) main.tex
	cd paper && $(BIBTEX) main
	cd paper && $(LATEX) main.tex
	cd paper && $(LATEX) main.tex
	@echo "Built paper/main.pdf"

clean:
	$(PYTHON) -c "import pathlib, shutil; p = pathlib.Path('paper'); [f.unlink() for e in ('aux','bbl','blg','log','out','toc','pdf') for f in p.glob('main.' + e)]; shutil.rmtree(p / 'generated', ignore_errors=True)"
