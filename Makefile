PYTHON ?= python3

.PHONY: changelog check compile test

changelog:
	git cliff --output CHANGELOG.md

check: compile test

compile:
	$(PYTHON) -m py_compile bootstrap.py main.py api/*.py models/*.py utils/*.py

test:
	$(PYTHON) -m unittest discover -s tests -v
