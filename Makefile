PYTHON ?= python

.PHONY: half_division uniform_search

half_division:
	$(PYTHON) -m half_division.main

uniform_search:
	$(PYTHON) -m uniform_search.main
