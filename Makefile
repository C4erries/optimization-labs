PYTHON ?= python

.PHONY: half_division uniform_search dichotomy golden_section

half_division:
	$(PYTHON) -m half_division.main

uniform_search:
	$(PYTHON) -m uniform_search.main

dichotomy:
	$(PYTHON) -m dichotomy.main

golden_section:
	$(PYTHON) -m golden_section.main
