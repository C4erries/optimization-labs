PYTHON ?= python

.PHONY: half_division uniform_search dichotomy golden_section fibonaccy svenn steepest_descent

half_division:
	$(PYTHON) -m half_division.main

uniform_search:
	$(PYTHON) -m uniform_search.main

dichotomy:
	$(PYTHON) -m dichotomy.main

golden_section:
	$(PYTHON) -m golden_section.main

fibonaccy:
	$(PYTHON) -m fibonaccy.main

svenn:
	$(PYTHON) -m svenn.main

steepest_descent:
	$(PYTHON) -m steepest_descent.main
