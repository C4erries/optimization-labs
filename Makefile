PYTHON ?= ~/myenv/bin/python3.12

.PHONY: half_division uniform_search dichotomy golden_section fibonaccy svenn steepest_descent fletcher_reeves newton newton_raphson marquardt powell lagrange_multipliers

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

fletcher_reeves:
	$(PYTHON) -m fletcher_reeves.main

newton:
	$(PYTHON) -m newton.main

newton_raphson:
	$(PYTHON) -m newton_raphson.main

marquardt:
	$(PYTHON) -m marquardt.main

powell:
	$(PYTHON) -m powell.main

lagrange_multipliers:
	$(PYTHON) -m lagrange_multipliers.main
