PYTHON ?= ~/myenv/bin/python3.12

.PHONY: half_division uniform_search dichotomy golden_section fibonaccy svenn steepest_descent fletcher_reeves dfp davidson_fletcher_powell newton newton_raphson marquardt powell lagrange_multipliers lagrange modified_lagrange_multipliers penalty penalty_method

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

dfp:
	$(PYTHON) -m davidson_fletcher_powell.main

davidson_fletcher_powell:
	$(PYTHON) -m davidson_fletcher_powell.main

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

lagrange:
	$(PYTHON) -m modified_lagrange_multipliers.main

modified_lagrange_multipliers:
	$(PYTHON) -m modified_lagrange_multipliers.main

penalty:
	$(PYTHON) -m penalty_method.main

penalty_method:
	$(PYTHON) -m penalty_method.main
