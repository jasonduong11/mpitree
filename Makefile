VENV = env

activate:
	. $(VENV)/bin/activate

build: requirements.txt
	python3.12 -m venv $(VENV)
	$(VENV)/bin/pip install -r requirements.txt

lint: activate
	black .
	isort mpitree/tree/_base.py
	isort mpitree/tree/decision_tree.py
	flake8 mpitree/tree/_base.py
	flake8 mpitree/tree/decision_tree.py
