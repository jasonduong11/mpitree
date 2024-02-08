VENV = env

activate:
	. $(VENV)/bin/activate

build: requirements.txt
	python3.12 -m venv $(VENV)
	$(VENV)/bin/pip install -r requirements.txt

lint: activate
	black .
	isort mpitree/tree/*.py
	flake8 mpitree/tree/*.py
