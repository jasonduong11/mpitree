VENV = env
PYTHON = $(VENV)/bin/python3

activate:
	. $(VENV)/bin/activate

build: requirements.txt
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install -r requirements.txt

lint: $(VENV) activate
	black .
	isort **/*.py
	flake8 **/*.py
	pylint **/*.py
