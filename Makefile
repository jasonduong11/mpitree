VENV = env
PYTHON = $(VENV)/bin/python3

install: requirements.txt
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install -r requirements.txt

activate:
	. $(VENV)/bin/activate

lint: $(VENV) activate
	black .
	isort **/*.py
	flake8 **/*.py
	pylint **/*.py
