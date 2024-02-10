VENV = env
PIP = $(VENV)/bin/pip


build: requirements.txt
	python3.12 -m venv $(VENV)
	$(PIP) install -r requirements.txt


lint: $(VENV)/bin/activate
	ruff check --fix
	ruff format


clean:
	rm -rf mpitree/__pycache__
	rm -rf mpitree/tree/__pycache__
	rm -rf .ruff_cache
