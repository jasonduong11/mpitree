build: environment.yml
	conda env create -f environment.yml

clean:
	rm -rf mpitree/__pycache__
	rm -rf mpitree/tree/__pycache__
	rm -rf .ipynb_checkpoints
	rm -rf .ruff_cache
