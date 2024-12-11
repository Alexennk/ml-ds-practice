.PHONY: install lint clean


REQUIREMENTS = requirements-dev.txt
PIP = pip


install:
	pip install --upgrade pip
	pip install -r $(REQUIREMENTS)


lint:
	@echo ">>> black files"
	black .
	@echo ">>> linting files"
	flake8


clean:
	rm -rf __pycache__