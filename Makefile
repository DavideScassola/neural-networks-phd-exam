.venv:
	python3.10 -m venv .venv

.PHONY: install
install:
	python -m pip install --upgrade pip
	python -m pip install --upgrade -r requirements.txt

.PHONY: clean
clean:
	find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete
	rm -R -f ./.mypy_cache

.PHONY: format
format:
	find . -type f -name '*.py' -exec reorder-python-imports --py310-plus "{}" \;
	black "$(realpath .)"
