.PHONY: lab run

lab:
	uv run jupyter lab --notebook-dir=.

run:
	uv run main.py