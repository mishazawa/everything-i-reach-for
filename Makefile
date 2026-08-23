.PHONY: lab run tensorboard

LOG_DIR := "./logs"

lab:
	uv run jupyter lab --notebook-dir=.

run:
	uv run main.py

tensorboard:
	uv run tensorboard --logdir $(LOG_DIR)