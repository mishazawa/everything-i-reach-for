.PHONY: lab run tensorboard

LOG_DIR := "./logs"
RELOAD_INTERVAL := 5

lab:
	uv run jupyter lab --notebook-dir=.

run:
	uv run main.py

tensorboard:
	uv run tensorboard --logdir $(LOG_DIR) --reload_interval $(RELOAD_INTERVAL)