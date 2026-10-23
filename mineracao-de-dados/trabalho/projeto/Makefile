.PHONY: analysis model

SHELL := /usr/bin/bash

analysis:
	source key-setup.sh && uv run jupyter lab

model:
	source key-setup.sh && uv run models.py
