.PHONY: analysis model best comp dashboard

SHELL := /usr/bin/bash

analysis:
	source key-setup.sh && uv run jupyter lab

model:
	source key-setup.sh && uv run models.py

best:
	source key-setup.sh && uv run models.py --step best

comp:
	uv run predict_kaggle.py

dashboard:
	uv run streamlit run dashboard.py