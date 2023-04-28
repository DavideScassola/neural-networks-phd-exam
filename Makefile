.ONESHELL:
SHELL := /bin/bash

.venv:
	python3.10 -m venv .venv

.PHONY: install
install:
	pip install --upgrade pip
	python -m pip install --upgrade -r requirements.txt
