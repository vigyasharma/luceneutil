# basic make safety: if error happens, delete output
.DELETE_ON_ERROR:

# explicitly declare shell used
SHELL := /bin/bash

# enforce that shell is picky
.SHELLFLAGS := -norc -euo pipefail -c

# don't litter up repo with bytecode
export PYTHONDONTWRITEBYTECODE=true

# don't self-check, don't ask questions
PIP_INSTALL_ARGS=--disable-pip-version-check --no-input --upgrade

# venv with dependencies in the standard location
VENV=${PWD}/.venv

# don't behave strangely if these files exist
.PHONY: lint format reformat ruff pyright env clean

# list of directories we check
SOURCES=src/python

# check formatting, linting, and types
#lint: format ruff pyright
lint: ruff pyright

# check all formatting
format: env
	# validate imports: if this fails, please run "make reformat" and commit changes.
	$(VENV)/bin/ruff check --select I $(SOURCES)
	# validate formatting: if this fails, please run "make reformat" and commit changes.
	$(VENV)/bin/ruff format --diff $(SOURCES)

# reformat code to conventions
reformat: env
	# organize imports
	$(VENV)/bin/ruff check --select I --fix $(SOURCES)
	# reformat sources
	$(VENV)/bin/ruff format $(SOURCES)

# lints sources
ruff: env
	# validate sources with ruff linter
	$(VENV)/bin/ruff check $(SOURCES)

# checks types
pyright: env
	# type-check sources with basedpyright
	$(VENV)/bin/basedpyright $(SOURCES)

# rebuild venv if dependencies change
env: $(VENV)/bin/activate
$(VENV)/bin/activate: requirements.txt
	# remove any existing venv
	rm -rf $(VENV)
	# create new venv
	python3 -m venv $(VENV)
	# install dependencies into venv
	$(VENV)/bin/pip install $(PIP_INSTALL_ARGS) -r requirements.txt
	# adjust timestamp for safety
	touch $(VENV)/bin/activate

# nuke venv
clean:
	rm -rf $(VENV)
