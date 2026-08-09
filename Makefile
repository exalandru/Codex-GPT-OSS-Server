ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
SERVER_DIR := $(ROOT)/src/server
DESKTOP_DIR := $(ROOT)/src/desktop
BUILD_DIR := $(ROOT)/build
DIST_DIR := $(ROOT)/dist

export UV_PROJECT_ENVIRONMENT := $(ROOT)/.venv

# Development models. The 20B keeps the edit-run loop fast; the 120B is the
# parity check, and the model to use for namespaced tools (see README).
#
# Override for your own layout:
#   make dev-server MODELS_DIR="$$HOME/models"
#   make dev-server MODEL_20B="/path/to/gpt-oss-20b-mxfp4-bf16"
MODELS_DIR ?= $(HOME)/models
MODEL_20B ?= $(MODELS_DIR)/gpt-oss-20b-mxfp4-bf16
MODEL_120B ?= $(MODELS_DIR)/gpt-oss-120b-mxfp4-bf16

PORT ?= 8123

.PHONY: help install install-server install-desktop dev-server dev-server-120b \
	dev-desktop build-desktop doctor test lint ci clean

help:
	@printf '%s\n' \
		'make install         Install server and desktop dependencies' \
		'make dev-server      Run the server on GPT-OSS-20B' \
		'make dev-server-120b Run the server on GPT-OSS-120B' \
		'make dev-desktop     Run the desktop control plane' \
		'make build-desktop   Build the macOS app and dmg' \
		'make doctor          Report environment readiness' \
		'make test            Run the Python test suite' \
		'make lint            Run ruff' \
		'make clean           Remove build/ and dist/'

install: install-server install-desktop

install-server:
	uv sync --project "$(SERVER_DIR)"

install-desktop:
	npm --prefix "$(DESKTOP_DIR)" install

dev-server:
	uv run --project "$(SERVER_DIR)" quantum-codex-server serve \
		--model "$(MODEL_20B)" --served-model-name gpt-oss-20b --port $(PORT)

dev-server-120b:
	uv run --project "$(SERVER_DIR)" quantum-codex-server serve \
		--model "$(MODEL_120B)" --served-model-name gpt-oss-120b --port $(PORT)

# The desktop app drives the same CLI a terminal would, so it needs to know
# how to invoke it from a development checkout rather than from PATH.
dev-desktop:
	QUANTUM_CODEX_COMMAND="uv run --project $(SERVER_DIR) quantum-codex-server" \
		npm --prefix "$(DESKTOP_DIR)" run app:dev

# `strip` removes symbols but not the dependency source paths rustc bakes into
# panic metadata, so a locally built release binary otherwise ships a few hundred
# strings naming the build machine's home directory. Cargo's own `trim-paths`
# is nightly-only as of 1.97; `--remap-path-prefix` is stable and does the same
# for the two roots that matter.
build-desktop:
	RUSTFLAGS="--remap-path-prefix=$(HOME)/.cargo=/cargo --remap-path-prefix=$(ROOT)=/src" \
		npm --prefix "$(DESKTOP_DIR)" run app:build

doctor:
	uv run --project "$(SERVER_DIR)" quantum-codex-server doctor

test:
	uv run --project "$(SERVER_DIR)" pytest

lint:
	uv run --project "$(SERVER_DIR)" ruff check "$(SERVER_DIR)"
	npm --prefix "$(DESKTOP_DIR)" run typecheck
	cd "$(DESKTOP_DIR)/src-tauri" && CARGO_TARGET_DIR="$(BUILD_DIR)/tauri" cargo check

# Exactly what .github/workflows/ci.yml runs, in the same order, so a green
# local run means a green CI run. It loads no model and reads no weights.
#
# What it deliberately does not cover: every witness that needs GPT-OSS weights
# on Apple Silicon. Those are run by hand
ci:
	uv run --project "$(SERVER_DIR)" ruff check "$(SERVER_DIR)"
	uv run --project "$(SERVER_DIR)" pytest
	npm --prefix "$(DESKTOP_DIR)" ci
	npm --prefix "$(DESKTOP_DIR)" run typecheck
	npm --prefix "$(DESKTOP_DIR)" test
	npm --prefix "$(DESKTOP_DIR)" run build
	cd "$(DESKTOP_DIR)/src-tauri" && cargo fmt --check
	cd "$(DESKTOP_DIR)/src-tauri" && cargo clippy --all-targets -- -D warnings
	cd "$(DESKTOP_DIR)/src-tauri" && cargo test

clean:
	rm -rf "$(BUILD_DIR)" "$(DIST_DIR)"
