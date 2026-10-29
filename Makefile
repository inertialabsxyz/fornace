MODEL ?= ./models/qwen2.5-0.5b-instruct
PROMPT ?= The capital of Spain is
FEATURES ?= $(if $(filter Darwin,$(shell uname -s)),--features metal,)

.DEFAULT_GOAL := help
.PHONY: help check fmt fmt-check clippy test build run bench model

help: ## Show this help
	@echo "Usage: make <target> [MODEL=dir] [PROMPT=text]"
	@echo
	@awk 'BEGIN {FS = ":.*## "} /^[a-z-]+:.*## / {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo
	@echo "MODEL defaults to $(MODEL)"
	@echo "On macOS, build, run, and bench use Metal."

check: fmt-check clippy test ## Run the format check, clippy, and the tests

fmt: ## Format the code
	cargo fmt --all

fmt-check: ## Check the format of the code
	cargo fmt --all --check

clippy: ## Run clippy with warnings as errors
	cargo clippy --workspace --all-targets -- -D warnings

test: ## Run the tests (set FORNACE_MODEL_DIR to run the model test)
	cargo test --workspace

build: ## Build the workspace in release mode
	cargo build --workspace --release $(FEATURES)

run: ## Generate text from PROMPT with the model in MODEL
	cargo run --release $(FEATURES) -p fornace-engine --bin fornace -- --model $(MODEL) --prompt "$(PROMPT)" --max-new-tokens 1024 --json --out out.json

bench: ## Run the benchmark harness on the model in MODEL
	cargo run --release $(FEATURES) -p fornace-bench -- --model $(MODEL)

model: ## Download Qwen2.5-0.5B to ./models/qwen2.5-0.5b
	scripts/get-model.sh
