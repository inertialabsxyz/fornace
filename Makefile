MODEL ?= ./models/qwen2.5-0.5b
PROMPT ?= The capital of Spain is
FEATURES ?= $(if $(filter Darwin,$(shell uname -s)),--features metal,)

.PHONY: check fmt fmt-check clippy test build run bench model

check: fmt-check clippy test

fmt:
	cargo fmt --all

fmt-check:
	cargo fmt --all --check

clippy:
	cargo clippy --workspace --all-targets -- -D warnings

test:
	cargo test --workspace

build:
	cargo build --workspace --release $(FEATURES)

run:
	cargo run --release $(FEATURES) -p fornace-engine --bin fornace -- --model $(MODEL) --prompt "$(PROMPT)"

bench:
	cargo run --release $(FEATURES) -p fornace-bench -- --model $(MODEL)

model:
	scripts/get-model.sh
