.PHONY: build release run test check fmt clippy clean install

build:
	cargo build

release:
	cargo build --release

run:
	cargo run

test:
	cargo test -- --skip serve_and_rotate

test-all:
	cargo test

check:
	cargo check --all-targets --all-features

fmt:
	cargo fmt --all

clippy:
	cargo clippy --all-targets --all-features

clean:
	cargo clean

install:
	cargo install --path .
