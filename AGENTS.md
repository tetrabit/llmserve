# PROJECT KNOWLEDGE BASE

**Generated:** 2026-03-28 22:47 EDT
**Commit:** 26cee6d
**Branch:** main

## OVERVIEW
Rust terminal UI for discovering local LLM models, detecting inference backends, and launching/monitoring local servers from one TUI.
Primary stack: `ratatui` + `crossterm` UI, `ureq` for backend/API checks, `serde`/`toml` config, `walkdir` model discovery.

## STRUCTURE
```text
./
├── src/                  # runtime app, backend detection, model discovery, launch logic, UI
├── tests/                # integration tests; ignored tests require local models/backends
├── .github/workflows/    # CI, release packaging, docs deploy
├── Cargo.toml            # crate metadata + deps
├── Makefile              # canonical local dev commands
└── README.md             # user-facing behavior, keybindings, config contract
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| TUI bootstrap / terminal lifecycle | `src/main.rs` | alternate screen + raw mode setup/restore |
| App state and navigation | `src/app.rs` | biggest state hub; filters, popups, serving lifecycle |
| Keybindings / mode dispatch | `src/events.rs` | vim-style input handling |
| Rendering | `src/ui.rs` | all panels + popups; read with `app.rs` |
| Backend detection and compatibility rules | `src/backends.rs` | seven backends, local-serve constraints |
| Config defaults / preset resolution | `src/config.rs` | global defaults merged with per-backend presets |
| Model discovery | `src/models.rs` | GGUF + MLX + Ollama listing |
| Process launch / logs / stop | `src/server.rs` | backend-specific command construction |
| Integration test behavior | `tests/serve_integration.rs` | CI-safe tests + ignored local-serve rotation |

## CODE MAP
| Symbol | Type | Location | Role |
|--------|------|----------|------|
| `App` | struct | `src/app.rs` | central state for filters, popups, serving, layout |
| `build_tree` | fn | `src/app.rs` | derives source tree from discovered models |
| `handle_normal` | fn | `src/events.rs` | main keybinding router |
| `draw_model_table` | fn | `src/ui.rs` | center pane rendering |
| `draw_confirm_popup` | fn | `src/ui.rs` | serve dialog rendering |
| `detect_backends` | fn | `src/backends.rs` | ordered backend discovery entry point |
| `Config::preset_for` | fn | `src/config.rs` | merges backend preset with globals |
| `discover_models` | fn | `src/models.rs` | scans model sources and API-backed registries |
| `launch` | fn | `src/server.rs` | dispatches to backend-specific launchers |
| `serve_and_rotate_backends` | test | `tests/serve_integration.rs` | ignored end-to-end local serving check |

## CONVENTIONS
- Flat module layout under `src/`; no nested feature directories.
- Backend order is intentional: detection vector order drives UI/backend selection order.
- Config is tolerant by design: load/save paths swallow IO parse failures and fall back to defaults.
- Serve dialog edits port only; context/batch/gpu/thread values come from resolved config presets.
- Local-file serving rules are explicit in `Backend::can_serve_local*` and `local_serve_reason()`; keep UI messaging aligned.

## ANTI-PATTERNS (THIS PROJECT)
- Do not treat ignored integration tests as CI-safe; `serve_and_rotate_backends` expects real models/backends.
- Do not add backend capability checks in UI only; capability logic belongs in `src/backends.rs`.
- Do not bypass `Config::preset_for`; launchers should use resolved presets, not raw `Option` fields.
- Do not duplicate keybinding docs in code comments unless behavior diverges from `README.md`.

## UNIQUE STYLES
- UX is vim-leaning: `j/k`, `g/G`, `Ctrl-d/u`, `h/l` where sensible.
- Right panel combines live server cards with preserved logs from dead processes.
- Model/source discovery mixes filesystem scans with API-sourced registries (notably Ollama).
- Large-model heuristics in `src/server.rs` tune batch size automatically when presets omit it.

## COMMANDS
```bash
make build
make test
make test-local
make check
make fmt
make clippy
make install
```

## NOTES
- `edition = "2024"` in `Cargo.toml`.
- CI runs `cargo test --verbose`, rustfmt check, clippy, and `cargo check --all-targets --all-features`.
- Release workflow builds many targets, publishes crates.io, and updates a separate Homebrew tap.
- There were no pre-existing `AGENTS.md`/`CLAUDE.md` files in this repo when generated.
