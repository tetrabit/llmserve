# SRC KNOWLEDGE BASE

## OVERVIEW
Runtime core for the TUI: state, input dispatch, rendering, discovery, backend detection, config, and server process management.

## STRUCTURE
```text
src/
├── app.rs       # state machine + interactions
├── ui.rs        # rendering layer
├── events.rs    # key dispatch by input mode
├── backends.rs  # backend detection + compatibility rules
├── config.rs    # config model + preset resolution
├── models.rs    # filesystem/API model discovery
├── server.rs    # launch, log capture, stop/check_exited
├── theme.rs     # theme palette cycling
└── lib.rs       # module exports for tests/main
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Add/change keybindings | `events.rs` + `README.md` | keep docs and behavior synced |
| Change popup behavior | `app.rs` + `ui.rs` | state in app, rendering in ui |
| Add backend | `backends.rs`, `config.rs`, `server.rs`, `app.rs`, `README.md` | detection + preset + launch + dialog messaging |
| Change model scanning | `models.rs` | preserve dedupe + source labeling |
| Change serve defaults | `config.rs` | prefer preset resolution over ad hoc logic |
| Change process/log handling | `server.rs` | nonblocking stdout/stderr logic lives here |

## CONVENTIONS
- `app.rs` is the orchestration hub; prefer thin event handlers that call `App` methods.
- `ui.rs` reads state; avoid burying business rules in rendering.
- Backend keys are lowercase labels (`llama-server`, `ollama`, `mlx`, `lm-studio`, `vllm`, `koboldcpp`, `localai`).
- If a new backend cannot serve local files, encode the reason in `local_serve_reason()` so the dialog can explain it.

## HOTSPOTS
- `app.rs` (~1100 lines): state-heavy; easiest place to create regressions in selection/filter/popup flow.
- `ui.rs` (~800 lines): layout and truncation logic; verify visually after changes.
- `models.rs` / `server.rs`: discovery and launching are side-effect heavy; prefer narrow edits.

## ANTI-PATTERNS
- Do not add direct mutation from `events.rs` when an `App` method already encapsulates behavior.
- Do not read raw preset `Option`s in launch code; resolve once through `Config::preset_for()`.
- Do not let README keybindings drift from `events.rs`.
- Do not mix rendering-only formatting with persistent app state unless the state is user-controlled.
