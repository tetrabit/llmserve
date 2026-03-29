# TESTS KNOWLEDGE BASE

## OVERVIEW
Integration coverage for backend detection, config preset behavior, model discovery, and optional end-to-end local serving.

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| CI-safe config/backend assertions | `serve_integration.rs` | plain unit-style integration checks |
| Full local serve rotation | `serve_integration.rs::serve_and_rotate_backends` | `#[ignore]`; requires local models/backends |
| Minimal launch config | `serve_integration.rs::test_config` | forces low ctx/port/resource usage |
| HTTP readiness and API probes | `wait_for_ready`, `check_openai_models`, `check_completion` | use existing helper style |

## CONVENTIONS
- Ignored tests are intentional: local models and inference daemons are not available in CI.
- Synthetic-data tests should keep running without local model directories.
- End-to-end coverage prefers the smallest available model to reduce runtime/resource cost.
- Test config binds to `127.0.0.1` and lowers context size to `512` for portability.

## UNIQUE STYLES
- One file owns all current integration coverage; helpers at the top keep later tests terse.
- Ollama is treated specially in the rotation test: API reachability is checked instead of launching a local file-backed server.
- Health endpoint differs by backend (`/health` for most, `/v1/models` for MLX); preserve that distinction when extending readiness checks.

## ANTI-PATTERNS
- Do not un-ignore local serving tests in CI workflows.
- Do not make non-local tests depend on real model paths or running daemons.
- Do not hardcode a fixed port when `free_port()` already exists.
- Do not remove skip paths just to make a local-only backend look green in automation.
