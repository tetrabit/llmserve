# OpenCode integration plan

## Goal

Let `llmserve` launch OpenCode against the currently selected or already-served model when the backend exposes a compatible API.

## User-facing behavior

### Primary entry point

- Add a new hotkey: `O`
- `O` should target the currently selected model

### If the selected model is already being served

- Reuse the active server when its backend exposes an OpenAI-compatible API
- Launch OpenCode pointed at that server and model
- Show a status message describing which model/backend/url OpenCode is using

### If the selected model is not being served

- Preferred UX: prompt the user to confirm **Serve + Open OpenCode**
- After a successful launch, start OpenCode against the newly started server
- If serving fails, do not launch OpenCode

## Compatibility scope

Start with backends that are already expected to expose OpenAI-compatible APIs:

- `llama-server`
- `vLLM`
- `LocalAI`

Backends to evaluate separately before enabling:

- `Ollama`
- `mlx-lm`
- `LM Studio`
- `KoboldCpp`

## Implementation plan

### 1. Capability modeling

Add a backend capability check for whether a backend can be used to launch OpenCode.

Possible API:

- `Backend::can_open_opencode()`

This should be distinct from local-file serving support because OpenCode integration depends on API compatibility, not just launchability.

### 2. Active server resolution

Add an app helper that finds the running server for the selected model and backend.

Needs to answer:

- is this model already running?
- which backend is serving it?
- what URL should OpenCode use?

### 3. OpenCode command builder

Add a small launcher/helper that prepares the OpenCode process.

Expected inputs:

- base URL
- model identifier
- backend metadata if needed

Likely environment/arguments:

- `OPENAI_BASE_URL=<server>/v1`
- `OPENAI_API_KEY=dummy` (or another harmless placeholder when required)
- model name passed in the way OpenCode expects

### 4. Serve-then-open flow

If the model is not running:

- open a confirmation flow
- launch the selected backend/model first
- wait until the server is actually usable
- only then start OpenCode

This should reuse the existing serve dialog and server lifecycle code where possible.

### 5. Status and errors

Add clear status messages for:

- unsupported backend
- selected model not yet served
- server launch failed
- OpenCode launch failed
- OpenCode launched successfully

## Safety constraints

- Never assume every backend uses the same API path or model naming behavior
- Only enable OpenCode launch for backends that have been explicitly validated
- Do not auto-serve without confirmation
- Prefer reusing an existing running server over starting a duplicate one

## Suggested development order

1. Add backend compatibility helpers
2. Add active server resolution in `app.rs`
3. Add OpenCode command/launcher helper
4. Add `O` hotkey for already-running compatible servers
5. Add confirm flow for **Serve + Open OpenCode**
6. Add docs and tests

## Nice follow-ups

- show `o:opencode` in the serve pane for running compatible servers
- cache the last successful OpenCode launch target
- support additional backends once their API compatibility is confirmed
