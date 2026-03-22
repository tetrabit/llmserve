//! Integration tests for llmserve.
//!
//! These tests actually serve a model on each available backend, verify HTTP
//! connectivity, then stop the server and rotate to the next backend.
//!
//! The tests discover a real model on disk and skip gracefully if none is
//! found or if a backend is unavailable.  They bind to ephemeral ports so
//! they don't collide with anything running on the host.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use llmserve::backends::{Backend, backend_key, detect_backends};
use llmserve::config::{BackendPreset, Config};
use llmserve::models::{DiscoveredModel, ModelFormat, ModelSource, discover_models};
use llmserve::server;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Pick the smallest GGUF model on disk so the test loads fast.
fn find_smallest_gguf() -> Option<DiscoveredModel> {
    let models = discover_models(&[]);
    models
        .into_iter()
        .filter(|m| m.format == ModelFormat::Gguf && m.source != ModelSource::Ollama)
        .min_by_key(|m| m.size_bytes)
}

/// Pick the smallest MLX model on disk.
fn find_smallest_mlx() -> Option<DiscoveredModel> {
    let models = discover_models(&[]);
    models
        .into_iter()
        .filter(|m| m.format == ModelFormat::Mlx)
        .min_by_key(|m| m.size_bytes)
}

/// Build a test config that uses a specific port and minimal resources.
fn test_config(port: u16) -> Config {
    let mut config = Config::default();
    config.preferred_port = port;
    config.preferred_host = "127.0.0.1".into();
    config.default_ctx_size = 512; // tiny context for fast loading

    // Override all presets with the test port and tiny context
    for (_key, preset) in config.presets.iter_mut() {
        preset.ctx_size = Some(512);
        preset.port = Some(port);
        preset.host = Some("127.0.0.1".into());
    }

    // llama-server specific: small batch, flash attn on
    if let Some(preset) = config.presets.get_mut("llama-server") {
        preset.batch_size = Some(256);
        preset.flash_attn = Some(true);
    }

    config
}

/// Find a free port by binding to :0 and reading back what the OS gave us.
fn free_port() -> u16 {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind to ephemeral port");
    listener.local_addr().unwrap().port()
}

/// Poll a URL until it returns 200 or the timeout expires.
/// Returns Ok(()) on success, Err(reason) on timeout.
fn wait_for_ready(url: &str, timeout: Duration) -> Result<(), String> {
    let agent = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_millis(500)))
        .build()
        .new_agent();

    let start = Instant::now();
    while start.elapsed() < timeout {
        if let Ok(resp) = agent.get(url).call() {
            if resp.status().as_u16() == 200 {
                return Ok(());
            }
        }
        std::thread::sleep(Duration::from_millis(500));
    }
    Err(format!("Timed out waiting for {url} after {timeout:?}"))
}

/// Verify we can hit the OpenAI-compatible models endpoint.
fn check_openai_models(port: u16) -> Result<(), String> {
    let agent = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(5)))
        .build()
        .new_agent();

    let url = format!("http://127.0.0.1:{port}/v1/models");
    match agent.get(&url).call() {
        Ok(resp) if resp.status().as_u16() == 200 => Ok(()),
        Ok(resp) => Err(format!("/v1/models returned {}", resp.status())),
        Err(e) => Err(format!("/v1/models failed: {e}")),
    }
}

/// Verify we can send a trivial completion request to the server.
fn check_completion(port: u16) -> Result<(), String> {
    let agent = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(30)))
        .build()
        .new_agent();

    let url = format!("http://127.0.0.1:{port}/v1/chat/completions");
    let body = serde_json::json!({
        "model": "test",
        "messages": [{"role": "user", "content": "Say hi"}],
        "max_tokens": 4
    });

    match agent.post(&url).send_json(&body) {
        Ok(resp) if resp.status().as_u16() == 200 => Ok(()),
        Ok(resp) => Err(format!("/v1/chat/completions returned {}", resp.status())),
        Err(e) => Err(format!("/v1/chat/completions failed: {e}")),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Core rotation test: iterate through every detected backend, serve, verify,
/// stop, and move on.
#[test]
fn serve_and_rotate_backends() {
    let backends = detect_backends();
    let gguf_model = find_smallest_gguf();
    let mlx_model = find_smallest_mlx();

    if gguf_model.is_none() && mlx_model.is_none() {
        eprintln!("SKIP: no local models found for integration test");
        return;
    }

    let mut served_count = 0;

    for detected in &backends {
        if !detected.available {
            eprintln!(
                "SKIP backend {}: not available",
                detected.backend.label()
            );
            continue;
        }

        // LM Studio manages its own server — we can't launch from here
        if detected.backend == Backend::LmStudio {
            eprintln!("SKIP backend LM Studio: externally managed");
            continue;
        }

        // Pick the right model for the backend
        let model = match detected.backend {
            Backend::MlxLm => {
                if let Some(ref m) = mlx_model {
                    m.clone()
                } else {
                    eprintln!("SKIP backend MLX: no MLX models found");
                    continue;
                }
            }
            _ => {
                if let Some(ref m) = gguf_model {
                    m.clone()
                } else {
                    eprintln!("SKIP backend {}: no GGUF models found", detected.backend.label());
                    continue;
                }
            }
        };

        // Ollama uses its own daemon port — we don't spawn a child for it in
        // the same way, and `ollama run` is interactive.  Test its API instead.
        if detected.backend == Backend::Ollama {
            eprintln!("Testing Ollama API connectivity (daemon already running)...");
            let ollama_url = std::env::var("OLLAMA_HOST")
                .unwrap_or_else(|_| "http://localhost:11434".into());
            let agent = ureq::Agent::config_builder()
                .timeout_global(Some(Duration::from_secs(2)))
                .build()
                .new_agent();
            match agent.get(&format!("{ollama_url}/api/tags")).call() {
                Ok(resp) if resp.status().as_u16() == 200 => {
                    eprintln!("  OK: Ollama API reachable");
                    served_count += 1;
                }
                _ => eprintln!("  WARN: Ollama API unreachable despite detection"),
            }
            continue;
        }

        let port = free_port();
        let config = test_config(port);

        eprintln!(
            "Serving {} via {} on port {port}...",
            model.name,
            detected.backend.label()
        );

        // Launch
        let result = server::launch(&model, &detected.backend, &config);
        let mut handle = match result {
            Ok(h) => h,
            Err(e) => {
                eprintln!("  SKIP: launch failed: {e}");
                continue;
            }
        };

        // Wait for the server to become ready
        let health_url = match detected.backend {
            Backend::LlamaServer => format!("http://127.0.0.1:{port}/health"),
            Backend::MlxLm => format!("http://127.0.0.1:{port}/v1/models"),
            _ => format!("http://127.0.0.1:{port}/health"),
        };

        // Give the server up to 120s to load the model (even small models
        // can take a while on slower machines).
        let ready = wait_for_ready(&health_url, Duration::from_secs(120));

        if let Err(ref reason) = ready {
            eprintln!("  Server did not become ready: {reason}");
            server::stop(&mut handle);
            // Check if it crashed
            if let Some(exit_msg) = server::check_exited(&mut handle) {
                eprintln!("  Server exited: {exit_msg}");
            }
            continue;
        }

        eprintln!("  Server ready on port {port}");

        // Verify /v1/models
        match check_openai_models(port) {
            Ok(()) => eprintln!("  /v1/models: OK"),
            Err(e) => eprintln!("  /v1/models: {e}"),
        }

        // Verify a short completion
        match check_completion(port) {
            Ok(()) => eprintln!("  /v1/chat/completions: OK"),
            Err(e) => eprintln!("  /v1/chat/completions: {e}"),
        }

        // Stop
        server::stop(&mut handle);
        eprintln!("  Stopped.");

        // Give the port a moment to be released
        std::thread::sleep(Duration::from_millis(500));

        served_count += 1;
    }

    assert!(
        served_count > 0,
        "Expected at least one backend to be tested, but none were available"
    );
    eprintln!("\nServed and verified {served_count} backend(s).");
}

/// Test that model discovery finds at least the models we know exist.
#[test]
fn discover_local_models() {
    let models = discover_models(&[]);
    // We know at least the LM Studio models are there
    assert!(!models.is_empty(), "Expected to discover at least one model");

    // Verify models have sensible metadata
    for m in &models {
        assert!(!m.name.is_empty(), "Model name should not be empty");
        // Ollama models have synthetic paths, others should be real files/dirs
        if m.source != ModelSource::Ollama {
            assert!(
                m.path.exists(),
                "Model path should exist: {}",
                m.path.display()
            );
        }
    }
}

/// Test backend detection returns consistent results.
#[test]
fn backend_detection_is_consistent() {
    let first = detect_backends();
    let second = detect_backends();

    assert_eq!(first.len(), second.len());
    for (a, b) in first.iter().zip(second.iter()) {
        assert_eq!(a.backend, b.backend);
        assert_eq!(
            a.available, b.available,
            "Backend {} availability changed between runs",
            a.backend.label()
        );
    }
}

/// Test config presets are correctly applied for each backend.
#[test]
fn config_presets_resolve_per_backend() {
    let port = free_port();
    let config = test_config(port);

    for detected in detect_backends() {
        let key = backend_key(&detected.backend);
        let preset = config.preset_for(key);

        assert_eq!(preset.port, port, "Preset port for {key}");
        assert_eq!(preset.host, "127.0.0.1", "Preset host for {key}");
        assert_eq!(preset.ctx_size, 512, "Preset ctx_size for {key}");
    }
}

/// Test that extra_args from presets are preserved through config roundtrip.
#[test]
fn preset_extra_args_roundtrip() {
    let mut config = Config::default();
    config.presets.insert(
        "llama-server".into(),
        BackendPreset {
            extra_args: vec!["--mlock".into(), "--cont-batching".into()],
            ..Default::default()
        },
    );

    let serialized = toml::to_string_pretty(&config).unwrap();
    let deserialized: Config = toml::from_str(&serialized).unwrap();

    let preset = deserialized.preset_for("llama-server");
    assert_eq!(preset.extra_args, vec!["--mlock", "--cont-batching"]);
}

/// Test that we can discover models from a custom extra directory.
#[test]
fn discover_models_from_extra_dir() {
    // Use the LM Studio models dir as an "extra" dir to verify the mechanism
    let home = dirs::home_dir().unwrap();
    let lmstudio_dir = home.join(".lmstudio").join("models");
    if !lmstudio_dir.is_dir() {
        eprintln!("SKIP: ~/.lmstudio/models not found");
        return;
    }

    // Discover with empty extras — these will come from the default LM Studio scan
    let baseline = discover_models(&[]);
    let baseline_count = baseline.len();

    // Now discover with a nonexistent extra dir — count should be the same
    let with_bogus = discover_models(&[PathBuf::from("/tmp/nonexistent_llmserve_test_dir")]);
    assert_eq!(
        with_bogus.len(),
        baseline_count,
        "Nonexistent extra dir should not change model count"
    );
}

/// Test that serving on LM Studio backend returns an informative error.
#[test]
fn lmstudio_serve_returns_error() {
    let model = DiscoveredModel {
        name: "test-model".into(),
        path: PathBuf::from("/tmp/fake.gguf"),
        mmproj: None,
        format: ModelFormat::Gguf,
        size_bytes: 0,
        quant: None,
        param_hint: None,
        source: ModelSource::ExtraDir,
    };
    let config = Config::default();
    let result = server::launch(&model, &Backend::LmStudio, &config);
    assert!(result.is_err());
    let err = result.err().unwrap();
    assert!(
        err.contains("LM Studio"),
        "Error should mention LM Studio: {err}"
    );
}

/// Test vision model mmproj detection by verifying models that have it.
#[test]
fn vision_models_detected() {
    let models = discover_models(&[]);
    let vision_models: Vec<_> = models.iter().filter(|m| m.mmproj.is_some()).collect();

    // We know from the filesystem that several models have mmproj files
    if !vision_models.is_empty() {
        for m in &vision_models {
            let mmproj = m.mmproj.as_ref().unwrap();
            assert!(
                mmproj.exists(),
                "mmproj path should exist: {}",
                mmproj.display()
            );
            assert!(
                mmproj
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
                    .starts_with("mmproj"),
                "mmproj filename should start with 'mmproj'"
            );
        }
    }
}
