use std::process::Command;
use std::time::Duration;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Backend {
    LlamaServer,
    Ollama,
    MlxLm,
    LmStudio,
}

impl Backend {
    pub fn label(&self) -> &'static str {
        match self {
            Backend::LlamaServer => "llama-server",
            Backend::Ollama => "Ollama",
            Backend::MlxLm => "MLX",
            Backend::LmStudio => "LM Studio",
        }
    }

    pub fn can_serve_gguf(&self) -> bool {
        matches!(self, Backend::LlamaServer | Backend::Ollama | Backend::LmStudio)
    }

    pub fn can_serve_mlx(&self) -> bool {
        matches!(self, Backend::MlxLm)
    }
}

#[derive(Debug, Clone)]
pub struct DetectedBackend {
    pub backend: Backend,
    pub available: bool,
    pub binary_path: Option<String>,
    pub api_url: Option<String>,
}

impl DetectedBackend {
    pub fn status_label(&self) -> &'static str {
        if self.available {
            "ready"
        } else {
            "not found"
        }
    }
}

fn http_agent() -> ureq::Agent {
    ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_millis(800)))
        .build()
        .new_agent()
}

pub fn detect_backends() -> Vec<DetectedBackend> {
    vec![
        detect_llama_server(),
        detect_ollama(),
        detect_mlx(),
        detect_lmstudio(),
    ]
}

/// Returns Ollama model list: Vec<(name, size_bytes)>
pub fn fetch_ollama_models(api_url: &str) -> Vec<(String, u64)> {
    let url = format!("{api_url}/api/tags");
    let agent = http_agent();
    let Ok(mut response) = agent.get(&url).call() else {
        return Vec::new();
    };
    let Ok(body) = response.body_mut().read_to_string() else {
        return Vec::new();
    };
    let Ok(json) = serde_json::from_str::<serde_json::Value>(&body) else {
        return Vec::new();
    };
    let Some(models) = json.get("models").and_then(|m| m.as_array()) else {
        return Vec::new();
    };
    models
        .iter()
        .filter_map(|m| {
            let name = m.get("name")?.as_str()?.to_string();
            let size = m.get("size")?.as_u64().unwrap_or(0);
            Some((name, size))
        })
        .collect()
}

fn detect_llama_server() -> DetectedBackend {
    let result = Command::new("which")
        .arg("llama-server")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string());

    DetectedBackend {
        backend: Backend::LlamaServer,
        available: result.is_some(),
        binary_path: result,
        api_url: None,
    }
}

fn detect_ollama() -> DetectedBackend {
    let url = std::env::var("OLLAMA_HOST").unwrap_or_else(|_| "http://localhost:11434".into());
    let agent = http_agent();
    let available = agent.get(&format!("{url}/api/tags")).call().is_ok();

    DetectedBackend {
        backend: Backend::Ollama,
        available,
        binary_path: None,
        api_url: Some(url),
    }
}

fn detect_mlx() -> DetectedBackend {
    if !cfg!(target_os = "macos") {
        return DetectedBackend {
            backend: Backend::MlxLm,
            available: false,
            binary_path: None,
            api_url: None,
        };
    }

    let available = Command::new("python3")
        .args(["-c", "import mlx_lm"])
        .output()
        .ok()
        .is_some_and(|o| o.status.success());

    DetectedBackend {
        backend: Backend::MlxLm,
        available,
        binary_path: None,
        api_url: None,
    }
}

pub fn backend_key(backend: &Backend) -> &'static str {
    match backend {
        Backend::LlamaServer => "llama-server",
        Backend::Ollama => "ollama",
        Backend::MlxLm => "mlx",
        Backend::LmStudio => "lm-studio",
    }
}

fn detect_lmstudio() -> DetectedBackend {
    let url = std::env::var("LMSTUDIO_HOST").unwrap_or_else(|_| "http://127.0.0.1:1234".into());
    let agent = http_agent();
    let available = agent.get(&format!("{url}/v1/models")).call().is_ok();

    DetectedBackend {
        backend: Backend::LmStudio,
        available,
        binary_path: None,
        api_url: Some(url),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_labels() {
        assert_eq!(Backend::LlamaServer.label(), "llama-server");
        assert_eq!(Backend::Ollama.label(), "Ollama");
        assert_eq!(Backend::MlxLm.label(), "MLX");
        assert_eq!(Backend::LmStudio.label(), "LM Studio");
    }

    #[test]
    fn backend_format_support() {
        assert!(Backend::LlamaServer.can_serve_gguf());
        assert!(!Backend::LlamaServer.can_serve_mlx());
        assert!(Backend::MlxLm.can_serve_mlx());
        assert!(!Backend::MlxLm.can_serve_gguf());
        assert!(Backend::Ollama.can_serve_gguf());
    }

    #[test]
    fn backend_keys() {
        assert_eq!(backend_key(&Backend::LlamaServer), "llama-server");
        assert_eq!(backend_key(&Backend::Ollama), "ollama");
        assert_eq!(backend_key(&Backend::MlxLm), "mlx");
        assert_eq!(backend_key(&Backend::LmStudio), "lm-studio");
    }

    #[test]
    fn detected_backend_status_labels() {
        let available = DetectedBackend {
            backend: Backend::LlamaServer,
            available: true,
            binary_path: Some("/usr/local/bin/llama-server".into()),
            api_url: None,
        };
        assert_eq!(available.status_label(), "ready");

        let unavailable = DetectedBackend {
            backend: Backend::Ollama,
            available: false,
            binary_path: None,
            api_url: None,
        };
        assert_eq!(unavailable.status_label(), "not found");
    }

    #[test]
    fn detect_backends_returns_four() {
        let backends = detect_backends();
        assert_eq!(backends.len(), 4);
        assert_eq!(backends[0].backend, Backend::LlamaServer);
        assert_eq!(backends[1].backend, Backend::Ollama);
        assert_eq!(backends[2].backend, Backend::MlxLm);
        assert_eq!(backends[3].backend, Backend::LmStudio);
    }
}
