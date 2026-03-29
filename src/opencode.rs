use crate::backends::Backend;
use crate::server::ServerHandle;
use serde_json::json;
use std::fs;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

pub struct OpenCodeSession {
    pub config_path: PathBuf,
    pub launch_command: String,
    pub base_url: String,
    pub model_id: String,
}

pub fn resolve_for_server(handle: &ServerHandle) -> Result<OpenCodeSession, String> {
    if !handle.backend.can_open_opencode() {
        return Err(format!(
            "{} is not yet supported for OpenCode launch",
            handle.backend.label()
        ));
    }

    let base_url = format!("{}/v1", handle.display_url());
    let model_id = fetch_openai_model_id(&base_url)?;
    let config_path = write_temp_config(&base_url, &model_id, &handle.backend)?;
    let launch_command = format!("OPENCODE_CONFIG={} opencode", config_path.display());

    Ok(OpenCodeSession {
        config_path,
        launch_command,
        base_url,
        model_id,
    })
}

pub fn launch(session: &OpenCodeSession) -> Result<bool, String> {
    if std::env::var_os("TMUX").is_some() {
        Command::new("tmux")
            .args([
                "new-window",
                "-n",
                "opencode",
                &format!("OPENCODE_CONFIG={} opencode", session.config_path.display()),
            ])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| format!("Failed to launch OpenCode in tmux: {e}"))?;
        return Ok(true);
    }

    Ok(false)
}

fn fetch_openai_model_id(base_url: &str) -> Result<String, String> {
    let url = format!("{base_url}/models");
    let mut response = ureq::get(&url)
        .call()
        .map_err(|e| format!("Failed to query running server models: {e}"))?;
    let body = response
        .body_mut()
        .read_to_string()
        .map_err(|e| format!("Failed to read server model list: {e}"))?;
    let json: serde_json::Value = serde_json::from_str(&body)
        .map_err(|e| format!("Failed to parse server model list: {e}"))?;
    json.get("data")
        .and_then(|data| data.as_array())
        .and_then(|models| models.first())
        .and_then(|model| model.get("id"))
        .and_then(|id| id.as_str())
        .map(|id| id.to_string())
        .ok_or_else(|| "Running server did not report an OpenAI-style model id".to_string())
}

fn write_temp_config(base_url: &str, model_id: &str, backend: &Backend) -> Result<PathBuf, String> {
    let provider_id = "llmserve-local";
    let payload = json!({
        "$schema": "https://opencode.ai/config.json",
        "model": format!("{provider_id}/{model_id}"),
        "provider": {
            provider_id: {
                "npm": "@ai-sdk/openai-compatible",
                "name": format!("{} (llmserve)", backend.label()),
                "options": {
                    "baseURL": base_url,
                    "apiKey": "llmserve"
                },
                "models": {
                    model_id: {
                        "name": format!("{} (llmserve)", model_id)
                    }
                }
            }
        }
    });

    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| format!("Clock error while preparing OpenCode config: {e}"))?
        .as_millis();
    let path = std::env::temp_dir().join(format!("llmserve-opencode-{stamp}.json"));
    let content = serde_json::to_vec_pretty(&payload)
        .map_err(|e| format!("Failed to encode OpenCode config: {e}"))?;
    fs::write(&path, content).map_err(|e| format!("Failed to write OpenCode config: {e}"))?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn launch_command_uses_opencode_config_env() {
        let session = OpenCodeSession {
            config_path: PathBuf::from("/tmp/example.json"),
            launch_command: "OPENCODE_CONFIG=/tmp/example.json opencode".into(),
            base_url: "http://127.0.0.1:8080/v1".into(),
            model_id: "test-model".into(),
        };

        assert!(session
            .launch_command
            .contains("OPENCODE_CONFIG=/tmp/example.json"));
    }

    #[test]
    fn write_temp_config_embeds_openai_compatible_provider() {
        let path = write_temp_config(
            "http://127.0.0.1:8080/v1",
            "qwen/test",
            &Backend::LlamaServer,
        )
        .unwrap();
        let content = fs::read_to_string(&path).unwrap();
        assert!(content.contains("@ai-sdk/openai-compatible"));
        assert!(content.contains("http://127.0.0.1:8080/v1"));
        assert!(content.contains("llmserve-local/qwen/test"));
        let _ = fs::remove_file(path);
    }
}
