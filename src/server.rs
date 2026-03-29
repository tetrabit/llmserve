use crate::backends::Backend;
use crate::config::Config;
use crate::models::DiscoveredModel;
use std::collections::VecDeque;
use std::io::Read;
use std::process::{Child, Command, Stdio};
use std::time::Instant;

const MAX_LOG_LINES: usize = 200;

pub struct ServerHandle {
    pub backend: Backend,
    pub model_name: String,
    pub pid: u32,
    pub port: u16,
    pub host: String,
    pub child: Child,
    pub started_at: Instant,
    /// Ring buffer of log lines (combined stdout + stderr).
    pub log_lines: VecDeque<String>,
    /// Partial line buffer for incomplete reads.
    partial: String,
}

impl ServerHandle {
    pub fn uptime_display(&self) -> String {
        let secs = self.started_at.elapsed().as_secs();
        if secs < 60 {
            format!("{secs}s")
        } else if secs < 3600 {
            format!("{}m {}s", secs / 60, secs % 60)
        } else {
            format!("{}h {}m", secs / 3600, (secs % 3600) / 60)
        }
    }

    pub fn display_url(&self) -> String {
        format!("http://{}:{}", self.host, self.port)
    }

    /// Read any available output from stderr (non-blocking).
    pub fn drain_output(&mut self) {
        let Some(stderr) = self.child.stderr.as_mut() else {
            return;
        };

        let mut buf = [0u8; 4096];
        // Non-blocking read — will return WouldBlock if nothing available
        loop {
            match stderr.read(&mut buf) {
                Ok(0) => break, // EOF
                Ok(n) => {
                    let chunk = String::from_utf8_lossy(&buf[..n]);
                    self.partial.push_str(&chunk);

                    // Split into complete lines
                    while let Some(pos) = self.partial.find('\n') {
                        let line: String = self.partial.drain(..=pos).collect();
                        let line = line.trim_end_matches('\n').trim_end_matches('\r');
                        self.log_lines.push_back(line.to_string());
                        if self.log_lines.len() > MAX_LOG_LINES {
                            self.log_lines.pop_front();
                        }
                    }

                    // Handle \r (carriage return) for progress lines
                    if self.partial.contains('\r') {
                        let last = self.partial.rsplit('\r').next().unwrap_or("").to_string();
                        if !last.is_empty() {
                            // Replace last line if it was a progress update
                            if let Some(back) = self.log_lines.back_mut() {
                                if back.contains('\r') || back.contains('%') || back.contains("...")
                                {
                                    *back = last.clone();
                                } else {
                                    self.log_lines.push_back(last.clone());
                                }
                            } else {
                                self.log_lines.push_back(last.clone());
                            }
                        }
                        self.partial.clear();
                        self.partial.push_str(&last);
                    }
                }
                Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => break,
                Err(_) => break,
            }
        }

        // Also try stdout
        let Some(stdout) = self.child.stdout.as_mut() else {
            return;
        };
        loop {
            match stdout.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => {
                    let chunk = String::from_utf8_lossy(&buf[..n]);
                    for line in chunk.lines() {
                        self.log_lines.push_back(line.to_string());
                        if self.log_lines.len() > MAX_LOG_LINES {
                            self.log_lines.pop_front();
                        }
                    }
                }
                Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => break,
                Err(_) => break,
            }
        }
    }
}

fn set_nonblocking(child: &mut Child) {
    #[cfg(unix)]
    {
        use std::os::unix::io::AsRawFd;
        if let Some(ref stderr) = child.stderr {
            let fd = stderr.as_raw_fd();
            unsafe {
                let flags = libc::fcntl(fd, libc::F_GETFL);
                libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK);
            }
        }
        if let Some(ref stdout) = child.stdout {
            let fd = stdout.as_raw_fd();
            unsafe {
                let flags = libc::fcntl(fd, libc::F_GETFL);
                libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK);
            }
        }
    }

    // Windows: would need winapi/windows-sys crate for SetNamedPipeHandleState(PIPE_NOWAIT)
    #[cfg(windows)]
    let _ = child;
}

fn make_handle(
    backend: Backend,
    model: &DiscoveredModel,
    port: u16,
    host: String,
    mut child: Child,
) -> ServerHandle {
    set_nonblocking(&mut child);
    let pid = child.id();
    ServerHandle {
        backend,
        model_name: model.name.clone(),
        pid,
        port,
        host,
        child,
        started_at: Instant::now(),
        log_lines: VecDeque::new(),
        partial: String::new(),
    }
}

#[cfg(test)]
pub(crate) fn make_test_handle(
    backend: Backend,
    model_name: String,
    host: String,
    port: u16,
    child: Child,
) -> ServerHandle {
    ServerHandle {
        backend,
        model_name,
        pid: child.id(),
        port,
        host,
        child,
        started_at: Instant::now(),
        log_lines: VecDeque::new(),
        partial: String::new(),
    }
}

pub fn launch(
    model: &DiscoveredModel,
    backend: &Backend,
    config: &Config,
) -> Result<ServerHandle, String> {
    if !backend.can_serve_model(model) {
        let reason = backend
            .serve_model_reason(model)
            .unwrap_or("incompatible model");
        return Err(format!(
            "{} cannot serve {}: {}",
            backend.label(),
            model.name,
            reason
        ));
    }

    match backend {
        Backend::LlamaServer => launch_llama_server(model, config),
        Backend::MlxLm => launch_mlx(model, config),
        Backend::Vllm => launch_vllm(model, config),
        Backend::KoboldCpp => launch_koboldcpp(model, config),
        Backend::LocalAi => launch_localai(model, config),
        // These are blocked by the can_serve_local check above,
        // but match exhaustively for safety.
        Backend::Ollama | Backend::LmStudio => Err(format!(
            "{} cannot serve local model files",
            backend.label()
        )),
    }
}

pub fn launch_on_port(
    model: &DiscoveredModel,
    backend: &Backend,
    config: &Config,
    port: u16,
) -> Result<ServerHandle, String> {
    let preset_ctx = config
        .preset_for(crate::backends::backend_key(backend))
        .ctx_size;
    launch_with_overrides(model, backend, config, port, preset_ctx)
}

pub fn launch_with_overrides(
    model: &DiscoveredModel,
    backend: &Backend,
    config: &Config,
    port: u16,
    ctx_size: u32,
) -> Result<ServerHandle, String> {
    let mut config = config.clone();
    let key = crate::backends::backend_key(backend);
    let preset = config.presets.entry(key.to_string()).or_default();
    preset.port = Some(port);
    preset.ctx_size = Some(ctx_size);
    config.preferred_port = port;
    config.default_ctx_size = ctx_size;
    launch(model, backend, &config)
}

fn launch_llama_server(model: &DiscoveredModel, config: &Config) -> Result<ServerHandle, String> {
    let preset = config.preset_for("llama-server");

    let mut cmd = Command::new("llama-server");
    cmd.arg("--model")
        .arg(&model.path)
        .arg("--ctx-size")
        .arg(preset.ctx_size.to_string())
        .arg("--host")
        .arg(&preset.host)
        .arg("--port")
        .arg(preset.port.to_string());

    if preset.flash_attn {
        cmd.arg("--flash-attn").arg("on");
    }

    if let Some(batch_size) = preset.batch_size {
        cmd.arg("--batch-size").arg(batch_size.to_string());
    } else if is_large_model(model) {
        cmd.arg("--batch-size").arg("512");
    }

    if let Some(gpu_layers) = preset.gpu_layers {
        cmd.arg("-ngl").arg(gpu_layers.to_string());
    }

    if let Some(threads) = preset.threads {
        cmd.arg("--threads").arg(threads.to_string());
    }

    if let Some(ref mmproj) = model.mmproj {
        cmd.arg("--mmproj").arg(mmproj);
    }

    for arg in &preset.extra_args {
        cmd.arg(arg);
    }

    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());

    let child = cmd
        .spawn()
        .map_err(|e| format!("Failed to start llama-server: {e}"))?;

    Ok(make_handle(
        Backend::LlamaServer,
        model,
        preset.port,
        preset.host,
        child,
    ))
}

fn launch_mlx(model: &DiscoveredModel, config: &Config) -> Result<ServerHandle, String> {
    let preset = config.preset_for("mlx");

    let mut cmd = Command::new("python3");
    cmd.arg("-m")
        .arg("mlx_lm.server")
        .arg("--model")
        .arg(&model.path)
        .arg("--port")
        .arg(preset.port.to_string());

    for arg in &preset.extra_args {
        cmd.arg(arg);
    }

    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());

    let child = cmd
        .spawn()
        .map_err(|e| format!("Failed to start mlx_lm.server: {e}"))?;

    Ok(make_handle(
        Backend::MlxLm,
        model,
        preset.port,
        preset.host,
        child,
    ))
}

fn build_vllm_command(model: &DiscoveredModel, preset: &crate::config::ResolvedPreset) -> Command {
    let mut cmd = Command::new("vllm");
    cmd.arg("serve")
        .arg(&model.path)
        .arg("--host")
        .arg(&preset.host)
        .arg("--port")
        .arg(preset.port.to_string())
        .arg("--max-model-len")
        .arg(preset.ctx_size.to_string())
        .arg("--generation-config")
        .arg("vllm");

    if model.format == crate::models::ModelFormat::Gguf {
        cmd.arg("--load-format").arg("gguf");
    }

    for arg in &preset.extra_args {
        cmd.arg(arg);
    }

    cmd
}

fn launch_vllm(model: &DiscoveredModel, config: &Config) -> Result<ServerHandle, String> {
    let preset = config.preset_for("vllm");
    let mut cmd = build_vllm_command(model, &preset);

    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());

    let child = cmd
        .spawn()
        .map_err(|e| format!("Failed to start vllm serve: {e}"))?;

    Ok(make_handle(
        Backend::Vllm,
        model,
        preset.port,
        preset.host,
        child,
    ))
}

fn launch_koboldcpp(model: &DiscoveredModel, config: &Config) -> Result<ServerHandle, String> {
    let preset = config.preset_for("koboldcpp");

    let mut cmd = Command::new("koboldcpp");
    cmd.arg("--model")
        .arg(&model.path)
        .arg("--host")
        .arg(&preset.host)
        .arg("--port")
        .arg(preset.port.to_string())
        .arg("--contextsize")
        .arg(preset.ctx_size.to_string());

    if let Some(gpu_layers) = preset.gpu_layers {
        cmd.arg("--gpulayers").arg(gpu_layers.to_string());
    }

    if let Some(threads) = preset.threads {
        cmd.arg("--threads").arg(threads.to_string());
    }

    for arg in &preset.extra_args {
        cmd.arg(arg);
    }

    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());

    let child = cmd
        .spawn()
        .map_err(|e| format!("Failed to start koboldcpp: {e}"))?;

    Ok(make_handle(
        Backend::KoboldCpp,
        model,
        preset.port,
        preset.host,
        child,
    ))
}

fn launch_localai(model: &DiscoveredModel, config: &Config) -> Result<ServerHandle, String> {
    let preset = config.preset_for("localai");

    // LocalAI serves models from a directory. We point --models-path at the
    // parent directory of the GGUF file so it discovers it automatically.
    let models_dir = model
        .path
        .parent()
        .ok_or_else(|| "Cannot determine model directory".to_string())?;

    let mut cmd = Command::new("local-ai");
    cmd.arg("run")
        .arg("--models-path")
        .arg(models_dir)
        .arg("--address")
        .arg(format!("{}:{}", preset.host, preset.port))
        .arg("--context-size")
        .arg(preset.ctx_size.to_string());

    if let Some(threads) = preset.threads {
        cmd.arg("--threads").arg(threads.to_string());
    }

    for arg in &preset.extra_args {
        cmd.arg(arg);
    }

    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());

    let child = cmd
        .spawn()
        .map_err(|e| format!("Failed to start local-ai: {e}"))?;

    Ok(make_handle(
        Backend::LocalAi,
        model,
        preset.port,
        preset.host,
        child,
    ))
}

pub fn stop(handle: &mut ServerHandle) {
    // Drain any remaining output before killing
    handle.drain_output();
    let _ = handle.child.kill();
    let _ = handle.child.wait();
}

pub fn check_exited(handle: &mut ServerHandle) -> Option<String> {
    match handle.child.try_wait() {
        Ok(Some(status)) => {
            // Final drain
            handle.drain_output();
            if status.success() {
                Some("Server exited normally".into())
            } else {
                Some(format!("Server exited with status: {status}"))
            }
        }
        Ok(None) => None,
        Err(e) => Some(format!("Error checking server status: {e}")),
    }
}

fn is_large_model(model: &DiscoveredModel) -> bool {
    if let Some(ref hint) = model.param_hint {
        let upper = hint.to_uppercase();
        for part in upper.split(|c: char| c == '-' || c == '_') {
            if part.ends_with('B') {
                if let Ok(n) = part[..part.len() - 1].parse::<f64>() {
                    return n >= 20.0;
                }
            }
        }
    }
    model.size_bytes > 12_000_000_000
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsStr;

    fn command_args(cmd: &Command) -> Vec<String> {
        cmd.get_args()
            .map(OsStr::to_string_lossy)
            .map(|arg| arg.to_string())
            .collect()
    }

    #[test]
    fn is_large_model_by_params() {
        let model = DiscoveredModel {
            name: "test".into(),
            path: "test.gguf".into(),
            mmproj: None,
            format: crate::models::ModelFormat::Gguf,
            size_bytes: 0,
            quant: None,
            param_hint: Some("27B".into()),
            max_context_size: None,
            kv_bytes_per_token: None,
            source: crate::models::ModelSource::ExtraDir,
        };
        assert!(is_large_model(&model));
    }

    #[test]
    fn is_not_large_model_by_params() {
        let model = DiscoveredModel {
            name: "test".into(),
            path: "test.gguf".into(),
            mmproj: None,
            format: crate::models::ModelFormat::Gguf,
            size_bytes: 0,
            quant: None,
            param_hint: Some("9B".into()),
            max_context_size: None,
            kv_bytes_per_token: None,
            source: crate::models::ModelSource::ExtraDir,
        };
        assert!(!is_large_model(&model));
    }

    #[test]
    fn is_large_model_by_size_fallback() {
        let model = DiscoveredModel {
            name: "test".into(),
            path: "test.gguf".into(),
            mmproj: None,
            format: crate::models::ModelFormat::Gguf,
            size_bytes: 15_000_000_000,
            quant: None,
            param_hint: None,
            max_context_size: None,
            kv_bytes_per_token: None,
            source: crate::models::ModelSource::ExtraDir,
        };
        assert!(is_large_model(&model));
    }

    #[test]
    fn lmstudio_backend_returns_error() {
        let config = Config::default();
        let model = DiscoveredModel {
            name: "test".into(),
            path: "test.gguf".into(),
            mmproj: None,
            format: crate::models::ModelFormat::Gguf,
            size_bytes: 0,
            quant: None,
            param_hint: None,
            max_context_size: None,
            kv_bytes_per_token: None,
            source: crate::models::ModelSource::ExtraDir,
        };
        let result = launch(&model, &Backend::LmStudio, &config);
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(err.contains("LM Studio"));
    }

    #[test]
    fn ollama_rejects_local_gguf() {
        let config = Config::default();
        let model = DiscoveredModel {
            name: "test".into(),
            path: "test.gguf".into(),
            mmproj: None,
            format: crate::models::ModelFormat::Gguf,
            size_bytes: 0,
            quant: None,
            param_hint: None,
            max_context_size: None,
            kv_bytes_per_token: None,
            source: crate::models::ModelSource::ExtraDir,
        };
        let result = launch(&model, &Backend::Ollama, &config);
        assert!(result.is_err());
        assert!(result.err().unwrap().contains("registry"));
    }

    #[test]
    fn vllm_builds_command_for_local_gguf() {
        let mut config = Config::default();
        config.presets.insert(
            "vllm".into(),
            crate::config::BackendPreset {
                ctx_size: Some(32768),
                host: Some("127.0.0.1".into()),
                port: Some(9000),
                extra_args: vec!["--tokenizer".into(), "Qwen/Qwen2.5-0.5B".into()],
                ..Default::default()
            },
        );
        let model = DiscoveredModel {
            name: "test".into(),
            path: "/tmp/test.gguf".into(),
            mmproj: None,
            format: crate::models::ModelFormat::Gguf,
            size_bytes: 0,
            quant: None,
            param_hint: None,
            max_context_size: None,
            kv_bytes_per_token: None,
            source: crate::models::ModelSource::ExtraDir,
        };
        let preset = config.preset_for("vllm");
        let cmd = build_vllm_command(&model, &preset);
        assert_eq!(cmd.get_program(), OsStr::new("vllm"));
        assert_eq!(
            command_args(&cmd),
            vec![
                "serve",
                "/tmp/test.gguf",
                "--host",
                "127.0.0.1",
                "--port",
                "9000",
                "--max-model-len",
                "32768",
                "--generation-config",
                "vllm",
                "--load-format",
                "gguf",
                "--tokenizer",
                "Qwen/Qwen2.5-0.5B",
            ]
        );
    }

    #[test]
    fn llama_server_rejects_ollama_registry_model() {
        let config = Config::default();
        let model = DiscoveredModel {
            name: "qwen3.5:latest".into(),
            path: "ollama:qwen3.5:latest".into(),
            mmproj: None,
            format: crate::models::ModelFormat::Gguf,
            size_bytes: 0,
            quant: None,
            param_hint: None,
            max_context_size: None,
            kv_bytes_per_token: None,
            source: crate::models::ModelSource::Ollama,
        };
        let result = launch(&model, &Backend::LlamaServer, &config);
        assert!(result.is_err());
        assert!(result
            .err()
            .unwrap()
            .contains("registry entries, not local model files"));
    }
}
