use std::fmt;
use std::fs::{self, File};
use std::io::{self, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelFormat {
    Gguf,
    Mlx,
}

impl fmt::Display for ModelFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelFormat::Gguf => write!(f, "GGUF"),
            ModelFormat::Mlx => write!(f, "MLX"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelSource {
    LmStudio,
    LlamaCppCache,
    HfCache,
    Ollama,
    LlmfitCache,
    ExtraDir,
}

impl fmt::Display for ModelSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelSource::LmStudio => write!(f, "LM Studio"),
            ModelSource::LlamaCppCache => write!(f, "llama.cpp"),
            ModelSource::HfCache => write!(f, "HF Cache"),
            ModelSource::Ollama => write!(f, "Ollama"),
            ModelSource::LlmfitCache => write!(f, "llmfit"),
            ModelSource::ExtraDir => write!(f, "Custom"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct DiscoveredModel {
    pub name: String,
    pub path: PathBuf,
    pub mmproj: Option<PathBuf>,
    pub format: ModelFormat,
    pub size_bytes: u64,
    pub quant: Option<String>,
    pub param_hint: Option<String>,
    pub max_context_size: Option<u32>,
    pub kv_bytes_per_token: Option<f64>,
    pub source: ModelSource,
}

impl DiscoveredModel {
    pub fn size_display(&self) -> String {
        let gb = self.size_bytes as f64 / 1_073_741_824.0;
        if gb >= 1.0 {
            format!("{:.1}G", gb)
        } else {
            let mb = self.size_bytes as f64 / 1_048_576.0;
            format!("{:.0}M", mb)
        }
    }
}

/// Model metadata extracted from GGUF or HF config for KV cache estimation.
#[derive(Debug, Clone)]
pub struct GgufMetadata {
    pub max_context: Option<u32>,
    pub block_count: Option<u32>,
    pub head_count_kv: Option<u32>,
    pub head_count: Option<u32>,
    pub embedding_length: Option<u32>,
}

impl GgufMetadata {
    /// Calculate KV cache bytes per token (FP16 by default).
    ///
    /// Formula: `2 * layers * kv_heads * head_dim * sizeof(fp16)`
    /// where `head_dim = embedding_length / head_count`.
    pub fn kv_bytes_per_token(&self) -> Option<f64> {
        let layers = self.block_count? as f64;
        let kv_heads = self.head_count_kv? as f64;
        let head_count = self.head_count? as f64;
        let embedding = self.embedding_length? as f64;
        if head_count == 0.0 {
            return None;
        }
        let head_dim = embedding / head_count;
        // 2 tensors (K+V) * layers * kv_heads * head_dim * 2 bytes (FP16)
        Some(2.0 * layers * kv_heads * head_dim * 2.0)
    }
}

/// Compute KV bytes-per-token from HF config.json fields.
pub fn kv_bytes_per_token_from_hf_config(value: &serde_json::Value) -> Option<f64> {
    let scopes = [
        value.get("text_config"),
        value.get("llm_config"),
        value.get("language_config"),
        Some(value),
    ];

    let mut layers: Option<u32> = None;
    let mut kv_heads: Option<u32> = None;
    let mut attn_heads: Option<u32> = None;
    let mut hidden_size: Option<u32> = None;

    for scope in scopes.into_iter().flatten() {
        if layers.is_none() {
            layers = find_json_u32(scope, "num_hidden_layers");
        }
        if kv_heads.is_none() {
            kv_heads = find_json_u32(scope, "num_key_value_heads");
        }
        if attn_heads.is_none() {
            attn_heads = find_json_u32(scope, "num_attention_heads");
        }
        if hidden_size.is_none() {
            hidden_size = find_json_u32(scope, "hidden_size");
        }
    }

    let layers = layers? as f64;
    let kv_heads = kv_heads.or(attn_heads)? as f64;
    let attn_heads = attn_heads? as f64;
    let hidden = hidden_size? as f64;
    if attn_heads == 0.0 {
        return None;
    }
    let head_dim = hidden / attn_heads;
    Some(2.0 * layers * kv_heads * head_dim * 2.0)
}

pub fn discover_models(extra_dirs: &[PathBuf]) -> Vec<DiscoveredModel> {
    let mut models = Vec::new();
    let mut seen_paths = std::collections::HashSet::new();

    let home = dirs::home_dir().unwrap_or_default();

    // LM Studio models — differs by platform
    let lmstudio_dir = if cfg!(windows) {
        // Windows: %USERPROFILE%/.lmstudio/models (newer) or %LOCALAPPDATA%/LM Studio/models
        let primary = home.join(".lmstudio").join("models");
        if primary.is_dir() {
            primary
        } else {
            dirs::data_local_dir()
                .unwrap_or_else(|| home.clone())
                .join("LM Studio")
                .join("models")
        }
    } else {
        home.join(".lmstudio").join("models")
    };
    scan_gguf_dir(
        &lmstudio_dir,
        ModelSource::LmStudio,
        &mut models,
        &mut seen_paths,
    );

    // llama.cpp cache — on Windows use %LOCALAPPDATA%/llm-models as fallback
    let llamacpp_dir = if cfg!(windows) {
        let cache = home.join(".cache").join("llm-models");
        if cache.is_dir() {
            cache
        } else {
            dirs::data_local_dir()
                .unwrap_or_else(|| home.clone())
                .join("llm-models")
        }
    } else {
        home.join(".cache").join("llm-models")
    };
    scan_gguf_dir(
        &llamacpp_dir,
        ModelSource::LlamaCppCache,
        &mut models,
        &mut seen_paths,
    );

    // llmfit cache — sister project model downloads
    let llmfit_dir = std::env::var("LLMFIT_MODELS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| home.join(".cache").join("llmfit").join("models"));
    scan_gguf_dir(
        &llmfit_dir,
        ModelSource::LlmfitCache,
        &mut models,
        &mut seen_paths,
    );

    // HuggingFace cache — MLX models
    let hf_hub = std::env::var("HF_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            if cfg!(windows) {
                home.join(".cache").join("huggingface").join("hub")
            } else {
                home.join(".cache").join("huggingface").join("hub")
            }
        });
    scan_mlx_models(&hf_hub, &mut models, &mut seen_paths);

    // Extra user-configured directories
    for dir in extra_dirs {
        scan_gguf_dir(dir, ModelSource::ExtraDir, &mut models, &mut seen_paths);
    }

    models.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
    models
}

fn scan_gguf_dir(
    dir: &Path,
    source: ModelSource,
    models: &mut Vec<DiscoveredModel>,
    seen: &mut std::collections::HashSet<PathBuf>,
) {
    if !dir.is_dir() {
        return;
    }

    for entry in WalkDir::new(dir).min_depth(1).into_iter().flatten() {
        let path = entry.path();
        if path.extension().is_some_and(|e| e == "gguf") {
            let fname = path.file_name().unwrap().to_string_lossy();
            if fname.starts_with("mmproj") {
                continue;
            }
            if !seen.insert(path.to_path_buf()) {
                continue;
            }

            let parent = path.parent().unwrap();
            let mmproj = find_mmproj(parent);
            let size_bytes = fs::metadata(path).map(|m| m.len()).unwrap_or(0);
            let model_name = gguf_display_name(path, dir);
            let metadata = read_gguf_metadata(path);
            let max_context_size = metadata.as_ref().and_then(|m| m.max_context);
            let kv_bytes_per_token = metadata.as_ref().and_then(|m| m.kv_bytes_per_token());

            models.push(DiscoveredModel {
                name: model_name.clone(),
                path: path.to_path_buf(),
                mmproj,
                format: ModelFormat::Gguf,
                size_bytes,
                quant: parse_quant(&fname),
                param_hint: parse_params(&model_name),
                max_context_size,
                kv_bytes_per_token,
                source: source.clone(),
            });
        }
    }
}

fn gguf_display_name(path: &Path, scan_root: &Path) -> String {
    let parent = path.parent().unwrap_or(scan_root);

    if parent == scan_root {
        return path
            .file_stem()
            .or_else(|| path.file_name())
            .map(|name| name.to_string_lossy().to_string())
            .unwrap_or_else(|| path.display().to_string());
    }

    parent
        .file_name()
        .map(|name| name.to_string_lossy().to_string())
        .unwrap_or_else(|| path.display().to_string())
}

fn scan_mlx_models(
    hf_hub: &Path,
    models: &mut Vec<DiscoveredModel>,
    seen: &mut std::collections::HashSet<PathBuf>,
) {
    if !hf_hub.is_dir() {
        return;
    }

    // HF cache structure: models--<owner>--<repo>/snapshots/<hash>/
    let Ok(entries) = fs::read_dir(hf_hub) else {
        return;
    };
    for entry in entries.flatten() {
        let dir_name = entry.file_name().to_string_lossy().to_string();
        if !dir_name.starts_with("models--") {
            continue;
        }
        // Check if it's an MLX model (owner is mlx-community or name contains mlx)
        let lower = dir_name.to_lowercase();
        if !lower.contains("mlx") {
            continue;
        }

        let snapshots_dir = entry.path().join("snapshots");
        if !snapshots_dir.is_dir() {
            continue;
        }

        // Find the latest snapshot (there's usually just one)
        let Some(snapshot) = fs::read_dir(&snapshots_dir)
            .into_iter()
            .flatten()
            .flatten()
            .filter(|e| e.path().is_dir())
            .max_by_key(|e| e.metadata().and_then(|m| m.modified()).ok())
        else {
            continue;
        };

        let snap_path = snapshot.path();

        // Verify it has config.json and safetensors
        if !snap_path.join("config.json").exists() {
            continue;
        }
        let has_safetensors = fs::read_dir(&snap_path)
            .into_iter()
            .flatten()
            .flatten()
            .any(|e| e.path().extension().is_some_and(|ext| ext == "safetensors"));
        if !has_safetensors {
            continue;
        }

        if !seen.insert(snap_path.clone()) {
            continue;
        }

        // Calculate total size of safetensors files
        let size_bytes: u64 = fs::read_dir(&snap_path)
            .into_iter()
            .flatten()
            .flatten()
            .filter(|e| e.path().extension().is_some_and(|ext| ext == "safetensors"))
            .filter_map(|e| e.metadata().ok())
            .map(|m| m.len())
            .sum();

        // Parse friendly name from "models--mlx-community--Qwen3.5-9B-4bit"
        let friendly = dir_name
            .strip_prefix("models--")
            .unwrap_or(&dir_name)
            .replace("--", "/");

        let quant = if lower.contains("8bit") || lower.contains("8-bit") {
            Some("8bit".into())
        } else if lower.contains("4bit") || lower.contains("4-bit") {
            Some("4bit".into())
        } else {
            None
        };

        let config_path = snap_path.join("config.json");
        let hf_config = fs::read_to_string(&config_path)
            .ok()
            .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok());
        let max_context_size = hf_config
            .as_ref()
            .and_then(|v| extract_hf_config_max_context(v));
        let kv_bytes_per_token = hf_config
            .as_ref()
            .and_then(|v| kv_bytes_per_token_from_hf_config(v));

        models.push(DiscoveredModel {
            name: friendly.clone(),
            path: snap_path,
            mmproj: None,
            format: ModelFormat::Mlx,
            size_bytes,
            quant,
            param_hint: parse_params(&friendly),
            max_context_size,
            kv_bytes_per_token,
            source: ModelSource::HfCache,
        });
    }
}

pub fn add_ollama_models(models: &mut Vec<DiscoveredModel>, ollama_models: Vec<(String, u64)>) {
    for (name, size) in ollama_models {
        models.push(DiscoveredModel {
            name: name.clone(),
            path: PathBuf::from(format!("ollama:{name}")),
            mmproj: None,
            format: ModelFormat::Gguf,
            size_bytes: size,
            quant: parse_quant(&name),
            param_hint: parse_params(&name),
            max_context_size: None,
            kv_bytes_per_token: None,
            source: ModelSource::Ollama,
        });
    }
    models.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
}

fn read_gguf_metadata(path: &Path) -> Option<GgufMetadata> {
    let mut file = File::open(path).ok()?;
    parse_gguf_metadata(&mut file).ok()
}

fn parse_gguf_metadata<R: Read + Seek>(reader: &mut R) -> io::Result<GgufMetadata> {
    let mut magic = [0_u8; 4];
    reader.read_exact(&mut magic)?;
    if &magic != b"GGUF" {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "not a GGUF file",
        ));
    }

    let version = read_u32_le(reader)?;
    let kv_count = match version {
        1 => {
            let _tensor_count = read_u32_le(reader)?;
            read_u32_le(reader)? as u64
        }
        2 | 3 => {
            let _tensor_count = read_u64_le(reader)?;
            read_u64_le(reader)?
        }
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "unsupported GGUF version",
            ))
        }
    };

    let mut architecture = None;
    let mut context_by_arch = std::collections::HashMap::new();
    let mut fallback_context = None;
    let mut block_count_by_arch = std::collections::HashMap::<String, u32>::new();
    let mut head_count_kv_by_arch = std::collections::HashMap::<String, u32>::new();
    let mut head_count_by_arch = std::collections::HashMap::<String, u32>::new();
    let mut embedding_by_arch = std::collections::HashMap::<String, u32>::new();

    for _ in 0..kv_count {
        let key = read_gguf_string(reader)?;
        let value_type = read_u32_le(reader)?;

        if key == "general.architecture" {
            architecture = read_gguf_string_value(reader, value_type)?;
            continue;
        }

        if let Some(prefix) = key.strip_suffix(".context_length") {
            if let Some(value) = read_gguf_u32_value(reader, value_type)? {
                context_by_arch.insert(prefix.to_string(), value);
                fallback_context.get_or_insert(value);
            }
            continue;
        }

        if let Some(prefix) = key.strip_suffix(".block_count") {
            if let Some(value) = read_gguf_u32_value(reader, value_type)? {
                block_count_by_arch.insert(prefix.to_string(), value);
            }
            continue;
        }

        if let Some(prefix) = key.strip_suffix(".attention.head_count_kv") {
            if let Some(value) = read_gguf_u32_value(reader, value_type)? {
                head_count_kv_by_arch.insert(prefix.to_string(), value);
            }
            continue;
        }

        if let Some(prefix) = key.strip_suffix(".attention.head_count") {
            if let Some(value) = read_gguf_u32_value(reader, value_type)? {
                head_count_by_arch.insert(prefix.to_string(), value);
            }
            continue;
        }

        if let Some(prefix) = key.strip_suffix(".embedding_length") {
            if let Some(value) = read_gguf_u32_value(reader, value_type)? {
                embedding_by_arch.insert(prefix.to_string(), value);
            }
            continue;
        }

        skip_gguf_value(reader, value_type)?;
    }

    let arch = architecture.unwrap_or_default();

    let max_context = context_by_arch.get(&arch).copied().or(fallback_context);

    Ok(GgufMetadata {
        max_context,
        block_count: block_count_by_arch.get(&arch).copied(),
        head_count_kv: head_count_kv_by_arch.get(&arch).copied(),
        head_count: head_count_by_arch.get(&arch).copied(),
        embedding_length: embedding_by_arch.get(&arch).copied(),
    })
}

#[cfg(test)]
fn parse_gguf_max_context<R: Read + Seek>(reader: &mut R) -> io::Result<Option<u32>> {
    parse_gguf_metadata(reader).map(|m| m.max_context)
}

fn extract_hf_config_max_context(value: &serde_json::Value) -> Option<u32> {
    let scopes = [
        value.get("text_config"),
        value.get("llm_config"),
        value.get("language_config"),
        Some(value),
    ];

    for scope in scopes.into_iter().flatten() {
        for key in [
            "max_position_embeddings",
            "model_max_length",
            "max_sequence_length",
            "n_positions",
            "seq_length",
        ] {
            if let Some(found) = find_json_u32(scope, key) {
                return Some(found);
            }
        }
    }

    None
}

fn find_json_u32(value: &serde_json::Value, target_key: &str) -> Option<u32> {
    match value {
        serde_json::Value::Object(map) => {
            if let Some(found) = map.get(target_key).and_then(json_value_to_u32) {
                return Some(found);
            }
            map.values()
                .find_map(|nested| find_json_u32(nested, target_key))
        }
        serde_json::Value::Array(values) => values
            .iter()
            .find_map(|nested| find_json_u32(nested, target_key)),
        _ => None,
    }
}

fn json_value_to_u32(value: &serde_json::Value) -> Option<u32> {
    value
        .as_u64()
        .and_then(|n| u32::try_from(n).ok())
        .or_else(|| value.as_i64().and_then(|n| u32::try_from(n).ok()))
}

fn read_u8(reader: &mut impl Read) -> io::Result<u8> {
    let mut buf = [0_u8; 1];
    reader.read_exact(&mut buf)?;
    Ok(buf[0])
}

fn read_u16_le(reader: &mut impl Read) -> io::Result<u16> {
    let mut buf = [0_u8; 2];
    reader.read_exact(&mut buf)?;
    Ok(u16::from_le_bytes(buf))
}

fn read_u32_le(reader: &mut impl Read) -> io::Result<u32> {
    let mut buf = [0_u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64_le(reader: &mut impl Read) -> io::Result<u64> {
    let mut buf = [0_u8; 8];
    reader.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_i8(reader: &mut impl Read) -> io::Result<i8> {
    Ok(read_u8(reader)? as i8)
}

fn read_i16_le(reader: &mut impl Read) -> io::Result<i16> {
    let mut buf = [0_u8; 2];
    reader.read_exact(&mut buf)?;
    Ok(i16::from_le_bytes(buf))
}

fn read_i32_le(reader: &mut impl Read) -> io::Result<i32> {
    let mut buf = [0_u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(i32::from_le_bytes(buf))
}

fn read_i64_le(reader: &mut impl Read) -> io::Result<i64> {
    let mut buf = [0_u8; 8];
    reader.read_exact(&mut buf)?;
    Ok(i64::from_le_bytes(buf))
}

fn read_gguf_string<R: Read>(reader: &mut R) -> io::Result<String> {
    let len = read_u64_le(reader)?;
    let len = usize::try_from(len)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "GGUF string too large"))?;
    let mut buf = vec![0_u8; len];
    reader.read_exact(&mut buf)?;
    Ok(String::from_utf8_lossy(&buf).to_string())
}

fn read_gguf_string_value<R: Read + Seek>(
    reader: &mut R,
    value_type: u32,
) -> io::Result<Option<String>> {
    if value_type == 8 {
        return read_gguf_string(reader).map(Some);
    }
    skip_gguf_value(reader, value_type)?;
    Ok(None)
}

fn read_gguf_u32_value<R: Read + Seek>(reader: &mut R, value_type: u32) -> io::Result<Option<u32>> {
    let value = match value_type {
        0 => Some(read_u8(reader)? as u32),
        1 => u32::try_from(read_i8(reader)?).ok(),
        2 => Some(read_u16_le(reader)? as u32),
        3 => u32::try_from(read_i16_le(reader)?).ok(),
        4 => Some(read_u32_le(reader)?),
        5 => u32::try_from(read_i32_le(reader)?).ok(),
        10 => u32::try_from(read_u64_le(reader)?).ok(),
        11 => u32::try_from(read_i64_le(reader)?).ok(),
        _ => {
            skip_gguf_value(reader, value_type)?;
            None
        }
    };

    Ok(value)
}

fn skip_gguf_value<R: Read + Seek>(reader: &mut R, value_type: u32) -> io::Result<()> {
    match value_type {
        0 | 1 | 7 => skip_bytes(reader, 1),
        2 | 3 => skip_bytes(reader, 2),
        4 | 5 | 6 => skip_bytes(reader, 4),
        8 | 10 | 11 | 12 => {
            if value_type == 8 {
                let len = read_u64_le(reader)?;
                skip_bytes(reader, len)
            } else {
                skip_bytes(reader, 8)
            }
        }
        9 => {
            let element_type = read_u32_le(reader)?;
            let len = read_u64_le(reader)?;
            skip_gguf_array(reader, element_type, len)
        }
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Unsupported GGUF value type {value_type}"),
        )),
    }
}

fn skip_gguf_array<R: Read + Seek>(reader: &mut R, element_type: u32, len: u64) -> io::Result<()> {
    if let Some(width) = gguf_fixed_width(element_type) {
        return skip_bytes(reader, len.saturating_mul(width));
    }

    match element_type {
        8 => {
            for _ in 0..len {
                let string_len = read_u64_le(reader)?;
                skip_bytes(reader, string_len)?;
            }
            Ok(())
        }
        9 => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Nested GGUF arrays are not supported",
        )),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Unsupported GGUF array element type {element_type}"),
        )),
    }
}

fn gguf_fixed_width(value_type: u32) -> Option<u64> {
    match value_type {
        0 | 1 | 7 => Some(1),
        2 | 3 => Some(2),
        4 | 5 | 6 => Some(4),
        10 | 11 | 12 => Some(8),
        _ => None,
    }
}

fn skip_bytes(reader: &mut impl Seek, count: u64) -> io::Result<()> {
    let delta = i64::try_from(count)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "seek overflow"))?;
    reader.seek(SeekFrom::Current(delta))?;
    Ok(())
}

fn find_mmproj(dir: &Path) -> Option<PathBuf> {
    fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .find(|e| {
            let n = e.file_name();
            let n = n.to_string_lossy();
            n.starts_with("mmproj") && n.ends_with(".gguf")
        })
        .map(|e| e.path())
}

fn parse_quant(s: &str) -> Option<String> {
    // Match patterns like Q4_K_M, Q8_0, IQ4_NL, Q5_K_S, etc.
    let upper = s.to_uppercase();
    for part in upper.split(|c: char| !c.is_alphanumeric() && c != '_') {
        if part.starts_with('Q')
            && part.len() >= 3
            && part.chars().nth(1).is_some_and(|c| c.is_ascii_digit())
        {
            return Some(part.to_string());
        }
        if part.starts_with("IQ")
            && part.len() >= 4
            && part.chars().nth(2).is_some_and(|c| c.is_ascii_digit())
        {
            return Some(part.to_string());
        }
    }
    // Check for "4bit" / "8bit" style
    for part in s.split(|c: char| !c.is_alphanumeric()) {
        let lower = part.to_lowercase();
        if lower == "4bit" || lower == "8bit" || lower == "fp16" || lower == "bf16" {
            return Some(lower);
        }
    }
    None
}

fn parse_params(s: &str) -> Option<String> {
    // Match patterns like "27B", "3.5B", "35B-A3B", "4B"
    let upper = s.to_uppercase();
    // First try MoE pattern like "35B-A3B"
    for window in upper
        .split(|c: char| c == '-' || c == '_')
        .collect::<Vec<_>>()
        .windows(2)
    {
        if let [total, active] = window {
            if total.ends_with('B')
                && active.starts_with('A')
                && active.ends_with('B')
                && total[..total.len() - 1].parse::<f64>().is_ok()
            {
                return Some(format!("{total}-{active}"));
            }
        }
    }
    // Then simple "NB" pattern
    for part in upper.split(|c: char| c == '-' || c == '_' || c == ' ') {
        if part.ends_with('B') && part.len() >= 2 {
            let num_part = &part[..part.len() - 1];
            if num_part.parse::<f64>().is_ok() {
                return Some(part.to_string());
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn push_u32(buf: &mut Vec<u8>, value: u32) {
        buf.extend_from_slice(&value.to_le_bytes());
    }

    fn push_u64(buf: &mut Vec<u8>, value: u64) {
        buf.extend_from_slice(&value.to_le_bytes());
    }

    fn push_string(buf: &mut Vec<u8>, value: &str) {
        push_u64(buf, value.len() as u64);
        buf.extend_from_slice(value.as_bytes());
    }

    fn sample_gguf_bytes(arch: &str, ctx: u32) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        push_u32(&mut buf, 3);
        push_u64(&mut buf, 0);
        push_u64(&mut buf, 2);

        push_string(&mut buf, "general.architecture");
        push_u32(&mut buf, 8);
        push_string(&mut buf, arch);

        push_string(&mut buf, &format!("{arch}.context_length"));
        push_u32(&mut buf, 4);
        push_u32(&mut buf, ctx);

        buf
    }

    /// Build a GGUF byte buffer with architecture metadata for KV cache testing.
    fn sample_gguf_with_kv_metadata(
        arch: &str,
        ctx: u32,
        block_count: u32,
        head_count_kv: u32,
        head_count: u32,
        embedding_length: u32,
    ) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        push_u32(&mut buf, 3); // version
        push_u64(&mut buf, 0); // tensor count
        push_u64(&mut buf, 6); // kv count

        // general.architecture
        push_string(&mut buf, "general.architecture");
        push_u32(&mut buf, 8); // type: string
        push_string(&mut buf, arch);

        // context_length
        push_string(&mut buf, &format!("{arch}.context_length"));
        push_u32(&mut buf, 4); // type: u32
        push_u32(&mut buf, ctx);

        // block_count
        push_string(&mut buf, &format!("{arch}.block_count"));
        push_u32(&mut buf, 4);
        push_u32(&mut buf, block_count);

        // attention.head_count_kv
        push_string(&mut buf, &format!("{arch}.attention.head_count_kv"));
        push_u32(&mut buf, 4);
        push_u32(&mut buf, head_count_kv);

        // attention.head_count
        push_string(&mut buf, &format!("{arch}.attention.head_count"));
        push_u32(&mut buf, 4);
        push_u32(&mut buf, head_count);

        // embedding_length
        push_string(&mut buf, &format!("{arch}.embedding_length"));
        push_u32(&mut buf, 4);
        push_u32(&mut buf, embedding_length);

        buf
    }

    #[test]
    fn parse_quant_q4_k_m() {
        assert_eq!(parse_quant("Qwen3.5-9B-Q4_K_M.gguf"), Some("Q4_K_M".into()));
    }

    #[test]
    fn parse_quant_q8_0() {
        assert_eq!(parse_quant("model-Q8_0.gguf"), Some("Q8_0".into()));
    }

    #[test]
    fn parse_quant_iq4_nl() {
        assert_eq!(parse_quant("model-IQ4_NL.gguf"), Some("IQ4_NL".into()));
    }

    #[test]
    fn parse_quant_4bit() {
        assert_eq!(parse_quant("model-4bit"), Some("4bit".into()));
    }

    #[test]
    fn parse_quant_fp16() {
        assert_eq!(parse_quant("model-fp16.gguf"), Some("fp16".into()));
    }

    #[test]
    fn parse_quant_none() {
        assert_eq!(parse_quant("just-a-model-name"), None);
    }

    #[test]
    fn parse_params_simple() {
        assert_eq!(parse_params("Qwen3.5-9B-Instruct"), Some("9B".into()));
    }

    #[test]
    fn parse_params_large() {
        assert_eq!(parse_params("Qwen3.5-27B-Claude"), Some("27B".into()));
    }

    #[test]
    fn parse_params_small() {
        assert_eq!(parse_params("NVIDIA-Nemotron-3-Nano-4B"), Some("4B".into()));
    }

    #[test]
    fn parse_params_moe() {
        assert_eq!(parse_params("Qwen3.5-35B-A3B-GGUF"), Some("35B-A3B".into()));
    }

    #[test]
    fn parse_params_decimal() {
        assert_eq!(parse_params("Model-3.5B-Instruct"), Some("3.5B".into()));
    }

    #[test]
    fn parse_params_none() {
        assert_eq!(parse_params("some-model-name"), None);
    }

    #[test]
    fn parse_gguf_max_context_reads_arch_specific_key() {
        let mut cursor = Cursor::new(sample_gguf_bytes("llama", 131072));
        assert_eq!(parse_gguf_max_context(&mut cursor).unwrap(), Some(131072));
    }

    #[test]
    fn parse_gguf_metadata_extracts_kv_params() {
        // Qwen2.5-7B-like: 28 layers, 4 KV heads, 28 attn heads, 3584 embedding
        let data = sample_gguf_with_kv_metadata("qwen2", 131072, 28, 4, 28, 3584);
        let mut cursor = Cursor::new(data);
        let meta = parse_gguf_metadata(&mut cursor).unwrap();

        assert_eq!(meta.max_context, Some(131072));
        assert_eq!(meta.block_count, Some(28));
        assert_eq!(meta.head_count_kv, Some(4));
        assert_eq!(meta.head_count, Some(28));
        assert_eq!(meta.embedding_length, Some(3584));

        // head_dim = 3584 / 28 = 128
        // kv_bytes = 2 * 28 * 4 * 128 * 2 = 57344
        let kv = meta.kv_bytes_per_token().unwrap();
        assert!((kv - 57344.0).abs() < 0.01);
    }

    #[test]
    fn kv_bytes_per_token_from_hf_config_extracts_values() {
        let config: serde_json::Value = serde_json::from_str(
            r#"{
                "num_hidden_layers": 32,
                "num_key_value_heads": 8,
                "num_attention_heads": 32,
                "hidden_size": 4096,
                "max_position_embeddings": 131072
            }"#,
        )
        .unwrap();

        // head_dim = 4096 / 32 = 128
        // kv_bytes = 2 * 32 * 8 * 128 * 2 = 131072
        let kv = kv_bytes_per_token_from_hf_config(&config).unwrap();
        assert!((kv - 131072.0).abs() < 0.01);
    }

    #[test]
    fn kv_bytes_per_token_hf_config_uses_text_config() {
        let config: serde_json::Value = serde_json::from_str(
            r#"{
                "text_config": {
                    "num_hidden_layers": 28,
                    "num_key_value_heads": 4,
                    "num_attention_heads": 28,
                    "hidden_size": 3584
                }
            }"#,
        )
        .unwrap();

        let kv = kv_bytes_per_token_from_hf_config(&config).unwrap();
        // head_dim = 3584/28 = 128, kv = 2*28*4*128*2 = 57344
        assert!((kv - 57344.0).abs() < 0.01);
    }

    #[test]
    fn extract_hf_config_max_context_prefers_max_position_embeddings() {
        let value: serde_json::Value = serde_json::from_str(
            r#"{
                "max_position_embeddings": 32768,
                "rope_scaling": {"original_max_position_embeddings": 8192}
            }"#,
        )
        .unwrap();

        assert_eq!(extract_hf_config_max_context(&value), Some(32768));
    }

    #[test]
    fn extract_hf_config_max_context_uses_text_config_when_present() {
        let value: serde_json::Value = serde_json::from_str(
            r#"{
                "vision_config": {"max_position_embeddings": 576},
                "text_config": {"max_position_embeddings": 65536}
            }"#,
        )
        .unwrap();

        assert_eq!(extract_hf_config_max_context(&value), Some(65536));
    }

    #[test]
    fn size_display_gigabytes() {
        let model = DiscoveredModel {
            name: "test".into(),
            path: PathBuf::from("test.gguf"),
            mmproj: None,
            format: ModelFormat::Gguf,
            size_bytes: 5_368_709_120, // 5 GB
            quant: None,
            param_hint: None,
            max_context_size: None,
            kv_bytes_per_token: None,
            source: ModelSource::ExtraDir,
        };
        assert_eq!(model.size_display(), "5.0G");
    }

    #[test]
    fn size_display_megabytes() {
        let model = DiscoveredModel {
            name: "test".into(),
            path: PathBuf::from("test.gguf"),
            mmproj: None,
            format: ModelFormat::Gguf,
            size_bytes: 524_288_000, // 500 MB
            quant: None,
            param_hint: None,
            max_context_size: None,
            kv_bytes_per_token: None,
            source: ModelSource::ExtraDir,
        };
        assert_eq!(model.size_display(), "500M");
    }

    #[test]
    fn model_format_display() {
        assert_eq!(ModelFormat::Gguf.to_string(), "GGUF");
        assert_eq!(ModelFormat::Mlx.to_string(), "MLX");
    }

    #[test]
    fn model_source_display() {
        assert_eq!(ModelSource::LmStudio.to_string(), "LM Studio");
        assert_eq!(ModelSource::LlamaCppCache.to_string(), "llama.cpp");
        assert_eq!(ModelSource::HfCache.to_string(), "HF Cache");
        assert_eq!(ModelSource::Ollama.to_string(), "Ollama");
        assert_eq!(ModelSource::LlmfitCache.to_string(), "llmfit");
        assert_eq!(ModelSource::ExtraDir.to_string(), "Custom");
    }

    #[test]
    fn add_ollama_models_sorts() {
        let mut models = vec![DiscoveredModel {
            name: "Zebra-Model".into(),
            path: PathBuf::from("z.gguf"),
            mmproj: None,
            format: ModelFormat::Gguf,
            size_bytes: 100,
            quant: None,
            param_hint: None,
            max_context_size: None,
            kv_bytes_per_token: None,
            source: ModelSource::ExtraDir,
        }];
        add_ollama_models(&mut models, vec![("alpha-model".into(), 200)]);
        assert_eq!(models[0].name, "alpha-model");
        assert_eq!(models[1].name, "Zebra-Model");
    }

    #[test]
    fn discover_models_with_empty_extra_dirs() {
        let models = discover_models(&[PathBuf::from("/nonexistent/path/12345")]);
        let _ = models.len();
    }

    #[test]
    fn gguf_display_name_uses_file_stem_for_flat_cache_files() {
        let root = Path::new("/home/test/.cache/llmfit/models");
        let file = root.join("Qwen2.5-7B-Instruct-1M-Q8_0.gguf");

        assert_eq!(
            gguf_display_name(&file, root),
            "Qwen2.5-7B-Instruct-1M-Q8_0"
        );
    }

    #[test]
    fn gguf_display_name_uses_parent_dir_for_nested_layouts() {
        let root = Path::new("/home/test/.lmstudio/models");
        let file = root
            .join("lmstudio-community")
            .join("Qwen3.5-9B-GGUF")
            .join("Qwen3.5-9B-Q4_K_M.gguf");

        assert_eq!(gguf_display_name(&file, root), "Qwen3.5-9B-GGUF");
    }
}
