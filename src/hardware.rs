//! Hardware detection: GPU VRAM and system RAM probing for max-context estimation.
//!
//! Uses shell commands to detect available memory:
//! - NVIDIA: `nvidia-smi` for per-GPU total/used VRAM
//! - macOS: `sysctl` for total system RAM (Metal shares unified memory)
//! - Fallback: `/proc/meminfo` on Linux for total system RAM

use std::process::Command;

/// Detected hardware memory info.
#[derive(Debug, Clone)]
pub struct HardwareInfo {
    /// Total VRAM/RAM in bytes.
    pub total_bytes: u64,
    /// Used VRAM/RAM in bytes (if detectable).
    pub used_bytes: Option<u64>,
    /// Source of the detection.
    pub source: MemorySource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemorySource {
    NvidiaGpu,
    MacosUnified,
    SystemRam,
}

impl MemorySource {
    pub fn label(&self) -> &'static str {
        match self {
            MemorySource::NvidiaGpu => "NVIDIA GPU",
            MemorySource::MacosUnified => "macOS unified",
            MemorySource::SystemRam => "system RAM",
        }
    }
}

impl HardwareInfo {
    /// Available memory in bytes (total minus used, or total if used is unknown).
    pub fn available_bytes(&self) -> u64 {
        match self.used_bytes {
            Some(used) => self.total_bytes.saturating_sub(used),
            None => self.total_bytes,
        }
    }
}

/// Detect the best available memory source.
///
/// Priority: NVIDIA GPU > macOS unified > Linux system RAM.
pub fn detect_hardware() -> Option<HardwareInfo> {
    detect_nvidia()
        .or_else(detect_macos_unified)
        .or_else(detect_linux_ram)
}

/// Estimate the maximum safe context size given hardware, model weight size, and KV cache cost.
///
/// Formula: `max_ctx = (available_vram * 0.95 - model_weight_bytes) / kv_bytes_per_token`
///
/// The 5% safety margin ensures the system retains enough VRAM for OS/desktop compositing.
pub fn estimate_max_context(
    hw: &HardwareInfo,
    model_size_bytes: u64,
    kv_bytes_per_token: f64,
) -> Option<u32> {
    if kv_bytes_per_token <= 0.0 {
        return None;
    }

    let available = hw.available_bytes() as f64;
    let safe_budget = available * 0.95;
    let remaining = safe_budget - model_size_bytes as f64;

    if remaining <= 0.0 {
        return None; // model doesn't fit
    }

    let max_tokens = remaining / kv_bytes_per_token;
    if max_tokens < 512.0 {
        return None; // too small to be useful
    }

    // Round down to nearest 256 for clean values
    let rounded = ((max_tokens as u64) / 256) * 256;
    Some(rounded as u32)
}

// ---------------------------------------------------------------------------
// Detection backends
// ---------------------------------------------------------------------------

fn detect_nvidia() -> Option<HardwareInfo> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Take the first GPU line (multi-GPU: pick the one with most free memory)
    let mut best: Option<HardwareInfo> = None;

    for line in stdout.lines() {
        let parts: Vec<&str> = line.split(',').map(str::trim).collect();
        if parts.len() < 2 {
            continue;
        }
        let total_mib: u64 = parts[0].parse().ok()?;
        let used_mib: u64 = parts[1].parse().ok()?;
        let total_bytes = total_mib * 1024 * 1024;
        let used_bytes = used_mib * 1024 * 1024;

        let candidate = HardwareInfo {
            total_bytes,
            used_bytes: Some(used_bytes),
            source: MemorySource::NvidiaGpu,
        };

        if best
            .as_ref()
            .is_none_or(|b| candidate.available_bytes() > b.available_bytes())
        {
            best = Some(candidate);
        }
    }

    best
}

fn detect_macos_unified() -> Option<HardwareInfo> {
    if !cfg!(target_os = "macos") {
        return None;
    }

    let output = Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let total_bytes: u64 = stdout.trim().parse().ok()?;

    Some(HardwareInfo {
        total_bytes,
        used_bytes: None, // macOS doesn't expose GPU-specific usage easily
        source: MemorySource::MacosUnified,
    })
}

fn detect_linux_ram() -> Option<HardwareInfo> {
    // Read /proc/meminfo for MemTotal and MemAvailable
    let contents = std::fs::read_to_string("/proc/meminfo").ok()?;

    let mut total_kb: Option<u64> = None;
    let mut available_kb: Option<u64> = None;

    for line in contents.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            total_kb = parse_meminfo_kb(rest);
        } else if let Some(rest) = line.strip_prefix("MemAvailable:") {
            available_kb = parse_meminfo_kb(rest);
        }
    }

    let total_bytes = total_kb? * 1024;
    let used_bytes = available_kb.map(|avail| total_bytes.saturating_sub(avail * 1024));

    Some(HardwareInfo {
        total_bytes,
        used_bytes,
        source: MemorySource::SystemRam,
    })
}

fn parse_meminfo_kb(s: &str) -> Option<u64> {
    // Format: "   12345 kB"
    s.trim().strip_suffix("kB")?.trim().parse().ok()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimate_basic_context_calculation() {
        let hw = HardwareInfo {
            total_bytes: 24 * 1024 * 1024 * 1024, // 24 GiB
            used_bytes: Some(2 * 1024 * 1024 * 1024), // 2 GiB used
            source: MemorySource::NvidiaGpu,
        };

        // 8 GiB model, 1024 bytes per token KV cache
        let ctx = estimate_max_context(&hw, 8 * 1024 * 1024 * 1024, 1024.0);
        assert!(ctx.is_some());
        let ctx = ctx.unwrap();

        // Available = 22 GiB, safe = 22*0.95 = 20.9 GiB, minus 8 GiB model = 12.9 GiB
        // 12.9 GiB / 1024 bytes = ~13.2M tokens — that's enormous, sanity check it's > 1M
        assert!(ctx > 1_000_000, "expected > 1M tokens, got {ctx}");
    }

    #[test]
    fn estimate_returns_none_when_model_too_large() {
        let hw = HardwareInfo {
            total_bytes: 8 * 1024 * 1024 * 1024,
            used_bytes: Some(2 * 1024 * 1024 * 1024),
            source: MemorySource::NvidiaGpu,
        };

        // Model is 10 GiB but only 6 GiB available after safety margin
        let ctx = estimate_max_context(&hw, 10 * 1024 * 1024 * 1024, 1024.0);
        assert!(ctx.is_none());
    }

    #[test]
    fn estimate_returns_none_for_zero_kv_cost() {
        let hw = HardwareInfo {
            total_bytes: 24 * 1024 * 1024 * 1024,
            used_bytes: None,
            source: MemorySource::NvidiaGpu,
        };
        let ctx = estimate_max_context(&hw, 0, 0.0);
        assert!(ctx.is_none());
    }

    #[test]
    fn estimate_rounds_to_256() {
        let hw = HardwareInfo {
            total_bytes: 4 * 1024 * 1024 * 1024, // 4 GiB
            used_bytes: Some(0),
            source: MemorySource::NvidiaGpu,
        };

        let ctx = estimate_max_context(&hw, 1 * 1024 * 1024 * 1024, 8192.0).unwrap();
        // Result should be divisible by 256
        assert_eq!(ctx % 256, 0, "expected multiple of 256, got {ctx}");
    }

    #[test]
    fn available_bytes_with_used() {
        let hw = HardwareInfo {
            total_bytes: 24_000,
            used_bytes: Some(4_000),
            source: MemorySource::NvidiaGpu,
        };
        assert_eq!(hw.available_bytes(), 20_000);
    }

    #[test]
    fn available_bytes_without_used() {
        let hw = HardwareInfo {
            total_bytes: 24_000,
            used_bytes: None,
            source: MemorySource::MacosUnified,
        };
        assert_eq!(hw.available_bytes(), 24_000);
    }

    #[test]
    fn parse_meminfo_kb_valid() {
        assert_eq!(parse_meminfo_kb("  16384000 kB"), Some(16384000));
    }

    #[test]
    fn parse_meminfo_kb_invalid() {
        assert_eq!(parse_meminfo_kb("not a number"), None);
    }

    #[test]
    fn detect_hardware_returns_something_on_this_system() {
        // This test just verifies the detection pipeline doesn't panic.
        // It may return None on systems without nvidia-smi or /proc/meminfo.
        let _hw = detect_hardware();
    }
}