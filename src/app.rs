use crate::backends::{backend_key, detect_backends, fetch_ollama_models, Backend, DetectedBackend};
use crate::config::Config;
use crate::hardware::{self, HardwareInfo};
use crate::models::{
    add_ollama_models, discover_models, DiscoveredModel, ModelFormat, ModelSource,
    load_probe_result, save_probe_result,
};
use crate::opencode;
use crate::server::{self, ServerHandle};
use crate::theme::Theme;
use std::collections::{BTreeSet, HashMap};
use std::collections::VecDeque;
use std::path::PathBuf;
use std::time::{Duration, Instant};

const PROBE_TIMEOUT: Duration = Duration::from_secs(10);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputMode {
    Normal,
    Search,
    BackendPopup,
    ConfirmServe,
    StopPopup,
    AddDir,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Focus {
    Tree,
    Table,
    Serve,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FormatFilter {
    All,
    Gguf,
    Mlx,
}

impl FormatFilter {
    pub fn next(&self) -> Self {
        match self {
            FormatFilter::All => FormatFilter::Gguf,
            FormatFilter::Gguf => FormatFilter::Mlx,
            FormatFilter::Mlx => FormatFilter::All,
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            FormatFilter::All => "All",
            FormatFilter::Gguf => "GGUF",
            FormatFilter::Mlx => "MLX",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortOrder {
    Name,
    Size,
    Source,
}

impl SortOrder {
    pub fn next(&self) -> Self {
        match self {
            SortOrder::Name => SortOrder::Size,
            SortOrder::Size => SortOrder::Source,
            SortOrder::Source => SortOrder::Name,
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            SortOrder::Name => "Name",
            SortOrder::Size => "Size",
            SortOrder::Source => "Source",
        }
    }
}

// -- Source tree --

/// A node in the source tree. Each represents a source category or a custom directory.
#[derive(Debug, Clone)]
pub struct TreeNode {
    pub label: String,
    pub path: Option<PathBuf>,
    pub source: Option<ModelSource>,
    pub model_count: usize,
    pub expanded: bool,
    pub removable: bool,
    /// Models belonging to this node (indices into App.models).
    pub model_indices: Vec<usize>,
}

struct ProbeSession {
    model: DiscoveredModel,
    backend: Backend,
    port: u16,
    candidates: Vec<u32>,
    phase: ProbePhase,
    mode: ProbeMode,
    current_idx: usize,
    current_ctx: u32,
    last_good: Option<u32>,
    first_bad: Option<(u32, String)>,
    refine_low: Option<u32>,
    refine_high: Option<u32>,
    refine_attempts: u8,
    started_at: Instant,
    handle: ServerHandle,
    keep_pinging: bool,
    ping_failure_count: u32,
    ping_success_count: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProbePhase {
    Ascending,
    Descending,
    Refining,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProbeMode {
    Step,
    Deep,
}

pub struct App {
    pub input_mode: InputMode,
    pub focus: Focus,
    pub should_quit: bool,

    pub models: Vec<DiscoveredModel>,
    pub filtered: Vec<usize>,
    pub selected: usize,
    pub scroll_offset: usize,
    pub visible_rows: usize,

    pub search_query: String,
    pub format_filter: FormatFilter,
    pub sort_order: SortOrder,

    pub backends: Vec<DetectedBackend>,
    pub selected_backend: usize,
    pub backend_popup_cursor: usize,

    pub servers: Vec<ServerHandle>,
    pub stop_popup_cursor: usize,
    /// Log lines from servers that have exited (so we can see crash output).
    pub dead_logs: VecDeque<String>,

    pub confirm_backend_idx: usize,
    pub confirm_port_input: String,
    pub confirm_editing_port: bool,
    pub confirm_use_model_max_ctx: bool,
    pub confirm_common_ctx_idx: Option<usize>,
    pub confirm_use_hw_guess: bool,
    pub confirm_probed_ctx: Option<u32>,
    probe_session: Option<ProbeSession>,
    /// In-memory cache of persisted probe results. Loaded at startup and updated
    /// whenever a probe completes. Keys are model file paths.
    probe_results: HashMap<PathBuf, u32>,

    // Source tree
    pub tree_nodes: Vec<TreeNode>,
    pub tree_cursor: usize,
    pub tree_source_filter: Option<ModelSource>,

    /// Width of the left source tree panel.
    pub tree_width: u16,
    /// Width of the right serve/logs panel.
    pub serve_width: u16,
    /// Whether log output wraps long lines.
    pub log_wrap: bool,
    /// Whether the source tree panel is visible.
    pub show_tree: bool,
    /// Whether the serve/logs panel is visible (user can toggle even when empty).
    pub show_serve: bool,

    // Add directory input
    pub add_dir_input: String,
    /// Tab-completion candidates for add-dir.
    pub add_dir_completions: Vec<String>,
    /// Currently selected completion index.
    pub add_dir_completion_idx: usize,

    pub status_message: Option<String>,

    pub config: Config,
    pub theme: Theme,

    /// Cached hardware info (detected once at startup).
    pub hardware_info: Option<HardwareInfo>,
}

impl App {
    pub fn new() -> Self {
        let config = Config::load();
        let theme = Theme::load();
        let backends = detect_backends();

        let mut models = discover_models(&config.extra_model_dirs);

        if let Some(ollama) = backends.iter().find(|b| b.backend == Backend::Ollama) {
            if ollama.available {
                if let Some(ref url) = ollama.api_url {
                    let ollama_models = fetch_ollama_models(url);
                    add_ollama_models(&mut models, ollama_models);
                }
            }
        }

        let selected_backend = if let Some(ref pref) = config.default_backend {
            backends
                .iter()
                .position(|b| {
                    b.backend.label().to_lowercase() == pref.to_lowercase() && b.available
                })
                .unwrap_or_else(|| first_available(&backends))
        } else {
            first_available(&backends)
        };

        let filtered: Vec<usize> = (0..models.len()).collect();
        let tree_nodes = build_tree(&models, &config);

        // Load persisted probe results for all discovered models
        let mut probe_results = HashMap::new();
        for model in &models {
            if let Some(result) = load_probe_result(&model.path) {
                probe_results.insert(model.path.clone(), result.context_size);
            }
        }

        App {
            input_mode: InputMode::Normal,
            focus: Focus::Table,
            should_quit: false,
            models,
            filtered,
            selected: 0,
            scroll_offset: 0,
            visible_rows: 20,
            search_query: String::new(),
            format_filter: FormatFilter::All,
            sort_order: SortOrder::Name,
            backends,
            selected_backend,
            backend_popup_cursor: selected_backend,
            servers: Vec::new(),
            stop_popup_cursor: 0,
            dead_logs: VecDeque::new(),
            confirm_backend_idx: selected_backend,
            confirm_port_input: String::new(),
            confirm_editing_port: false,
            confirm_use_model_max_ctx: false,
            confirm_common_ctx_idx: None,
            confirm_use_hw_guess: false,
            confirm_probed_ctx: None,
            probe_session: None,
            probe_results,
            tree_nodes,
            tree_cursor: 0,
            tree_source_filter: None,
            tree_width: 30,
            serve_width: 38,
            log_wrap: false,
            show_tree: true,
            show_serve: false,
            add_dir_input: String::new(),
            add_dir_completions: Vec::new(),
            add_dir_completion_idx: 0,
            status_message: None,
            config,
            theme,
            hardware_info: hardware::detect_hardware(),
        }
    }

    pub fn active_backend(&self) -> Option<&DetectedBackend> {
        self.backends.get(self.selected_backend)
    }

    pub fn selected_model(&self) -> Option<&DiscoveredModel> {
        self.filtered.get(self.selected).map(|&i| &self.models[i])
    }

    pub fn is_model_served(&self, model_name: &str) -> bool {
        self.servers.iter().any(|s| s.model_name == model_name)
    }

    pub fn open_opencode_for_selected(&mut self) {
        let Some(model) = self.selected_model() else {
            return;
        };

        let Some(server_idx) = self.selected_model_opencode_server_idx() else {
            let backend_label = self
                .active_backend()
                .map(|b| b.backend.label())
                .unwrap_or("this backend");
            self.status_message = Some(format!(
                "Start {} via a running OpenCode-compatible server first (current backend: {})",
                model.name, backend_label
            ));
            return;
        };

        let Some(server) = self.servers.get(server_idx) else {
            self.status_message = Some("OpenCode target server disappeared".into());
            return;
        };

        match opencode::resolve_for_server(server) {
            Ok(session) => {
                self.show_serve = true;
                self.push_dead_log(format!(
                    "[opencode] prepared {} at {} using {}",
                    session.model_id,
                    session.base_url,
                    session.config_path.display()
                ));

                match opencode::launch(&session) {
                    Ok(true) => {
                        self.push_dead_log(format!(
                            "[opencode] launched in tmux with {}",
                            session.launch_command
                        ));
                        self.status_message = Some(format!(
                            "Opened OpenCode for {} in a tmux window",
                            session.model_id
                        ));
                    }
                    Ok(false) => {
                        self.push_dead_log(format!(
                            "[opencode] run manually: {}",
                            session.launch_command
                        ));
                        self.status_message = Some(
                            "Prepared OpenCode launch command in logs (auto-launch requires tmux)"
                                .into(),
                        );
                    }
                    Err(err) => {
                        self.push_dead_log(format!("[opencode] launch failed: {}", err));
                        self.status_message = Some(err);
                    }
                }
            }
            Err(err) => {
                self.status_message = Some(err);
            }
        }
    }

    pub fn next_available_port(&self) -> u16 {
        let base = self.config.preferred_port;
        let used: std::collections::HashSet<u16> = self.servers.iter().map(|s| s.port).collect();
        (base..).find(|p| !used.contains(p)).unwrap_or(base)
    }

    fn selected_model_opencode_server_idx(&self) -> Option<usize> {
        let model = self.selected_model()?;
        let preferred_backend = self.active_backend().map(|b| &b.backend);
        choose_opencode_server_idx(&self.servers, &model.name, preferred_backend)
    }

    // -- Focus --

    pub fn toggle_focus(&mut self) {
        // Build the list of visible panels and cycle through them
        let mut panels = Vec::new();
        if self.show_tree {
            panels.push(Focus::Tree);
        }
        panels.push(Focus::Table); // table is always visible
        if self.show_serve {
            panels.push(Focus::Serve);
        }

        if let Some(pos) = panels.iter().position(|&f| f == self.focus) {
            self.focus = panels[(pos + 1) % panels.len()];
        } else {
            self.focus = Focus::Table;
        }
    }

    pub fn toggle_tree(&mut self) {
        self.show_tree = !self.show_tree;
        if !self.show_tree && self.focus == Focus::Tree {
            self.focus = Focus::Table;
        }
    }

    pub fn toggle_serve_panel(&mut self) {
        self.show_serve = !self.show_serve;
        if !self.show_serve && self.focus == Focus::Serve {
            self.focus = Focus::Table;
        }
    }

    // -- Navigation --

    pub fn move_down(&mut self) {
        match self.focus {
            Focus::Table => {
                if !self.filtered.is_empty() && self.selected < self.filtered.len() - 1 {
                    self.selected += 1;
                    self.ensure_visible();
                }
            }
            Focus::Tree => {
                if self.tree_cursor < self.tree_nodes.len().saturating_sub(1) {
                    self.tree_cursor += 1;
                }
            }
            Focus::Serve => {}
        }
    }

    pub fn move_up(&mut self) {
        match self.focus {
            Focus::Table => {
                if self.selected > 0 {
                    self.selected -= 1;
                    self.ensure_visible();
                }
            }
            Focus::Tree => {
                if self.tree_cursor > 0 {
                    self.tree_cursor -= 1;
                }
            }
            Focus::Serve => {}
        }
    }

    pub fn home(&mut self) {
        match self.focus {
            Focus::Table => {
                self.selected = 0;
                self.scroll_offset = 0;
            }
            Focus::Tree => {
                self.tree_cursor = 0;
            }
            Focus::Serve => {}
        }
    }

    pub fn end(&mut self) {
        match self.focus {
            Focus::Table => {
                if !self.filtered.is_empty() {
                    self.selected = self.filtered.len() - 1;
                    self.ensure_visible();
                }
            }
            Focus::Tree => {
                if !self.tree_nodes.is_empty() {
                    self.tree_cursor = self.tree_nodes.len() - 1;
                }
            }
            Focus::Serve => {}
        }
    }

    pub fn half_page_down(&mut self) {
        if self.focus == Focus::Table {
            let jump = self.visible_rows / 2;
            self.selected = (self.selected + jump).min(self.filtered.len().saturating_sub(1));
            self.ensure_visible();
        }
    }

    pub fn half_page_up(&mut self) {
        if self.focus == Focus::Table {
            let jump = self.visible_rows / 2;
            self.selected = self.selected.saturating_sub(jump);
            self.ensure_visible();
        }
    }

    fn ensure_visible(&mut self) {
        if self.selected < self.scroll_offset {
            self.scroll_offset = self.selected;
        } else if self.selected >= self.scroll_offset + self.visible_rows {
            self.scroll_offset = self.selected - self.visible_rows + 1;
        }
    }

    // -- Pane resize (Shift+Left/Right grows/shrinks the focused pane) --

    pub fn grow_focused_pane(&mut self) {
        match self.focus {
            Focus::Tree => {
                if self.tree_width < 60 {
                    self.tree_width += 2;
                }
            }
            Focus::Serve => {
                if self.serve_width < 80 {
                    self.serve_width += 2;
                }
            }
            Focus::Table => {} // table uses remaining space
        }
    }

    pub fn shrink_focused_pane(&mut self) {
        match self.focus {
            Focus::Tree => {
                if self.tree_width > 16 {
                    self.tree_width -= 2;
                }
            }
            Focus::Serve => {
                if self.serve_width > 24 {
                    self.serve_width -= 2;
                }
            }
            Focus::Table => {} // table uses remaining space
        }
    }

    pub fn toggle_log_wrap(&mut self) {
        self.log_wrap = !self.log_wrap;
    }

    // -- Tree actions --

    pub fn tree_toggle_expand(&mut self) {
        if let Some(node) = self.tree_nodes.get_mut(self.tree_cursor) {
            node.expanded = !node.expanded;
        }
    }

    pub fn tree_select_source(&mut self) {
        if let Some(node) = self.tree_nodes.get(self.tree_cursor) {
            if node.source.is_some() {
                if self.tree_source_filter == node.source {
                    // Deselect — show all
                    self.tree_source_filter = None;
                } else {
                    self.tree_source_filter = node.source.clone();
                }
                self.apply_filters();
            }
        }
    }

    pub fn tree_remove_dir(&mut self) {
        let Some(node) = self.tree_nodes.get(self.tree_cursor) else {
            return;
        };
        if !node.removable {
            self.status_message = Some("Cannot remove built-in source".into());
            return;
        }
        let Some(path) = node.path.clone() else {
            return;
        };
        self.config.extra_model_dirs.retain(|d| *d != path);
        self.config.save();
        self.rebuild_models();
        self.status_message = Some(format!("Removed {}", path.display()));
    }

    pub fn open_add_dir(&mut self) {
        self.add_dir_input.clear();
        self.add_dir_completions.clear();
        self.add_dir_completion_idx = 0;
        self.input_mode = InputMode::AddDir;
    }

    pub fn add_dir_push(&mut self, c: char) {
        self.add_dir_input.push(c);
        self.refresh_completions();
    }

    pub fn add_dir_pop(&mut self) {
        self.add_dir_input.pop();
        self.refresh_completions();
    }

    pub fn add_dir_accept_completion(&mut self) {
        if let Some(completion) = self.add_dir_completions.get(self.add_dir_completion_idx) {
            self.add_dir_input = completion.clone();
            // Append / if it's a directory so user can keep drilling
            if !self.add_dir_input.ends_with('/') {
                self.add_dir_input.push('/');
            }
            self.refresh_completions();
        }
    }

    pub fn add_dir_next_completion(&mut self) {
        if !self.add_dir_completions.is_empty() {
            self.add_dir_completion_idx =
                (self.add_dir_completion_idx + 1) % self.add_dir_completions.len();
        }
    }

    pub fn add_dir_prev_completion(&mut self) {
        if !self.add_dir_completions.is_empty() {
            self.add_dir_completion_idx = if self.add_dir_completion_idx == 0 {
                self.add_dir_completions.len() - 1
            } else {
                self.add_dir_completion_idx - 1
            };
        }
    }

    fn refresh_completions(&mut self) {
        self.add_dir_completions = compute_completions(&self.add_dir_input);
        self.add_dir_completion_idx = 0;
    }

    pub fn confirm_add_dir(&mut self) {
        let raw = self.add_dir_input.trim().to_string();
        if raw.is_empty() {
            self.input_mode = InputMode::Normal;
            return;
        }

        // Expand ~ to home dir
        let expanded = if raw.starts_with('~') {
            if let Some(home) = dirs::home_dir() {
                home.join(raw.strip_prefix("~/").unwrap_or(&raw[1..]))
            } else {
                PathBuf::from(&raw)
            }
        } else {
            PathBuf::from(&raw)
        };

        if !expanded.is_dir() {
            self.status_message = Some(format!("Not a directory: {}", expanded.display()));
            self.input_mode = InputMode::Normal;
            return;
        }

        if self.config.extra_model_dirs.contains(&expanded) {
            self.status_message = Some("Directory already added".into());
            self.input_mode = InputMode::Normal;
            return;
        }

        self.config.extra_model_dirs.push(expanded.clone());
        self.config.save();
        self.rebuild_models();
        self.status_message = Some(format!("Added {}", expanded.display()));
        self.input_mode = InputMode::Normal;
    }

    pub fn cancel_add_dir(&mut self) {
        self.input_mode = InputMode::Normal;
    }

    // -- Filtering --

    pub fn apply_filters(&mut self) {
        self.filtered = (0..self.models.len())
            .filter(|&i| {
                let m = &self.models[i];
                // Source filter from tree
                if let Some(ref src) = self.tree_source_filter {
                    if m.source != *src {
                        return false;
                    }
                }
                match self.format_filter {
                    FormatFilter::All => {}
                    FormatFilter::Gguf => {
                        if m.format != ModelFormat::Gguf {
                            return false;
                        }
                    }
                    FormatFilter::Mlx => {
                        if m.format != ModelFormat::Mlx {
                            return false;
                        }
                    }
                }
                if !self.search_query.is_empty() {
                    let q = self.search_query.to_lowercase();
                    let name_lower = m.name.to_lowercase();
                    if !name_lower.contains(&q) {
                        return false;
                    }
                }
                true
            })
            .collect();

        match self.sort_order {
            SortOrder::Name => {
                self.filtered.sort_by(|&a, &b| {
                    self.models[a]
                        .name
                        .to_lowercase()
                        .cmp(&self.models[b].name.to_lowercase())
                });
            }
            SortOrder::Size => {
                self.filtered
                    .sort_by(|&a, &b| self.models[b].size_bytes.cmp(&self.models[a].size_bytes));
            }
            SortOrder::Source => {
                self.filtered.sort_by(|&a, &b| {
                    self.models[a]
                        .source
                        .to_string()
                        .cmp(&self.models[b].source.to_string())
                });
            }
        }

        if self.filtered.is_empty() {
            self.selected = 0;
        } else if self.selected >= self.filtered.len() {
            self.selected = self.filtered.len() - 1;
        }
        self.scroll_offset = 0;
    }

    pub fn cycle_format_filter(&mut self) {
        self.format_filter = self.format_filter.next();
        self.apply_filters();
    }

    pub fn cycle_sort(&mut self) {
        self.sort_order = self.sort_order.next();
        self.apply_filters();
    }

    // -- Search --

    pub fn enter_search(&mut self) {
        self.input_mode = InputMode::Search;
    }

    pub fn search_push(&mut self, c: char) {
        self.search_query.push(c);
        self.apply_filters();
    }

    pub fn search_pop(&mut self) {
        self.search_query.pop();
        self.apply_filters();
    }

    pub fn exit_search(&mut self) {
        self.input_mode = InputMode::Normal;
    }

    pub fn clear_search(&mut self) {
        self.search_query.clear();
        self.apply_filters();
        self.input_mode = InputMode::Normal;
    }

    // -- Backend popup --

    pub fn open_backend_popup(&mut self) {
        self.backend_popup_cursor = self.selected_backend;
        self.input_mode = InputMode::BackendPopup;
    }

    pub fn backend_popup_down(&mut self) {
        if self.backend_popup_cursor < self.backends.len() - 1 {
            self.backend_popup_cursor += 1;
        }
    }

    pub fn backend_popup_up(&mut self) {
        if self.backend_popup_cursor > 0 {
            self.backend_popup_cursor -= 1;
        }
    }

    pub fn select_backend(&mut self) {
        self.selected_backend = self.backend_popup_cursor;
        self.input_mode = InputMode::Normal;
    }

    // -- Serve --

    pub fn open_confirm_serve(&mut self) {
        self.stop_probe_session(false);
        let Some(model) = self.selected_model() else {
            return;
        };

        // Pre-select the first compatible + available backend
        let best = self
            .backends
            .iter()
            .position(|b| b.can_launch() && b.backend.can_serve_model(&model))
            .unwrap_or(self.selected_backend);

        // Clone what we need before the borrow is released
        let model_path = model.path.clone();

        self.confirm_backend_idx = best;
        self.confirm_port_input = self.next_available_port().to_string();
        self.confirm_editing_port = false;
        self.confirm_use_model_max_ctx = false;
        self.confirm_common_ctx_idx = None;
        self.confirm_use_hw_guess = false;
        // Restore persisted probe result for this model, if any
        self.confirm_probed_ctx = self.probe_results.get(&model_path).copied();
        self.input_mode = InputMode::ConfirmServe;
    }

    pub fn confirm_backend(&self) -> Option<&DetectedBackend> {
        self.backends.get(self.confirm_backend_idx)
    }

    pub fn confirm_port(&self) -> u16 {
        self.confirm_port_input
            .parse()
            .unwrap_or(self.next_available_port())
    }

    pub fn confirm_model_max_ctx(&self) -> Option<u32> {
        self.selected_model()
            .and_then(|model| model.max_context_size)
    }

    pub fn confirm_can_use_model_max_ctx(&self) -> bool {
        self.confirm_model_max_ctx().is_some()
            && self
                .confirm_backend()
                .is_some_and(|backend| backend.backend.supports_ctx_size_override())
    }

    pub fn confirm_common_ctx_sizes(&self) -> Vec<u32> {
        const COMMON_CONTEXT_SIZES: [u32; 9] = [
            4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1_048_576,
        ];

        let supports_override = self
            .confirm_backend()
            .is_some_and(|backend| backend.backend.supports_ctx_size_override());
        if !supports_override {
            return Vec::new();
        }

        let model_max = self.confirm_model_max_ctx();
        COMMON_CONTEXT_SIZES
            .into_iter()
            .filter(|size| model_max.is_none_or(|max| *size <= max))
            .collect()
    }

    pub fn confirm_can_cycle_common_ctx(&self) -> bool {
        !self.confirm_common_ctx_sizes().is_empty()
    }

    pub fn confirm_ctx_source_label(&self) -> &'static str {
        if self.confirm_probed_ctx.is_some() {
            "probe"
        } else if self.confirm_use_hw_guess && self.confirm_can_use_hw_guess() {
            "hw guess"
        } else if self.confirm_use_model_max_ctx && self.confirm_can_use_model_max_ctx() {
            "model max"
        } else if self
            .confirm_common_ctx_idx
            .is_some_and(|idx| idx < self.confirm_common_ctx_sizes().len())
        {
            "common"
        } else {
            "preset default"
        }
    }

    pub fn confirm_ctx_size(&self) -> u32 {
        let backend_key = self
            .confirm_backend()
            .map(|backend| crate::backends::backend_key(&backend.backend))
            .unwrap_or("unknown");
        let preset_ctx = self.config.preset_for(backend_key).ctx_size;
        if let Some(ctx) = self.confirm_probed_ctx {
            ctx
        } else if self.confirm_use_hw_guess && self.confirm_can_use_hw_guess() {
            self.confirm_hw_guess_ctx().unwrap_or(preset_ctx)
        } else if self.confirm_use_model_max_ctx && self.confirm_can_use_model_max_ctx() {
            self.confirm_model_max_ctx().unwrap_or(preset_ctx)
        } else if let Some(idx) = self.confirm_common_ctx_idx {
            self.confirm_common_ctx_sizes()
                .get(idx)
                .copied()
                .unwrap_or(preset_ctx)
        } else {
            preset_ctx
        }
    }

    pub fn confirm_toggle_max_context(&mut self) {
        if self.confirm_can_use_model_max_ctx() {
            self.stop_probe_session(false);
            self.confirm_use_model_max_ctx = !self.confirm_use_model_max_ctx;
            if self.confirm_use_model_max_ctx {
                self.confirm_common_ctx_idx = None;
                self.confirm_use_hw_guess = false;
            }
        }
    }

    pub fn confirm_cycle_common_context(&mut self) {
        self.stop_probe_session(false);
        let options = self.confirm_common_ctx_sizes();
        if options.is_empty() {
            return;
        }

        let current = self.confirm_ctx_size();
        let next_idx = if let Some(idx) = self.confirm_common_ctx_idx {
            (idx + 1) % options.len()
        } else {
            options.iter().position(|size| *size > current).unwrap_or(0)
        };

        self.confirm_use_model_max_ctx = false;
        self.confirm_probed_ctx = None;
        self.confirm_use_model_max_ctx = false;
        self.confirm_common_ctx_idx = Some(next_idx);
        self.confirm_use_hw_guess = false;
        self.confirm_use_hw_guess = false;
    }

    pub fn confirm_hw_guess_ctx(&self) -> Option<u32> {
        let hw = self.hardware_info.as_ref()?;
        let model = self.selected_model()?;
        let kv_cost = model.kv_bytes_per_token?;
        hardware::estimate_max_context(hw, model.size_bytes, kv_cost)
    }

    pub fn confirm_can_use_hw_guess(&self) -> bool {
        self.confirm_hw_guess_ctx().is_some()
            && self
                .confirm_backend()
                .is_some_and(|backend| backend.backend.supports_ctx_size_override())
    }

    pub fn confirm_toggle_hw_guess(&mut self) {
        if self.confirm_can_use_hw_guess() {
            self.stop_probe_session(false);
            self.confirm_use_hw_guess = !self.confirm_use_hw_guess;
            if self.confirm_use_hw_guess {
                self.confirm_use_model_max_ctx = false;
                self.confirm_common_ctx_idx = None;
            }
        }
    }

    pub fn confirm_can_probe_ctx(&self) -> bool {
        self.confirm_backend().is_some_and(|backend| {
            backend.available
                && backend.can_launch()
                && backend.backend.supports_ctx_size_override()
                && self
                    .selected_model()
                    .is_some_and(|model| backend.backend.can_serve_model(model))
        })
    }

    pub fn confirm_probe_context(&mut self) {
        self.start_probe(ProbeMode::Step);
    }

    pub fn confirm_deep_probe_context(&mut self) {
        self.start_probe(ProbeMode::Deep);
    }

    fn start_probe(&mut self, mode: ProbeMode) {
        self.stop_probe_session(false);

        if !self.confirm_can_probe_ctx() {
            self.status_message =
                Some("Context probing is unavailable for this backend/model".into());
            return;
        }

        let Some(model) = self.selected_model().cloned() else {
            return;
        };
        let Some(backend) = self.confirm_backend().map(|b| b.backend.clone()) else {
            return;
        };

        let current_ctx = self.confirm_ctx_size();
        let candidates = probe_candidate_ctx_sizes(
            current_ctx,
            &self.confirm_common_ctx_sizes(),
            self.confirm_model_max_ctx(),
        );

        if candidates.is_empty() {
            self.status_message = Some("No probeable context sizes are available".into());
            return;
        }

        self.show_serve = true;
        self.push_dead_log(format!(
            "--- probing {} via {} ---",
            model.name,
            backend.label()
        ));

        let probe_port = if self.servers.iter().any(|s| s.port == self.confirm_port()) {
            self.next_available_port()
        } else {
            self.confirm_port()
        };

        let start_idx = candidates
            .iter()
            .position(|size| *size >= current_ctx)
            .unwrap_or(candidates.len().saturating_sub(1));

        self.continue_probe_attempt(
            model,
            backend,
            probe_port,
            candidates,
            ProbePhase::Ascending,
            mode,
            start_idx,
            None,
            None,
            None,
            None,
            0,
        );
    }

    fn continue_probe_attempt(
        &mut self,
        model: DiscoveredModel,
        backend: Backend,
        port: u16,
        candidates: Vec<u32>,
        mut phase: ProbePhase,
        mode: ProbeMode,
        mut idx: usize,
        last_good: Option<u32>,
        mut first_bad: Option<(u32, String)>,
        mut refine_low: Option<u32>,
        mut refine_high: Option<u32>,
        mut refine_attempts: u8,
    ) {
        loop {
            let ctx = match phase {
                ProbePhase::Refining => {
                    let Some(low) = refine_low else {
                        self.finish_probe(last_good, first_bad, (Some(model.path.clone()), Some(backend_key(&backend).to_string()), matches!(mode, ProbeMode::Deep)));
                        return;
                    };
                    let Some(high) = refine_high else {
                        self.finish_probe(last_good, first_bad, (Some(model.path.clone()), Some(backend_key(&backend).to_string()), matches!(mode, ProbeMode::Deep)));
                        return;
                    };
                    let Some(mid) = refine_midpoint(low, high) else {
                        self.finish_probe(last_good, first_bad, (Some(model.path.clone()), Some(backend_key(&backend).to_string()), matches!(mode, ProbeMode::Deep)));
                        return;
                    };
                    mid
                }
                _ => candidates[idx],
            };
            let label = if matches!(phase, ProbePhase::Descending) {
                format!("[probe] backpedal to ctx {} on port {}", ctx, port)
            } else if matches!(phase, ProbePhase::Refining) {
                format!("[deep-probe] refining with ctx {} on port {}", ctx, port)
            } else {
                match mode {
                    ProbeMode::Deep => format!("[deep-probe] trying ctx {} on port {}", ctx, port),
                    ProbeMode::Step => format!("[probe] trying ctx {} on port {}", ctx, port),
                }
            };
            self.push_dead_log(label);
            self.status_message = Some(match mode {
                ProbeMode::Deep if matches!(phase, ProbePhase::Refining) => {
                    format!("Deep probing midpoint ctx {}", ctx)
                }
                ProbeMode::Deep => format!("Deep probing ctx {}", ctx),
                ProbeMode::Step => format!("Probing ctx {}", ctx),
            });

            match server::launch_with_overrides(&model, &backend, &self.config, port, ctx) {
                Ok(handle) => {
                    self.probe_session = Some(ProbeSession {
                        model,
                        backend,
                        port,
                        candidates,
                        phase,
                        mode,
                        current_idx: idx,
                        current_ctx: ctx,
                        last_good,
                        first_bad,
                        refine_low,
                        refine_high,
                        refine_attempts,
                        started_at: Instant::now(),
                        handle,
                        keep_pinging: false,
                        ping_failure_count: 0,
                        ping_success_count: 0,
                    });
                    return;
                }
                Err(err) => {
                    self.push_dead_log(match mode {
                        ProbeMode::Deep if matches!(phase, ProbePhase::Refining) => {
                            format!("[deep-probe] midpoint ctx {} failed: {}", ctx, err)
                        }
                        ProbeMode::Deep => format!("[deep-probe] ctx {} failed: {}", ctx, err),
                        ProbeMode::Step => format!("[probe] ctx {} failed: {}", ctx, err),
                    });

                    if matches!(phase, ProbePhase::Refining) {
                        first_bad = Some((ctx, err));
                        refine_high = Some(ctx);
                        refine_attempts += 1;

                        if should_stop_refining(refine_low, refine_high, refine_attempts) {
                            self.finish_probe(last_good, first_bad, (Some(model.path.clone()), Some(backend_key(&backend).to_string()), matches!(mode, ProbeMode::Deep)));
                            return;
                        }

                        continue;
                    }

                    if let Some(good) = last_good {
                        if matches!(mode, ProbeMode::Deep) {
                            refine_low = Some(good);
                            refine_high = Some(ctx);
                            first_bad = Some((ctx, err));
                            refine_attempts = 0;
                            phase = ProbePhase::Refining;
                            continue;
                        }
                        self.finish_probe(Some(good), Some((ctx, err)), (Some(model.path.clone()), Some(backend_key(&backend).to_string()), matches!(mode, ProbeMode::Deep)));
                        return;
                    }

                    first_bad.get_or_insert((ctx, err));

                    if idx == 0 {
                        self.finish_probe(None, first_bad, (Some(model.path.clone()), Some(backend_key(&backend).to_string()), matches!(mode, ProbeMode::Deep)));
                        return;
                    }

                    phase = ProbePhase::Descending;
                    idx -= 1;
                }
            }
        }
    }

    fn finish_probe(
        &mut self,
        last_good: Option<u32>,
        first_bad: Option<(u32, String)>,
        meta: (Option<PathBuf>, Option<String>, bool),
    ) {
        if let Some(ctx) = last_good {
            self.confirm_probed_ctx = Some(ctx);
            self.confirm_use_model_max_ctx = false;
            self.confirm_common_ctx_idx = None;
            self.confirm_use_hw_guess = false;
            self.push_dead_log(format!("[probe] selected ctx {}", ctx));
            self.status_message = Some(match first_bad {
                Some((failed_ctx, _)) if failed_ctx != ctx => {
                    format!("Probe selected ctx {} after {} failed", ctx, failed_ctx)
                }
                _ => format!("Probe selected ctx {}", ctx),
            });

            // Persist successful probe result to disk and update in-memory cache
            if let (Some(model_path), Some(backend), is_deep) = meta {
                save_probe_result(&model_path, ctx, &backend, is_deep);
                self.probe_results.insert(model_path, ctx);
            }
        } else {
            self.status_message = Some(match first_bad {
                Some((ctx, err)) => format!("Probe failed at ctx {}: {}", ctx, err),
                None => "Context probe failed".into(),
            });
        }
    }

    fn stop_probe_session(&mut self, cancelled: bool) {
        if let Some(mut probe) = self.probe_session.take() {
            probe.handle.drain_output();
            if cancelled {
                self.archive_probe_attempt(
                    &probe,
                    format!("[probe] cancelled at ctx {}", probe.current_ctx),
                );
            }
            server::stop(&mut probe.handle);
            if cancelled {
                self.status_message = Some(format!("Cancelled probe at ctx {}", probe.current_ctx));
            }
        }
    }

    fn archive_probe_attempt(&mut self, probe: &ProbeSession, trailer: String) {
        for line in &probe.handle.log_lines {
            self.push_dead_log(line.clone());
        }
        self.push_dead_log(trailer);
    }

    fn push_dead_log(&mut self, line: String) {
        self.dead_logs.push_back(line);
        self.cap_dead_logs();
    }

    fn cap_dead_logs(&mut self) {
        while self.dead_logs.len() > 200 {
            self.dead_logs.pop_front();
        }
    }

    /// Whether the selected backend is compatible with the selected model's format.
    pub fn confirm_compatible(&self) -> bool {
        let Some(model) = self.selected_model() else {
            return false;
        };
        let Some(backend) = self.confirm_backend() else {
            return false;
        };
        backend.backend.can_serve_model(model)
    }

    /// Reason the selected backend can't serve local files, if any.
    pub fn confirm_incompatible_reason(&self) -> Option<&'static str> {
        self.confirm_backend()
            .and_then(|b| b.backend.serve_model_reason(self.selected_model()?))
    }

    pub fn confirm_already_serving(&self) -> bool {
        let Some(model) = self.selected_model() else {
            return false;
        };
        let Some(backend) = self.confirm_backend() else {
            return false;
        };
        self.servers
            .iter()
            .any(|s| s.model_name == model.name && s.backend == backend.backend)
    }

    pub fn confirm_cycle_backend_right(&mut self) {
        if !self.backends.is_empty() {
            self.confirm_backend_idx = (self.confirm_backend_idx + 1) % self.backends.len();
            if !self.confirm_can_use_model_max_ctx() {
                self.confirm_use_model_max_ctx = false;
            }
            if !self.confirm_can_use_hw_guess() {
                self.confirm_use_hw_guess = false;
            }
            if !self.confirm_can_cycle_common_ctx() {
                self.confirm_common_ctx_idx = None;
            } else if self
                .confirm_common_ctx_idx
                .is_some_and(|idx| idx >= self.confirm_common_ctx_sizes().len())
            {
                self.confirm_common_ctx_idx = Some(self.confirm_common_ctx_sizes().len() - 1);
            }
        }
    }

    pub fn confirm_cycle_backend_left(&mut self) {
        if !self.backends.is_empty() {
            self.confirm_backend_idx = if self.confirm_backend_idx == 0 {
                self.backends.len() - 1
            } else {
                self.confirm_backend_idx - 1
            };
            if !self.confirm_can_use_model_max_ctx() {
                self.confirm_use_model_max_ctx = false;
            }
            if !self.confirm_can_use_hw_guess() {
                self.confirm_use_hw_guess = false;
            }
            if !self.confirm_can_cycle_common_ctx() {
                self.confirm_common_ctx_idx = None;
            } else if self
                .confirm_common_ctx_idx
                .is_some_and(|idx| idx >= self.confirm_common_ctx_sizes().len())
            {
                self.confirm_common_ctx_idx = Some(self.confirm_common_ctx_sizes().len() - 1);
            }
        }
    }

    pub fn confirm_toggle_port_edit(&mut self) {
        self.confirm_editing_port = !self.confirm_editing_port;
    }

    pub fn confirm_port_push(&mut self, c: char) {
        if c.is_ascii_digit() && self.confirm_port_input.len() < 5 {
            self.confirm_port_input.push(c);
        }
    }

    pub fn confirm_port_pop(&mut self) {
        self.confirm_port_input.pop();
    }

    pub fn do_serve(&mut self) {
        self.stop_probe_session(true);
        self.input_mode = InputMode::Normal;

        let Some(model) = self.selected_model().cloned() else {
            return;
        };
        let Some(backend) = self.backends.get(self.confirm_backend_idx) else {
            self.status_message = Some("No backend selected".into());
            return;
        };

        if !backend.available {
            self.status_message = Some(format!("{} is not available", backend.backend.label()));
            return;
        }

        if !backend.can_launch() {
            self.status_message = Some(format!(
                "{} is detectable but not launchable from llmserve on this system",
                backend.backend.label()
            ));
            return;
        }

        if !backend.backend.can_serve_model(&model) {
            let reason = backend
                .backend
                .serve_model_reason(&model)
                .unwrap_or("incompatible model");
            self.status_message = Some(format!("{}: {}", backend.backend.label(), reason));
            return;
        }

        if self.confirm_already_serving() {
            self.status_message = Some(format!(
                "{} is already being served via {}",
                model.name,
                backend.backend.label()
            ));
            return;
        }

        let port = self.confirm_port();
        let ctx_size = self.confirm_ctx_size();
        if self.servers.iter().any(|s| s.port == port) {
            self.status_message = Some(format!("Port {port} is already in use"));
            return;
        }

        match server::launch_with_overrides(&model, &backend.backend, &self.config, port, ctx_size)
        {
            Ok(handle) => {
                self.status_message = Some(format!(
                    "Started {} via {} on port {} (ctx {})",
                    handle.model_name,
                    handle.backend.label(),
                    handle.port,
                    ctx_size
                ));
                self.servers.push(handle);
                self.show_serve = true; // auto-show serve panel
            }
            Err(e) => {
                self.status_message = Some(e);
                self.show_serve = true; // show panel so error is visible
            }
        }
    }

    // -- Stop --

    pub fn open_stop_popup(&mut self) {
        if self.servers.is_empty() {
            self.status_message = Some("No servers running".into());
            return;
        }
        if self.servers.len() == 1 {
            self.stop_server_at(0);
            return;
        }
        self.stop_popup_cursor = 0;
        self.input_mode = InputMode::StopPopup;
    }

    pub fn stop_popup_down(&mut self) {
        if self.stop_popup_cursor < self.servers.len() {
            self.stop_popup_cursor += 1;
        }
    }

    pub fn stop_popup_up(&mut self) {
        if self.stop_popup_cursor > 0 {
            self.stop_popup_cursor -= 1;
        }
    }

    pub fn confirm_stop(&mut self) {
        if self.stop_popup_cursor == self.servers.len() {
            self.stop_all_servers();
        } else {
            self.stop_server_at(self.stop_popup_cursor);
        }
        self.input_mode = InputMode::Normal;
    }

    fn stop_server_at(&mut self, idx: usize) {
        if idx < self.servers.len() {
            let mut handle = self.servers.remove(idx);
            let name = handle.model_name.clone();
            let port = handle.port;
            server::stop(&mut handle);
            self.status_message = Some(format!("Stopped {name} on port {port}"));
        }
    }

    pub fn stop_all_servers(&mut self) {
        let count = self.servers.len();
        for handle in &mut self.servers {
            server::stop(handle);
        }
        self.servers.clear();
        self.status_message = Some(format!("Stopped {count} server(s)"));
    }

    pub fn cancel_popup(&mut self) {
        self.stop_probe_session(true);
        self.input_mode = InputMode::Normal;
    }

    // -- Tick --

    pub fn tick(&mut self) {
        let mut probe_next = None;
        let mut probe_done = None;
        let mut probe_keep = None;

        if let Some(mut probe) = self.probe_session.take() {
            probe.handle.drain_output();

            if let Some(msg) = server::check_exited(&mut probe.handle) {
                let err = probe_failure_message(&probe.handle, msg);
                self.archive_probe_attempt(
                    &probe,
                    probe_log_line(
                        probe.mode,
                        probe.phase,
                        format!("ctx {} failed: {}", probe.current_ctx, err),
                    ),
                );

                if matches!(probe.phase, ProbePhase::Refining) {
                    let refine_low = probe.refine_low;
                    let refine_high = Some(probe.current_ctx);
                    let refine_attempts = probe.refine_attempts + 1;
                    let first_bad = Some((probe.current_ctx, err.clone()));

                    if should_stop_refining(refine_low, refine_high, refine_attempts) {
                        probe_done = Some((probe.last_good, first_bad, probe.model.path.clone(), backend_key(&probe.backend).to_string(), matches!(probe.mode, ProbeMode::Deep)));
                    } else {
                        probe_next = Some((
                            probe.model.clone(),
                            probe.backend.clone(),
                            probe.port,
                            probe.candidates.clone(),
                            ProbePhase::Refining,
                            probe.mode,
                            probe.current_idx,
                            probe.last_good,
                            first_bad,
                            refine_low,
                            refine_high,
                            refine_attempts,
                        ));
                    }
                } else if let Some(last_good) = probe.last_good {
                    if matches!(probe.mode, ProbeMode::Deep) {
                        probe_next = Some((
                            probe.model.clone(),
                            probe.backend.clone(),
                            probe.port,
                            probe.candidates.clone(),
                            ProbePhase::Refining,
                            probe.mode,
                            probe.current_idx,
                            Some(last_good),
                            Some((probe.current_ctx, err)),
                            Some(last_good),
                            Some(probe.current_ctx),
                            0,
                        ));
                    } else {
                        probe_done = Some((Some(last_good), Some((probe.current_ctx, err)), probe.model.path.clone(), backend_key(&probe.backend).to_string(), matches!(probe.mode, ProbeMode::Deep)));
                    }
                } else if probe.current_idx > 0 {
                    probe_next = Some((
                        probe.model.clone(),
                        probe.backend.clone(),
                        probe.port,
                        probe.candidates.clone(),
                        ProbePhase::Descending,
                        probe.mode,
                        probe.current_idx - 1,
                        None,
                        Some((probe.current_ctx, err)),
                        None,
                        Some(probe.current_ctx),
                        0,
                    ));
                } else {
                    probe_done = Some((None, Some((probe.current_ctx, err)), probe.model.path.clone(), backend_key(&probe.backend).to_string(), matches!(probe.mode, ProbeMode::Deep)));
                }
            } else if probe.started_at.elapsed() >= PROBE_TIMEOUT {
                let ctx = probe.current_ctx;
                self.archive_probe_attempt(
                    &probe,
                    probe_log_line(
                        probe.mode,
                        probe.phase,
                        format!("ctx {} stayed up for 10s", ctx),
                    ),
                );
                server::stop(&mut probe.handle);

                match probe.phase {
                    ProbePhase::Refining => {
                        let last_good = Some(ctx);
                        let first_bad = probe.first_bad.clone();
                        let refine_low = Some(ctx);
                        let refine_high = probe.refine_high;
                        let refine_attempts = probe.refine_attempts + 1;

                        if should_stop_refining(refine_low, refine_high, refine_attempts) {
                            probe_done = Some((last_good, first_bad, probe.model.path.clone(), backend_key(&probe.backend).to_string(), matches!(probe.mode, ProbeMode::Deep)));
                        } else {
                            probe_next = Some((
                                probe.model.clone(),
                                probe.backend.clone(),
                                probe.port,
                                probe.candidates.clone(),
                                ProbePhase::Refining,
                                probe.mode,
                                probe.current_idx,
                                last_good,
                                first_bad,
                                refine_low,
                                refine_high,
                                refine_attempts,
                            ));
                        }
                    }
                    ProbePhase::Ascending if probe.current_idx + 1 < probe.candidates.len() => {
                        probe_next = Some((
                            probe.model.clone(),
                            probe.backend.clone(),
                            probe.port,
                            probe.candidates.clone(),
                            ProbePhase::Ascending,
                            probe.mode,
                            probe.current_idx + 1,
                            Some(ctx),
                            probe.first_bad.clone(),
                            probe.refine_low,
                            probe.refine_high,
                            probe.refine_attempts,
                        ));
                    }
                    _ => {
                        probe_done = Some((Some(ctx), probe.first_bad.clone(), probe.model.path.clone(), backend_key(&probe.backend).to_string(), matches!(probe.mode, ProbeMode::Deep)));
                    }
                }
            } else {
                probe_keep = Some(probe);
            }
        }

        if let Some(probe) = probe_keep {
            self.probe_session = Some(probe);
        }

        if let Some((last_good, first_bad, model_path, backend, is_deep)) = probe_done {
            self.finish_probe(last_good, first_bad, (Some(model_path), Some(backend), is_deep));
        } else if let Some((
            model,
            backend,
            port,
            candidates,
            phase,
            mode,
            idx,
            last_good,
            first_bad,
            refine_low,
            refine_high,
            refine_attempts,
        )) = probe_next
        {
            self.continue_probe_attempt(
                model,
                backend,
                port,
                candidates,
                phase,
                mode,
                idx,
                last_good,
                first_bad,
                refine_low,
                refine_high,
                refine_attempts,
            );
        }

        // Drain output from all running servers
        for handle in &mut self.servers {
            handle.drain_output();
        }

        // Check for exits
        let mut exited = Vec::new();
        for (i, handle) in self.servers.iter_mut().enumerate() {
            if let Some(msg) = server::check_exited(handle) {
                exited.push((i, msg));
            }
        }
        for (i, msg) in exited.into_iter().rev() {
            let handle = self.servers.remove(i);
            // Preserve logs from the dead server
            self.dead_logs.push_back(format!(
                "--- {} (:{}) exited: {msg} ---",
                handle.model_name, handle.port
            ));
            for line in &handle.log_lines {
                self.dead_logs.push_back(line.clone());
            }
            self.cap_dead_logs();
            self.status_message = Some(format!(
                "{} (port {}): {msg}",
                handle.model_name, handle.port
            ));
        }
    }

    /// Get all log lines to display — combines live server logs and dead logs.
    pub fn all_log_lines(&self) -> Vec<(&str, &str)> {
        let mut lines = Vec::new();

        // Dead logs first (historical)
        for line in &self.dead_logs {
            lines.push(("", line.as_str()));
        }

        if let Some(probe) = &self.probe_session {
            if !probe.handle.log_lines.is_empty() {
                lines.push((probe.handle.model_name.as_str(), "─── probe ───"));
                for line in &probe.handle.log_lines {
                    lines.push((probe.handle.model_name.as_str(), line.as_str()));
                }
            }
        }

        // Then live server logs
        for s in &self.servers {
            if !s.log_lines.is_empty() {
                lines.push((s.model_name.as_str(), "─── live ───"));
                for line in &s.log_lines {
                    lines.push((s.model_name.as_str(), line.as_str()));
                }
            }
        }

        lines
    }

    pub fn clear_dead_logs(&mut self) {
        self.dead_logs.clear();
    }

    /// Whether there are any logs to show (live or dead).
    pub fn has_logs(&self) -> bool {
        !self.dead_logs.is_empty()
            || self
                .probe_session
                .as_ref()
                .is_some_and(|probe| !probe.handle.log_lines.is_empty())
            || self.servers.iter().any(|s| !s.log_lines.is_empty())
    }

    // -- Refresh / rebuild --

    pub fn refresh(&mut self) {
        self.backends = detect_backends();
        self.rebuild_models();
        self.status_message = Some(format!("Found {} models", self.models.len()));
    }

    fn rebuild_models(&mut self) {
        self.models = discover_models(&self.config.extra_model_dirs);

        if let Some(ollama) = self.backends.iter().find(|b| b.backend == Backend::Ollama) {
            if ollama.available {
                if let Some(ref url) = ollama.api_url {
                    let ollama_models = fetch_ollama_models(url);
                    add_ollama_models(&mut self.models, ollama_models);
                }
            }
        }

        self.tree_nodes = build_tree(&self.models, &self.config);
        self.apply_filters();
    }

    // -- Theme --

    pub fn cycle_theme(&mut self) {
        self.theme = self.theme.next();
        self.theme.save();
    }
}

fn first_available(backends: &[DetectedBackend]) -> usize {
    backends.iter().position(|b| b.available).unwrap_or(0)
}

fn probe_candidate_ctx_sizes(current: u32, common: &[u32], model_max: Option<u32>) -> Vec<u32> {
    let mut sizes = BTreeSet::new();
    sizes.insert(current);

    for &size in common {
        sizes.insert(size);
    }

    if let Some(max_ctx) = model_max {
        sizes.insert(max_ctx);
    }

    sizes.into_iter().collect()
}

fn choose_opencode_server_idx(
    servers: &[ServerHandle],
    model_name: &str,
    preferred_backend: Option<&Backend>,
) -> Option<usize> {
    let matches: Vec<usize> = servers
        .iter()
        .enumerate()
        .filter(|(_, server)| {
            server.model_name == model_name && server.backend.can_open_opencode()
        })
        .map(|(idx, _)| idx)
        .collect();

    if matches.is_empty() {
        return None;
    }

    if let Some(preferred) = preferred_backend
        && preferred.can_open_opencode()
    {
        let exact: Vec<usize> = matches
            .iter()
            .copied()
            .filter(|idx| servers[*idx].backend == *preferred)
            .collect();
        if exact.len() == 1 {
            return exact.first().copied();
        }
        if exact.len() > 1 {
            return None;
        }
    }

    (matches.len() == 1).then(|| matches[0])
}

fn refine_midpoint(low: u32, high: u32) -> Option<u32> {
    if high <= low + 256 {
        return None;
    }

    let midpoint = low + ((high - low) / 2);
    let rounded = (midpoint / 256) * 256;
    if rounded <= low || rounded >= high {
        None
    } else {
        Some(rounded)
    }
}

fn should_stop_refining(low: Option<u32>, high: Option<u32>, attempts: u8) -> bool {
    attempts >= 8
        || low
            .zip(high)
            .is_none_or(|(low, high)| high.saturating_sub(low) < 2048)
        || refine_midpoint(low.unwrap_or(0), high.unwrap_or(0)).is_none()
}

fn probe_log_line(mode: ProbeMode, phase: ProbePhase, message: String) -> String {
    match (mode, phase) {
        (ProbeMode::Deep, ProbePhase::Refining) => format!("[deep-probe] midpoint {message}"),
        (ProbeMode::Deep, _) => format!("[deep-probe] {message}"),
        _ => format!("[probe] {message}"),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        choose_opencode_server_idx, probe_candidate_ctx_sizes, probe_log_line, refine_midpoint,
        should_stop_refining, ProbeMode, ProbePhase,
    };
    use crate::backends::Backend;
    use crate::server::{make_test_handle, ServerHandle};
    use std::process::{Command, Stdio};

    fn dummy_server(backend: Backend, model_name: &str, port: u16) -> ServerHandle {
        let child = Command::new("sh")
            .arg("-c")
            .arg("sleep 1")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();

        make_test_handle(backend, model_name.into(), "127.0.0.1".into(), port, child)
    }

    #[test]
    fn probe_candidates_include_current_common_and_model_max() {
        let sizes = probe_candidate_ctx_sizes(8192, &[4096, 8192, 16384], Some(20000));
        assert_eq!(sizes, vec![4096, 8192, 16384, 20000]);
    }

    #[test]
    fn probe_candidates_dedup_and_sort() {
        let sizes = probe_candidate_ctx_sizes(16384, &[4096, 16384, 32768], Some(32768));
        assert_eq!(sizes, vec![4096, 16384, 32768]);
    }

    #[test]
    fn refine_midpoint_rounds_down_to_256_boundary() {
        assert_eq!(refine_midpoint(131072, 262144), Some(196608));
    }

    #[test]
    fn refine_midpoint_returns_none_when_gap_too_small() {
        assert_eq!(refine_midpoint(8192, 8448), None);
    }

    #[test]
    fn refining_stops_on_small_gap() {
        assert!(should_stop_refining(Some(131072), Some(132096), 0));
    }

    #[test]
    fn refining_stops_after_attempt_limit() {
        assert!(should_stop_refining(Some(131072), Some(262144), 8));
    }

    #[test]
    fn deep_probe_log_line_marks_refinement() {
        assert_eq!(
            probe_log_line(
                ProbeMode::Deep,
                ProbePhase::Refining,
                "ctx 196608 ok".into()
            ),
            "[deep-probe] midpoint ctx 196608 ok"
        );
    }

    #[test]
    fn choose_opencode_server_prefers_exact_backend_match() {
        let mut servers = vec![
            dummy_server(Backend::LlamaServer, "qwen", 8080),
            dummy_server(Backend::LlamaServer, "other", 8081),
        ];
        assert_eq!(
            choose_opencode_server_idx(&servers, "qwen", Some(&Backend::LlamaServer)),
            Some(0)
        );
        for server in &mut servers {
            let _ = server.child.kill();
            let _ = server.child.wait();
        }
    }

    #[test]
    fn choose_opencode_server_rejects_unsupported_backend_matches() {
        let mut servers = vec![dummy_server(Backend::LocalAi, "qwen", 8080)];
        assert_eq!(
            choose_opencode_server_idx(&servers, "qwen", Some(&Backend::LocalAi)),
            None
        );
        for server in &mut servers {
            let _ = server.child.kill();
            let _ = server.child.wait();
        }
    }
}

/// Build the source tree from discovered models.
fn build_tree(models: &[DiscoveredModel], config: &Config) -> Vec<TreeNode> {
    let home = dirs::home_dir().unwrap_or_default();
    let mut nodes = Vec::new();

    // Built-in sources
    let builtins: Vec<(ModelSource, &str, Option<PathBuf>)> = vec![
        (
            ModelSource::LmStudio,
            "LM Studio",
            Some(home.join(".lmstudio").join("models")),
        ),
        (
            ModelSource::LlamaCppCache,
            "llama.cpp",
            Some(home.join(".cache").join("llm-models")),
        ),
        (
            ModelSource::HfCache,
            "HF/MLX",
            Some(
                std::env::var("HF_HOME")
                    .map(PathBuf::from)
                    .unwrap_or_else(|_| home.join(".cache").join("huggingface").join("hub")),
            ),
        ),
        (
            ModelSource::LlmfitCache,
            "llmfit",
            Some(
                std::env::var("LLMFIT_MODELS_DIR")
                    .map(PathBuf::from)
                    .unwrap_or_else(|_| home.join(".cache").join("llmfit").join("models")),
            ),
        ),
        (ModelSource::Ollama, "Ollama", None),
    ];

    for (source, label, path) in builtins {
        let model_indices: Vec<usize> = models
            .iter()
            .enumerate()
            .filter(|(_, m)| m.source == source)
            .map(|(i, _)| i)
            .collect();
        let count = model_indices.len();

        let display = if let Some(ref p) = path {
            let short = shorten_path(p, &home);
            format!("{label} ({short})")
        } else {
            format!("{label} (API)")
        };

        nodes.push(TreeNode {
            label: display,
            path,
            source: Some(source),
            model_count: count,
            expanded: true,
            removable: false,
            model_indices,
        });
    }

    // Custom directories
    for dir in &config.extra_model_dirs {
        let model_indices: Vec<usize> = models
            .iter()
            .enumerate()
            .filter(|(_, m)| m.source == ModelSource::ExtraDir && m.path.starts_with(dir))
            .map(|(i, _)| i)
            .collect();
        let count = model_indices.len();
        let short = shorten_path(dir, &home);

        nodes.push(TreeNode {
            label: format!("{short}"),
            path: Some(dir.clone()),
            source: Some(ModelSource::ExtraDir),
            model_count: count,
            expanded: true,
            removable: true,
            model_indices,
        });
    }

    nodes
}

/// Compute filesystem completions for the current input.
fn compute_completions(input: &str) -> Vec<String> {
    if input.is_empty() {
        return Vec::new();
    }

    // Expand ~ to home dir for lookup
    let expanded = if input.starts_with('~') {
        if let Some(home) = dirs::home_dir() {
            let rest = input.strip_prefix("~/").unwrap_or(&input[1..]);
            home.join(rest).to_string_lossy().to_string()
        } else {
            input.to_string()
        }
    } else {
        input.to_string()
    };

    let path = std::path::Path::new(&expanded);

    // If input ends with /, list contents of that directory
    // Otherwise, treat parent as directory and filename prefix as filter
    let (dir, prefix) = if expanded.ends_with('/') || path.is_dir() && input.ends_with('/') {
        (path.to_path_buf(), String::new())
    } else {
        let parent = path.parent().unwrap_or(std::path::Path::new("/"));
        let prefix = path
            .file_name()
            .map(|f| f.to_string_lossy().to_string())
            .unwrap_or_default();
        (parent.to_path_buf(), prefix)
    };

    let Ok(entries) = std::fs::read_dir(&dir) else {
        return Vec::new();
    };

    let home = dirs::home_dir();
    let prefix_lower = prefix.to_lowercase();

    let mut results: Vec<String> = entries
        .flatten()
        .filter(|e| {
            // Only show directories
            e.file_type().is_ok_and(|ft| ft.is_dir())
        })
        .filter(|e| {
            if prefix.is_empty() {
                true
            } else {
                e.file_name()
                    .to_string_lossy()
                    .to_lowercase()
                    .starts_with(&prefix_lower)
            }
        })
        .filter(|e| {
            // Hide hidden dirs unless user typed a dot
            let name = e.file_name().to_string_lossy().to_string();
            !name.starts_with('.') || prefix.starts_with('.')
        })
        .map(|e| {
            let full = e.path();
            // Show with ~ if original input used ~
            if input.starts_with('~') {
                if let Some(ref h) = home {
                    if let Ok(stripped) = full.strip_prefix(h) {
                        return format!("~/{}", stripped.display());
                    }
                }
            }
            full.display().to_string()
        })
        .collect();

    results.sort();
    results.truncate(10); // Cap at 10 completions
    results
}

fn shorten_path(path: &PathBuf, home: &PathBuf) -> String {
    if let Ok(stripped) = path.strip_prefix(home) {
        format!("~/{}", stripped.display())
    } else {
        path.display().to_string()
    }
}

fn probe_failure_message(handle: &ServerHandle, msg: String) -> String {
    handle
        .log_lines
        .iter()
        .rev()
        .find(|line| !line.trim().is_empty())
        .map(|line| format!("{msg}; last log: {line}"))
        .unwrap_or(msg)
}
