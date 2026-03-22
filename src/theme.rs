use ratatui::style::Color;
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Theme {
    Default,
    Dracula,
    Solarized,
    Nord,
    Monokai,
    Gruvbox,
    CatppuccinMocha,
}

impl Theme {
    pub fn label(&self) -> &'static str {
        match self {
            Theme::Default => "Default",
            Theme::Dracula => "Dracula",
            Theme::Solarized => "Solarized",
            Theme::Nord => "Nord",
            Theme::Monokai => "Monokai",
            Theme::Gruvbox => "Gruvbox",
            Theme::CatppuccinMocha => "Catppuccin Mocha",
        }
    }

    pub fn next(&self) -> Self {
        match self {
            Theme::Default => Theme::Dracula,
            Theme::Dracula => Theme::Solarized,
            Theme::Solarized => Theme::Nord,
            Theme::Nord => Theme::Monokai,
            Theme::Monokai => Theme::Gruvbox,
            Theme::Gruvbox => Theme::CatppuccinMocha,
            Theme::CatppuccinMocha => Theme::Default,
        }
    }

    pub fn colors(&self) -> ThemeColors {
        match self {
            Theme::Default => default_colors(),
            Theme::Dracula => dracula_colors(),
            Theme::Solarized => solarized_colors(),
            Theme::Nord => nord_colors(),
            Theme::Monokai => monokai_colors(),
            Theme::Gruvbox => gruvbox_colors(),
            Theme::CatppuccinMocha => catppuccin_mocha_colors(),
        }
    }

    fn config_path() -> Option<PathBuf> {
        dirs::config_dir().map(|d| d.join("llmserve").join("theme"))
    }

    pub fn save(&self) {
        if let Some(path) = Self::config_path() {
            if let Some(parent) = path.parent() {
                let _ = fs::create_dir_all(parent);
            }
            let _ = fs::write(&path, self.label());
        }
    }

    pub fn load() -> Self {
        Self::config_path()
            .and_then(|path| fs::read_to_string(path).ok())
            .map(|s| Self::from_label(s.trim()))
            .unwrap_or(Theme::Default)
    }

    fn from_label(s: &str) -> Self {
        match s {
            "Dracula" => Theme::Dracula,
            "Solarized" => Theme::Solarized,
            "Nord" => Theme::Nord,
            "Monokai" => Theme::Monokai,
            "Gruvbox" => Theme::Gruvbox,
            "Catppuccin Mocha" => Theme::CatppuccinMocha,
            _ => Theme::Default,
        }
    }
}

pub struct ThemeColors {
    pub bg: Color,
    pub fg: Color,
    pub muted: Color,
    pub border: Color,
    pub title: Color,
    pub highlight_bg: Color,
    pub accent: Color,
    pub good: Color,
    pub warning: Color,
    pub error: Color,
    pub info: Color,
    pub status_bg: Color,
    pub status_fg: Color,
}

fn default_colors() -> ThemeColors {
    ThemeColors {
        bg: Color::Reset,
        fg: Color::Reset,
        muted: Color::DarkGray,
        border: Color::DarkGray,
        title: Color::Green,
        highlight_bg: Color::LightBlue,
        accent: Color::Cyan,
        good: Color::Green,
        warning: Color::Yellow,
        error: Color::Red,
        info: Color::Cyan,
        status_bg: Color::Green,
        status_fg: Color::Black,
    }
}

fn dracula_colors() -> ThemeColors {
    ThemeColors {
        bg: Color::Rgb(40, 42, 54),
        fg: Color::Rgb(248, 248, 242),
        muted: Color::Rgb(98, 114, 164),
        border: Color::Rgb(68, 71, 90),
        title: Color::Rgb(80, 250, 123),
        highlight_bg: Color::Rgb(68, 71, 90),
        accent: Color::Rgb(139, 233, 253),
        good: Color::Rgb(80, 250, 123),
        warning: Color::Rgb(241, 250, 140),
        error: Color::Rgb(255, 85, 85),
        info: Color::Rgb(139, 233, 253),
        status_bg: Color::Rgb(189, 147, 249),
        status_fg: Color::Rgb(40, 42, 54),
    }
}

fn solarized_colors() -> ThemeColors {
    ThemeColors {
        bg: Color::Rgb(0, 43, 54),
        fg: Color::Rgb(131, 148, 150),
        muted: Color::Rgb(88, 110, 117),
        border: Color::Rgb(88, 110, 117),
        title: Color::Rgb(133, 153, 0),
        highlight_bg: Color::Rgb(7, 54, 66),
        accent: Color::Rgb(38, 139, 210),
        good: Color::Rgb(133, 153, 0),
        warning: Color::Rgb(181, 137, 0),
        error: Color::Rgb(220, 50, 47),
        info: Color::Rgb(38, 139, 210),
        status_bg: Color::Rgb(38, 139, 210),
        status_fg: Color::Rgb(253, 246, 227),
    }
}

fn nord_colors() -> ThemeColors {
    ThemeColors {
        bg: Color::Rgb(46, 52, 64),
        fg: Color::Rgb(216, 222, 233),
        muted: Color::Rgb(76, 86, 106),
        border: Color::Rgb(67, 76, 94),
        title: Color::Rgb(163, 190, 140),
        highlight_bg: Color::Rgb(59, 66, 82),
        accent: Color::Rgb(136, 192, 208),
        good: Color::Rgb(163, 190, 140),
        warning: Color::Rgb(235, 203, 139),
        error: Color::Rgb(191, 97, 106),
        info: Color::Rgb(136, 192, 208),
        status_bg: Color::Rgb(129, 161, 193),
        status_fg: Color::Rgb(46, 52, 64),
    }
}

fn monokai_colors() -> ThemeColors {
    ThemeColors {
        bg: Color::Rgb(39, 40, 34),
        fg: Color::Rgb(248, 248, 242),
        muted: Color::Rgb(117, 113, 94),
        border: Color::Rgb(73, 72, 62),
        title: Color::Rgb(166, 226, 46),
        highlight_bg: Color::Rgb(73, 72, 62),
        accent: Color::Rgb(102, 217, 239),
        good: Color::Rgb(166, 226, 46),
        warning: Color::Rgb(230, 219, 116),
        error: Color::Rgb(249, 38, 114),
        info: Color::Rgb(102, 217, 239),
        status_bg: Color::Rgb(253, 151, 31),
        status_fg: Color::Rgb(39, 40, 34),
    }
}

fn gruvbox_colors() -> ThemeColors {
    ThemeColors {
        bg: Color::Rgb(40, 40, 40),
        fg: Color::Rgb(235, 219, 178),
        muted: Color::Rgb(146, 131, 116),
        border: Color::Rgb(80, 73, 69),
        title: Color::Rgb(184, 187, 38),
        highlight_bg: Color::Rgb(60, 56, 54),
        accent: Color::Rgb(131, 165, 152),
        good: Color::Rgb(184, 187, 38),
        warning: Color::Rgb(250, 189, 47),
        error: Color::Rgb(251, 73, 52),
        info: Color::Rgb(131, 165, 152),
        status_bg: Color::Rgb(214, 93, 14),
        status_fg: Color::Rgb(40, 40, 40),
    }
}

fn catppuccin_mocha_colors() -> ThemeColors {
    ThemeColors {
        bg: Color::Rgb(30, 30, 46),
        fg: Color::Rgb(205, 214, 244),
        muted: Color::Rgb(127, 132, 156),
        border: Color::Rgb(88, 91, 112),
        title: Color::Rgb(166, 227, 161),
        highlight_bg: Color::Rgb(49, 50, 68),
        accent: Color::Rgb(137, 180, 250),
        good: Color::Rgb(166, 227, 161),
        warning: Color::Rgb(249, 226, 175),
        error: Color::Rgb(243, 139, 168),
        info: Color::Rgb(137, 220, 235),
        status_bg: Color::Rgb(180, 190, 254),
        status_fg: Color::Rgb(17, 17, 27),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn theme_cycle_wraps() {
        let mut theme = Theme::Default;
        let start = theme;
        // Cycle through all themes and back to start
        for _ in 0..7 {
            theme = theme.next();
        }
        assert_eq!(theme, start);
    }

    #[test]
    fn theme_from_label_roundtrip() {
        let themes = [
            Theme::Default,
            Theme::Dracula,
            Theme::Solarized,
            Theme::Nord,
            Theme::Monokai,
            Theme::Gruvbox,
            Theme::CatppuccinMocha,
        ];
        for t in themes {
            assert_eq!(Theme::from_label(t.label()), t);
        }
    }

    #[test]
    fn theme_from_unknown_label_defaults() {
        assert_eq!(Theme::from_label("NonExistent"), Theme::Default);
    }

    #[test]
    fn all_themes_produce_colors() {
        let mut theme = Theme::Default;
        for _ in 0..7 {
            let _ = theme.colors();
            theme = theme.next();
        }
    }
}
