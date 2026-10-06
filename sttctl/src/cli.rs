use clap::{Parser, Subcommand, ValueEnum};

#[derive(Parser)]
#[command(name = "sttctl", about = "Control sttd daemon via D-Bus")]
pub struct Cli {
    /// Output in JSON format
    #[arg(long, global = true)]
    pub json: bool,

    /// Timeout in seconds for operations
    #[arg(long, global = true, value_name = "SECONDS")]
    pub timeout: Option<u64>,

    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand)]
pub enum Command {
    /// Control pause state
    Daemon {
        #[command(subcommand)]
        action: DaemonAction,
    },
    /// Control dictionary
    Dictionary {
        #[command(subcommand)]
        action: DictionaryAction,
    },
    /// Control translation
    Translation {
        #[command(subcommand)]
        action: TranslationAction,
    },
    /// Full status (pause + translation)
    Status,
    /// Watch D-Bus signals
    Watch {
        /// Filter signals by name
        #[arg(long)]
        signals: Vec<String>,
    },
    /// Generate shell completions
    Completions {
        #[arg(value_enum)]
        shell: Shell,
    },
}

#[derive(Subcommand)]
pub enum DaemonAction {
    /// Pause transcription
    Pause,
    /// Resume transcription
    Resume,
    /// Toggle pause state
    Toggle,
    /// Get pause state
    Paused,
}

#[derive(Subcommand)]
pub enum DictionaryAction {
    /// Load dictionary from file
    Load {
        /// Path to dictionary file
        path: String,
    },
    /// Replace all words
    Replace {
        /// Words to set
        words: Vec<String>,
    },
    /// Add words to dictionary
    Add {
        /// Words to add
        words: Vec<String>,
    },
    /// Remove words from dictionary
    Remove {
        /// Words to remove
        words: Vec<String>,
    },
    /// Get all words
    Get,
    /// Check if word exists
    Contains {
        /// Word to check
        word: String,
    },
    /// Get word count
    Count,
    /// Check if dictionary is dirty
    Dirty,
}

#[derive(Subcommand)]
pub enum TranslationAction {
    /// Enable translation
    Enable,
    /// Disable translation
    Disable {
        /// Unload model after disabling
        #[arg(long)]
        unload: bool,
    },
    /// Toggle translation
    Toggle,
    /// Set target language
    SetTarget {
        /// Target language code
        language: String,
    },
    /// Get translation state
    State,
    /// Check if translation is enabled
    Enabled,
    /// Get target language
    TargetLanguage,
    /// Get requested language
    RequestedLanguage,
    /// Get model
    Model,
}

#[derive(ValueEnum, Clone)]
pub enum Shell {
    Bash,
    Zsh,
    Fish,
    PowerShell,
}
