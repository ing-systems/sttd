use crate::cli::*;
use crate::dbus_client::SttdClient;
use crate::output::*;
use anyhow::Result;
use serde_json::json;

pub fn execute(cli: &Cli, client: &SttdClient) -> Result<()> {
    let print = |fields: &[(&str, serde_json::Value)]| print_fields(cli.json, fields);
    match &cli.command {
        Command::Daemon { action } => {
            let paused = match action {
                DaemonAction::Pause => client.pause()?,
                DaemonAction::Resume => client.resume()?,
                DaemonAction::Toggle => client.toggle_pause()?,
                DaemonAction::Paused => client.is_paused()?,
            };
            print(&[("paused", json!(paused))]);
        }
        Command::Dictionary { action } => match action {
            DictionaryAction::Load { path } => {
                let (ok, count, message) = client.dict_load(path)?;
                if cli.json {
                    print_json(&json!({
                        "ok": ok,
                        "count": count,
                        "message": message,
                    }));
                } else if ok {
                    print_text("count", count);
                } else {
                    print_text("error", message);
                }
            }
            DictionaryAction::Replace { words } => {
                let count = client.dict_replace(words)?;
                print(&[("count", json!(count))]);
            }
            DictionaryAction::Add { words } => {
                let (count, added) = client.dict_add(words)?;
                print(&[("count", json!(count)), ("added", json!(added))]);
            }
            DictionaryAction::Remove { words } => {
                let (count, removed) = client.dict_remove(words)?;
                print(&[("count", json!(count)), ("removed", json!(removed))]);
            }
            DictionaryAction::Get => {
                let (words, path, count, dirty) = client.dict_get()?;
                if cli.json {
                    print_json(&json!({
                        "words": words,
                        "path": path,
                        "count": count,
                        "dirty": dirty,
                    }));
                } else {
                    print_text("path", path);
                    print_text("count", count);
                    print_text("dirty", dirty);
                    for word in words {
                        println!("  {word}");
                    }
                }
            }
            DictionaryAction::Contains { word } => {
                print(&[("contains", json!(client.dict_contains(word)?))]);
            }
            DictionaryAction::Count => print(&[("count", json!(client.dict_count()?))]),
            DictionaryAction::Dirty => print(&[("dirty", json!(client.dict_dirty()?))]),
        },
        Command::Translation { action } => match action {
            TranslationAction::Enable => print(&[("enabled", json!(client.trans_enable()?))]),
            TranslationAction::Disable { unload } => {
                print(&[("enabled", json!(client.trans_disable(*unload)?))]);
            }
            TranslationAction::Toggle => print(&[("enabled", json!(client.trans_toggle()?))]),
            TranslationAction::SetTarget { language } => {
                let (ok, target, message) = client.trans_set_target(language)?;
                if cli.json {
                    print_json(&json!({
                        "ok": ok,
                        "target": target,
                        "message": message,
                    }));
                } else if ok {
                    print_text("target", target);
                } else {
                    print_text("error", message);
                }
            }
            TranslationAction::State => {
                let (enabled, target, requested, model) = client.trans_state()?;
                print(&[
                    ("enabled", json!(enabled)),
                    ("target", json!(target)),
                    ("requested", json!(requested)),
                    ("model", json!(model)),
                ]);
            }
            TranslationAction::Enabled => print(&[("enabled", json!(client.trans_enabled()?))]),
            TranslationAction::TargetLanguage => {
                print(&[("target", json!(client.trans_target_language()?))]);
            }
            TranslationAction::RequestedLanguage => {
                print(&[("requested", json!(client.trans_requested_language()?))]);
            }
            TranslationAction::Model => print(&[("model", json!(client.trans_model()?))]),
        },
        Command::Status => {
            let paused = client.is_paused()?;
            let (enabled, target, requested, model) = client.trans_state()?;
            if cli.json {
                print_json(&json!({
                    "paused": paused,
                    "translation": {
                        "enabled": enabled,
                        "target": target,
                        "requested": requested,
                        "model": model,
                    },
                }));
            } else {
                print_text("paused", paused);
                print_text("translation.enabled", enabled);
                print_text("translation.target", target);
                print_text("translation.requested", requested);
                print_text("translation.model", model);
            }
        }
        Command::Watch { signals } => {
            client.watch_signals(signals, cli.json)?;
        }
        Command::Completions { shell } => {
            let mut cmd = <Cli as clap::CommandFactory>::command();
            let name = cmd.get_name().to_string();
            let shell = match shell {
                Shell::Bash => clap_complete::Shell::Bash,
                Shell::Zsh => clap_complete::Shell::Zsh,
                Shell::Fish => clap_complete::Shell::Fish,
                Shell::PowerShell => clap_complete::Shell::PowerShell,
            };
            clap_complete::generate(shell, &mut cmd, name, &mut std::io::stdout());
        }
    }
    Ok(())
}
