use crate::cli::*;
use crate::dbus_client::SttdClient;
use crate::output::*;
use anyhow::Result;

pub fn execute(cli: &Cli, client: &SttdClient) -> Result<()> {
    match &cli.command {
        Command::Daemon { action } => match action {
            DaemonAction::Pause => {
                let paused = client.pause()?;
                if cli.json {
                    print_json_field("paused", paused);
                } else {
                    print_text_field("paused", paused);
                }
            }
            DaemonAction::Resume => {
                let paused = client.resume()?;
                if cli.json {
                    print_json_field("paused", paused);
                } else {
                    print_text_field("paused", paused);
                }
            }
            DaemonAction::Toggle => {
                let paused = client.toggle_pause()?;
                if cli.json {
                    print_json_field("paused", paused);
                } else {
                    print_text_field("paused", paused);
                }
            }
            DaemonAction::Paused => {
                let paused = client.is_paused()?;
                if cli.json {
                    print_json_field("paused", paused);
                } else {
                    print_text_field("paused", paused);
                }
            }
        },
        Command::Dictionary { action } => match action {
            DictionaryAction::Load { path } => {
                let (ok, count, message) = client.dict_load(path)?;
                if cli.json {
                    print_json(&serde_json::json!({
                        "ok": ok,
                        "count": count,
                        "message": message,
                    }));
                } else {
                    if ok {
                        print_text_field("count", count);
                    } else {
                        print_text_field("error", message);
                    }
                }
            }
            DictionaryAction::Replace { words } => {
                let count = client.dict_replace(words.clone())?;
                if cli.json {
                    print_json_field("count", count);
                } else {
                    print_text_field("count", count);
                }
            }
            DictionaryAction::Add { words } => {
                let (count, added) = client.dict_add(words.clone())?;
                if cli.json {
                    print_json(&serde_json::json!({
                        "count": count,
                        "added": added,
                    }));
                } else {
                    print_text_field("count", count);
                    print_text_field("added", added);
                }
            }
            DictionaryAction::Remove { words } => {
                let (count, removed) = client.dict_remove(words.clone())?;
                if cli.json {
                    print_json(&serde_json::json!({
                        "count": count,
                        "removed": removed,
                    }));
                } else {
                    print_text_field("count", count);
                    print_text_field("removed", removed);
                }
            }
            DictionaryAction::Get => {
                let (words, path, count, dirty) = client.dict_get()?;
                if cli.json {
                    print_json(&serde_json::json!({
                        "words": words,
                        "path": path,
                        "count": count,
                        "dirty": dirty,
                    }));
                } else {
                    print_text_field("path", path);
                    print_text_field("count", count);
                    print_text_field("dirty", dirty);
                    for word in words {
                        println!("  {word}");
                    }
                }
            }
            DictionaryAction::Contains { word } => {
                let contains = client.dict_contains(word)?;
                if cli.json {
                    print_json_field("contains", contains);
                } else {
                    print_text_field("contains", contains);
                }
            }
            DictionaryAction::Count => {
                let count = client.dict_count()?;
                if cli.json {
                    print_json_field("count", count);
                } else {
                    print_text_field("count", count);
                }
            }
            DictionaryAction::Dirty => {
                let dirty = client.dict_dirty()?;
                if cli.json {
                    print_json_field("dirty", dirty);
                } else {
                    print_text_field("dirty", dirty);
                }
            }
        },
        Command::Translation { action } => match action {
            TranslationAction::Enable => {
                let enabled = client.trans_enable()?;
                if cli.json {
                    print_json_field("enabled", enabled);
                } else {
                    print_text_field("enabled", enabled);
                }
            }
            TranslationAction::Disable { unload } => {
                let enabled = client.trans_disable(*unload)?;
                if cli.json {
                    print_json_field("enabled", enabled);
                } else {
                    print_text_field("enabled", enabled);
                }
            }
            TranslationAction::Toggle => {
                let enabled = client.trans_toggle()?;
                if cli.json {
                    print_json_field("enabled", enabled);
                } else {
                    print_text_field("enabled", enabled);
                }
            }
            TranslationAction::SetTarget { language } => {
                let (ok, target, message) = client.trans_set_target(language)?;
                if cli.json {
                    print_json(&serde_json::json!({
                        "ok": ok,
                        "target": target,
                        "message": message,
                    }));
                } else {
                    if ok {
                        print_text_field("target", target);
                    } else {
                        print_text_field("error", message);
                    }
                }
            }
            TranslationAction::State => {
                let (enabled, target, requested, model) = client.trans_state()?;
                if cli.json {
                    print_json(&serde_json::json!({
                        "enabled": enabled,
                        "target": target,
                        "requested": requested,
                        "model": model,
                    }));
                } else {
                    print_text_field("enabled", enabled);
                    print_text_field("target", target);
                    print_text_field("requested", requested);
                    print_text_field("model", model);
                }
            }
            TranslationAction::Enabled => {
                let enabled = client.trans_enabled()?;
                if cli.json {
                    print_json_field("enabled", enabled);
                } else {
                    print_text_field("enabled", enabled);
                }
            }
            TranslationAction::TargetLanguage => {
                let target = client.trans_target_language()?;
                if cli.json {
                    print_json_field("target", target);
                } else {
                    print_text_field("target", target);
                }
            }
            TranslationAction::RequestedLanguage => {
                let requested = client.trans_requested_language()?;
                if cli.json {
                    print_json_field("requested", requested);
                } else {
                    print_text_field("requested", requested);
                }
            }
            TranslationAction::Model => {
                let model = client.trans_model()?;
                if cli.json {
                    print_json_field("model", model);
                } else {
                    print_text_field("model", model);
                }
            }
        },
        Command::Status => {
            let paused = client.is_paused()?;
            let (enabled, target, requested, model) = client.trans_state()?;
            if cli.json {
                print_json(&serde_json::json!({
                    "paused": paused,
                    "translation": {
                        "enabled": enabled,
                        "target": target,
                        "requested": requested,
                        "model": model,
                    },
                }));
            } else {
                print_text_field("paused", paused);
                print_text_field("translation.enabled", enabled);
                print_text_field("translation.target", target);
                print_text_field("translation.requested", requested);
                print_text_field("translation.model", model);
            }
        }
        Command::Watch { signals } => {
            client.watch_signals(signals, cli.json)?;
        }
        Command::Completions { shell } => {
            let mut cmd = <Cli as clap::CommandFactory>::command();
            let name = cmd.get_name().to_string();
            match shell {
                Shell::Bash => clap_complete::generate(
                    clap_complete::Shell::Bash,
                    &mut cmd,
                    name,
                    &mut std::io::stdout(),
                ),
                Shell::Zsh => clap_complete::generate(
                    clap_complete::Shell::Zsh,
                    &mut cmd,
                    name,
                    &mut std::io::stdout(),
                ),
                Shell::Fish => clap_complete::generate(
                    clap_complete::Shell::Fish,
                    &mut cmd,
                    name,
                    &mut std::io::stdout(),
                ),
                Shell::PowerShell => clap_complete::generate(
                    clap_complete::Shell::PowerShell,
                    &mut cmd,
                    name,
                    &mut std::io::stdout(),
                ),
            }
        }
    }
    Ok(())
}
