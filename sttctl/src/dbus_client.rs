use crate::output::print_json;
use anyhow::{Context, Result};
use std::time::Duration;
use zbus::blocking::connection::Builder;
use zbus::blocking::Connection;

use sttd_dbus_names::{
    DAEMON_INTERFACE, DAEMON_PATH, DICT_INTERFACE, DICT_PATH, SERVICE_NAME, TRANSLATION_INTERFACE,
    TRANSLATION_PATH,
};

pub struct SttdClient {
    conn: Connection,
}

impl SttdClient {
    pub fn new(timeout: Duration) -> Result<Self> {
        let conn = Builder::session()
            .context("Failed to create session bus builder")?
            .method_timeout(timeout)
            .build()
            .context("Failed to build connection")?;
        Ok(Self { conn })
    }

    // Daemon methods
    pub fn pause(&self) -> Result<bool> {
        let proxy = self.daemon_proxy()?;
        let paused: bool = proxy.call("Pause", &())?;
        Ok(paused)
    }

    pub fn resume(&self) -> Result<bool> {
        let proxy = self.daemon_proxy()?;
        let paused: bool = proxy.call("Resume", &())?;
        Ok(paused)
    }

    pub fn toggle_pause(&self) -> Result<bool> {
        let proxy = self.daemon_proxy()?;
        let paused: bool = proxy.call("Toggle", &())?;
        Ok(paused)
    }

    pub fn is_paused(&self) -> Result<bool> {
        let proxy = self.daemon_proxy()?;
        let paused: bool = proxy.get_property("Paused")?;
        Ok(paused)
    }

    // Dictionary methods
    pub fn dict_load(&self, path: &str) -> Result<(bool, u32, String)> {
        let proxy = self.dict_proxy()?;
        let result: (bool, u32, String) = proxy.call("LoadFile", &(path,))?;
        Ok(result)
    }

    pub fn dict_replace(&self, words: Vec<String>) -> Result<u32> {
        let proxy = self.dict_proxy()?;
        let count: u32 = proxy.call("ReplaceWords", &(words,))?;
        Ok(count)
    }

    pub fn dict_add(&self, words: Vec<String>) -> Result<(u32, u32)> {
        let proxy = self.dict_proxy()?;
        let result: (u32, u32) = proxy.call("AddWords", &(words,))?;
        Ok(result)
    }

    pub fn dict_remove(&self, words: Vec<String>) -> Result<(u32, u32)> {
        let proxy = self.dict_proxy()?;
        let result: (u32, u32) = proxy.call("RemoveWords", &(words,))?;
        Ok(result)
    }

    pub fn dict_get(&self) -> Result<(Vec<String>, String, u32, bool)> {
        let proxy = self.dict_proxy()?;
        let result: (Vec<String>, String, u32, bool) = proxy.call("GetWords", &())?;
        Ok(result)
    }

    pub fn dict_contains(&self, word: &str) -> Result<bool> {
        let proxy = self.dict_proxy()?;
        let contains: bool = proxy.call("Contains", &(word,))?;
        Ok(contains)
    }

    pub fn dict_count(&self) -> Result<u32> {
        let proxy = self.dict_proxy()?;
        let count: u32 = proxy.get_property("Count")?;
        Ok(count)
    }

    pub fn dict_dirty(&self) -> Result<bool> {
        let proxy = self.dict_proxy()?;
        let dirty: bool = proxy.get_property("Dirty")?;
        Ok(dirty)
    }

    // Translation methods
    pub fn trans_enable(&self) -> Result<bool> {
        let proxy = self.trans_proxy()?;
        let enabled: bool = proxy.call("Enable", &())?;
        Ok(enabled)
    }

    pub fn trans_disable(&self, unload: bool) -> Result<bool> {
        let proxy = self.trans_proxy()?;
        let enabled: bool = proxy.call("Disable", &(unload,))?;
        Ok(enabled)
    }

    pub fn trans_toggle(&self) -> Result<bool> {
        let proxy = self.trans_proxy()?;
        let enabled: bool = proxy.call("Toggle", &())?;
        Ok(enabled)
    }

    pub fn trans_set_target(&self, lang: &str) -> Result<(bool, String, String)> {
        let proxy = self.trans_proxy()?;
        let result: (bool, String, String) = proxy.call("SetTargetLanguage", &(lang,))?;
        Ok(result)
    }

    pub fn trans_state(&self) -> Result<(bool, String, String, String)> {
        let proxy = self.trans_proxy()?;
        let result: (bool, String, String, String) = proxy.call("GetState", &())?;
        Ok(result)
    }

    pub fn trans_enabled(&self) -> Result<bool> {
        let proxy = self.trans_proxy()?;
        let enabled: bool = proxy.get_property("Enabled")?;
        Ok(enabled)
    }

    pub fn trans_target_language(&self) -> Result<String> {
        let proxy = self.trans_proxy()?;
        let target: String = proxy.get_property("TargetLanguage")?;
        Ok(target)
    }

    pub fn trans_requested_language(&self) -> Result<String> {
        let proxy = self.trans_proxy()?;
        let requested: String = proxy.get_property("RequestedLanguage")?;
        Ok(requested)
    }

    pub fn trans_model(&self) -> Result<String> {
        let proxy = self.trans_proxy()?;
        let model: String = proxy.get_property("Model")?;
        Ok(model)
    }

    // Watch signals
    pub fn watch_signals(&self, filter: &[String], json: bool) -> Result<()> {
        let proxy = self.daemon_proxy()?;
        let iter = proxy.receive_all_signals()?;

        for msg in iter {
            let header = msg.header();
            let interface = header.interface().map_or("", |v| v).to_string();
            let member = header.member().map_or("", |v| v).to_string();

            if filter.is_empty() || filter.contains(&member) {
                let body = msg.body();
                match body.deserialize::<Vec<String>>() {
                    Ok(args) => {
                        if json {
                            print_json(&serde_json::json!({
                                "interface": interface,
                                "member": member,
                                "args": args,
                            }));
                        } else {
                            println!(
                                "[{}] {}.{}({})",
                                chrono::Local::now().format("%Y-%m-%dT%H:%M:%S"),
                                interface,
                                member,
                                args.join(", ")
                            );
                        }
                    }
                    Err(e) => {
                        if json {
                            print_json(&serde_json::json!({
                                "interface": interface,
                                "member": member,
                                "error": format!("{e}"),
                            }));
                        } else {
                            println!(
                                "[{}] {}.{} (deserialization failed: {e})",
                                chrono::Local::now().format("%Y-%m-%dT%H:%M:%S"),
                                interface,
                                member,
                            );
                        }
                    }
                }
            }
        }
        Ok(())
    }

    // Proxy helpers
    fn daemon_proxy(&self) -> Result<zbus::blocking::Proxy<'_>> {
        zbus::blocking::Proxy::new(&self.conn, SERVICE_NAME, DAEMON_PATH, DAEMON_INTERFACE)
            .map_err(Into::into)
    }

    fn dict_proxy(&self) -> Result<zbus::blocking::Proxy<'_>> {
        zbus::blocking::Proxy::new(&self.conn, SERVICE_NAME, DICT_PATH, DICT_INTERFACE)
            .map_err(Into::into)
    }

    fn trans_proxy(&self) -> Result<zbus::blocking::Proxy<'_>> {
        zbus::blocking::Proxy::new(
            &self.conn,
            SERVICE_NAME,
            TRANSLATION_PATH,
            TRANSLATION_INTERFACE,
        )
        .map_err(Into::into)
    }
}
