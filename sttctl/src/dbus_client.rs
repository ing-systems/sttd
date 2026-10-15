use crate::output::print_json;
use anyhow::{Context, Result};
use std::time::Duration;
use zbus::blocking::connection::Builder;
use zbus::blocking::{Connection, MessageIterator};

use sttd_dbus_names::{
    DAEMON_INTERFACE, DAEMON_PATH, DICT_INTERFACE, DICT_PATH, OBJECT_ROOT, SERVICE_NAME,
    TRANSLATION_INTERFACE, TRANSLATION_PATH,
};

/// One of the daemon's objects: its path and the interface served there.
type Object = (&'static str, &'static str);

const DAEMON: Object = (DAEMON_PATH, DAEMON_INTERFACE);
const DICT: Object = (DICT_PATH, DICT_INTERFACE);
const TRANSLATION: Object = (TRANSLATION_PATH, TRANSLATION_INTERFACE);

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
        self.call(DAEMON, "Pause", &())
    }

    pub fn resume(&self) -> Result<bool> {
        self.call(DAEMON, "Resume", &())
    }

    pub fn toggle_pause(&self) -> Result<bool> {
        self.call(DAEMON, "Toggle", &())
    }

    pub fn is_paused(&self) -> Result<bool> {
        self.property(DAEMON, "Paused")
    }

    // Dictionary methods
    pub fn dict_load(&self, path: &str) -> Result<(bool, u32, String)> {
        self.call(DICT, "LoadFile", &(path,))
    }

    pub fn dict_replace(&self, words: &[String]) -> Result<u32> {
        self.call(DICT, "ReplaceWords", &(words,))
    }

    pub fn dict_add(&self, words: &[String]) -> Result<(u32, u32)> {
        self.call(DICT, "AddWords", &(words,))
    }

    pub fn dict_remove(&self, words: &[String]) -> Result<(u32, u32)> {
        self.call(DICT, "RemoveWords", &(words,))
    }

    pub fn dict_get(&self) -> Result<(Vec<String>, String, u32, bool)> {
        self.call(DICT, "GetWords", &())
    }

    pub fn dict_contains(&self, word: &str) -> Result<bool> {
        self.call(DICT, "Contains", &(word,))
    }

    pub fn dict_count(&self) -> Result<u32> {
        self.property(DICT, "Count")
    }

    pub fn dict_dirty(&self) -> Result<bool> {
        self.property(DICT, "Dirty")
    }

    // Translation methods
    pub fn trans_enable(&self) -> Result<bool> {
        self.call(TRANSLATION, "Enable", &())
    }

    pub fn trans_disable(&self, unload: bool) -> Result<bool> {
        self.call(TRANSLATION, "Disable", &(unload,))
    }

    pub fn trans_toggle(&self) -> Result<bool> {
        self.call(TRANSLATION, "Toggle", &())
    }

    pub fn trans_set_target(&self, lang: &str) -> Result<(bool, String, String)> {
        self.call(TRANSLATION, "SetTargetLanguage", &(lang,))
    }

    pub fn trans_state(&self) -> Result<(bool, String, String, String)> {
        self.call(TRANSLATION, "GetState", &())
    }

    pub fn trans_enabled(&self) -> Result<bool> {
        self.property(TRANSLATION, "Enabled")
    }

    pub fn trans_target_language(&self) -> Result<String> {
        self.property(TRANSLATION, "TargetLanguage")
    }

    pub fn trans_requested_language(&self) -> Result<String> {
        self.property(TRANSLATION, "RequestedLanguage")
    }

    pub fn trans_model(&self) -> Result<String> {
        self.property(TRANSLATION, "Model")
    }

    // Watch signals
    pub fn watch_signals(&self, filter: &[String], json: bool) -> Result<()> {
        // Every object the daemon serves lives under one root, so one rule
        // covers the signals of all three interfaces.
        let rule = zbus::MatchRule::builder()
            .msg_type(zbus::message::Type::Signal)
            .path_namespace(OBJECT_ROOT)?
            .build();
        for msg in MessageIterator::for_match_rule(rule, &self.conn, None)? {
            let msg = msg?;
            let header = msg.header();
            let interface = header.interface().map_or("", |v| v).to_string();
            let member = header.member().map_or("", |v| v).to_string();
            if !filter.is_empty() && !filter.contains(&member) {
                continue;
            }
            match signal_args(&msg) {
                Ok(args) => {
                    if json {
                        print_json(&serde_json::json!({
                            "interface": interface,
                            "member": member,
                            "args": args,
                        }));
                    } else {
                        let args: Vec<String> = args.iter().map(ToString::to_string).collect();
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
        Ok(())
    }

    // Proxy helpers
    fn proxy(&self, (path, interface): Object) -> Result<zbus::blocking::Proxy<'_>> {
        zbus::blocking::Proxy::new(&self.conn, SERVICE_NAME, path, interface).map_err(Into::into)
    }

    /// Call `method` on one of the daemon's objects and read its reply.
    fn call<B, R>(&self, object: Object, method: &str, body: &B) -> Result<R>
    where
        B: serde::Serialize + zbus::zvariant::DynamicType,
        R: for<'d> zbus::zvariant::DynamicDeserialize<'d>,
    {
        Ok(self.proxy(object)?.call(method, body)?)
    }

    /// Read one property of one of the daemon's objects.
    fn property<T>(&self, object: Object, name: &str) -> Result<T>
    where
        T: TryFrom<zbus::zvariant::OwnedValue>,
        T::Error: Into<zbus::Error>,
    {
        Ok(self.proxy(object)?.get_property(name)?)
    }
}

/// A signal's arguments as JSON values, whatever its signature: the daemon's
/// signals carry `b`, `ubs` and `bssbs` bodies.
fn signal_args(msg: &zbus::Message) -> Result<Vec<serde_json::Value>> {
    let body = msg.body();
    if body.signature().to_string().is_empty() {
        return Ok(Vec::new());
    }
    let fields: zbus::zvariant::Structure = body.deserialize()?;
    Ok(fields.fields().iter().map(to_json).collect())
}

/// One D-Bus value as plain JSON: numbers, booleans and strings as themselves,
/// arrays and structures as arrays. Anything else (dicts, fds) is spelled the
/// way zvariant displays it.
fn to_json(value: &zbus::zvariant::Value<'_>) -> serde_json::Value {
    use zbus::zvariant::Value;
    match value {
        Value::Bool(b) => (*b).into(),
        Value::U8(n) => (*n).into(),
        Value::I16(n) => (*n).into(),
        Value::U16(n) => (*n).into(),
        Value::I32(n) => (*n).into(),
        Value::U32(n) => (*n).into(),
        Value::I64(n) => (*n).into(),
        Value::U64(n) => (*n).into(),
        Value::F64(n) => (*n).into(),
        Value::Str(s) => s.as_str().into(),
        Value::ObjectPath(p) => p.as_str().into(),
        Value::Signature(s) => s.to_string().into(),
        Value::Value(inner) => to_json(inner),
        Value::Array(items) => items.iter().map(to_json).collect(),
        Value::Structure(fields) => fields.fields().iter().map(to_json).collect(),
        other => other.to_string().into(),
    }
}
