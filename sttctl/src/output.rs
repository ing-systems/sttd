use serde_json::{Map, Value};

pub fn print_json<T: serde::Serialize>(value: &T) {
    println!("{}", serde_json::to_string_pretty(value).unwrap());
}

pub fn print_text(key: &str, value: impl std::fmt::Display) {
    println!("{key}: {value}");
}

/// Print `fields` as one JSON object (`json`) or as `key: value` lines, in
/// order. Text mode writes strings bare, the way `Display` would.
pub fn print_fields(json: bool, fields: &[(&str, Value)]) {
    if json {
        let object: Map<String, Value> = fields
            .iter()
            .map(|(key, value)| (key.to_string(), value.clone()))
            .collect();
        print_json(&object);
        return;
    }
    for (key, value) in fields {
        match value {
            Value::String(text) => print_text(key, text),
            other => print_text(key, other),
        }
    }
}
