use serde_json::json;

pub fn print_json<T: serde::Serialize>(value: &T) {
    println!("{}", serde_json::to_string_pretty(value).unwrap());
}

pub fn print_text(key: &str, value: impl std::fmt::Display) {
    println!("{key}: {value}");
}

pub fn print_json_field(key: &str, value: impl serde::Serialize) {
    print_json(&json!({ key: value }));
}

pub fn print_text_field(key: &str, value: impl std::fmt::Display) {
    print_text(key, value);
}
