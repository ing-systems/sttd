//! The session-bus names `sttd` serves and `sttctl` calls, in one place so the
//! two binaries cannot drift apart.
//!
//! One well-known name carries three object paths, one interface each. The
//! `#[zbus::interface(name = ...)]` attributes in `sttd`'s `dbus` module spell
//! the interface names as literals (an attribute cannot take a constant); a
//! test there pins them to these.

/// Well-known bus name. All three interfaces are served on it.
pub const SERVICE_NAME: &str = "org.sttd.Daemon";

/// The path namespace every object below lives in (what a client matches to
/// hear all of the daemon's signals).
pub const OBJECT_ROOT: &str = "/org/sttd";

/// Pause control.
pub const DAEMON_PATH: &str = "/org/sttd/Daemon";
pub const DAEMON_INTERFACE: &str = "org.sttd.Daemon";

/// Live dictionary control.
pub const DICT_PATH: &str = "/org/sttd/Dictionary";
pub const DICT_INTERFACE: &str = "org.sttd.Dictionary";

/// Translation control.
pub const TRANSLATION_PATH: &str = "/org/sttd/Translation";
pub const TRANSLATION_INTERFACE: &str = "org.sttd.Translation";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_object_lives_under_the_root() {
        for path in [DAEMON_PATH, DICT_PATH, TRANSLATION_PATH] {
            assert!(path.starts_with(&format!("{OBJECT_ROOT}/")), "{path}");
        }
    }
}
