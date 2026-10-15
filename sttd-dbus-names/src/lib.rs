//! The session-bus names `sttd` serves and `sttctl` calls, in one place so the
//! two binaries cannot drift apart.
//!
//! One well-known name carries three object paths, one interface each. The
//! `#[zbus::interface(name = ...)]` attributes in `sttd`'s `dbus` module spell
//! the interface names as literals (an attribute cannot take a constant); a
//! test there pins them to these.

/// Well-known bus name. All three interfaces are served on it.
pub const SERVICE_NAME: &str = "org.sttd.Daemon";

/// Pause control.
pub const DAEMON_PATH: &str = "/org/sttd/Daemon";
pub const DAEMON_INTERFACE: &str = "org.sttd.Daemon";

/// Live dictionary control.
pub const DICT_PATH: &str = "/org/sttd/Dictionary";
pub const DICT_INTERFACE: &str = "org.sttd.Dictionary";

/// Translation control.
pub const TRANSLATION_PATH: &str = "/org/sttd/Translation";
pub const TRANSLATION_INTERFACE: &str = "org.sttd.Translation";
