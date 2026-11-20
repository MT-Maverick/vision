//! Log lines in the same style the Python version produced: `2026-11-18 12:00:00,123  message`.
//!
//! Everything goes to stderr (like Python's `logging`), so it appears in `docker logs` next to
//! llama-server's own output and the launcher's lines. Warnings carry a `WARNING:` marker.
//! Declared with `#[macro_use]` in `main.rs` so `info!` and `warn!` work in every module.

use std::{fmt, io::Write};

/// The current local time, formatted like Python's default `asctime`.
pub fn timestamp() -> String {
    chrono::Local::now()
        .format("%Y-%m-%d %H:%M:%S,%3f")
        .to_string()
}

pub fn emit(args: fmt::Arguments<'_>) {
    let _ = writeln!(std::io::stderr().lock(), "{}  {args}", timestamp());
}

macro_rules! info {
    ($($arg:tt)*) => { $crate::logging::emit(format_args!($($arg)*)) };
}

macro_rules! warn {
    ($($arg:tt)*) => { $crate::logging::emit(format_args!("WARNING: {}", format_args!($($arg)*))) };
}
