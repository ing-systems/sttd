mod cli;
mod commands;
mod dbus_client;
mod output;

use anyhow::{Context, Result};
use clap::Parser;
use std::time::Duration;

use cli::Cli;
use dbus_client::SttdClient;

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let cli = Cli::parse();

    let timeout = Duration::from_secs(cli.timeout.unwrap_or(30));
    let client = SttdClient::new(timeout).context("Failed to connect to sttd daemon")?;

    commands::execute(&cli, &client)
}
