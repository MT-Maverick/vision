use std::{
    env,
    fs::OpenOptions,
    io::Write,
    time::{Duration, Instant},
};

use chrono::Local;
use reqwest::Client;

struct Check {
    name: &'static str,
    url: String,
}

async fn probe(client: &Client, url: &str) -> (bool, String) {
    let started = Instant::now();
    let (ok, detail) = match client.get(url).timeout(Duration::from_secs(3)).send().await {
        Ok(response) => {
            let status = response.status();
            (status.as_u16() == 200, format!("HTTP {status}"))
        }
        Err(error) if error.is_timeout() => (false, "timeout".to_owned()),
        Err(error) if error.is_connect() => (false, "connection refused".to_owned()),
        Err(error) => (false, error.to_string()),
    };
    (
        ok,
        format!(
            "{detail}, {:.0} ms",
            started.elapsed().as_secs_f64() * 1000.0
        ),
    )
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let port = env::var("PORT").unwrap_or_else(|_| "10000".to_owned());
    let checks = [
        Check {
            name: "app",
            url: format!("http://127.0.0.1:{port}/healthz"),
        },
        Check {
            name: "model",
            url: env::var("LLM_HEALTH_URL")
                .unwrap_or_else(|_| "http://127.0.0.1:8080/health".to_owned()),
        },
    ];
    let client = Client::builder().build()?;
    let mut results = Vec::with_capacity(checks.len());
    for check in &checks {
        let (ok, detail) = probe(&client, &check.url).await;
        results.push((check.name, ok, detail));
    }
    let healthy = results.iter().all(|(_, ok, _)| *ok);
    let parts = results
        .iter()
        .map(|(name, ok, detail)| format!("{name}={} ({detail})", if *ok { "ok" } else { "FAIL" }))
        .collect::<Vec<_>>()
        .join(" ");
    let state = if healthy { "healthy" } else { "unhealthy" };
    let line = format!(
        "{}  healthcheck: {parts} -> {}",
        Local::now().format("%Y-%m-%d %H:%M:%S%.3f"),
        if healthy { "healthy" } else { "UNHEALTHY" }
    );
    println!("{line}");

    let mode = env::var("HEALTH_LOG")
        .unwrap_or_else(|_| "all".to_owned())
        .to_lowercase();
    if mode != "off" {
        let state_file = "/tmp/healthcheck.last";
        let previous = match tokio::fs::read_to_string(state_file).await {
            Ok(value) => value,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => String::new(),
            Err(error) => {
                eprintln!("Could not read health state file: {error}");
                String::new()
            }
        };
        if let Err(error) = tokio::fs::write(state_file, state).await {
            eprintln!("Could not update health state file: {error}");
        }
        if mode == "all" || state != previous {
            let target =
                env::var("HEALTH_LOG_TARGET").unwrap_or_else(|_| "/proc/1/fd/1".to_owned());
            match OpenOptions::new().append(true).open(&target) {
                Ok(mut file) => {
                    if let Err(error) = writeln!(file, "{line}") {
                        eprintln!("Could not write health log: {error}");
                    }
                }
                Err(error) => eprintln!("Could not open health log target {target}: {error}"),
            }
        }
    }
    if healthy {
        Ok(())
    } else {
        std::process::exit(1);
    }
}
