//! `vision-snapshot healthcheck`: the command behind the Docker `HEALTHCHECK`.
//!
//! Checks the web app and the model server, prints one line per run, and exits 0 if both are
//! healthy and 1 otherwise. Docker keeps the printed line (`docker inspect` shows the last few).
//! The line is also written to the container's main log, so every check shows up in `docker logs`:
//!
//! * `HEALTH_LOG=all` logs every check (default)
//! * `HEALTH_LOG=changes` logs only when the result changes
//! * `HEALTH_LOG=off` doesn't log, it only reports to Docker
//!
//! `HEALTH_LOG_TARGET` (default `/proc/1/fd/1`) is where the line is written.

use std::{
    env, fs,
    io::Write,
    time::{Duration, Instant},
};

const STATE_FILE: &str = "/tmp/healthcheck.last";

struct Probe {
    name: &'static str,
    ok: bool,
    detail: String,
}

async fn probe(client: &reqwest::Client, name: &'static str, url: String) -> Probe {
    let started = Instant::now();
    let what = match client
        .get(&url)
        .timeout(Duration::from_secs(3))
        .send()
        .await
    {
        Ok(response) => {
            let status = response.status();
            (status.as_u16() == 200, format!("HTTP {}", status.as_u16()))
        }
        Err(error) if error.is_timeout() => (false, "timed out".to_owned()),
        Err(error) if error.is_connect() => (false, "cannot connect".to_owned()),
        Err(_) => (false, "request failed".to_owned()),
    };
    Probe {
        name,
        ok: what.0,
        detail: format!("{}, {} ms", what.1, started.elapsed().as_millis()),
    }
}

fn summary(probes: &[Probe], healthy: bool) -> String {
    let parts = probes
        .iter()
        .map(|p| {
            format!(
                "{}={} ({})",
                p.name,
                if p.ok { "ok" } else { "FAIL" },
                p.detail
            )
        })
        .collect::<Vec<_>>()
        .join(" ");
    format!(
        "healthcheck: {parts} -> {}",
        if healthy { "healthy" } else { "UNHEALTHY" }
    )
}

/// Whether this run should be written to the container log.
fn should_log(mode: &str, state: &str, previous: &str) -> bool {
    match mode {
        "off" => false,
        "changes" => state != previous,
        _ => true,
    }
}

/// The model server's health URL as the app sees it. Docker starts health checks outside the
/// launcher script, so work it out from `LLAMA_PORT` when `LLM_HEALTH_URL` isn't set.
fn model_health_url() -> String {
    env::var("LLM_HEALTH_URL").unwrap_or_else(|_| {
        let port = env::var("LLAMA_PORT").unwrap_or_else(|_| "8080".to_owned());
        format!("http://127.0.0.1:{port}/health")
    })
}

pub async fn run() -> i32 {
    let port = env::var("PORT").unwrap_or_else(|_| "10000".to_owned());
    let Ok(client) = reqwest::Client::builder().build() else {
        println!("healthcheck: could not create an HTTP client -> UNHEALTHY");
        return 1;
    };
    // Both probes at once, so the whole check stays inside Docker's timeout.
    let (app, model) = tokio::join!(
        probe(&client, "app", format!("http://127.0.0.1:{port}/healthz")),
        probe(&client, "model", model_health_url()),
    );
    let probes = [app, model];
    let healthy = probes.iter().all(|p| p.ok);
    let line = format!(
        "{}  {}",
        crate::logging::timestamp(),
        summary(&probes, healthy)
    );
    println!("{line}");

    let mode = env::var("HEALTH_LOG")
        .unwrap_or_else(|_| "all".to_owned())
        .to_lowercase();
    if mode != "off" {
        let state = if healthy { "healthy" } else { "unhealthy" };
        let previous = fs::read_to_string(STATE_FILE).unwrap_or_default();
        let _ = fs::write(STATE_FILE, state);
        if should_log(&mode, state, &previous) {
            let target =
                env::var("HEALTH_LOG_TARGET").unwrap_or_else(|_| "/proc/1/fd/1".to_owned());
            // Not allowed to write there? The line printed above still reaches Docker.
            if let Ok(mut out) = fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(target)
            {
                let _ = writeln!(out, "{line}");
            }
        }
    }
    if healthy {
        0
    } else {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn probe_result(name: &'static str, ok: bool, detail: &str) -> Probe {
        Probe {
            name,
            ok,
            detail: detail.to_owned(),
        }
    }

    #[test]
    fn summary_names_each_check_and_the_verdict() {
        let probes = [
            probe_result("app", true, "HTTP 200, 3 ms"),
            probe_result("model", false, "cannot connect, 1 ms"),
        ];
        assert_eq!(
            summary(&probes, false),
            "healthcheck: app=ok (HTTP 200, 3 ms) model=FAIL (cannot connect, 1 ms) -> UNHEALTHY"
        );
    }

    #[test]
    fn log_modes_behave_like_the_python_healthcheck() {
        assert!(should_log("all", "healthy", "healthy"));
        assert!(!should_log("off", "unhealthy", ""));
        assert!(should_log("changes", "unhealthy", "healthy"));
        assert!(!should_log("changes", "healthy", "healthy"));
    }
}
