#[macro_use]
mod logging;
mod healthcheck;

use std::{
    collections::HashMap,
    env,
    error::Error,
    net::{IpAddr, SocketAddr},
    path::PathBuf,
    str::FromStr,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        ConnectInfo, Request, State,
    },
    http::{header, HeaderMap, HeaderValue, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::get,
    Json, Router,
};
use base64::{engine::general_purpose::STANDARD, Engine};
use image::{codecs::jpeg::JpegEncoder, imageops::FilterType};
use regex::Regex;
use reqwest::{Client, StatusCode as ReqwestStatusCode};
use serde::Serialize;
use serde_json::{json, Value};
use tokio::sync::{Mutex, Semaphore};
use tower::ServiceBuilder;
use tower_http::{services::ServeDir, set_header::SetResponseHeaderLayer};

const MAX_FRAMES: usize = 6;
const MAX_FRAME_BYTES: usize = 4_000_000;
static NEXT_REQUEST_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Clone)]
struct Config {
    llm_base_url: String,
    health_url: String,
    model: String,
    max_dim: u32,
    timeout: Duration,
    max_concurrent: usize,
    /// Folder holding `index.html` and `static/`.
    web_dir: PathBuf,
}

/// Reads a numeric setting, naming the setting if it isn't a valid number.
fn env_number<T: FromStr>(name: &str, default: T) -> Result<T, String> {
    match env::var(name) {
        Ok(value) => value
            .trim()
            .parse()
            .map_err(|_| format!("{name} must be a whole number, but it is {value:?}")),
        Err(_) => Ok(default),
    }
}

impl Config {
    fn from_env() -> Result<Self, String> {
        let max_dim = env_number::<u32>("MAX_DIM", 512)?;
        let timeout = env_number::<u64>("INFERENCE_TIMEOUT", 300)?;
        let max_concurrent = env_number::<usize>("MAX_CONCURRENT", 1)?;
        if max_dim == 0 || timeout == 0 || max_concurrent == 0 {
            return Err("MAX_DIM, INFERENCE_TIMEOUT, and MAX_CONCURRENT must be positive".into());
        }

        Ok(Self {
            llm_base_url: env::var("LLM_BASE_URL")
                .unwrap_or_else(|_| "http://127.0.0.1:8080/v1".to_owned())
                .trim_end_matches('/')
                .to_owned(),
            health_url: env::var("LLM_HEALTH_URL")
                .unwrap_or_else(|_| "http://127.0.0.1:8080/health".to_owned()),
            model: env::var("LLM_MODEL").unwrap_or_else(|_| "SmolVLM-256M".to_owned()),
            max_dim,
            timeout: Duration::from_secs(timeout),
            max_concurrent,
            web_dir: PathBuf::from(env::var("WEB_DIR").unwrap_or_else(|_| "web".to_owned())),
        })
    }
}

#[derive(Clone)]
struct AppState {
    client: Client,
    config: Config,
    api_slots: Arc<Semaphore>,
    geo_cache: Arc<Mutex<HashMap<(i32, i32), String>>>,
}

#[derive(Serialize)]
struct Status {
    model: String,
    ready: bool,
}

#[tokio::main]
async fn main() {
    match env::args().nth(1).as_deref() {
        Some("healthcheck") => std::process::exit(healthcheck::run().await),
        Some("--version" | "-V") => {
            println!("vision-snapshot {}", env!("CARGO_PKG_VERSION"));
            return;
        }
        Some(other) => {
            eprintln!(
                "Unknown argument {other:?}. Usage: vision-snapshot [healthcheck | --version]"
            );
            std::process::exit(2);
        }
        None => {}
    }
    if let Err(error) = serve().await {
        warn!("{error}");
        std::process::exit(1);
    }
}

async fn serve() -> Result<(), Box<dyn Error>> {
    let config = Config::from_env()?;
    info!("Vision Snapshot {} starting", env!("CARGO_PKG_VERSION"));
    info!(
        "Settings: model='{}' llm_base_url={} health_url={} max_dim={} inference_timeout={}s max_concurrent={} web_dir={}",
        config.model,
        config.llm_base_url,
        config.health_url,
        config.max_dim,
        config.timeout.as_secs(),
        config.max_concurrent,
        config.web_dir.display()
    );
    let index_file = config.web_dir.join("index.html");
    if !index_file.is_file() {
        warn!(
            "{} not found, so the page can't be served. Set WEB_DIR to the folder that holds index.html and static/.",
            index_file.display()
        );
    }

    let static_dir = config.web_dir.join("static");
    let client = Client::builder().build()?;
    let state = AppState {
        client,
        api_slots: Arc::new(Semaphore::new(config.max_concurrent)),
        geo_cache: Arc::new(Mutex::new(HashMap::new())),
        config,
    };

    tokio::spawn(watch_startup(state.clone()));
    tokio::spawn(log_stop_signal());

    let static_service = ServiceBuilder::new()
        .layer(SetResponseHeaderLayer::overriding(
            header::CACHE_CONTROL,
            HeaderValue::from_static("no-cache"),
        ))
        .service(ServeDir::new(static_dir));
    let app = Router::new()
        .route("/", get(index))
        .route("/healthz", get(healthz))
        .route("/api/status", get(status))
        .route("/ws/vision", get(vision_websocket))
        .nest_service("/static", static_service)
        .layer(middleware::from_fn(access_log))
        .with_state(state);

    let port = env::var("PORT").unwrap_or_else(|_| "10000".to_owned());
    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{port}"))
        .await
        .map_err(|error| format!("Can't listen on port {port}: {error}"))?;
    info!("Vision Snapshot listening on port {port}");
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .await?;
    Ok(())
}

/// Logs the stop request from Docker or the launcher script, then exits right away. (A graceful
/// shutdown would wait for open camera connections, which can stay open indefinitely.)
async fn log_stop_signal() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{signal, SignalKind};
        if let Ok(mut terminate) = signal(SignalKind::terminate()) {
            tokio::select! {
                _ = terminate.recv() => {}
                _ = tokio::signal::ctrl_c() => {}
            }
        } else {
            let _ = tokio::signal::ctrl_c().await;
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
    info!("Stop signal received. Shutting down.");
    std::process::exit(0);
}

/// One line per HTTP request, like the uvicorn access log of the Python version.
async fn access_log(
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    request: Request,
    next: Next,
) -> Response {
    let started = Instant::now();
    let client = client_address(request.headers(), peer);
    let method = request.method().clone();
    let target = request.uri().path_and_query().map_or_else(
        || request.uri().path().to_owned(),
        |t| t.as_str().to_owned(),
    );
    let response = next.run(request).await;
    info!(
        "{client} - \"{method} {target}\" {} ({} ms)",
        response.status().as_u16(),
        started.elapsed().as_millis()
    );
    response
}

/// Behind a proxy (Render, a load balancer) every connection comes from the proxy, which names
/// the real client in `X-Forwarded-For`. Only used to label log lines, so it must parse as an
/// IP address to be shown.
fn client_address(headers: &HeaderMap, peer: SocketAddr) -> IpAddr {
    headers
        .get("x-forwarded-for")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.split(',').next())
        .and_then(|first| first.trim().parse().ok())
        .unwrap_or_else(|| peer.ip())
}

async fn index(State(state): State<AppState>) -> Response {
    let path = state.config.web_dir.join("index.html");
    match tokio::fs::read(&path).await {
        Ok(contents) => (
            [(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"))],
            [(
                header::CONTENT_TYPE,
                HeaderValue::from_static("text/html; charset=utf-8"),
            )],
            contents,
        )
            .into_response(),
        Err(error) => {
            warn!("Could not read {}: {error}", path.display());
            StatusCode::INTERNAL_SERVER_ERROR.into_response()
        }
    }
}

async fn healthz() -> Json<Value> {
    Json(json!({ "ok": true }))
}

async fn status(State(state): State<AppState>) -> Json<Status> {
    Json(Status {
        model: state.config.model.clone(),
        ready: probe_model(&state).await.0,
    })
}

async fn probe_model(state: &AppState) -> (bool, String) {
    match state
        .client
        .get(&state.config.health_url)
        .timeout(Duration::from_secs(2))
        .send()
        .await
    {
        Ok(response) => (
            response.status() == ReqwestStatusCode::OK,
            format!("health check answered HTTP {}", response.status()),
        ),
        Err(error) if error.is_connect() => (
            false,
            "nothing is listening on that port, so the model server isn't running".to_owned(),
        ),
        Err(error) => (false, error.to_string()),
    }
}

async fn watch_startup(state: AppState) {
    let mut reason = String::new();
    for attempt in 0..90 {
        let (ready, why) = probe_model(&state).await;
        reason = why;
        if ready {
            info!("Model '{}' is ready.", state.config.model);
            return;
        }
        if attempt % 10 == 0 {
            info!("Waiting for the model server: {reason}");
        }
        tokio::time::sleep(Duration::from_secs(2)).await;
    }
    warn!(
        "The model server did not become ready at {} ({reason}).",
        state.config.health_url
    );
}

async fn vision_websocket(
    State(state): State<AppState>,
    upgrade: WebSocketUpgrade,
) -> impl IntoResponse {
    upgrade
        .max_message_size(32 * 1024 * 1024)
        .on_upgrade(move |socket| handle_socket(socket, state))
}

async fn handle_socket(socket: WebSocket, state: AppState) {
    info!("Client connected");
    serve_socket(socket, state).await;
    info!("Client disconnected");
}

async fn serve_socket(mut socket: WebSocket, state: AppState) {
    while let Some(Ok(message)) = socket.recv().await {
        let Message::Text(text) = message else {
            continue;
        };
        let Ok(message) = serde_json::from_str::<Value>(&text) else {
            continue;
        };
        if message.get("type").and_then(Value::as_str) == Some("scout") {
            run_scout(&mut socket, &state, &message).await;
            continue;
        }

        let request_id = NEXT_REQUEST_ID.fetch_add(1, Ordering::Relaxed);
        let request_started = Instant::now();
        let id = message.get("id").cloned().unwrap_or(Value::Null);
        let mut images = message
            .get("images")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        if images.is_empty() {
            if let Some(image) = message.get("image") {
                images.push(image.clone());
            }
        }
        let frames = prepare_frames(&images, state.config.max_dim);
        if frames.is_empty() {
            warn!("request={request_id} rejected reason=no_usable_image");
            send_reply(&mut socket, &id, "error", "No usable image received", None).await;
            continue;
        }

        let requested_mode = message.get("mode").and_then(Value::as_str).unwrap_or("");
        let mode = if is_valid_mode(requested_mode) {
            requested_mode
        } else {
            "identify"
        };
        let question = clean_text(message.get("question"), 300);
        let skills: Vec<String> = message
            .get("skills")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .filter(|skill| matches!(*skill, "explain" | "ocr" | "translate" | "web_search"))
            .map(ToOwned::to_owned)
            .collect();
        let photo_context = message
            .get("photo_context")
            .and_then(Value::as_str)
            .map(|value| value.chars().take(300).collect::<String>());
        if mode == "ask" && question.is_empty() {
            warn!("request={request_id} rejected mode=ask reason=empty_question");
            send_reply(&mut socket, &id, "error", "Type a question first", None).await;
            continue;
        }

        info!(
            "request={request_id} started mode={mode} frames={} skills={skills:?} model={} user_question={question:?} selected_photo_summary={photo_context:?}",
            frames.len(),
            state.config.model
        );
        let coords = parse_location(message.get("location"));
        let typed_place = clean_text(message.get("place"), 80);
        let place = if !typed_place.is_empty() {
            Some(typed_place)
        } else if let Some((latitude, longitude)) = coords {
            Some(reverse_geocode(&state, latitude, longitude).await)
        } else {
            None
        };

        if !send_reply(&mut socket, &id, "working", "", None).await {
            return;
        }
        info!(
            "{mode}: {} frame(s){}",
            frames.len(),
            place
                .as_deref()
                .map(|name| format!(" near {name}"))
                .unwrap_or_default()
        );

        let result = match state.api_slots.clone().acquire_owned().await {
            Ok(_permit) => {
                let prompt = build_prompt(
                    mode,
                    frames.len(),
                    place.as_deref(),
                    &question,
                    &skills,
                    photo_context.as_deref(),
                );
                ask_model(
                    &state, &prompt, &frames, mode, request_id, "vision", &skills,
                )
                .await
            }
            Err(error) => Err(format!("Inference is unavailable: {error}")),
        };
        match result {
            Ok(label) => {
                info!("request={request_id} result: {label}");
                info!(
                    "request={request_id} completed status=success elapsed_ms={}",
                    request_started.elapsed().as_millis(),
                );
                let extra = json!({
                    "location_name": place,
                    "coords": coords.map(|(latitude, longitude)| {
                        json!({ "latitude": latitude, "longitude": longitude })
                    }),
                });
                if !send_reply(&mut socket, &id, "success", &label, Some(extra)).await {
                    return;
                }
            }
            Err(error) => {
                warn!(
                    "request={request_id} completed status=error elapsed_ms={} detail={error}",
                    request_started.elapsed().as_millis()
                );
                if !send_reply(&mut socket, &id, "error", &error, None).await {
                    return;
                }
            }
        }
    }
}

async fn send_reply(
    socket: &mut WebSocket,
    id: &Value,
    status: &str,
    label: &str,
    extra: Option<Value>,
) -> bool {
    let mut reply = json!({ "id": id, "status": status, "label": label });
    if let Some(extra) = extra {
        if let (Some(reply), Some(extra)) = (reply.as_object_mut(), extra.as_object()) {
            reply.extend(extra.clone());
        }
    }
    socket.send(Message::Text(reply.to_string())).await.is_ok()
}

fn is_valid_mode(mode: &str) -> bool {
    matches!(
        mode,
        "identify" | "describe" | "narrate" | "story" | "funny" | "funfact" | "ask"
    )
}

fn clean_text(value: Option<&Value>, max_chars: usize) -> String {
    value
        .and_then(Value::as_str)
        .unwrap_or("")
        .chars()
        .map(|character| {
            if character.is_control() {
                ' '
            } else {
                character
            }
        })
        .take(max_chars)
        .collect::<String>()
        .trim()
        .to_owned()
}

fn prepare_frames(images: &[Value], max_dim: u32) -> Vec<String> {
    images
        .iter()
        .take(MAX_FRAMES)
        .filter_map(|raw| {
            let raw = raw.as_str()?;
            let encoded = raw.split_once(',').map_or(raw, |(_, data)| data);
            let bytes = STANDARD.decode(encoded).ok()?;
            if bytes.is_empty() || bytes.len() > MAX_FRAME_BYTES {
                return None;
            }

            let image = image::load_from_memory(&bytes).ok()?;
            let (width, height) = (image.width(), image.height());
            let output = if width.max(height) > max_dim {
                let scale = f64::from(max_dim) / f64::from(width.max(height));
                let new_width = (f64::from(width) * scale).round().max(1.0) as u32;
                let new_height = (f64::from(height) * scale).round().max(1.0) as u32;
                let resized = image.resize_exact(new_width, new_height, FilterType::Triangle);
                let mut jpeg = Vec::new();
                JpegEncoder::new_with_quality(&mut jpeg, 80)
                    .encode_image(&resized)
                    .ok()?;
                jpeg
            } else {
                bytes
            };
            Some(STANDARD.encode(output))
        })
        .collect()
}

const STYLE_TASKS: &[(&str, &str)] = &[
    (
        "identify",
        "Identify the main object, person, animal, landmark, or activity. Reply with only its name in 1 to 5 words, with no punctuation.",
    ),
    (
        "describe",
        "Describe this in three or four sentences. Cover who or what is in the frame, what they are doing, their clothing, expression and posture if visible, the objects around them, the setting, and the overall mood or atmosphere.",
    ),
    (
        "narrate",
        "Narrate this like a calm nature-documentary presenter, in two short sentences. Only mention things that are visible.",
    ),
    (
        "story",
        "Write a tiny three-sentence story inspired by what you see, in the style of a film-noir detective. Only use things that are visible.",
    ),
    (
        "funny",
        "Make one light-hearted, affectionate joke about the scene in a single sentence. Never insult anyone's appearance.",
    ),
    (
        "funfact",
        "Name the main object, animal or landmark, then give one short fun fact about it. If you are not sure what it is, say what it looks like instead of guessing.",
    ),
];

fn build_prompt(
    mode: &str,
    frame_count: usize,
    place: Option<&str>,
    question: &str,
    skills: &[String],
    photo_context: Option<&str>,
) -> String {
    let source = if frame_count > 1 {
        format!("These {frame_count} images are frames from one short camera clip, in order.")
    } else {
        "This is a single photo from a camera.".to_owned()
    };
    let where_taken = place
        .map(|place| format!(" It was taken near {place}."))
        .unwrap_or_default();
    let explain = skills.iter().any(|skill| skill == "explain");
    let task = if mode == "ask" {
        let noun = if frame_count > 1 { "images" } else { "image" };
        let prior_summary = photo_context
            .map(|summary| {
                format!(
                    " The user selected this same photo from recent results. Its earlier summary was: {summary:?}. Treat that summary only as a hint to resolve references or clarify what the user means."
                )
            })
            .unwrap_or_default();
        format!(
            "Answer this question about the {noun}: {question} Reply briefly and only use what is visible. If the answer cannot be seen, say so.{}{prior_summary}",
            if explain {
                " Briefly explain what the identified object is."
            } else {
                ""
            }
        )
    } else if mode == "identify" && explain {
        "Identify the main visible object, person, animal, landmark, or activity, then explain what it is in one short sentence. Only state what is visible.".to_owned()
    } else {
        STYLE_TASKS
            .iter()
            .find_map(|(name, task)| (*name == mode).then_some(*task))
            .unwrap_or(STYLE_TASKS[0].1)
            .to_owned()
    };
    format!("{source}{where_taken} {task}")
}

fn max_tokens(mode: &str, skills: &[String]) -> u64 {
    let base = match mode {
        "ask" => 120,
        "identify" => 40,
        "describe" => 220,
        "narrate" => 150,
        "story" => 170,
        "funny" => 70,
        "funfact" => 110,
        "scout" => 6,
        _ => 40,
    };
    if skills.iter().any(|skill| {
        matches!(
            skill.as_str(),
            "explain" | "ocr" | "translate" | "web_search"
        )
    }) {
        base.max(220)
    } else {
        base
    }
}

fn is_creative(mode: &str) -> bool {
    matches!(mode, "narrate" | "story" | "funny" | "funfact")
}

async fn ask_model(
    state: &AppState,
    prompt: &str,
    frames: &[String],
    mode: &str,
    request_id: u64,
    phase: &str,
    skills: &[String],
) -> Result<String, String> {
    let mut content = vec![json!({ "type": "text", "text": prompt })];
    content.extend(frames.iter().map(|frame| {
        json!({
            "type": "image_url",
            "image_url": { "url": format!("data:image/jpeg;base64,{frame}") }
        })
    }));
    let payload = json!({
        "model": state.config.model,
        "messages": [{ "role": "user", "content": content }],
        "max_tokens": max_tokens(mode, skills),
        "temperature": if is_creative(mode) { 0.7 } else { 0.1 },
        "stream": false,
    });

    let started = Instant::now();
    let trace_model = phase != "watch-scout";
    if trace_model {
        info!(
            "request={request_id} model_phase={phase} started frames={} prompt_chars={}",
            frames.len(),
            prompt.chars().count()
        );
    }
    let response = state
        .client
        .post(format!("{}/chat/completions", state.config.llm_base_url))
        .json(&payload)
        .timeout(state.config.timeout)
        .send()
        .await
        .map_err(|error| {
            if trace_model {
                warn!(
                    "request={request_id} model_phase={phase} transport_error timeout={} connect={}",
                    error.is_timeout(),
                    error.is_connect()
                );
            }
            if error.is_connect() {
                warn!(
                    "Can't connect to the model server at {}",
                    state.config.llm_base_url
                );
                "The model isn't ready yet. Give it a minute and try again.".to_owned()
            } else if error.is_timeout() {
                "The model took too long to answer".to_owned()
            } else {
                "Couldn't reach the model service.".to_owned()
            }
        })?;

    if response.status().is_client_error() || response.status().is_server_error() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        let detail = serde_json::from_str::<Value>(&body)
            .ok()
            .and_then(|value| {
                value
                    .get("error")
                    .and_then(|error| {
                        error
                            .get("message")
                            .and_then(Value::as_str)
                            .map(ToOwned::to_owned)
                            .or_else(|| error.as_str().map(ToOwned::to_owned))
                    })
                    .map(|message| message.chars().take(200).collect::<String>())
            })
            .unwrap_or_else(|| body.chars().take(200).collect());
        warn!("Model server returned {status}: {detail}");
        warn!("request={request_id} model_phase={phase} failed http_status={status}");
        if status == ReqwestStatusCode::SERVICE_UNAVAILABLE {
            return Err("The model is still loading. Give it a minute and try again.".to_owned());
        }
        return Err(format!(
            "Model error: {}",
            if detail.is_empty() {
                status.to_string()
            } else {
                detail
            }
        ));
    }

    let data: Value = response
        .json()
        .await
        .map_err(|error| format!("Invalid response from the model server: {error}"))?;
    if trace_model {
        log_timings(request_id, phase, &data, started.elapsed().as_secs_f64());
    }
    let content = data
        .pointer("/choices/0/message/content")
        .cloned()
        .unwrap_or(Value::Null);
    let text = if let Some(text) = content.as_str() {
        text.to_owned()
    } else if let Some(parts) = content.as_array() {
        parts
            .iter()
            .filter_map(|part| part.get("text").and_then(Value::as_str))
            .collect::<String>()
    } else {
        String::new()
    };
    let thought = Regex::new(r"(?s)<think>.*?</think>").expect("constant regex");
    let answer = thought.replace_all(&text, "").trim().to_owned();
    let answer = if answer.is_empty() {
        "Nothing recognised".to_owned()
    } else {
        answer
    };
    if trace_model {
        info!(
            "request={request_id} model_phase={phase} completed elapsed_ms={} answer_chars={}",
            started.elapsed().as_millis(),
            answer.chars().count()
        );
    }
    Ok(answer)
}

fn log_timings(request_id: u64, phase: &str, data: &Value, elapsed: f64) {
    let Some(timings) = data.get("timings") else {
        info!("request={request_id} model_phase={phase} elapsed_s={elapsed:.1}");
        return;
    };
    let number = |name: &str| timings.get(name).and_then(Value::as_f64).unwrap_or(0.0);
    let predicted = timings
        .get("predicted_n")
        .and_then(Value::as_i64)
        .unwrap_or(0);
    info!(
        "request={request_id} model_phase={phase} elapsed_s={elapsed:.1} prompt_s={:.1} prompt_tokens_per_s={:.0} generation_s={:.1} generation_tokens_per_s={:.1} generated_tokens={predicted}",
        number("prompt_ms") / 1000.0,
        number("prompt_per_second"),
        number("predicted_ms") / 1000.0,
        number("predicted_per_second"),
    );
}

fn parse_location(location: Option<&Value>) -> Option<(f64, f64)> {
    let location = location?;
    let latitude = location.get("latitude")?.as_f64()?;
    let longitude = location.get("longitude")?.as_f64()?;
    if (-90.0..=90.0).contains(&latitude) && (-180.0..=180.0).contains(&longitude) {
        Some((latitude, longitude))
    } else {
        None
    }
}

fn place_name(address: &Value) -> String {
    let ward = Regex::new(r"(?i)\bward\s*\d+\b").expect("constant regex");
    let suburb = ["suburb", "neighbourhood", "quarter"]
        .iter()
        .filter_map(|key| address.get(key).and_then(Value::as_str))
        .find(|value| !ward.is_match(value))
        .unwrap_or("");
    let town = ["town", "village", "city", "municipality"]
        .iter()
        .find_map(|key| address.get(key).and_then(Value::as_str))
        .unwrap_or("");
    let suburb = if suburb == town { "" } else { suburb };
    [
        suburb,
        town,
        address.get("country").and_then(Value::as_str).unwrap_or(""),
    ]
    .into_iter()
    .filter(|part| !part.is_empty())
    .collect::<Vec<_>>()
    .join(", ")
}

async fn reverse_geocode(state: &AppState, latitude: f64, longitude: f64) -> String {
    let key = (
        (latitude * 1000.0).round() as i32,
        (longitude * 1000.0).round() as i32,
    );
    if let Some(name) = state.geo_cache.lock().await.get(&key).cloned() {
        return name;
    }
    let result = state
        .client
        .get("https://nominatim.openstreetmap.org/reverse")
        .query(&[
            ("format", "json".to_owned()),
            ("lat", latitude.to_string()),
            ("lon", longitude.to_string()),
            ("zoom", "16".to_owned()),
        ])
        .header("User-Agent", "SnapshotVisionEngine/3.0")
        .timeout(Duration::from_secs(5))
        .send()
        .await;
    match result {
        Ok(response) if response.status().is_success() => match response.json::<Value>().await {
            Ok(data) => {
                let name = place_name(&data["address"]);
                if !name.is_empty() {
                    state.geo_cache.lock().await.insert(key, name.clone());
                    return name;
                }
            }
            Err(error) => warn!("Reverse geocoding unavailable: {error}"),
        },
        Ok(response) => warn!("Reverse geocoding returned HTTP {}", response.status()),
        Err(error) => warn!("Reverse geocoding unavailable: {error}"),
    }
    format!("{latitude:.4}, {longitude:.4}")
}

const SCOUT_TARGETS: &[(&str, &str)] = &[
    ("person", "a person (face or body)"),
    (
        "landmark",
        "a well-known landmark, monument, or notable building",
    ),
];

async fn run_scout(socket: &mut WebSocket, state: &AppState, message: &Value) {
    let request_id = NEXT_REQUEST_ID.fetch_add(1, Ordering::Relaxed);
    let started = Instant::now();
    let id = message.get("id").cloned().unwrap_or(Value::Null);
    let targets: Vec<_> = message
        .get("targets")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .filter(|target| SCOUT_TARGETS.iter().any(|(name, _)| name == target))
        .collect();
    let frames = message
        .get("images")
        .and_then(Value::as_array)
        .map(|images| prepare_frames(images, state.config.max_dim))
        .unwrap_or_default();

    let send_scout = |status: &str, hit: Option<&str>| json!({ "id": id, "type": "scout", "status": status, "hit": hit });
    if frames.is_empty() || targets.is_empty() {
        let _ = socket
            .send(Message::Text(send_scout("skipped", None).to_string()))
            .await;
        return;
    }
    let Ok(_permit) = state.api_slots.clone().try_acquire_owned() else {
        let _ = socket
            .send(Message::Text(send_scout("skipped", None).to_string()))
            .await;
        return;
    };

    let options = targets
        .iter()
        .map(|target| target.to_uppercase())
        .chain(std::iter::once("NONE".to_owned()))
        .collect::<Vec<_>>();
    let descriptions = targets
        .iter()
        .filter_map(|target| {
            SCOUT_TARGETS
                .iter()
                .find(|(name, _)| name == target)
                .map(|(_, description)| format!("{} = {description}", target.to_uppercase()))
        })
        .collect::<Vec<_>>();
    let prompt = format!(
        "Look at this image. Is any of these clearly visible: {}? If several apply, pick the most prominent. Reply with exactly one word: {}.",
        descriptions.join("; "),
        options.join(", ")
    );
    match ask_model(
        state,
        &prompt,
        &frames[..1],
        "scout",
        request_id,
        "watch-scout",
        &[],
    )
    .await
    {
        Ok(text) => {
            let upper = text.to_uppercase();
            let hit = options
                .iter()
                .filter_map(|option| upper.find(option).map(|position| (position, option)))
                .min_by_key(|(position, _)| *position)
                .map(|(_, option)| option.to_lowercase())
                .filter(|hit| targets.contains(&hit.as_str()));
            match hit.as_deref() {
                Some(hit) => info!(
                    "request={request_id} watch_scout=hit target={hit} elapsed_ms={}",
                    started.elapsed().as_millis()
                ),
                None => info!(
                    "request={request_id} watch_scout=nothing elapsed_ms={}",
                    started.elapsed().as_millis()
                ),
            }
            let _ = socket
                .send(Message::Text(
                    send_scout("success", hit.as_deref()).to_string(),
                ))
                .await;
        }
        Err(error) => {
            warn!(
                "request={request_id} watch_scout=error elapsed_ms={} detail={error}",
                started.elapsed().as_millis()
            );
            let _ = socket
                .send(Message::Text(send_scout("error", None).to_string()))
                .await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_includes_clip_count_and_place() {
        let prompt = build_prompt("identify", 4, Some("Cape Town"), "", &[], None);
        assert!(
            prompt.starts_with("These 4 images are frames from one short camera clip, in order.")
        );
        assert!(prompt.contains("taken near Cape Town"));
        assert!(prompt.ends_with("Reply with only its name in 1 to 5 words, with no punctuation."));
    }

    #[test]
    fn location_is_rejected_outside_geographic_bounds() {
        assert_eq!(
            parse_location(Some(&json!({ "latitude": -33.9, "longitude": 18.4 }))),
            Some((-33.9, 18.4))
        );
        assert_eq!(
            parse_location(Some(&json!({ "latitude": 91, "longitude": 18.4 }))),
            None
        );
    }

    #[test]
    fn place_name_omits_ward_and_duplicate_town() {
        let address = json!({
            "suburb": "Ward 4",
            "neighbourhood": "Sea Point",
            "city": "Cape Town",
            "country": "South Africa"
        });
        assert_eq!(place_name(&address), "Sea Point, Cape Town, South Africa");
    }

    #[test]
    fn client_address_prefers_a_valid_forwarded_ip() {
        let peer: SocketAddr = "10.0.0.2:5000".parse().unwrap();
        let mut headers = HeaderMap::new();
        assert_eq!(client_address(&headers, peer).to_string(), "10.0.0.2");
        headers.insert("x-forwarded-for", "203.0.113.9, 10.0.0.2".parse().unwrap());
        assert_eq!(client_address(&headers, peer).to_string(), "203.0.113.9");
        // Anything that isn't an IP address is ignored, so it can't dress up a log line.
        headers.insert("x-forwarded-for", "not-an-ip".parse().unwrap());
        assert_eq!(client_address(&headers, peer).to_string(), "10.0.0.2");
    }

    #[test]
    fn log_timestamps_match_the_python_format() {
        let stamp = crate::logging::timestamp();
        let shape = Regex::new(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}$").unwrap();
        assert!(shape.is_match(&stamp), "unexpected timestamp {stamp:?}");
    }

    #[test]
    fn text_cleaning_removes_controls_and_applies_character_limit() {
        assert_eq!(clean_text(Some(&json!("hi\nthere")), 5), "hi th");
    }
}
