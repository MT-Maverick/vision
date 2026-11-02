(() => {
  const $ = (id) => document.getElementById(id);

  // Tunables
  const SHOT_DIM = 512, BURST_DIM = 512, SCOUT_DIM = 384, THUMB_DIM = 128; // image sizes sent to the model
  const BURST_GAP = 350, MAX_HISTORY = 12;
  const TICK_MS = 700;         // how often Watch samples the camera
  const CHANGE_THRESHOLD = 9;  // average pixel change (0-255) that counts as "the scene changed"
  const RESCOUT_MS = 20000;    // re-check an unchanged scene this often
  const HOLD_MS = 30000;       // after a hit, wait for a scene change or this long before triggering again

  const el = Object.fromEntries(
    ["stage","video","still","empty","emptyMsg","retry","dot","conn","pill","result","meta","shutter",
     "flip","watch","loc","place","speak","card","ask","askBtn","useLast","file","history","model","sources","opts","optsBtn","tPerson","tLandmark"].map((k) => [k, $(k)]));
  const opts = { mode: "identify", frames: 1 };
  const allowedSkills = ["explain", "ocr", "translate", "web_search"];
  const skills = new Set(["explain", "ocr", "translate"]);
  const targets = { person: true, landmark: true };

  let ws, stream, facing = "environment", busy = false, pending = null, nextId = 1, retries = 0;
  let watchOn = false, watchTimer = null, scouting = false, scoutPending = null;
  let ref = null, lastScoutAt = 0, holdUntil = 0, holdRef = null;
  let locOn = false, locError = 0, lastLoc = null, lastLocAt = 0, roughM = 0;
  const history = [];
  let speakOn = false, shown = null;
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const activeTargets = () => Object.keys(targets).filter((k) => targets[k]);

  /* ---------- connection ---------- */
  function setConn(on) {
    el.dot.classList.toggle("on", on);
    el.conn.textContent = on ? "Connected" : "Reconnecting";
  }
  function connect() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${location.host}/ws/vision`);
    ws.onopen = () => { setConn(true); retries = 0; };
    ws.onclose = () => {
      setConn(false);
      if (scoutPending) { scoutPending = null; scouting = false; }
      if (pending) finish({ status: "error", label: "Connection lost. Try again." });
      setTimeout(connect, Math.min(1000 * 2 ** retries++, 8000));
    };
    ws.onmessage = (e) => {
      const m = JSON.parse(e.data);
      if (m.type === "scout") { if (scoutPending && m.id === scoutPending.id) onScout(m); return; }
      if (!pending || m.id !== pending.id || m.status === "working") return;
      finish(m);
    };
  }

  /* ---------- camera ---------- */
  const showEmpty = (msg) => { el.emptyMsg.textContent = msg; el.empty.hidden = false; };
  function stopCamera() { stream?.getTracks().forEach((t) => t.stop()); stream = null; }
  async function startCamera() {
    if (!navigator.mediaDevices?.getUserMedia) {
      showEmpty(window.isSecureContext
        ? "This browser can't open the camera. Use Options to take a photo instead."
        : "The camera needs a secure connection. Open the https:// address, or use Options to take a photo.");
      return;
    }
    stopCamera();
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: facing }, width: { ideal: 1920 }, height: { ideal: 1080 } },
        audio: false,
      });
      el.video.srcObject = stream;
      el.video.classList.toggle("mirror", facing === "user");
      el.empty.hidden = true;
      el.still.hidden = true;
    } catch (err) {
      showEmpty(err.name === "NotAllowedError"
        ? "Camera access is blocked. Allow it in your browser settings, or use Options to take a photo."
        : "No camera found. Use Options to take a photo instead.");
    }
  }
  const cameraLive = () => !!stream && el.video.videoWidth > 0;

  function toJpeg(src, w, h, max, q = 0.85) {
    const s = Math.min(1, max / Math.max(w, h));
    const c = document.createElement("canvas");
    c.width = Math.round(w * s);
    c.height = Math.round(h * s);
    c.getContext("2d").drawImage(src, 0, 0, c.width, c.height);
    return c.toDataURL("image/jpeg", q);
  }
  const grab = (max, q) => toJpeg(el.video, el.video.videoWidth, el.video.videoHeight, max, q);

  /* ---------- location ---------- */
  const MAX_ACCURACY_M = 5000;  // a fix worse than this is a guess, not a position
  const GOOD_ENOUGH_M = 100;    // stop waiting once we are this close
  const fmtDist = (m) => (m < 1000 ? `${m} m` : `${(m / 1000).toFixed(1)} km`);

  // Asks for a real GPS fix and keeps the most accurate reading for a few seconds.
  const gpsFix = (timeout) => new Promise((res) => {
    if (!navigator.geolocation || !window.isSecureContext) { locError = -1; return res(null); }
    let best = null, done = false;
    const finish = () => {
      if (done) return;
      done = true;
      navigator.geolocation.clearWatch(id);
      clearTimeout(timer);
      if (!best && !locError) locError = 3;
      res(best);
    };
    const id = navigator.geolocation.watchPosition(
      (p) => {
        const a = Math.round(p.coords.accuracy);
        if (!best || a < best.accuracy) best = { latitude: p.coords.latitude, longitude: p.coords.longitude, accuracy: a };
        if (best.accuracy <= GOOD_ENOUGH_M) finish();
      },
      (e) => { locError = e.code; finish(); },
      { enableHighAccuracy: true, maximumAge: 0, timeout });
    const timer = setTimeout(finish, timeout);
  });
  const getLoc = async (timeout = 8000) => {
    if (lastLoc && Date.now() - lastLocAt < 120000) return lastLoc;
    locError = 0;
    const loc = await gpsFix(timeout);
    if (loc && loc.accuracy > MAX_ACCURACY_M) { locError = -2; roughM = loc.accuracy; return null; }
    if (loc) { lastLoc = loc; lastLocAt = Date.now(); }
    return loc;
  };
  const locProblem = () => ({
    "-1": "Location needs a secure page. Open the https:// address.",
    "-2": `Only a rough position was available (±${fmtDist(roughM)}), so it isn't used. Type your town under Options instead.`,
    1: "Location is blocked for this site. Click the lock icon in the address bar and allow it.",
    2: "This device couldn't work out its position. Type your town under Options instead.",
    3: "No GPS fix yet. Try outside or near a window, or type your town under Options.",
  }[locError] || "Location is unavailable.");

  /* ---------- analysis ---------- */
  function setBusy(b) {
    busy = b;
    el.shutter.classList.toggle("busy", b);
    el.stage.classList.toggle("looking", b || scouting);
    el.shutter.disabled = b;
  }
  function show(text, kind = "", meta = "") {
    el.result.textContent = text;
    el.result.className = "result " + (kind || (text.length > 40 ? "long" : ""));
    el.meta.textContent = meta;
    el.sources.replaceChildren();
    el.sources.hidden = true;
    el.card.hidden = !!kind || !shown;
    el.useLast.hidden = !shown;
  }
  function setPill(text) {
    el.pill.textContent = text;
    el.pill.hidden = !text;
    el.stage.classList.toggle("looking", busy || scouting);
  }

  async function send(images, thumb, question = "") {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      show("Not connected to the server yet. Try again in a moment.", "error");
      setBusy(false);
      return;
    }
    show("Looking…", "idle");
    const place = el.place.value.trim();
    const location = !place && locOn ? await getLoc() : null;
    pending = { id: nextId++, thumb, frame: images[0], question, mode: question ? "ask" : opts.mode, gps: location ? location.accuracy : null };
    ws.send(JSON.stringify({ id: pending.id, mode: pending.mode, question, images, location, place, skills: [...skills] }));
  }

  async function captureFrames() {
    const dim = opts.frames > 1 ? BURST_DIM : SHOT_DIM;
    const thumb = grab(THUMB_DIM, 0.7), images = [];
    for (let i = 0; i < opts.frames; i++) {
      images.push(grab(dim));
      if (i < opts.frames - 1) await sleep(BURST_GAP);
    }
    return { images, thumb };
  }

  async function analyze() {
    if (busy || !cameraLive()) return;
    window.speechSynthesis?.cancel();
    setBusy(true);
    const { images, thumb } = await captureFrames();
    send(images, thumb);
  }

  // Asks the model a typed question about the live view (or the last photo, if that chip is on).
  async function askQuestion() {
    const question = el.ask.value.trim();
    if (!question || busy) return;
    window.speechSynthesis?.cancel();
    let images, thumb;
    if (el.useLast.getAttribute("aria-pressed") === "true" && shown?.frame) {
      images = [shown.frame]; thumb = shown.thumb;
    } else if (!el.still.hidden) {
      images = [el.still.src]; thumb = el.still.src;
    } else if (!cameraLive()) {
      show("The camera isn't running. Allow it, or take a photo in Options.", "error");
      return;
    }
    setBusy(true);
    if (!images) ({ images, thumb } = await captureFrames());
    el.ask.value = "";
    send(images, thumb, question);
  }

  function finish(m) {
    const p = pending;
    pending = null;
    setBusy(false);
    if (m.status === "success") {
      const when = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
      const acc = p.gps != null ? `±${fmtDist(p.gps)}` : "";
      const base = [m.location_name, acc, when].filter(Boolean).join(" · ");
      const meta = p.question ? `Q: ${p.question} · ${base}` : base;
      const sources = Array.isArray(m.sources) ? m.sources.filter((s) => s && typeof s.title === "string" && typeof s.url === "string") : [];
      shown = { thumb: p.thumb, frame: p.frame, label: m.label, meta, base, sources };
      show(m.label, "", meta);
      renderSources(sources);
      history.unshift(shown);
      speak(m.label, p.mode);
      history.length = Math.min(history.length, MAX_HISTORY);
      renderHistory(0);
    } else {
      show(m.label || "Something went wrong", "error");
    }
    setPill(watchOn ? "Watching…" : "");
  }

  function renderSources(sources) {
    const links = sources.flatMap((source, i) => {
      try {
        const url = new URL(source.url);
        if (url.protocol !== "https:" && url.protocol !== "http:") return [];
        const link = document.createElement("a");
        link.href = url.href;
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        link.textContent = `[${i + 1}] ${source.title}`;
        return [link];
      } catch {
        return [];
      }
    });
    el.sources.replaceChildren(...links);
    el.sources.hidden = !links.length;
  }

  /* ---------- watch mode: scout cheaply, analyze only when something is there ---------- */
  const sc = document.createElement("canvas");
  sc.width = 32; sc.height = 24;
  const sctx = sc.getContext("2d", { willReadFrequently: true });
  function sample() {
    sctx.drawImage(el.video, 0, 0, 32, 24);
    const d = sctx.getImageData(0, 0, 32, 24).data, g = new Float32Array(768);
    for (let i = 0; i < 768; i++) g[i] = 0.299 * d[i * 4] + 0.587 * d[i * 4 + 1] + 0.114 * d[i * 4 + 2];
    return g;
  }
  function diff(a, b) {
    if (!a || !b) return 255;
    let s = 0;
    for (let i = 0; i < a.length; i++) s += Math.abs(a[i] - b[i]);
    return s / a.length;
  }

  function tick() {
    if (!watchOn || busy || scouting || !cameraLive() || ws?.readyState !== WebSocket.OPEN) return;
    const now = Date.now(), cur = sample();
    if (holdUntil > now) {
      if (diff(cur, holdRef) < CHANGE_THRESHOLD * 1.5) return; // same scene we just reported
      holdUntil = 0;
    }
    if (diff(cur, ref) <= CHANGE_THRESHOLD && now - lastScoutAt < RESCOUT_MS) return;
    ref = cur;
    lastScoutAt = now;
    scout(cur);
  }
  function scout(cur) {
    const t = activeTargets();
    if (!t.length) return;
    scouting = true;
    setPill("Checking…");
    scoutPending = { id: nextId++, cur };
    ws.send(JSON.stringify({ id: scoutPending.id, type: "scout", targets: t, images: [grab(SCOUT_DIM, 0.75)] }));
  }
  function onScout(m) {
    const p = scoutPending;
    scoutPending = null;
    scouting = false;
    if (m.hit && watchOn) {
      holdUntil = Date.now() + HOLD_MS;
      holdRef = p.cur;
      setPill(m.hit === "person" ? "Person spotted" : "Landmark spotted");
      analyze();
    } else {
      setPill(watchOn ? "Watching…" : "");
    }
  }

  /* ---------- voice and cards ---------- */
  function speak(text, mode) {
    if (!speakOn || !("speechSynthesis" in window)) return;
    speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    if (mode === "narrate") { u.rate = 0.88; u.pitch = 0.8; }
    if (mode === "story") { u.rate = 0.92; u.pitch = 0.7; }
    speechSynthesis.speak(u);
  }

  function wrapLines(ctx, text, maxW) {
    const lines = [];
    let line = "";
    for (const w of text.split(/\s+/)) {
      const t = line ? `${line} ${w}` : w;
      if (ctx.measureText(t).width > maxW && line) { lines.push(line); line = w; } else line = t;
    }
    if (line) lines.push(line);
    return lines;
  }

  // Draws the photo and its caption onto a card, then shares it (phones) or downloads it (computers).
  async function saveCard() {
    if (!shown) return;
    const img = new Image();
    img.src = shown.frame;
    await img.decode();
    const W = 1080, pad = 56, font = "600 46px system-ui, sans-serif";
    const probe = document.createElement("canvas").getContext("2d");
    probe.font = font;
    const lines = wrapLines(probe, shown.label, W - pad * 2).slice(0, 9);
    const photoH = Math.round((W - pad * 2) * img.height / img.width);
    const H = pad + photoH + 48 + lines.length * 62 + 110;
    const cv = document.createElement("canvas");
    cv.width = W; cv.height = H;
    const ctx = cv.getContext("2d");
    ctx.fillStyle = "#0E1822";
    ctx.fillRect(0, 0, W, H);
    ctx.save();
    if (ctx.roundRect) { ctx.beginPath(); ctx.roundRect(pad, pad, W - pad * 2, photoH, 28); ctx.clip(); }
    ctx.drawImage(img, pad, pad, W - pad * 2, photoH);
    ctx.restore();
    ctx.textBaseline = "top";
    ctx.fillStyle = "#EEF3F6";
    ctx.font = font;
    lines.forEach((l, i) => ctx.fillText(l, pad, pad + photoH + 48 + i * 62));
    ctx.font = "500 30px system-ui, sans-serif";
    ctx.fillStyle = "#F2C14E";
    ctx.fillText(shown.base || shown.meta || "", pad, H - 76, 600);
    ctx.fillStyle = "#A9BAC7";
    ctx.textAlign = "right";
    ctx.fillText("Vision Snapshot", W - pad, H - 76);
    const blob = await new Promise((r) => cv.toBlob(r, "image/png"));
    const file = new File([blob], "snapshot.png", { type: "image/png" });
    if (navigator.canShare?.({ files: [file] })) {
      try { await navigator.share({ files: [file] }); return; } catch { /* cancelled: fall through to download */ }
    }
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "snapshot.png";
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 4000);
  }

  /* ---------- history ---------- */
  function renderHistory(active) {
    el.history.hidden = !history.length;
    el.history.replaceChildren(...history.map((h, i) => {
      const li = document.createElement("li"), b = document.createElement("button"), img = document.createElement("img");
      img.src = h.thumb;
      img.alt = h.label;
      b.append(img);
      b.setAttribute("aria-label", `Show result: ${h.label}`);
      if (i === active) b.setAttribute("aria-current", "true");
      b.onclick = () => { shown = h; show(h.label, "", h.meta); renderSources(h.sources || []); renderHistory(i); };
      li.append(b);
      return li;
    }));
  }

  /* ---------- controls ---------- */
  document.querySelectorAll(".seg button[data-group]").forEach((b) => b.addEventListener("click", () => {
    const g = b.dataset.group;
    opts[g] = g === "frames" ? Number(b.dataset.value) : b.dataset.value;
    document.querySelectorAll(`[data-group="${g}"]`).forEach((x) => x.setAttribute("aria-pressed", x === b));
  }));
  try {
    const savedSkills = JSON.parse(localStorage.getItem("skills") || "null");
    if (Array.isArray(savedSkills)) {
      skills.clear();
      savedSkills.filter((skill) => allowedSkills.includes(skill)).forEach((skill) => skills.add(skill));
    }
  } catch {}
  document.querySelectorAll(".seg button[data-skill]").forEach((button) => {
    const skill = button.dataset.skill;
    button.setAttribute("aria-pressed", skills.has(skill));
    button.onclick = () => {
      if (skills.has(skill)) skills.delete(skill);
      else skills.add(skill);
      button.setAttribute("aria-pressed", skills.has(skill));
      try { localStorage.setItem("skills", JSON.stringify([...skills])); } catch {}
    };
  });
  [["tPerson", "person"], ["tLandmark", "landmark"]].forEach(([id, k]) => {
    el[id].onclick = () => { targets[k] = !targets[k]; el[id].setAttribute("aria-pressed", targets[k]); };
  });

  el.optsBtn.onclick = () => {
    el.opts.hidden = !el.opts.hidden;
    el.optsBtn.setAttribute("aria-expanded", !el.opts.hidden);
  };
  el.shutter.onclick = () => analyze();
  el.retry.onclick = startCamera;
  el.flip.onclick = () => { facing = facing === "environment" ? "user" : "environment"; startCamera(); };

  el.watch.onclick = () => {
    if (!watchOn && !activeTargets().length) {
      show("Pick People or Landmarks in Options first.", "error");
      return;
    }
    watchOn = !watchOn;
    el.watch.setAttribute("aria-pressed", watchOn);
    clearInterval(watchTimer);
    ref = null; lastScoutAt = 0; holdUntil = 0;
    if (watchOn) watchTimer = setInterval(tick, TICK_MS);
    setPill(watchOn ? "Watching…" : "");
  };

  el.loc.onclick = async () => {
    if (locOn) {
      locOn = false;
    } else {
      el.loc.textContent = "Finding you…";
      const loc = await getLoc(12000);
      locOn = !!loc;
      if (!loc) show(locProblem(), "error");
      else show(`Location on (±${fmtDist(loc.accuracy)}).`, "idle");
    }
    el.loc.setAttribute("aria-pressed", locOn);
    el.loc.textContent = locOn ? "Location on" : "Location off";
  };

  el.file.onchange = async () => {
    const f = el.file.files[0];
    el.file.value = "";
    if (!f || busy) return;
    setBusy(true);
    try {
      const bmp = await createImageBitmap(f);
      const url = toJpeg(bmp, bmp.width, bmp.height, SHOT_DIM);
      el.still.src = url;
      el.still.hidden = false;
      el.empty.hidden = true;
      el.opts.hidden = true;
      el.optsBtn.setAttribute("aria-expanded", "false");
      send([url], toJpeg(bmp, bmp.width, bmp.height, THUMB_DIM, 0.7));
    } catch {
      setBusy(false);
      show("That file couldn't be read as an image.", "error");
    }
  };

  document.addEventListener("visibilitychange", () => {
    if (document.hidden) { stopCamera(); window.speechSynthesis?.cancel(); }
    else if (el.still.hidden) startCamera();
  });

  fetch("/api/status").then((r) => r.json()).then((s) => {
    el.model.textContent = s.ready ? `Model: ${s.model}`
      : "The model is still loading. Give it a minute, then refresh.";
  }).catch(() => {});

  // Ask bar
  el.askBtn.onclick = askQuestion;
  el.ask.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); askQuestion(); } });
  el.useLast.onclick = () => el.useLast.setAttribute("aria-pressed", el.useLast.getAttribute("aria-pressed") !== "true");

  // Voice and card buttons
  if (!("speechSynthesis" in window)) el.speak.hidden = true;
  try { speakOn = localStorage.getItem("speak") === "1"; } catch {}
  el.speak.setAttribute("aria-pressed", speakOn);
  el.speak.onclick = () => {
    speakOn = !speakOn;
    el.speak.setAttribute("aria-pressed", speakOn);
    try { localStorage.setItem("speak", speakOn ? "1" : "0"); } catch {}
    if (speakOn) speak("Voice on.", ""); else window.speechSynthesis?.cancel(); // the tap also unlocks audio on phones
  };
  el.card.onclick = saveCard;

  // Remember the typed place between visits
  try { el.place.value = localStorage.getItem("place") || ""; } catch {}
  el.place.addEventListener("input", () => { try { localStorage.setItem("place", el.place.value.trim()); } catch {} });

  connect();
  startCamera();
})();