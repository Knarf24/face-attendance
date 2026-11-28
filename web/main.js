// The page only handles the camera, drawing and browser storage.
// Face detection, recognition, grouping and the register are Python (attendance.py, register.py).
const $ = (id) => document.getElementById(id);
const PY_FILES = ["attendance.py", "register.py"];
const MODEL_FILES = [
  "models/face_detection_yunet_2023mar.onnx",
  "models/face_recognition_sface_2021dec_int8.onnx",
];
const STORE_KEY = "face-attendance-v1";

let py = null, reg = null, stream = null, mode = "cam", uploaded = false;
let lastShot = null;       // canvas holding the last scanned photo (for face thumbnails)
let poolPhotos = [];       // canvases of the class photos used for grouping
let state = { students: [], dates: [] };

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const today = () => { const d = new Date(); return new Date(d - d.getTimezoneOffset() * 60000).toISOString().slice(0, 10); };
function setLoading(text) { $("loadText").textContent = text; }
function pyCall(fn, ...args) { const r = reg[fn](...args); return typeof r === "string" ? JSON.parse(r) : r; }

function save() {
  try { localStorage.setItem(STORE_KEY, reg.export_state()); }
  catch (e) { console.warn("Could not save in this browser", e); }
}

// The page is usable right away (camera, upload); Python, OpenCV and the models load in the background,
// all at the same time, and attendance taken before they are ready waits for them.
async function startPython() {
  try {
    const files = Promise.all([...PY_FILES, ...MODEL_FILES].map(async (f) => {
      const r = await fetch(f);
      if (!r.ok) throw new Error(`Could not load ${f}`);
      return [f, new Uint8Array(await r.arrayBuffer())];
    }));
    files.catch(() => {}); // reported below
    py = await loadPyodideFast();
    setLoading("Loading OpenCV and NumPy…");
    await py.loadPackage(["numpy", "opencv-python"]);
    setLoading("Loading the face models…");
    py.FS.mkdirTree("/home/pyodide/models");
    for (const [f, bytes] of await files) py.FS.writeFile(`/home/pyodide/${f}`, bytes);
    setLoading("Opening your class register…");
    reg = py.pyimport("register");
    let saved = "";
    try { saved = localStorage.getItem(STORE_KEY) || ""; } catch (e) { /* private window */ }
    state = JSON.parse(reg.import_state(saved));
    renderRegister();
    $("loading").classList.add("done");
    document.documentElement.dataset.py = "ready";
    updateReady();
  } catch (e) {
    console.error(e);
    $("loading").classList.add("error");
    setLoading("Sorry, Python could not start in this browser. Try Chrome, Edge or Firefox. (" + e.message + ")");
    document.querySelector(".spinner").style.display = "none";
    updateReady();
  }
  // Keeps Python saved on this device so the next visit starts without downloading it (see sw.js).
  if ("serviceWorker" in navigator && isSecureContext) navigator.serviceWorker.register("sw.js").catch(() => {});
}

// ---- camera and photos ---------------------------------------------------------
function updateReady() {
  const haveImage = mode === "cam" ? !!stream : uploaded;
  $("takeBtn").disabled = !haveImage || $("loading").classList.contains("error");
  $("poolAdd").disabled = !reg;
}

async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1920 }, height: { ideal: 1080 } }, audio: false });
    $("video").srcObject = stream;
    await $("video").play();
    $("camHint").hidden = true;
    $("startCam").textContent = "Camera on";
    $("startCam").disabled = true;
  } catch (e) {
    $("camHint").innerHTML = "Couldn't open the camera. Allow camera access, or upload a photo instead.";
  }
  updateReady();
}

function setMode(m) {
  mode = m;
  $("tabCam").classList.toggle("active", m === "cam");
  $("tabUpload").classList.toggle("active", m === "upload");
  $("video").hidden = m !== "cam";
  $("preview").hidden = m !== "upload" || !uploaded;
  $("startCam").hidden = m !== "cam";
  $("camHint").hidden = m === "cam" ? !!stream : uploaded;
  if (m === "upload") {
    $("camHint").innerHTML = "Choose a photo of the class (or of one student to enroll them).";
    if (!uploaded) $("file").click();
  } else if (!stream) {
    $("camHint").innerHTML = "Press <b>Start camera</b>, or upload a photo.";
  }
  updateReady();
}

// Draws an image/video into a new canvas, at most 2000 px on the long side.
function toCanvas(src, w0, h0) {
  const scale = Math.min(1, 2000 / Math.max(w0, h0));
  const c = document.createElement("canvas");
  c.width = Math.round(w0 * scale); c.height = Math.round(h0 * scale);
  c.getContext("2d", { willReadFrequently: true }).drawImage(src, 0, 0, c.width, c.height);
  return c;
}

function grabFrame() {
  return mode === "cam"
    ? toCanvas($("video"), $("video").videoWidth, $("video").videoHeight)
    : toCanvas($("preview"), $("preview").naturalWidth, $("preview").naturalHeight);
}

function pixels(c) { return c.getContext("2d", { willReadFrequently: true }).getImageData(0, 0, c.width, c.height).data; }

// A small square picture of one face, as a data URL (stored with the student).
function thumb(c, box, size = 96) {
  const [x, y, w, h] = box;
  const side = Math.max(w, h) * 1.25, cx = x + w / 2, cy = y + h / 2;
  const t = document.createElement("canvas");
  t.width = t.height = size;
  t.getContext("2d").drawImage(c, cx - side / 2, cy - side / 2, side, side, 0, 0, size, size);
  return t.toDataURL("image/jpeg", 0.8);
}

function scanCanvas(c) {
  const res = reg.scan(pixels(c), c.width, c.height);
  const [rgba, json] = res.toJs();
  res.destroy();
  return { rgba, result: JSON.parse(json) };
}

// ---- 1 + 2: take attendance ---------------------------------------------------------
async function takeAttendance() {
  $("takeBtn").disabled = true;
  $("takeBtn").textContent = reg ? "Looking for faces…" : "Getting the face models ready…";
  await new Promise((r) => setTimeout(r, 30)); // let the button repaint
  try {
    await pyReady;
    if (!reg) throw new Error("Python could not start");
    $("takeBtn").textContent = "Looking for faces…";
    await new Promise((r) => setTimeout(r, 30));
    const c = grabFrame();
    lastShot = c;
    const { rgba, result } = scanCanvas(c);
    const out = $("out");
    out.width = c.width; out.height = c.height;
    out.getContext("2d").putImageData(new ImageData(new Uint8ClampedArray(rgba.buffer, rgba.byteOffset, c.width * c.height * 4), c.width, c.height), 0, 0);
    out.hidden = false;
    const date = $("date").value || today();
    const marked = pyCall("mark_present", date);
    state = marked;
    save();
    const known = result.faces.filter((f) => f.id).length;
    const unknown = result.count - known;
    const b = $("banner");
    if (!result.count) {
      b.className = "banner bad";
      b.textContent = "No faces found. Try a closer or brighter photo.";
    } else {
      b.className = "banner ok";
      b.textContent = `👥 ${result.count} ${result.count === 1 ? "face" : "faces"} counted · ✅ ${known} recognised` +
        (unknown ? ` · ❓ ${unknown} new` : "");
    }
    showFaces(result.faces);
    renderRegister();
    window.lastResult = { ...result, newly: marked.newly }; // handy for testing
  } catch (e) {
    console.error(e);
    $("banner").className = "banner bad";
    $("banner").textContent = "Something went wrong: " + e.message;
  }
  $("takeBtn").textContent = "Take attendance";
  updateReady();
}

function showFaces(faces) {
  $("faces").innerHTML = faces.map((f) => {
    const img = `<img src="${thumb(lastShot, f.box)}" alt="">`;
    if (f.id) return `<div class="face known">${img}<b>${esc(f.name)}</b><div class="small">Present ✓</div></div>`;
    return `<div class="face unknown" data-i="${f.index}">${img}<div class="small">New face #${f.index + 1}</div>
      <input placeholder="Student name" aria-label="Name for face ${f.index + 1}">
      <button class="secondary">Add &amp; mark present</button></div>`;
  }).join("");
  $("faces").querySelectorAll(".face.unknown").forEach((el) => {
    const input = el.querySelector("input");
    const go = () => {
      const i = +el.dataset.i;
      try {
        const face = faces.find((f) => f.index === i);
        reg.enroll(i, input.value, thumb(lastShot, face.box));
        state = JSON.parse(reg.set_present($("date").value || today(), studentIdByName(input.value), true));
        save();
        el.className = "face known";
        el.innerHTML = `<img src="${thumb(lastShot, face.box)}" alt=""><b>${esc(input.value.trim())}</b><div class="small">Added ✓</div>`;
        renderRegister();
      } catch (e) { input.placeholder = "Please type a name"; input.focus(); }
    };
    el.querySelector("button").onclick = go;
    input.onkeydown = (e) => { if (e.key === "Enter") go(); };
  });
}

function studentIdByName(name) {
  const s = JSON.parse(reg.summary()).students.find((s) => s.name.toLowerCase() === name.trim().replace(/\s+/g, " ").toLowerCase());
  return s ? s.id : "";
}

// ---- 3: register ---------------------------------------------------------------------
function renderRegister() {
  const { students, dates } = state;
  const date = $("date").value || today();
  const cols = dates.includes(date) ? dates : [...dates, date].sort();
  const shown = cols.slice(-10); // the last 10 school days
  if (!students.length) {
    $("reg").innerHTML = '<tr><td class="small">No students yet. Take a class photo and name the faces, or use Quick setup below.</td></tr>';
    $("regSummary").textContent = "";
    return;
  }
  const head = `<tr><th>Student</th>${shown.map((d) => `<th class="c${d === date ? " today" : ""}">${d.slice(5)}</th>`).join("")}<th class="c">Rate</th><th></th></tr>`;
  const rows = students.map((s) => `<tr data-id="${s.id}">
    <td><div class="who">${s.thumb ? `<img src="${s.thumb}" alt="">` : ""}<span class="name" title="Click to rename">${esc(s.name)}</span>
      <span class="small">${s.photos} ${s.photos === 1 ? "photo" : "photos"}</span></div></td>
    ${shown.map((d) => `<td class="c${d === date ? " today" : ""}"><input type="checkbox" data-d="${d}" ${s.days[d] ? "checked" : ""} title="${esc(s.days[d] || "Absent")}"></td>`).join("")}
    <td class="c">${s.rate == null ? "–" : s.rate + "%"}</td>
    <td><button class="del" title="Remove student">✕</button></td></tr>`).join("");
  $("reg").innerHTML = head + rows;
  const presentToday = students.filter((s) => s.days[date]).length;
  $("regSummary").textContent = `${students.length} students · ${presentToday} present on ${date} · ${dates.length} school ${dates.length === 1 ? "day" : "days"} recorded`;
  $("reg").querySelectorAll("tr[data-id]").forEach((tr) => {
    const id = tr.dataset.id;
    tr.querySelectorAll("input[type=checkbox]").forEach((cb) => {
      cb.onchange = () => { state = JSON.parse(reg.set_present(cb.dataset.d, id, cb.checked)); save(); renderRegister(); };
    });
    tr.querySelector(".name").onclick = (e) => {
      const span = e.target;
      const input = document.createElement("input");
      input.value = span.textContent;
      span.replaceWith(input);
      input.focus();
      const done = () => { state = JSON.parse(reg.rename_student(id, input.value)); save(); renderRegister(); };
      input.onblur = done;
      input.onkeydown = (k) => { if (k.key === "Enter") input.blur(); };
    };
    const del = tr.querySelector(".del");
    del.onclick = () => {
      if (del.dataset.armed) { state = JSON.parse(reg.remove_student(id)); save(); renderRegister(); return; }
      del.dataset.armed = "1"; del.textContent = "Remove?";
      setTimeout(() => { del.textContent = "✕"; delete del.dataset.armed; }, 3000);
    };
  });
}

function download(name, text, type) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type }));
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

// ---- 4: quick setup by grouping faces ----------------------------------------------------
function loadImage(file) {
  return new Promise((res, rej) => {
    const img = new Image();
    img.onload = () => res(img);
    img.onerror = () => rej(new Error("Could not open " + file.name));
    img.src = URL.createObjectURL(file);
  });
}

async function addPoolPhotos(files) {
  $("poolInfo").textContent = "Finding faces…";
  for (const f of files) {
    await new Promise((r) => setTimeout(r, 20));
    try {
      const img = await loadImage(f);
      const c = toCanvas(img, img.naturalWidth, img.naturalHeight);
      reg.pool_add(pixels(c), c.width, c.height, poolPhotos.length);
      poolPhotos.push(c);
    } catch (e) { console.error(e); }
  }
  showGroups(pyCall("pool_group"));
}

function showGroups(g) {
  $("poolInfo").textContent = g.groups.length
    ? `${g.faces} faces in ${poolPhotos.length} ${poolPhotos.length === 1 ? "photo" : "photos"}, grouped into ${g.groups.length} ${g.groups.length === 1 ? "person" : "people"}.`
    : "";
  $("groups").innerHTML = g.groups.map((gr) => {
    const imgs = gr.faces.slice(0, 8).map((m) => `<img src="${thumb(poolPhotos[m.photo], m.box, 64)}" alt="">`).join("");
    return `<div class="group${gr.known ? " done" : ""}" data-g="${gr.group}">
      <div class="thumbs">${imgs}</div>
      <div class="small">${gr.size} ${gr.size === 1 ? "photo" : "photos"}${gr.known ? ` · looks like <b>${esc(gr.known)}</b>` : ""}</div>
      <input placeholder="Student name" value="${esc(gr.known || "")}">
      <button class="secondary">${gr.known ? "Add these photos" : "Add student"}</button></div>`;
  }).join("");
  $("groups").querySelectorAll(".group").forEach((el) => {
    const input = el.querySelector("input");
    const gr = g.groups.find((x) => x.group === +el.dataset.g);
    const go = () => {
      if (!input.value.trim()) { input.focus(); return; }
      const m = gr.faces[0];
      state = JSON.parse(reg.pool_enroll_group(gr.group, input.value, thumb(poolPhotos[m.photo], m.box)));
      save();
      renderRegister();
      el.classList.add("done");
      el.querySelector("button").textContent = "Added ✓";
      el.querySelector("button").disabled = true;
    };
    el.querySelector("button").onclick = go;
    input.onkeydown = (e) => { if (e.key === "Enter") go(); };
  });
}

// ---- wiring ----------------------------------------------------------------------------------
$("date").value = today();
$("date").onchange = () => reg && renderRegister();
$("startCam").onclick = startCamera;
$("tabCam").onclick = () => setMode("cam");
$("tabUpload").onclick = () => setMode("upload");
$("file").onchange = () => {
  const f = $("file").files[0];
  if (!f) return;
  const img = $("preview");
  img.onload = () => { uploaded = true; setMode("upload"); };
  img.src = URL.createObjectURL(f);
};
$("preview").onclick = () => $("file").click();
$("takeBtn").onclick = takeAttendance;
$("csvBtn").onclick = () => reg && download(`attendance-${today()}.csv`, reg.export_csv(), "text/csv");
$("backupBtn").onclick = () => reg && download(`class-register-backup-${today()}.json`, reg.export_state(), "application/json");
$("restoreBtn").onclick = () => $("restoreFile").click();
$("restoreFile").onchange = async () => {
  const f = $("restoreFile").files[0];
  if (!f || !reg) return;
  try { state = JSON.parse(reg.import_state(await f.text())); save(); renderRegister(); }
  catch (e) { $("regSummary").textContent = "That file is not a Face Attendance backup."; }
  $("restoreFile").value = "";
};
$("clearBtn").onclick = () => {
  const b = $("clearBtn");
  if (!b.dataset.armed) {
    b.dataset.armed = "1"; b.textContent = "⚠️ Click again to delete all";
    setTimeout(() => { b.textContent = "🗑️ Delete everything"; delete b.dataset.armed; }, 4000);
    return;
  }
  if (reg) state = JSON.parse(reg.clear_all());
  try { localStorage.removeItem(STORE_KEY); } catch (e) { /* ignore */ }
  poolPhotos = []; $("groups").innerHTML = ""; $("poolInfo").textContent = ""; $("faces").innerHTML = "";
  b.textContent = "🗑️ Delete everything"; delete b.dataset.armed;
  renderRegister();
};
$("poolAdd").onclick = () => $("poolFiles").click();
$("poolFiles").onchange = async () => {
  const files = [...$("poolFiles").files];
  $("poolFiles").value = "";
  if (files.length && reg) await addPoolPhotos(files);
};
$("poolClear").onclick = () => { if (reg) reg.pool_clear(); poolPhotos = []; $("groups").innerHTML = ""; $("poolInfo").textContent = ""; };

const pyReady = startPython();
document.documentElement.dataset.ui = "ready";
