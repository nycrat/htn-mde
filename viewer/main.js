import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const SCAN_DIR = "../scans/";

// PLY parsing + decorating + spacing estimation all happen in a worker so the
// render loop never blocks on load work.
const loaderWorker = new Worker(
  new URL("./loader.worker.js", import.meta.url),
  { type: "module" },
);
let decodeSeq = 0;
const decodePending = new Map();
loaderWorker.onmessage = (e) => {
  const { id, ok } = e.data;
  const job = decodePending.get(id);
  if (!job) return;
  decodePending.delete(id);
  if (!ok) {
    job.reject(new Error(e.data.error));
    return;
  }
  try {
    job.resolve(wrapDecoded(e.data));
  } catch (err) {
    job.reject(err);
  }
};

function wrapDecoded(d) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(d.position, 3));
  geometry.setAttribute("viewZ", new THREE.BufferAttribute(d.viewZ, 3));
  geometry.setAttribute("aColor", new THREE.BufferAttribute(d.aColor, 3));
  geometry.userData.spacing = d.spacing;
  geometry.userData.thumb = d.thumb;
  return geometry;
}

function decodeOnWorker(buf) {
  return new Promise((resolve, reject) => {
    const id = ++decodeSeq;
    decodePending.set(id, { resolve, reject });
    loaderWorker.postMessage({ id, buf }, [buf]);
  });
}

const app = document.getElementById("app");
const statsEl = document.getElementById("stats");

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x121212);

const camera = new THREE.PerspectiveCamera(
  55,
  innerWidth / innerHeight,
  0.001,
  1000,
);
camera.position.set(1.6, 1.0, 1.6);

const renderer = new THREE.WebGLRenderer({
  antialias: true,
  powerPreference: "high-performance",
});
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
app.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;

const IDLE_MS = 500;
const SPIN_SPEED = 0.7;
let idleTimer = null;
let spinEnabled = true;

function updateSpinButton() {
  const btn = document.getElementById("spin");
  if (!btn) return;
  btn.classList.toggle("spin-on", spinEnabled && controls.autoRotate);
  btn.classList.toggle("spin-off", !spinEnabled);
  btn.textContent = spinEnabled
    ? controls.autoRotate
      ? "spin: on"
      : "spin: auto"
    : "spin: off";
}

function stopIdleSpin() {
  clearTimeout(idleTimer);
  controls.autoRotate = false;
  updateSpinButton();
}

function scheduleIdleSpin() {
  clearTimeout(idleTimer);
  if (!spinEnabled) {
    controls.autoRotate = false;
    updateSpinButton();
    return;
  }
  idleTimer = setTimeout(() => {
    controls.autoRotate = true;
    updateSpinButton();
  }, IDLE_MS);
}

controls.autoRotateSpeed = SPIN_SPEED;
controls.addEventListener("start", stopIdleSpin);
controls.addEventListener("end", scheduleIdleSpin);
controls.autoRotate = true;
updateSpinButton();

document.getElementById("spin").addEventListener("click", () => {
  spinEnabled = !spinEnabled;
  clearTimeout(idleTimer);
  if (spinEnabled) {
    controls.autoRotate = true;
  } else {
    controls.autoRotate = false;
  }
  updateSpinButton();
});

const VERT = `
attribute vec3 aColor;
attribute vec3 viewZ;
uniform float uSize;
varying float vViewZ;
varying vec3 vColor;
void main() {
  vViewZ = viewZ.z;
  vColor = aColor;
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  gl_PointSize = min(uSize * (240.0 / max(0.01, -mv.z)), 256.0);
  gl_Position = projectionMatrix * mv;
}
`;

const FRAG = `
uniform float uNear;
uniform float uFar;
varying float vViewZ;
varying vec3 vColor;
void main() {
  if (vViewZ < uNear || vViewZ > uFar) discard;
  vec2 c = gl_PointCoord - 0.5;
  if (dot(c, c) > 0.25) discard;
  gl_FragColor = vec4(vColor, 1.0);
}
`;

let points = null;

function makeMaterial() {
  return new THREE.ShaderMaterial({
    uniforms: {
      uSize: { value: parseFloat(document.getElementById("size").value) },
      uNear: { value: 0.0 },
      uFar: { value: 1.0 },
    },
    vertexShader: VERT,
    fragmentShader: FRAG,
  });
}

function frameObject(geometry, cam, front = false) {
  geometry.computeBoundingBox();
  const center = geometry.boundingBox.getCenter(new THREE.Vector3());
  const radius = geometry.boundingBox.getBoundingSphere(
    new THREE.Sphere(),
  ).radius;
  const dist = Math.max(radius, 1e-3) * 2.4;
  cam.near = Math.max(dist / 20000, 1e-6);
  cam.far = dist * 200;
  // Scans are stored with the capture camera at the origin looking down +Z
  // (depth); the subject's rendered front faces +Z, so the original image's
  // perspective is viewed from +Z looking back toward -Z, yawed -20deg so the
  // initial framing is on the same side as auto-rotation.
  const yaw = (20 * Math.PI) / 180;
  const dir = front
    ? new THREE.Vector3(Math.sin(yaw), 0, Math.cos(yaw))
    : new THREE.Vector3(0.7, 0.5, 0.7).normalize();
  cam.position.copy(center).add(dir.multiplyScalar(dist));
  cam.lookAt(center);
  cam.updateProjectionMatrix();
}

function fitTo(geometry) {
  geometry.computeBoundingBox();
  const center = geometry.boundingBox.getCenter(new THREE.Vector3());
  controls.target.copy(center);
  frameObject(geometry, camera, true);
  controls.update();
}

let lastSpacing = null;
let sizeTouched = false;

const SIZE_OVERLAP = 0.6;

function sizeFromSpacing(spacing, heightPx, dpr) {
  const uSize =
    (spacing *
      (heightPx || innerHeight) *
      (dpr || devicePixelRatio || 1) *
      SIZE_OVERLAP) /
    250;
  return THREE.MathUtils.clamp(uSize, 0.002, 10);
}

function applyAutoSize(geometry) {
  lastSpacing = (geometry.userData && geometry.userData.spacing) || 0.05;
  document.getElementById("size").value = String(sizeFromSpacing(lastSpacing));
  sizeTouched = false;
}

function processGeometry(geometry, name) {
  applyAutoSize(geometry);
  if (points) {
    scene.remove(points);
    points.geometry.dispose();
    points.material.dispose();
  }
  points = new THREE.Points(geometry, makeMaterial());
  scene.add(points);
  fitTo(geometry);
  scheduleIdleSpin();
  statsEl.textContent =
    geometry.attributes.position.count.toLocaleString() +
    " pts · " +
    (name || "");
}

function loadPly(blob, name) {
  blob.arrayBuffer().then((buf) => {
    return decodeOnWorker(buf)
      .then((geometry) => {
        geometry.userData.name = name;
        processGeometry(geometry, name);
      })
      .catch((err) => {
        statsEl.textContent = "failed to load ply: " + err.message;
      });
  });
}

// ---- scans directory + gallery ----

const scanCache = new Map();
const inflight = new Map();
let activeName = null;

function scanUrl(name) {
  return SCAN_DIR + encodeURIComponent(name);
}

function loadScanGeometry(name) {
  if (scanCache.has(name)) return Promise.resolve(scanCache.get(name));
  if (inflight.has(name)) return inflight.get(name);
  const p = fetch(scanUrl(name))
    .then((r) => {
      if (!r.ok) throw new Error(r.status);
      return r.arrayBuffer();
    })
    .then((buf) => decodeOnWorker(buf))
    .then((geometry) => {
      geometry.name = name;
      scanCache.set(name, geometry);
      return geometry;
    })
    .finally(() => inflight.delete(name));
  inflight.set(name, p);
  return p;
}

async function discoverScans() {
  try {
    const res = await fetch(SCAN_DIR);
    if (!res.ok) throw new Error(res.status);
    const html = await res.text();
    const names = [...html.matchAll(/href="([^"]+\.ply)"/gi)]
      .map((m) => m[1])
      .filter((href) => !href.startsWith("?") && !href.startsWith("#"))
      .map((href) => decodeURIComponent(href.replace(/.*\//, "")));
    return [...new Set(names)].sort();
  } catch {
    return [];
  }
}

// ---- thumbnail rendering ----

let thumbGenToken = 0;

const thumbRenderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
thumbRenderer.setSize(128, 128);
thumbRenderer.setPixelRatio(1);

const thumbScene = new THREE.Scene();
thumbScene.background = new THREE.Color(0x0d0f13);
const thumbCamera = new THREE.PerspectiveCamera(55, 1, 1e-4, 1000);
let thumbPoints = null;

function thumbGeometry(data) {
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(data.position, 3));
  g.setAttribute("viewZ", new THREE.BufferAttribute(data.viewZ, 3));
  g.setAttribute("aColor", new THREE.BufferAttribute(data.color, 3));
  return g;
}

function generateThumb(geometry) {
  if (!geometry.userData.thumb) return "";
  if (thumbPoints) {
    thumbScene.remove(thumbPoints);
    thumbPoints.geometry.dispose();
    thumbPoints.material.dispose();
  }
  const material = makeMaterial();
  material.uniforms.uSize.value = sizeFromSpacing(
    geometry.userData.spacing,
    128,
    1,
  );
  const tg = thumbGeometry(geometry.userData.thumb);
  thumbPoints = new THREE.Points(tg, material);
  thumbScene.add(thumbPoints);
  frameObject(tg, thumbCamera);
  thumbRenderer.render(thumbScene, thumbCamera);
  return thumbRenderer.domElement.toDataURL();
}

function buildGallery(names) {
  const strip = document.getElementById("gallery-strip");
  const cards = new Map();
  names.forEach((name) => {
    const item = document.createElement("button");
    item.type = "button";
    item.className = "gallery-item";
    item.title = name;
    const img = document.createElement("img");
    img.className = "thumb";
    img.alt = name;
    const label = document.createElement("span");
    label.textContent = name;
    item.append(img, label);
    item.addEventListener("click", () => selectScan(name, item));
    strip.appendChild(item);
    cards.set(name, { item, img });
  });
  return cards;
}

function selectScan(name, item) {
  if (name === activeName && points) return Promise.resolve(points.geometry);
  activeName = name;
  thumbGenToken++; // pause background thumbnail work, prioritize the click
  document.querySelectorAll(".gallery-item").forEach((el) => {
    el.classList.toggle("active", el === item || el.title === name);
  });
  return loadScanGeometry(name)
    .then((geometry) => processGeometry(geometry, name))
    .then(() => restartThumbs())
    .catch((err) => {
      statsEl.textContent = "failed to load " + name + ": " + err.message;
    });
}

async function runThumbLoop(names, cards) {
  const token = ++thumbGenToken;
  const halted = () => token !== thumbGenToken || document.hidden;
  for (const name of names) {
    const card = cards.get(name);
    if (!card || card.img.dataset.done) continue;
    if (halted()) return;
    await new Promise((r) => requestAnimationFrame(r));
    if (halted()) return;
    let ok = false;
    try {
      const geometry = await loadScanGeometry(name);
      if (halted()) return;
      card.img.src = generateThumb(geometry);
      ok = true;
    } catch {
      ok = false;
    }
    card.img.dataset.done = ok ? "1" : "fail";
  }
  if (!halted()) {
    const pending = names.filter((n) => {
      const c = cards.get(n);
      return c && !c.img.dataset.done;
    });
    if (pending.length) runThumbLoop(pending, cards);
  }
}

function restartThumbs() {
  if (document.hidden) return;
  const pending = galleryNames.filter((n) => {
    const c = galleryCards.get(n);
    return c && !c.img.dataset.done;
  });
  if (pending.length) runThumbLoop(pending, galleryCards);
}

document.addEventListener("visibilitychange", () => {
  if (!document.hidden) restartThumbs();
});

let galleryNames = [];
let galleryCards = new Map();

async function init() {
  const names = await discoverScans();
  if (!names.length) {
    document.body.classList.add("no-gallery");
    return;
  }
  galleryNames = names;
  galleryCards = buildGallery(names);
  syncGalleryAnchor();
  const primary = names.includes("scan.ply") ? "scan.ply" : names[0];
  const primaryCard = galleryCards.get(primary);
  if (primaryCard) primaryCard.item.classList.add("active");
  await selectScan(primary, primaryCard ? primaryCard.item : null);
  restartThumbs();
}

function syncGalleryAnchor() {
  const g = document.getElementById("gallery");
  const root = document.documentElement;
  root.style.setProperty("--gallery-h", (g ? g.offsetHeight : 0) + "px");
}

// ---- UI wiring ----

const menuBtn = document.getElementById("menu");
if (menuBtn) {
  const updateMenu = () => {
    const open = document.body.classList.toggle("controls-open");
    menuBtn.classList.toggle("open", open);
    menuBtn.setAttribute("aria-expanded", open ? "true" : "false");
    menuBtn.innerHTML = open ? "&#10005;" : "&#9776;";
  };
  menuBtn.addEventListener("click", updateMenu);
}
document.getElementById("size").addEventListener("input", (e) => {
  sizeTouched = true;
  if (points) points.material.uniforms.uSize.value = parseFloat(e.target.value);
});

document.addEventListener("keydown", (e) => {
  if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
  const el = document.activeElement;
  if (
    el &&
    (el.tagName === "INPUT" ||
      el.tagName === "SELECT" ||
      el.tagName === "TEXTAREA")
  )
    return;
  const idx = galleryNames.indexOf(activeName);
  if (idx < 0) return;
  const step = e.key === "ArrowRight" ? 1 : -1;
  const name =
    galleryNames[(idx + step + galleryNames.length) % galleryNames.length];
  const card = galleryCards.get(name);
  if (!card) return;
  e.preventDefault();
  selectScan(name, card.item);
  card.item.scrollIntoView({
    behavior: "smooth",
    block: "nearest",
    inline: "center",
  });
});

const drop = document.getElementById("drop");
let dragDepth = 0;
window.addEventListener("dragenter", (e) => {
  e.preventDefault();
  dragDepth++;
  drop.className = "over";
});
window.addEventListener("dragleave", (e) => {
  e.preventDefault();
  if (--dragDepth <= 0) drop.className = "";
});
window.addEventListener("dragover", (e) => e.preventDefault());
window.addEventListener("drop", (e) => {
  e.preventDefault();
  dragDepth = 0;
  drop.className = "";
  const file = [...e.dataTransfer.files].find((f) =>
    f.name.toLowerCase().endsWith(".ply"),
  );
  if (file) loadPly(file, file.name);
});

function resize() {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
  syncGalleryAnchor();
  if (lastSpacing && !sizeTouched) {
    document.getElementById("size").value = String(
      sizeFromSpacing(lastSpacing),
    );
    if (points)
      points.material.uniforms.uSize.value = parseFloat(
        document.getElementById("size").value,
      );
  }
}
window.addEventListener("resize", resize);

renderer.setAnimationLoop(() => {
  controls.update();
  renderer.render(scene, camera);
});

init();
