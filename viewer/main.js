import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { PLYLoader } from "three/addons/loaders/PLYLoader.js";

const SCAN_DIR = "../scans/";

const app = document.getElementById("app");
const statsEl = document.getElementById("stats");

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0d0f13);

const camera = new THREE.PerspectiveCamera(55, innerWidth / innerHeight, 0.001, 1000);
camera.position.set(1.6, 1.0, 1.6);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(devicePixelRatio);
renderer.setSize(innerWidth, innerHeight);
app.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;

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
  gl_PointSize = uSize * (240.0 / max(0.01, -mv.z));
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

function decorateGeometry(geometry) {
  if (geometry.hasAttribute("viewZ") && geometry.hasAttribute("aColor")) return geometry;
  geometry.computeBoundingBox();
  const box = geometry.boundingBox;
  const lo = box.min.z, hi = box.max.z, span = Math.max(hi - lo, 1e-9);
  const pos = geometry.attributes.position;
  const n = pos.count;
  const viewZ = pos.array.slice();
  for (let i = 0; i < n; i++) viewZ[i * 3 + 2] = (pos.getZ(i) - lo) / span;
  geometry.setAttribute("viewZ", new THREE.BufferAttribute(viewZ, 3));

  // Vertex colors: PLYLoader emits a 3- or 4-component normalized `color`
  // attribute (or none). Fold it into a stable vec3 `aColor`.
  const hasColor = geometry.hasAttribute("color") && geometry.getAttribute("color").array.length >= n * 3;
  const aColor = new Float32Array(n * 3);
  if (hasColor) {
    const src = geometry.getAttribute("color").array;
    const comps = Math.floor(geometry.getAttribute("color").array.length / n);
    for (let i = 0; i < n; i++) {
      aColor[i * 3] = src[i * comps];
      aColor[i * 3 + 1] = src[i * comps + 1];
      aColor[i * 3 + 2] = src[i * comps + 2];
    }
  } else {
    aColor.fill(1.0); // light gray fallback
  }
  geometry.setAttribute("aColor", new THREE.BufferAttribute(aColor, 3));
  geometry.deleteAttribute("color");
  return geometry;
}

function makeMaterial() {
  return new THREE.ShaderMaterial({
    uniforms: {
      uSize: { value: parseFloat(document.getElementById("size").value) },
      uNear: { value: parseFloat(document.getElementById("near").value) },
      uFar: { value: parseFloat(document.getElementById("far").value) },
    },
    vertexShader: VERT,
    fragmentShader: FRAG,
  });
}

function frameObject(geometry, cam) {
  geometry.computeBoundingBox();
  const center = geometry.boundingBox.getCenter(new THREE.Vector3());
  const radius = geometry.boundingBox.getBoundingSphere(new THREE.Sphere()).radius;
  const dist = Math.max(radius, 1e-3) * 2.4;
  cam.near = Math.max(dist / 20000, 1e-6);
  cam.far = dist * 200;
  cam.position.copy(center).add(new THREE.Vector3(0.7, 0.5, 0.7).normalize().multiplyScalar(dist));
  cam.lookAt(center);
  cam.updateProjectionMatrix();
}

function fitTo(geometry) {
  geometry.computeBoundingBox();
  const center = geometry.boundingBox.getCenter(new THREE.Vector3());
  controls.target.copy(center);
  frameObject(geometry, camera);
  controls.update();
}

function applyFilters() {
  if (!points) return;
  points.material.uniforms.uNear.value = parseFloat(document.getElementById("near").value);
  points.material.uniforms.uFar.value = parseFloat(document.getElementById("far").value);
}

function processGeometry(geometry, name) {
  decorateGeometry(geometry);
  if (points) { scene.remove(points); points.geometry.dispose(); points.material.dispose(); }
  points = new THREE.Points(geometry, makeMaterial());
  scene.add(points);
  fitTo(geometry);
  statsEl.textContent = geometry.attributes.position.count.toLocaleString() + " pts · " + (name || "");
}

function loadPly(blob, name) {
  const loader = new PLYLoader();
  blob.arrayBuffer().then((buf) => {
    processGeometry(loader.parse(buf), name);
  }).catch((err) => {
    statsEl.textContent = "failed to load ply: " + err.message;
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
    .then((r) => { if (!r.ok) throw new Error(r.status); return r.blob(); })
    .then((b) => b.arrayBuffer())
    .then((buf) => {
      const g = decorateGeometry(new PLYLoader().parse(buf));
      scanCache.set(name, g);
      return g;
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

const thumbRenderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
thumbRenderer.setSize(128, 128);
thumbRenderer.setPixelRatio(1);

const thumbScene = new THREE.Scene();
thumbScene.background = new THREE.Color(0x0d0f13);
const thumbCamera = new THREE.PerspectiveCamera(55, 1, 1e-4, 1000);
let thumbPoints = null;

function generateThumb(geometry) {
  if (thumbPoints) { thumbScene.remove(thumbPoints); thumbPoints.material.dispose(); }
  thumbPoints = new THREE.Points(geometry, makeMaterial());
  thumbScene.add(thumbPoints);
  frameObject(geometry, thumbCamera);
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
  document.querySelectorAll(".gallery-item").forEach((el) => {
    el.classList.toggle("active", el === item || el.title === name);
  });
  return loadScanGeometry(name)
    .then((geometry) => processGeometry(geometry, name))
    .catch((err) => { statsEl.textContent = "failed to load " + name + ": " + err.message; });
}

async function generateThumbs(names, cards) {
  for (const name of names) {
    const card = cards.get(name);
    if (!card) continue;
    try {
      const geometry = await loadScanGeometry(name);
      card.img.src = generateThumb(geometry);
    } catch {
      card.img.alt = "failed to load";
    }
    await new Promise((r) => setTimeout(r, 0));
  }
}

async function init() {
  const names = await discoverScans();
  if (!names.length) {
    document.body.classList.add("no-gallery");
    return;
  }
  const cards = buildGallery(names);
  const primary = names.includes("scan.ply") ? "scan.ply" : names[0];
  const primaryCard = cards.get(primary);
  if (primaryCard) primaryCard.item.classList.add("active");
  selectScan(primary, primaryCard ? primaryCard.item : null);
  generateThumbs(names, cards);
}

// ---- UI wiring ----

document.getElementById("fit").addEventListener("click", () => points && fitTo(points.geometry));
document.getElementById("bg").addEventListener("click", () => {
  const dark = scene.background.getHex() === 0x0d0f13;
  scene.background = new THREE.Color(dark ? 0xeceff4 : 0x0d0f13);
});
document.getElementById("size").addEventListener("input", (e) => {
  if (points) points.material.uniforms.uSize.value = parseFloat(e.target.value);
});
document.getElementById("near").addEventListener("input", applyFilters);
document.getElementById("far").addEventListener("input", applyFilters);

const drop = document.getElementById("drop");
let dragDepth = 0;
window.addEventListener("dragenter", (e) => { e.preventDefault(); dragDepth++; drop.className = "over"; });
window.addEventListener("dragleave", (e) => { e.preventDefault(); if (--dragDepth <= 0) drop.className = ""; });
window.addEventListener("dragover", (e) => e.preventDefault());
window.addEventListener("drop", (e) => {
  e.preventDefault(); dragDepth = 0; drop.className = "";
  const file = [...e.dataTransfer.files].find((f) => f.name.toLowerCase().endsWith(".ply"));
  if (file) loadPly(file, file.name);
});

function resize() {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
}
window.addEventListener("resize", resize);

renderer.setAnimationLoop(() => {
  controls.update();
  renderer.render(scene, camera);
});

init();