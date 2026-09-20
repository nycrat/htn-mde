/**
 * PLY decode worker (mirrors three@0.160 PLYLoader semantics: colors are
 * divided by 255 and converted sRGB -> linear working space). Runs entirely
 * off the main thread so scene rendering never blocks while a scan is
 * parsed / sized / thumbnailed.
 *
 * Message in:  { id, buf }                       buf = raw PLY ArrayBuffer
 * Message out: { id, position, viewZ, aColor, spacing, thumb } + transferables
 */
globalThis.onmessage = (e) => {
  const { id, buf } = e.data;
  try {
    const result = decode(buf);
    globalThis.postMessage({ id, ok: true, ...result }, [
      result.position.buffer,
      result.viewZ.buffer,
      result.aColor.buffer,
      result.thumb.position.buffer,
      result.thumb.viewZ.buffer,
      result.thumb.color.buffer,
    ]);
  } catch (err) {
    globalThis.postMessage({ id, ok: false, error: String(err.message || err) });
  }
};

const SRGB_LUT = new Float32Array(256);
for (let i = 0; i < 256; i++) {
  const c = i / 255;
  SRGB_LUT[i] = c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}

const TYPE_BYTES = {
  int8: 1, char: 1, uint8: 1, uchar: 1,
  int16: 2, short: 2, uint16: 2, ushort: 2,
  int32: 4, int: 4, uint32: 4, uint: 4,
  float32: 4, float: 4, float64: 8, double: 8,
};

const TYPE_GETTER = {
  int8: "getInt8", char: "getInt8",
  uint8: "getUint8", uchar: "getUint8",
  int16: "getInt16", short: "getInt16",
  uint16: "getUint16", ushort: "getUint16",
  int32: "getInt32", int: "getInt32",
  uint32: "getUint32", uint: "getUint32",
  float32: "getFloat32", float: "getFloat32",
  float64: "getFloat64", double: "getFloat64",
};

function decode(buf) {
  const bytes = new Uint8Array(buf);
  const view = new DataView(buf);
  let headerEnd = -1;
  for (let i = 0; i < bytes.length; i++) {
    const s = String.fromCharCode(bytes[i]);
    let match = true;
    const word = "end_header";
    if (i + word.length - 1 >= bytes.length) break;
    for (let j = 0; j < word.length; j++) {
      if (String.fromCharCode(bytes[i + j]) !== word[j]) { match = false; break; }
    }
    if (match) {
      // advance past "end_header\n"
      let k = i + word.length;
      while (k < bytes.length && (bytes[k] === 13 || bytes[k] === 10)) k++;
      headerEnd = k;
      break;
    }
  }
  if (headerEnd < 0) throw new Error("no end_header found");

  const headerText = new TextDecoder("latin1").decode(bytes.subarray(0, headerEnd));
  const formatLine = headerText.match(/^format\s+(\S+)/m);
  if (!formatLine) throw new Error("no format line");
  const littleEndian = formatLine[1] === "binary_little_endian";
  if (!littleEndian && formatLine[1] !== "binary_big_endian") {
    throw new Error("ascii PLY not supported");
  }
  const vertexCountM = headerText.match(/^element vertex (\d+)/m);
  if (!vertexCountM) throw new Error("no vertex element");
  const n = parseInt(vertexCountM[1], 10);
  if (n <= 0 || !Number.isFinite(n)) throw new Error("bad vertex count");

  const props = [];
  const propRe = /^property\s+(\S+)\s+(\S+)/gm;
  let m;
  while ((m = propRe.exec(headerText)) !== null) {
    props.push({ type: m[1], name: m[2] });
  }
  if (!props.length) throw new Error("no properties on vertex element");

  // Only vertex properties counted from the header: find x/y/z and rgb.
  const find = (names) => {
    for (const name of names) {
      const i = props.findIndex((p) => p.name === name);
      if (i >= 0) return i;
    }
    return -1;
  };
  const iX = find(["x", "px", "posx"]);
  const iY = find(["y", "py", "posy"]);
  const iZ = find(["z", "pz", "posz"]);
  const iR = find(["red", "diffuse_red", "r", "diffuse_r"]);
  const iG = find(["green", "diffuse_green", "g", "diffuse_g"]);
  const iB = find(["blue", "diffuse_blue", "b", "diffuse_b"]);
  if (iX < 0 || iY < 0 || iZ < 0) throw new Error("vertex x/y/z not found");

  let rowSize = 0;
  const offsets = [];
  for (const p of props) {
    const size = TYPE_BYTES[p.type];
    if (!size) throw new Error("unsupported property type: " + p.type);
    offsets.push(rowSize);
    rowSize += size;
  }

  const read = (at, getter, le) => view[getter](at, le);
  const position = new Float32Array(n * 3);
  const aColor = new Float32Array(n * 3);
  if (iR >= 0 && iG >= 0 && iB >= 0) {
    for (let r = 0; r < n; r++) {
      const base = headerEnd + r * rowSize;
      position[r * 3] = read(base + offsets[iX], TYPE_GETTER[props[iX].type], littleEndian);
      position[r * 3 + 1] = read(base + offsets[iY], TYPE_GETTER[props[iY].type], littleEndian);
      position[r * 3 + 2] = read(base + offsets[iZ], TYPE_GETTER[props[iZ].type], littleEndian);
      aColor[r * 3] = SRGB_LUT[read(base + offsets[iR], "getUint8", false) & 0xff];
      aColor[r * 3 + 1] = SRGB_LUT[read(base + offsets[iG], "getUint8", false) & 0xff];
      aColor[r * 3 + 2] = SRGB_LUT[read(base + offsets[iB], "getUint8", false) & 0xff];
    }
  } else {
    for (let r = 0; r < n; r++) {
      const base = headerEnd + r * rowSize;
      position[r * 3] = read(base + offsets[iX], TYPE_GETTER[props[iX].type], littleEndian);
      position[r * 3 + 1] = read(base + offsets[iY], TYPE_GETTER[props[iY].type], littleEndian);
      position[r * 3 + 2] = read(base + offsets[iZ], TYPE_GETTER[props[iZ].type], littleEndian);
      aColor[r * 3] = 1;
      aColor[r * 3 + 1] = 1;
      aColor[r * 3 + 2] = 1;
    }
  }

  const spacing = estimateSpacing(position);
  return {
    position,
    viewZ: computeViewZ(position),
    aColor,
    spacing,
    thumb: sampleThumb(position, aColor),
  };
}

function computeViewZ(position) {
  const n = position.length / 3;
  let lo = Infinity, hi = -Infinity;
  const viewZ = new Float32Array(position.length);
  for (let i = 0; i < n; i++) {
    const z = position[i * 3 + 2];
    viewZ[i * 3 + 2] = z;
    if (z < lo) lo = z;
    if (z > hi) hi = z;
  }
  const span = Math.max(hi - lo, 1e-9);
  for (let i = 0; i < n; i++) {
    viewZ[i * 3] = 0;
    viewZ[i * 3 + 1] = 0;
    viewZ[i * 3 + 2] = (viewZ[i * 3 + 2] - lo) / span;
  }
  return viewZ;
}

/** Mean nearest-neighbor spacing, same result as viewer's estimateSpacing. */
function estimateSpacing(position) {
  const n = position.length / 3;
  if (n < 2) return 1;
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (let i = 0; i < n; i++) {
    const x = position[i * 3], y = position[i * 3 + 1], z = position[i * 3 + 2];
    if (x < minX) minX = x; if (x > maxX) maxX = x;
    if (y < minY) minY = y; if (y > maxY) maxY = y;
    if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
  }
  const cell = Math.cbrt(Math.max((maxX - minX) * (maxY - minY) * (maxZ - minZ), 1e-9) / n);
  const inv = 1 / Math.max(cell, 1e-9);
  const MAX_SAMPLE = 20000;
  const stride = Math.max(1, Math.round(n / MAX_SAMPLE));
  const MAX_PER_CELL = 8;
  const grid = new Map();
  const key = (a, b, c) => ((a * 73856093) ^ (b * 19349663) ^ (c * 83492791)) >>> 0;

  const sampledIdx = [];
  const bx = [], by = [], bz = [];
  for (let i = 0; i < n; i += stride) {
    const gx = Math.floor(position[i * 3] * inv);
    const gy = Math.floor(position[i * 3 + 1] * inv);
    const gz = Math.floor(position[i * 3 + 2] * inv);
    const k = key(gx, gy, gz);
    let bucket = grid.get(k);
    if (!bucket) { bucket = []; grid.set(k, bucket); }
    if (bucket.length < MAX_PER_CELL) bucket.push(i);
    sampledIdx.push(i);
    bx.push(gx); by.push(gy); bz.push(gz);
  }

  let sum = 0, count = 0;
  for (let s = 0; s < sampledIdx.length; s++) {
    const i = sampledIdx[s];
    const px = position[i * 3], py = position[i * 3 + 1], pz = position[i * 3 + 2];
    let best = Infinity;
    for (let dx = -1; dx <= 1; dx++) {
      for (let dy = -1; dy <= 1; dy++) {
        for (let dz = -1; dz <= 1; dz++) {
          const bucket = grid.get(key(bx[s] + dx, by[s] + dy, bz[s] + dz));
          if (!bucket) continue;
          for (let t = 0; t < bucket.length; t++) {
            const j = bucket[t];
            if (j === i) continue;
            const dxv = px - position[j * 3];
            const dyv = py - position[j * 3 + 1];
            const dzv = pz - position[j * 3 + 2];
            const d2 = dxv * dxv + dyv * dyv + dzv * dzv;
            if (d2 < best) best = d2;
          }
        }
      }
    }
    if (best < Infinity) { sum += Math.sqrt(best); count++; }
  }
  return count ? sum / count : cell;
}

/** ~120k-point evenly strided subset + its viewZ for cheap thumbnail rendering. */
function sampleThumb(position, aColor) {
  const n = position.length / 3;
  const MAX = 120000;
  const step = Math.max(1, Math.ceil(n / MAX));
  const out = Math.ceil(n / step);
  const pos = new Float32Array(out * 3);
  const col = new Float32Array(out * 3);
  const vz = new Float32Array(out * 3);
  for (let i = 0, o = 0; i < n; i += step, o++) {
    pos[o * 3] = position[i * 3];
    pos[o * 3 + 1] = position[i * 3 + 1];
    pos[o * 3 + 2] = position[i * 3 + 2];
    vz[o * 3 + 2] = position[i * 3 + 2];
    col[o * 3] = aColor[i * 3];
    col[o * 3 + 1] = aColor[i * 3 + 1];
    col[o * 3 + 2] = aColor[i * 3 + 2];
  }
  let lo = Infinity, hi = -Infinity;
  for (let o = 0; o < out; o++) {
    const z = vz[o * 3 + 2];
    if (z < lo) lo = z;
    if (z > hi) hi = z;
  }
  const span = Math.max(hi - lo, 1e-9);
  for (let o = 0; o < out; o++) {
    vz[o * 3] = 0;
    vz[o * 3 + 1] = 0;
    vz[o * 3 + 2] = (vz[o * 3 + 2] - lo) / span;
  }
  return { position: pos, viewZ: vz, color: col };
}