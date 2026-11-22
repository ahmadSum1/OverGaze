// Motion files made outside the browser (tools/motion_opencv.py, OpenCV's DIS optical flow): a NumPy .npz archive
// with t_a, t_b (seconds), flow (fields × height × width × 2, pixels moved from the frame at t_a to the frame at t_b)
// and optionally wrap, fps and method. Read into the same store "Analyse motion" fills.
import { FlowStore } from './flow.js';
import { parseNpyHeader } from './reference.js';

/** The arrays of an .npz file (a ZIP of .npy files, stored or deflated): { name: { shape, descr, data } }. */
export async function readNpz(file) {
  const size = file.size, tailLen = Math.min(size, 65536 + 22);
  const tail = new DataView(await file.slice(size - tailLen, size).arrayBuffer());
  let eocd = -1;
  for (let i = tailLen - 22; i >= 0; i--) if (tail.getUint32(i, true) === 0x06054b50) { eocd = i; break; }
  if (eocd < 0) throw new Error(`${file.name} is not an .npz (ZIP) file`);
  let count = tail.getUint16(eocd + 10, true), cdSize = tail.getUint32(eocd + 12, true), cdStart = tail.getUint32(eocd + 16, true);
  if (cdStart === 0xFFFFFFFF || count === 0xFFFF) {
    // ZIP64: the end record locator sits just before the ordinary end record
    const loc = eocd - 20;
    if (loc < 0 || tail.getUint32(loc, true) !== 0x07064b50) throw new Error(`${file.name}: damaged ZIP64 archive`);
    const recAt = Number(tail.getBigUint64(loc + 8, true));
    const rec = new DataView(await file.slice(recAt, recAt + 56).arrayBuffer());
    count = Number(rec.getBigUint64(32, true)); cdSize = Number(rec.getBigUint64(40, true)); cdStart = Number(rec.getBigUint64(48, true));
  }
  const cd = new DataView(await file.slice(cdStart, cdStart + cdSize).arrayBuffer());
  const arrays = {};
  for (let p = 0, k = 0; k < count && p + 46 <= cd.byteLength; k++) {
    if (cd.getUint32(p, true) !== 0x02014b50) break;
    const method = cd.getUint16(p + 10, true), nameLen = cd.getUint16(p + 28, true), extraLen = cd.getUint16(p + 30, true), commentLen = cd.getUint16(p + 32, true);
    let compSize = cd.getUint32(p + 20, true), rawSize = cd.getUint32(p + 24, true), offset = cd.getUint32(p + 42, true);
    const name = new TextDecoder().decode(new Uint8Array(cd.buffer, p + 46, nameLen));
    // ZIP64 extra field: the 64-bit values, in this order, for the 32-bit fields that are 0xFFFFFFFF
    for (let e = p + 46 + nameLen; e + 4 <= p + 46 + nameLen + extraLen;) {
      const id = cd.getUint16(e, true), len = cd.getUint16(e + 2, true);
      if (id === 1) {
        let q = e + 4;
        if (rawSize === 0xFFFFFFFF) { rawSize = Number(cd.getBigUint64(q, true)); q += 8; }
        if (compSize === 0xFFFFFFFF) { compSize = Number(cd.getBigUint64(q, true)); q += 8; }
        if (offset === 0xFFFFFFFF) { offset = Number(cd.getBigUint64(q, true)); }
      }
      e += 4 + len;
    }
    p += 46 + nameLen + extraLen + commentLen;
    if (!name.endsWith('.npy')) continue;
    const local = new DataView(await file.slice(offset, offset + 30).arrayBuffer());
    const dataAt = offset + 30 + local.getUint16(26, true) + local.getUint16(28, true);
    let bytes = await file.slice(dataAt, dataAt + compSize).arrayBuffer();
    if (method === 8) bytes = await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('deflate-raw'))).arrayBuffer();
    else if (method !== 0) throw new Error(`${file.name}: ${name} uses an unsupported compression (${method})`);
    if (bytes.byteLength !== rawSize) throw new Error(`${file.name}: ${name} is damaged`);
    const head = parseNpyHeader(bytes);
    arrays[name.slice(0, -4)] = { shape: head.shape, descr: head.descr, fortran: head.fortran, data: bytes.slice(head.offset) };
  }
  return arrays;
}

const READERS = {
  '<f2': [2, (dv, o) => half(dv.getUint16(o, true))], '<f4': [4, (dv, o) => dv.getFloat32(o, true)], '<f8': [8, (dv, o) => dv.getFloat64(o, true)],
  '|b1': [1, (dv, o) => dv.getUint8(o)], '|u1': [1, (dv, o) => dv.getUint8(o)]
};
function half(h) {
  const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
  if (e === 0) return s * m * 2 ** -24;
  if (e === 31) return m ? NaN : s * Infinity;
  return s * (1 + m / 1024) * 2 ** (e - 15);
}
function numbers(a, what) {
  const r = READERS[a?.descr];
  if (!r) throw new Error(`${what}: data type ${a?.descr} is not supported (float16, float32 or float64)`);
  const dv = new DataView(a.data), n = a.data.byteLength / r[0], out = new Float64Array(n);
  for (let i = 0; i < n; i++) out[i] = r[1](dv, i * r[0]);
  return out;
}
function text(a) {
  if (!a) return '';
  const m = /^<U(\d+)$/.exec(a.descr);
  if (m) { const u = new Uint32Array(a.data.slice(0, 4 * Number(m[1]))); return String.fromCodePoint(...[...u].filter(c => c)); }
  if (/^\|S\d+$/.test(a.descr)) return new TextDecoder().decode(a.data).replace(/\0+$/, '');
  return '';
}

/** Build a FlowStore from an opened motion file; returns { store, method, fps, fields }. */
export function motionFromArrays(arrays, name = 'motion file') {
  const { t_a: ta, t_b: tb, flow } = arrays;
  if (!ta || !tb || !flow) throw new Error(`${name}: expected the arrays t_a, t_b and flow (see tools/motion_opencv.py)`);
  if (flow.fortran) throw new Error(`${name}: Fortran-ordered arrays are not supported`);
  const [N, H, W, two] = flow.shape;
  if (flow.shape.length !== 4 || two !== 2) throw new Error(`${name}: flow has shape (${flow.shape.join(', ')}); expected (fields, height, width, 2)`);
  const tA = numbers(ta, name), tB = numbers(tb, name);
  if (tA.length !== N || tB.length !== N) throw new Error(`${name}: ${N} flow fields but ${tA.length} / ${tB.length} times`);
  const wrap = arrays.wrap ? numbers(arrays.wrap, name)[0] !== 0 : true;
  const fps = arrays.fps ? numbers(arrays.fps, name)[0] : null;
  const r = READERS[flow.descr];
  if (!r || flow.descr === '|b1' || flow.descr === '|u1') throw new Error(`${name}: flow data type ${flow.descr} is not supported (float16, float32 or float64)`);
  const dv = new DataView(flow.data), [size, read] = r, n = W * H;
  const store = new FlowStore(W, H, { wrap });
  const u = new Float32Array(n), v = new Float32Array(n);
  for (let k = 0; k < N; k++) {
    const base = k * n * 2 * size;
    for (let i = 0; i < n; i++) { u[i] = read(dv, base + 2 * i * size); v[i] = read(dv, base + (2 * i + 1) * size); }
    store.add(tA[k], tB[k], u, v);
  }
  return { store, method: text(arrays.method), fps, fields: N };
}

export async function openMotionFile(file) {
  return motionFromArrays(await readNpz(file), file.name);
}
