// Reference (ground-truth) saliency maps to compare with: Salient360 .bin sequences, .npy arrays, a single image,
// or one image per frame. Frames are read on demand (File.slice), so multi-gigabyte files never sit in memory.

const BIN_NAME = /_(\d+)x(\d+)x(\d+)_32b\.bin$/i;
const IMAGE = /\.(png|jpe?g|webp|bmp)$/i;

/** Open reference files. Returns { kind, name, N, W, H, frame(k) → Promise<Float32Array H·W> }; N = 1 for a static map. */
export async function openReference(files) {
  files = [...files];
  if (!files.length) throw new Error('No file chosen.');
  if (files.length === 1) {
    const f = files[0];
    if (/\.bin$/i.test(f.name)) return openBin(f);
    if (/\.npy$/i.test(f.name)) return openNpy(f);
    if (IMAGE.test(f.name)) return openImages([f]);
    throw new Error(`${f.name}: expected a .bin (Salient360 <name>_<W>x<H>x<frames>_32b.bin), a .npy array or an image.`);
  }
  const images = files.filter(f => IMAGE.test(f.name));
  if (!images.length) throw new Error('Several files were chosen but none is an image; choose one .bin or .npy file, or one image per frame.');
  return openImages(images);
}

function openBin(f) {
  const m = BIN_NAME.exec(f.name);
  if (!m) throw new Error(`${f.name}: the size is read from the name, which must end in _<width>x<height>x<frames>_32b.bin.`);
  const W = Number(m[1]), H = Number(m[2]), N = Number(m[3]), bytes = W * H * 4;
  if (f.size < bytes * N) throw new Error(`${f.name}: ${f.size} bytes, but ${N} frames of ${W} × ${H} float32 need ${bytes * N}.`);
  return {
    kind: 'bin', name: f.name, N, W, H, file: f,
    // Salient360: bin frame 0 is empty and bin frame k belongs to decoded video frame k − 1.
    suggestedOffset: 1,
    async frame(k) { return new Float32Array(await f.slice(k * bytes, (k + 1) * bytes).arrayBuffer()); }
  };
}

/** Parse the header of a .npy file (versions 1–3). */
export function parseNpyHeader(buf) {
  const u8 = new Uint8Array(buf);
  if (u8[0] !== 0x93 || String.fromCharCode(...u8.slice(1, 6)) !== 'NUMPY') throw new Error('not a .npy file');
  const major = u8[6], dv = new DataView(buf);
  const len = major === 1 ? dv.getUint16(8, true) : dv.getUint32(8, true), start = major === 1 ? 10 : 12;
  const text = new TextDecoder('latin1').decode(u8.slice(start, start + len));
  const descr = /'descr'\s*:\s*'([^']+)'/.exec(text)?.[1];
  const fortran = /'fortran_order'\s*:\s*True/.test(text);
  const shape = (/'shape'\s*:\s*\(([^)]*)\)/.exec(text)?.[1] || '').split(',').map(x => x.trim()).filter(Boolean).map(Number);
  return { descr, fortran, shape, offset: start + len };
}

const DTYPES = { '<f4': [4, (dv, o) => dv.getFloat32(o, true)], '<f8': [8, (dv, o) => dv.getFloat64(o, true)], '<f2': [2, (dv, o) => half(dv.getUint16(o, true))],
  '|u1': [1, (dv, o) => dv.getUint8(o)], '<u1': [1, (dv, o) => dv.getUint8(o)], '<u2': [2, (dv, o) => dv.getUint16(o, true)], '|b1': [1, (dv, o) => dv.getUint8(o)] };

function half(h) {
  const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
  if (e === 0) return s * m * 2 ** -24;
  if (e === 31) return m ? NaN : s * Infinity;
  return s * (1 + m / 1024) * 2 ** (e - 15);
}

async function openNpy(f) {
  const head = parseNpyHeader(await f.slice(0, 4096).arrayBuffer());
  const dt = DTYPES[head.descr];
  if (!dt) throw new Error(`${f.name}: data type ${head.descr} is not supported (float16/32/64, uint8, uint16).`);
  if (head.fortran) throw new Error(`${f.name}: Fortran-ordered arrays are not supported.`);
  const shape = head.shape.filter(d => d !== 1);
  if (shape.length < 2 || shape.length > 3) throw new Error(`${f.name}: shape (${head.shape.join(', ')}); expected (frames, height, width) or (height, width).`);
  const [N, H, W] = shape.length === 3 ? shape : [1, ...shape];
  const [size, read] = dt, bytes = H * W * size;
  return {
    kind: 'npy', name: f.name, N, W, H, file: f, suggestedOffset: 0,
    async frame(k) {
      const buf = await f.slice(head.offset + k * bytes, head.offset + (k + 1) * bytes).arrayBuffer();
      const dv = new DataView(buf), out = new Float32Array(H * W);
      for (let i = 0; i < out.length; i++) out[i] = read(dv, i * size);
      return out;
    }
  };
}

async function openImages(files) {
  // one image per frame, ordered by the last number in the name (frame_0001.png, 00000037.png …)
  const num = f => Number((/(\d+)(?=\D*$)/.exec(f.name) || [0, NaN])[1]);
  files.sort((a, b) => (num(a) - num(b)) || a.name.localeCompare(b.name));
  const first = await imageToGrey(files[0]);
  return {
    kind: files.length === 1 ? 'image' : 'images', name: files.length === 1 ? files[0].name : `${files.length} images (${files[0].name} …)`,
    N: files.length, W: first.W, H: first.H, files, suggestedOffset: 0,
    async frame(k) { return k === 0 ? first.data : (await imageToGrey(files[k])).data; }
  };
}

async function imageToGrey(file) {
  const bmp = await createImageBitmap(file);
  const c = new OffscreenCanvas(bmp.width, bmp.height), g = c.getContext('2d');
  g.drawImage(bmp, 0, 0);
  const d = g.getImageData(0, 0, bmp.width, bmp.height).data, out = new Float32Array(bmp.width * bmp.height);
  for (let i = 0, j = 0; i < out.length; i++, j += 4) out[i] = (0.299 * d[j] + 0.587 * d[j + 1] + 0.114 * d[j + 2]) / 255;
  return { W: bmp.width, H: bmp.height, data: out };
}

/** Resample a map to W×H: area average when shrinking, bilinear when enlarging. */
export function resample(src, sw, sh, W, H) {
  if (sw === W && sh === H) return src;
  const out = new Float32Array(W * H);
  if (sw >= W && sh >= H) {
    const cnt = new Float32Array(W * H);
    for (let y = 0; y < sh; y++) {
      const row = Math.min(H - 1, Math.floor(y * H / sh)) * W;
      for (let x = 0; x < sw; x++) { const i = row + Math.min(W - 1, Math.floor(x * W / sw)); out[i] += src[y * sw + x]; cnt[i]++; }
    }
    for (let i = 0; i < out.length; i++) out[i] /= cnt[i] || 1;
    return out;
  }
  for (let y = 0; y < H; y++) {
    const fy = Math.min(sh - 1, Math.max(0, (y + 0.5) * sh / H - 0.5)), y0 = Math.floor(fy), y1 = Math.min(sh - 1, y0 + 1), ay = fy - y0;
    for (let x = 0; x < W; x++) {
      const fx = Math.min(sw - 1, Math.max(0, (x + 0.5) * sw / W - 0.5)), x0 = Math.floor(fx), x1 = Math.min(sw - 1, x0 + 1), ax = fx - x0;
      out[y * W + x] = (src[y0 * sw + x0] * (1 - ax) + src[y0 * sw + x1] * ax) * (1 - ay) + (src[y1 * sw + x0] * (1 - ax) + src[y1 * sw + x1] * ax) * ay;
    }
  }
  return out;
}

/** Compare two maps of equal size: correlation, coverage of each at `frac` of its maximum, and overlap of those areas. */
export function compareMaps(a, b, frac = 0.1) {
  const n = a.length;
  let ma = 0, mb = 0, xa = 0, xb = 0;
  for (let i = 0; i < n; i++) { ma += a[i]; mb += b[i]; if (a[i] > xa) xa = a[i]; if (b[i] > xb) xb = b[i]; }
  ma /= n; mb /= n;
  let sab = 0, saa = 0, sbb = 0, ca = 0, cb = 0, both = 0, either = 0;
  const ta = frac * xa, tb = frac * xb;
  for (let i = 0; i < n; i++) {
    const da = a[i] - ma, db = b[i] - mb;
    sab += da * db; saa += da * da; sbb += db * db;
    const ia = xa > 0 && a[i] >= ta, ib = xb > 0 && b[i] >= tb;
    if (ia) ca++; if (ib) cb++; if (ia && ib) both++; if (ia || ib) either++;
  }
  return { cc: saa > 0 && sbb > 0 ? sab / Math.sqrt(saa * sbb) : NaN, coverageA: ca / n, coverageB: cb / n, iou: either ? both / either : NaN, maxB: xb };
}
