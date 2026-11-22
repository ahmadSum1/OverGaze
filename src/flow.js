// Dense optical flow (pyramidal Lucas–Kanade) on small greyscale frames, with optional horizontal
// wrap-around for equirectangular video, and a time-indexed store of velocity fields used to move
// gaze points between frames (motion-compensated temporal pooling).

const TAU = Math.PI * 2;

/** RGBA ImageData-like → Float32 luminance in [0, 1]. */
export function toGray(rgba, W, H) {
  const out = new Float32Array(W * H);
  for (let i = 0, j = 0; i < out.length; i++, j += 4) out[i] = (0.299 * rgba[j] + 0.587 * rgba[j + 1] + 0.114 * rgba[j + 2]) / 255;
  return out;
}

function downsample(img, W, H, wrap) {
  const w = Math.max(1, W >> 1), h = Math.max(1, H >> 1);
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const x0 = 2 * x, x1 = wrap ? (2 * x + 1) % W : Math.min(W - 1, 2 * x + 1), y0 = 2 * y, y1 = Math.min(H - 1, 2 * y + 1);
    out[y * w + x] = 0.25 * (img[y0 * W + x0] + img[y0 * W + x1] + img[y1 * W + x0] + img[y1 * W + x1]);
  }
  return [out, w, h];
}

function blur3(img, W, H, wrap) {
  // Separable [1 2 1]/4 smoothing to stabilise gradients.
  const tmp = new Float32Array(W * H), out = new Float32Array(W * H);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const l = wrap ? (x - 1 + W) % W : Math.max(0, x - 1), r = wrap ? (x + 1) % W : Math.min(W - 1, x + 1);
    tmp[y * W + x] = 0.25 * img[y * W + l] + 0.5 * img[y * W + x] + 0.25 * img[y * W + r];
  }
  for (let y = 0; y < H; y++) {
    const u = Math.max(0, y - 1), d = Math.min(H - 1, y + 1);
    for (let x = 0; x < W; x++) out[y * W + x] = 0.25 * tmp[u * W + x] + 0.5 * tmp[y * W + x] + 0.25 * tmp[d * W + x];
  }
  return out;
}

export function sampleBilinear(img, W, H, x, y, wrap) {
  if (wrap) x -= W * Math.floor(x / W); else x = Math.min(W - 1, Math.max(0, x));
  y = Math.min(H - 1, Math.max(0, y));
  const x0 = Math.floor(x), y0 = Math.floor(y), fx = x - x0, fy = y - y0;
  const x1 = wrap ? (x0 + 1) % W : Math.min(W - 1, x0 + 1), y1 = Math.min(H - 1, y0 + 1);
  return (img[y0 * W + x0] * (1 - fx) + img[y0 * W + x1] * fx) * (1 - fy) + (img[y1 * W + x0] * (1 - fx) + img[y1 * W + x1] * fx) * fy;
}

function boxSum(src, W, H, r, wrap) {
  const tmp = new Float32Array(W * H), out = new Float32Array(W * H);
  for (let y = 0; y < H; y++) {
    const row = y * W;
    for (let x = 0; x < W; x++) {
      let s = 0;
      for (let k = -r; k <= r; k++) {
        const xx = wrap ? (x + k + W) % W : Math.min(W - 1, Math.max(0, x + k));
        s += src[row + xx];
      }
      tmp[row + x] = s;
    }
  }
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    let s = 0;
    for (let k = -r; k <= r; k++) s += tmp[Math.min(H - 1, Math.max(0, y + k)) * W + x];
    out[y * W + x] = s;
  }
  return out;
}

function lkLevel(I0, I1, W, H, u, v, { iterations, radius, wrap }) {
  const n = W * H, Ix = new Float32Array(n), Iy = new Float32Array(n);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const l = wrap ? (x - 1 + W) % W : Math.max(0, x - 1), r = wrap ? (x + 1) % W : Math.min(W - 1, x + 1);
    const up = Math.max(0, y - 1), dn = Math.min(H - 1, y + 1), i = y * W + x;
    Ix[i] = 0.5 * (I0[y * W + r] - I0[y * W + l]);
    Iy[i] = 0.5 * (I0[dn * W + x] - I0[up * W + x]);
  }
  const A = boxSum(Ix.map((g, i) => g * g), W, H, radius, wrap);
  const B = boxSum(Ix.map((g, i) => g * Iy[i]), W, H, radius, wrap);
  const C = boxSum(Iy.map(g => g * g), W, H, radius, wrap);
  const area = (2 * radius + 1) ** 2, eps = 1e-5 * area;
  const D0 = new Float32Array(n), E0 = new Float32Array(n);
  for (let it = 0; it < iterations; it++) {
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const i = y * W + x;
      const It = sampleBilinear(I1, W, H, x + u[i], y + v[i], wrap) - I0[i];
      D0[i] = Ix[i] * It; E0[i] = Iy[i] * It;
    }
    const D = boxSum(D0, W, H, radius, wrap), E = boxSum(E0, W, H, radius, wrap);
    for (let i = 0; i < n; i++) {
      const a = A[i] + eps, c = C[i] + eps, b = B[i], det = a * c - b * b;
      if (det <= 1e-12) continue;
      let du = -(c * D[i] - b * E[i]) / det, dv = -(a * E[i] - b * D[i]) / det;
      const m = Math.hypot(du, dv);
      if (m > 2) { du *= 2 / m; dv *= 2 / m; } // damp per-iteration steps
      u[i] += du; v[i] += dv;
    }
  }
}

/**
 * Dense flow from I0 to I1 (both W×H Float32 in [0, 1]): I1(x + u, y + v) ≈ I0(x, y).
 * Returns { u, v } in pixels of the input resolution.
 */
export function opticalFlow(I0, I1, W, H, opts = {}) {
  const o = { levels: 4, iterations: 4, radius: 3, wrap: true, smooth: 3, select: true, selectRadius: 2, selectReach: 3, ...opts };
  const p0 = [[blur3(I0, W, H, o.wrap), W, H]], p1 = [[blur3(I1, W, H, o.wrap), W, H]];
  for (let l = 1; l < o.levels; l++) {
    const [a, w, h] = p0[l - 1], [b] = p1[l - 1];
    if (Math.min(w, h) < 16) break;
    p0.push(downsample(a, w, h, o.wrap)); p1.push(downsample(b, w, h, o.wrap));
  }
  let [, cw, ch] = p0[p0.length - 1];
  let u = new Float32Array(cw * ch), v = new Float32Array(cw * ch);
  for (let l = p0.length - 1; l >= 0; l--) {
    const [a, w, h] = p0[l], [b] = p1[l];
    if (w !== cw || h !== ch) {
      const nu = new Float32Array(w * h), nv = new Float32Array(w * h);
      for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
        const sx = (x + 0.5) * cw / w - 0.5, sy = (y + 0.5) * ch / h - 0.5;
        nu[y * w + x] = sampleBilinear(u, cw, ch, sx, sy, o.wrap) * (w / cw);
        nv[y * w + x] = sampleBilinear(v, cw, ch, sx, sy, o.wrap) * (h / ch);
      }
      u = nu; v = nv; cw = w; ch = h;
    }
    lkLevel(a, b, w, h, u, v, { iterations: o.iterations, radius: o.radius, wrap: o.wrap });
    if (o.select) selectCandidates(a, b, w, h, u, v, o);
  }
  if (o.smooth > 0) confidenceSmooth(p0[0][0], W, H, u, v, o);
  if (o.select) selectCandidates(p0[0][0], p1[0][0], W, H, u, v, o);
  return { u, v };
}

/**
 * Each pixel keeps whichever movement explains its neighbourhood best: no movement at all, its own, or that of a
 * pixel a few steps away. The cost is the mean absolute difference between the first frame and the second frame
 * moved back by the candidate, over a (2·radius+1)² window. This stops motion spreading from a moving neighbour or
 * from the coarse levels of the pyramid into parts of the picture that do not change (sand next to surf), while a
 * flat area that really moves with the camera keeps the movement of its textured neighbours. On a tie, "no
 * movement" wins. Checked against OpenCV's DIS flow on Salient360 clips (see the pull request that added this).
 */
function selectCandidates(I0, I1, W, H, u, v, o) {
  const n = W * H, r = o.selectRadius, d = o.selectReach;
  const offsets = [[0, 0, true], [0, 0, false], [d, 0], [-d, 0], [0, d], [0, -d], [d, d], [-d, -d], [d, -d], [-d, d]];
  const bestCost = new Float32Array(n).fill(Infinity), bu = new Float32Array(n), bv = new Float32Array(n);
  const res = new Float32Array(n), cu = new Float32Array(n), cv = new Float32Array(n);
  for (const [dx, dy, zero] of offsets) {
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const i = y * W + x;
      if (zero) { cu[i] = 0; cv[i] = 0; } else {
        const xx = o.wrap ? (x + dx + W) % W : Math.min(W - 1, Math.max(0, x + dx)), yy = Math.min(H - 1, Math.max(0, y + dy));
        cu[i] = u[yy * W + xx]; cv[i] = v[yy * W + xx];
      }
    }
    // the candidate of pixel i is tried over i's whole window: residual of every window pixel moved by i's candidate
    // is approximated by moving each pixel by its own candidate field, which equals it where the field is smooth
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const i = y * W + x;
      res[i] = Math.abs(sampleBilinear(I1, W, H, x + cu[i], y + cv[i], o.wrap) - I0[i]);
    }
    const sum = boxSum(res, W, H, r, o.wrap), area = (2 * r + 1) ** 2;
    for (let i = 0; i < n; i++) {
      const c = sum[i] / area;
      if (c < bestCost[i]) { bestCost[i] = c; bu[i] = cu[i]; bv[i] = cv[i]; }
    }
  }
  u.set(bu); v.set(bv);
}

/**
 * Normalised convolution: average the flow over a (2·smooth+1)² window, weighting each pixel by the smaller
 * eigenvalue of its structure tensor (how well its motion is determined). Flat, textureless areas take the
 * motion of their textured neighbours instead of noise.
 */
function confidenceSmooth(I0, W, H, u, v, o) {
  const n = W * H, Ix = new Float32Array(n), Iy = new Float32Array(n);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const l = o.wrap ? (x - 1 + W) % W : Math.max(0, x - 1), r = o.wrap ? (x + 1) % W : Math.min(W - 1, x + 1);
    const i = y * W + x;
    Ix[i] = 0.5 * (I0[y * W + r] - I0[y * W + l]);
    Iy[i] = 0.5 * (I0[Math.min(H - 1, y + 1) * W + x] - I0[Math.max(0, y - 1) * W + x]);
  }
  const A = boxSum(Ix.map(g => g * g), W, H, 1, o.wrap), B = boxSum(Ix.map((g, i) => g * Iy[i]), W, H, 1, o.wrap), C = boxSum(Iy.map(g => g * g), W, H, 1, o.wrap);
  const c = new Float32Array(n), cu = new Float32Array(n), cv = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const tr = A[i] + C[i], det = A[i] * C[i] - B[i] * B[i];
    c[i] = Math.max(0, tr / 2 - Math.sqrt(Math.max(0, tr * tr / 4 - det))) + 1e-9;
    cu[i] = c[i] * u[i]; cv[i] = c[i] * v[i];
  }
  const sc = boxSum(c, W, H, o.smooth, o.wrap), su = boxSum(cu, W, H, o.smooth, o.wrap), sv = boxSum(cv, W, H, o.smooth, o.wrap);
  for (let i = 0; i < n; i++) { u[i] = su[i] / sc[i]; v[i] = sv[i] / sc[i]; }
}

/**
 * Velocity fields (pixels per second at the store resolution) at sorted sample times.
 * Stored as Int16 with a fixed scale to keep memory low (≈ W·H·4 bytes per sample).
 */
export class FlowStore {
  constructor(W, H, { wrap = true, scale = 32 } = {}) {
    this.W = W; this.H = H; this.wrap = wrap; this.scale = scale;
    this.times = []; this.fields = []; this.step = 0.1;
  }
  get size() { return this.times.length; }
  get bytes() { return this.fields.length * this.W * this.H * 4; }
  /** Add a field measured between tA and tB (velocity = displacement / (tB − tA)). */
  add(tA, tB, u, v) {
    const dt = tB - tA;
    if (!(dt > 0)) return;
    const n = this.W * this.H, q = new Int16Array(2 * n), k = this.scale / dt;
    for (let i = 0; i < n; i++) {
      q[2 * i] = Math.max(-32767, Math.min(32767, Math.round(u[i] * k)));
      q[2 * i + 1] = Math.max(-32767, Math.min(32767, Math.round(v[i] * k)));
    }
    const t = 0.5 * (tA + tB);
    let i = this.times.length;
    while (i > 0 && this.times[i - 1] > t) i--;
    if (i > 0 && Math.abs(this.times[i - 1] - t) < 1e-6) { this.fields[i - 1] = { q, dt, tA, tB }; return; }
    this.times.splice(i, 0, t); this.fields.splice(i, 0, { q, dt, tA, tB });
    this.step = this.medianStep();
  }
  medianStep() {
    if (this.times.length < 2) return this.fields[0]?.dt || 0.1;
    const d = [];
    for (let i = 1; i < this.times.length; i++) d.push(this.times[i] - this.times[i - 1]);
    d.sort((a, b) => a - b);
    return d[d.length >> 1];
  }
  /** Fraction of [a, b] that has flow samples within one step. */
  coverage(a, b) {
    if (!this.times.length || b <= a) return 0;
    let covered = 0;
    const n = 200;
    for (let i = 0; i < n; i++) if (this.nearest(a + (i + 0.5) * (b - a) / n) >= 0) covered++;
    return covered / n;
  }
  nearest(t) {
    const ts = this.times;
    if (!ts.length) return -1;
    let lo = 0, hi = ts.length - 1;
    while (lo < hi) { const m = (lo + hi) >> 1; if (ts[m] < t) lo = m + 1; else hi = m; }
    let best = lo;
    if (lo > 0 && Math.abs(ts[lo - 1] - t) < Math.abs(ts[lo] - t)) best = lo - 1;
    const f = this.fields[best];
    // A field stands in for the time around it; 1.6 steps bridge one field that was missed while the video played.
    return t >= f.tA - 1.6 * this.step && t <= f.tB + 1.6 * this.step ? best : -1;
  }
  /** Velocity (px/s, store resolution) at time t and store-pixel position (x, y). */
  velocity(t, x, y) {
    const i = this.nearest(t);
    if (i < 0) return [0, 0];
    const { q } = this.fields[i], W = this.W, H = this.H, wrap = this.wrap;
    if (wrap) x -= W * Math.floor(x / W); else x = Math.min(W - 1, Math.max(0, x));
    y = Math.min(H - 1, Math.max(0, y));
    const x0 = Math.floor(x), y0 = Math.floor(y), fx = x - x0, fy = y - y0;
    const x1 = wrap ? (x0 + 1) % W : Math.min(W - 1, x0 + 1), y1 = Math.min(H - 1, y0 + 1);
    const at = (xx, yy, c) => q[2 * (yy * W + xx) + c];
    const s = 1 / this.scale;
    const lerp = c => ((at(x0, y0, c) * (1 - fx) + at(x1, y0, c) * fx) * (1 - fy) + (at(x0, y1, c) * (1 - fx) + at(x1, y1, c) * fx) * fy) * s;
    return [lerp(0), lerp(1)];
  }
  /** Move a point (lon, lat radians) from time `from` to time `to` along the flow (midpoint integration). */
  propagate(lon, lat, from, to) {
    const W = this.W, H = this.H;
    let x = (lon / TAU + 0.5) * W - 0.5, y = (0.5 - lat / Math.PI) * H - 0.5;
    const span = to - from, steps = Math.max(1, Math.ceil(Math.abs(span) / Math.max(1e-3, this.step)));
    const h = span / steps;
    for (let k = 0; k < steps; k++) {
      const t = from + (k + 0.5) * h;
      const [u1, v1] = this.velocity(t, x, y);
      const [u2, v2] = this.velocity(t, x + 0.5 * h * u1, y + 0.5 * h * v1);
      x += h * u2; y += h * v2;
      if (this.wrap) x -= W * Math.floor((x + 0.5) / W);
      y = Math.min(H - 0.5, Math.max(-0.5, y));
    }
    let outLon = ((x + 0.5) / W - 0.5) * TAU;
    outLon -= TAU * Math.floor((outLon + Math.PI) / TAU);
    return [outLon, (0.5 - (y + 0.5) / H) * Math.PI];
  }
  /** Plain transferable form for workers. */
  serialize() {
    return { W: this.W, H: this.H, wrap: this.wrap, scale: this.scale, step: this.step, times: this.times.slice(), fields: this.fields.map(f => ({ ...f, q: f.q })) };
  }
  static from(data) {
    const s = new FlowStore(data.W, data.H, { wrap: data.wrap, scale: data.scale });
    s.times = data.times; s.fields = data.fields; s.step = data.step;
    return s;
  }
}
