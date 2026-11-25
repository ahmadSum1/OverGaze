// Fixation-to-saliency conversion: temporal pooling (optionally motion compensated) and Gaussian splatting,
// either on the sphere (equirectangular / ERP frames) or in the image plane (ordinary 2D video).
import { lowerBound } from './gaze.js';

const TAU = Math.PI * 2, DEG = Math.PI / 180;

/** Pixel-centre lon/lat tables for a W×H equirectangular grid. */
export function makeGrid(W, H) {
  const sinLat = new Float64Array(H), cosLat = new Float64Array(H), sinLon = new Float64Array(W), cosLon = new Float64Array(W);
  for (let y = 0; y < H; y++) {
    const lat = (0.5 - (y + 0.5) / H) * Math.PI;
    sinLat[y] = Math.sin(lat); cosLat[y] = Math.cos(lat);
  }
  for (let x = 0; x < W; x++) {
    const lon = ((x + 0.5) / W - 0.5) * TAU;
    sinLon[x] = Math.sin(lon); cosLon[x] = Math.cos(lon);
  }
  return { W, H, sinLat, cosLat, sinLon, cosLon };
}

/** Default settings; every field is a knob in the UI and is written into exported settings.json. */
export const DEFAULTS = Object.freeze({
  projection: 'erp',        // 'erp' (360°, sphere Gaussian) or 'planar' (2D video)
  sigmaDeg: 2,              // Gaussian standard deviation in degrees
  truncate: 3,              // kernel cut-off in σ
  planarFovDeg: 90,         // planar only: horizontal field of view used to turn degrees into pixels
  windowSec: 0,             // temporal pooling half-width: frames within ±windowSec are pooled
  kernel: 'box',            // 'box' | 'triangle' | 'gaussian' (σ_t = windowSec / 2) | 'peaked' (exp(−|Δt| / peakTauSec))
  peakTauSec: 0.25,         // 'peaked' only: how fast the weight falls away from the current frame
  memoryWeight: 0,          // scene memory: + w × the map of everything looked at since the clip started (0 = off)
  causal: false,            // only use gaze from the past (as a deployed, real-time mask would)
  durationWeight: false,    // weight each fixation by its duration
  motionComp: false,        // warp pooled gaze along optical flow into the current frame
  lonOffsetDeg: 0, flipX: false, flipY: false, // orientation fixes applied to the gaze
  normalize: 'max',         // 'max' (divide by frame max) | 'sum' (probability map) | 'none'
  coverageThreshold: 0.1    // coverage = share of the frame at or above this fraction of the frame maximum
});

/** Frame index ↔ frame-centre time. Frame k covers [k/fps, (k+1)/fps); its centre is used for pooling. */
export const frameCentre = (k, fps) => (k + 0.5) / fps;
export const frameOf = (t, fps) => Math.max(0, Math.floor(t * fps + 1e-6));

export function temporalWeight(d, reach, s) {
  if (s.kernel === 'triangle') return Math.max(0, 1 - d / reach);
  if (s.kernel === 'peaked') return Math.exp(-d / Math.max(1e-3, s.peakTauSec ?? 0.25));
  if (s.kernel === 'gaussian') {
    const st = s.windowSec / 2;
    return st > 0 ? Math.exp(-(d * d) / (2 * st * st)) : 1;
  }
  return 1;
}

/**
 * Collect the gaze points that contribute to the map at time t (seconds).
 * A point (or fixation interval [t0, t1]) counts if its time distance to t is within windowSec + tol,
 * tol = half a frame, so windowSec = 0 means "active in this frame".
 * Returns { n, lon, lat, w, obs } (typed arrays, reused between calls through `out`).
 */
export function collectPoints(g, t, s, ctx = {}, out = makePointBuffer()) {
  out.n = 0;
  if (!g || !g.n) return out;
  const fps = ctx.fps || 30;
  const tol = ctx.tol ?? 0.5 / fps;
  const reach = Math.max(0, s.windowSec) + tol;
  const start = lowerBound(g.t0, g.n, t - reach - g.maxDur - 1e-9);
  const offset = (s.lonOffsetDeg || 0) * DEG;
  const obsOn = ctx.observerEnabled, condOn = ctx.conditionEnabled;
  const flow = s.motionComp ? ctx.flow : null;
  // ctx.moveCache (live view only): points moved to an earlier time, usable when t is a little later (up to 0.25 s)
  let cache = null;
  if (flow && ctx.moveCache) {
    const c = ctx.moveCache, key = [s.flipX, s.flipY, s.lonOffsetDeg, fps].join('|');
    const ok = c.g === g && c.flow === flow && c.key === key && t > c.t && t - c.t <= 0.25;
    cache = { prev: ok ? c.map : new Map(), t: c.t, next: new Map() };
    Object.assign(c, { g, flow, key });
  }
  // A fixation (an interval longer than half a frame) counts once for every frame it covers, like the per-frame
  // fixation sets the dataset maps are built from; a gaze sample counts once at its own time.
  const isFixation = (a, b) => b - a > 0.5 / fps;
  const add = (i, lon, lat, w, anchor) => {
    lon = s.flipX ? -lon : lon; lat = s.flipY ? -lat : lat;
    lon += offset;
    lon -= TAU * Math.floor((lon + Math.PI) / TAU);
    if (flow && Math.abs(anchor - t) > 1e-6) {
      // live view while playing: a point already moved to the previous frame's time is moved on from there
      // (one short step) instead of from its own moment
      const key = i * 8192 + Math.floor(anchor * fps) % 8192, prev = cache && cache.prev.get(key);   // anchor × fps = frame + 0.5
      const p = prev ? flow.propagate(prev[0], prev[1], cache.t, t) : flow.propagate(lon, lat, anchor, t);
      lon = p[0]; lat = p[1];
      if (cache) cache.next.set(key, [lon, lat]);
    }
    push(out, lon, lat, w, g.obs[i]);
  };
  for (let i = start; i < g.n; i++) {
    const a = g.t0[i];
    if (a > t + reach) break;
    if (s.causal && a > t + tol) continue;
    const b = g.t1[i];
    if (obsOn && !obsOn[g.obs[i]]) continue;
    if (condOn && !condOn[g.cond[i]]) continue;
    const base = g.w[i] * (s.durationWeight && b > a ? (b - a) : 1);
    if (isFixation(a, b)) {
      // frames j with centre (j + 0.5) / fps inside [a, b) and inside the window around t
      const lo = Math.max(Math.ceil(a * fps - 0.5), Math.ceil((t - reach) * fps - 0.5 - 1e-9));
      const hiTime = s.causal ? Math.min(b, t + tol) : Math.min(b, t + reach);
      const hi = Math.ceil(hiTime * fps - 0.5 - 1e-9) - 1;
      for (let j = lo; j <= hi; j++) {
        const c = (j + 0.5) / fps;
        if (c < a - 1e-9 || c >= b - 1e-9 || c > t + reach + 1e-9) continue;
        const d = s.causal ? Math.max(0, t - c) : Math.abs(c - t);
        if (d > reach + 1e-9) continue;
        const w = base * temporalWeight(d, reach, s);
        if (w > 0) add(i, g.lon[i], g.lat[i], w, c);
      }
      continue;
    }
    const d = s.causal ? Math.max(0, t - b) : Math.max(0, a - t, t - b);
    if (d > reach + 1e-9) continue;
    const w = base * temporalWeight(d, reach, s);
    if (!(w > 0)) continue;
    add(i, g.lon[i], g.lat[i], w, Math.min(b, Math.max(a, t)));
  }
  if (cache) Object.assign(ctx.moveCache, { map: cache.next, t });
  return out;
}

export function makePointBuffer(capacity = 1024) {
  return { n: 0, lon: new Float64Array(capacity), lat: new Float64Array(capacity), w: new Float64Array(capacity), obs: new Uint16Array(capacity) };
}

function push(out, lon, lat, w, obs) {
  if (out.n === out.lon.length) {
    for (const key of ['lon', 'lat', 'w', 'obs']) {
      const bigger = new out[key].constructor(out[key].length * 2);
      bigger.set(out[key]); out[key] = bigger;
    }
  }
  out.lon[out.n] = lon; out.lat[out.n] = lat; out.w[out.n] = w; out.obs[out.n] = obs; out.n++;
}

/** Accumulate sphere Gaussians (σ in radians, geodesic distance) into `map` (W×H, ERP). */
export function splatSphere(map, grid, pts, sigma, truncate = 3, lut = null) {
  const { W, H, sinLat, cosLat, sinLon, cosLon } = grid;
  const cut = Math.min(Math.PI, truncate * sigma);
  const cosCut = Math.cos(cut), inv2s2 = 1 / (2 * sigma * sigma);
  const rowScale = H / Math.PI, colScale = W / TAU;
  // fast path (live view while playing): the Gaussian as a table over the cosine of the angle, read with linear
  // interpolation, instead of acos and exp for every pixel; exports use the exact formula
  const table = lut ? gaussTable(lut, cosCut, inv2s2) : null, tScale = table ? (table.length - 2) / (1 - cosCut) : 0;
  for (let p = 0; p < pts.n; p++) {
    const lon = pts.lon[p], lat = pts.lat[p], w = pts.w[p];
    const sp = Math.sin(lat), cp = Math.cos(lat), sl = Math.sin(lon), cl = Math.cos(lon);
    const y0 = Math.max(0, Math.floor((Math.PI / 2 - lat - cut) * rowScale));
    const y1 = Math.min(H - 1, Math.ceil((Math.PI / 2 - lat + cut) * rowScale));
    const span = cp > Math.sin(cut) ? Math.asin(Math.min(1, Math.sin(cut) / cp)) : Math.PI;
    const full = span >= Math.PI - 1e-9;
    const cx = (lon / TAU + 0.5) * W - 0.5;
    const x0 = full ? 0 : Math.floor(cx - span * colScale) - 1;
    const x1 = full ? W - 1 : Math.ceil(cx + span * colScale) + 1;
    const count = Math.min(W, x1 - x0 + 1);
    const xs = ((x0 % W) + W) % W;
    for (let y = y0; y <= y1; y++) {
      const a = sp * sinLat[y], b = cp * cosLat[y], row = y * W;
      for (let k = 0, x = xs; k < count; k++, x = x + 1 === W ? 0 : x + 1) {
        const dot = a + b * (cosLon[x] * cl + sinLon[x] * sl);
        if (dot <= cosCut) continue;
        if (table) {
          const f = (dot - cosCut) * tScale, i = f | 0, r = f - i;
          map[row + x] += w * (table[i] + (table[i + 1] - table[i]) * r);
        } else {
          const th = Math.acos(dot > 1 ? 1 : dot);
          map[row + x] += w * Math.exp(-th * th * inv2s2);
        }
      }
    }
  }
}
/** exp(−acos(c)² / 2σ²) at evenly spaced c from cos(cut) to 1 (4096 steps), kept in `lut` for the current σ. */
function gaussTable(lut, cosCut, inv2s2) {
  if (lut.table && lut.cosCut === cosCut && lut.inv2s2 === inv2s2) return lut.table;
  const n = 4096, t = new Float32Array(n + 2);
  for (let i = 0; i <= n + 1; i++) { const c = Math.min(1, cosCut + (1 - cosCut) * i / n), th = Math.acos(c); t[i] = Math.exp(-th * th * inv2s2); }
  Object.assign(lut, { table: t, cosCut, inv2s2 });
  return t;
}

/** Accumulate 2D Gaussians (σ in map pixels) into `map` (W×H, planar). Points use the lon/lat → u/v mapping. */
export function splatPlanar(map, W, H, pts, sigmaPx, truncate = 3) {
  const r = Math.ceil(truncate * sigmaPx), inv2s2 = 1 / (2 * sigmaPx * sigmaPx);
  for (let p = 0; p < pts.n; p++) {
    const cx = (pts.lon[p] / TAU + 0.5) * W - 0.5, cy = (0.5 - pts.lat[p] / Math.PI) * H - 0.5, w = pts.w[p];
    const x0 = Math.max(0, Math.floor(cx - r)), x1 = Math.min(W - 1, Math.ceil(cx + r));
    const y0 = Math.max(0, Math.floor(cy - r)), y1 = Math.min(H - 1, Math.ceil(cy + r));
    for (let y = y0; y <= y1; y++) {
      const dy2 = (y - cy) * (y - cy), row = y * W;
      for (let x = x0; x <= x1; x++) {
        const d2 = (x - cx) * (x - cx) + dy2;
        if (d2 <= r * r) map[row + x] += w * Math.exp(-d2 * inv2s2);
      }
    }
  }
}

/** Planar σ in map pixels: degrees → pixels through the horizontal field of view. */
export function planarSigmaPx(s, W) { return Math.max(0.3, s.sigmaDeg * W / Math.max(1, s.planarFovDeg)); }

/**
 * Render one saliency map. Returns { map, max, sum, points } where map is normalised as requested.
 * `work` caches grids and point buffers between calls: pass the same object every frame.
 */
export function renderMap(g, t, s, W, H, ctx = {}, work = {}) {
  if (!work.map || work.map.length !== W * H) work.map = new Float32Array(W * H);
  const map = work.map;
  map.fill(0);
  work.points = collectPoints(g, t, s, ctx, work.points || makePointBuffer());
  if (s.projection === 'planar') splatPlanar(map, W, H, work.points, planarSigmaPx(s, W), s.truncate);
  else {
    if (!work.grid || work.grid.W !== W || work.grid.H !== H) work.grid = makeGrid(W, H);
    splatSphere(map, work.grid, work.points, Math.max(0.05, s.sigmaDeg) * DEG, s.truncate, ctx.fast ? (work.lut ||= {}) : null);
  }
  work.plain = null;
  if (s.memoryWeight > 0) {
    // scene memory: the pooled map and the memory map, each divided by its own maximum, added with weight w
    // drawn at a quarter of the width (a broad map of the whole clip; thousands of points), then enlarged
    const mw = Math.max(32, Math.round(W / 4)), mh = Math.max(16, Math.round(H * mw / W));
    if (!work.memSmall || work.memSmall.length !== mw * mh) work.memSmall = new Float32Array(mw * mh);
    if (!work.mem || work.mem.length !== W * H) work.mem = new Float32Array(W * H);
    const small = work.memSmall, mem = work.mem; small.fill(0);
    work.memPoints = memoryPoints(g, t, s, ctx, work.memPoints || makePointBuffer(), work.memCache || (work.memCache = {}));
    if (s.projection === 'planar') splatPlanar(small, mw, mh, work.memPoints, planarSigmaPx(s, mw), s.truncate);
    else {
      if (!work.memGrid || work.memGrid.W !== mw || work.memGrid.H !== mh) work.memGrid = makeGrid(mw, mh);
      splatSphere(small, work.memGrid, work.memPoints, Math.max(0.05, s.sigmaDeg) * DEG, s.truncate, ctx.fast ? (work.lut ||= {}) : null);
    }
    enlarge(small, mw, mh, mem, W, H, s.projection !== 'planar');
    let bm = 0, mm = 0;
    for (let i = 0; i < map.length; i++) { if (map[i] > bm) bm = map[i]; if (mem[i] > mm) mm = mem[i]; }
    if (!work.plainBuf || work.plainBuf.length !== W * H) work.plainBuf = new Float32Array(W * H);
    const plain = work.plainBuf;
    for (let i = 0; i < map.length; i++) { plain[i] = bm > 0 ? map[i] / bm : 0; map[i] = plain[i] + (mm > 0 ? s.memoryWeight * mem[i] / mm : 0); }
    work.plain = plain;
  }
  let max = 0, sum = 0;
  for (let i = 0; i < map.length; i++) { const v = map[i]; if (v > max) max = v; sum += v; }
  if (s.normalize === 'max' && max > 0) for (let i = 0; i < map.length; i++) map[i] /= max;
  else if (s.normalize === 'sum' && sum > 0) for (let i = 0; i < map.length; i++) map[i] /= sum;
  return { map, max, sum, points: work.points, plain: work.plain, memoryPoints: s.memoryWeight > 0 ? work.memPoints.n : 0 };
}

/** Bilinear enlargement of a small map (pixel centres aligned), joining the left and right edges for 360° maps. */
function enlarge(src, sw, sh, dst, W, H, wrap) {
  for (let y = 0; y < H; y++) {
    const fy = Math.min(sh - 1, Math.max(0, (y + 0.5) * sh / H - 0.5)), y0 = Math.floor(fy), y1 = Math.min(sh - 1, y0 + 1), ay = fy - y0;
    for (let x = 0; x < W; x++) {
      let fx = (x + 0.5) * sw / W - 0.5;
      let x0 = Math.floor(fx), ax = fx - x0, x1 = x0 + 1;
      if (wrap) { x0 = (x0 + sw) % sw; x1 = x1 % sw; } else { x0 = Math.max(0, x0); x1 = Math.min(sw - 1, x1); if (fx < 0) ax = 0; }
      dst[y * W + x] = (src[y0 * sw + x0] * (1 - ax) + src[y0 * sw + x1] * ax) * (1 - ay) + (src[y1 * sw + x0] * (1 - ax) + src[y1 * sw + x1] * ax) * ay;
    }
  }
}

/**
 * Scene memory: everything looked at from the start of the clip up to time t (past only, so it also works live),
 * one point per fixation weighted by how long it had lasted by t (in frames), or one per gaze sample; each moved
 * along the motion (ctx.memoryFlow, when analysed or loaded) from its last moment to t.
 * `cache` makes playback fast: finished fixations already moved to an earlier time are only moved on from there
 * (moving a→b then b→c is moving a→c), so each frame moves them one frame further instead of from where they were
 * recorded; fixations still going are kept aside until they end. Going back, other observers or other settings
 * start again from the beginning.
 */
export function memoryPoints(g, t, s, ctx = {}, out = makePointBuffer(), cache = null) {
  out.n = 0;
  if (!g || !g.n) return out;
  const fps = ctx.fps || 30, tol = ctx.tol ?? 0.5 / fps, flow = ctx.memoryFlow || null;
  const offset = (s.lonOffsetDeg || 0) * DEG, obsOn = ctx.observerEnabled, condOn = ctx.conditionEnabled;
  const key = [s.lonOffsetDeg, s.flipX, s.flipY, fps, obsOn ? obsOn.join('') : '', condOn ? condOn.join('') : ''].join('|');
  const usable = cache && cache.g === g && cache.flow === flow && cache.key === key && cache.t <= t;
  // done: finished fixations (and samples), already moved to cache.t; pending: rows started but not finished then
  const done = usable ? cache.done : { n: 0, lon: [], lat: [], w: [], obs: [] };
  const wasPending = usable ? cache.pending : [];
  let next = usable ? cache.next : 0;
  if (usable && flow && t > cache.t) for (let k = 0; k < done.n; k++) { const p = flow.propagate(done.lon[k], done.lat[k], cache.t, t); done.lon[k] = p[0]; done.lat[k] = p[1]; }
  const place = i => {
    let lon = s.flipX ? -g.lon[i] : g.lon[i], lat = s.flipY ? -g.lat[i] : g.lat[i];
    lon += offset; lon -= TAU * Math.floor((lon + Math.PI) / TAU);
    return [lon, lat];
  };
  const frames = (i, last) => g.w[i] * (g.t1[i] - g.t0[i] > 0.5 / fps ? Math.max(1, (last - g.t0[i]) * fps) : 1);
  const pending = [];
  const take = i => {
    if ((obsOn && !obsOn[g.obs[i]]) || (condOn && !condOn[g.cond[i]])) return;
    if (g.t1[i] > t) { pending.push(i); return; }
    let [lon, lat] = place(i);
    if (flow && Math.abs(g.t1[i] - t) > 1e-6) [lon, lat] = flow.propagate(lon, lat, g.t1[i], t);
    done.lon.push(lon); done.lat.push(lat); done.w.push(frames(i, g.t1[i])); done.obs.push(g.obs[i]); done.n++;
  };
  for (const i of wasPending) take(i);
  for (; next < g.n && g.t0[next] <= t + tol; next++) take(next);
  if (cache) Object.assign(cache, { g, flow, key, t, done, pending, next });
  for (let k = 0; k < done.n; k++) push(out, done.lon[k], done.lat[k], done.w[k], done.obs[k]);
  // still going at t: from where they are now
  for (const i of pending) {
    let [lon, lat] = place(i);
    push(out, lon, lat, frames(i, t), g.obs[i]);
  }
  return out;
}

/**
 * Coverage: share of the frame whose value is ≥ frac × frame max (0.1 by default, the measure used for the
 * training-map density checks), both per pixel and weighted by pixel area on the sphere (cos latitude).
 */
export function coverage(map, W, H, frac = 0.1, erp = true) {
  let max = 0;
  for (let i = 0; i < map.length; i++) if (map[i] > max) max = map[i];
  if (!(max > 0)) return { pixels: 0, sphere: 0 };
  const thr = frac * max;
  let hit = 0, hitW = 0, totalW = 0;
  for (let y = 0; y < H; y++) {
    const wgt = erp ? Math.cos((0.5 - (y + 0.5) / H) * Math.PI) : 1;
    totalW += wgt * W;
    const row = y * W;
    for (let x = 0; x < W; x++) if (map[row + x] >= thr) { hit++; hitW += wgt; }
  }
  return { pixels: hit / (W * H), sphere: hitW / totalW };
}

/** Map dimensions for a given width: 2:1 for ERP, the video's aspect ratio for planar. */
export function mapSize(width, s, videoW = 2, videoH = 1) {
  const W = Math.max(8, Math.round(width));
  const H = s.projection === 'planar' ? Math.max(4, Math.round(W * videoH / Math.max(1, videoW))) : Math.max(4, Math.round(W / 2));
  return [W, H];
}
