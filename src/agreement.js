// Observer agreement: how well each observer's map matches the map of all the other observers (leave-one-out),
// averaged over the clip. A flag for the person deciding, never a filter: nothing is removed here.
import { collectPoints, makeGrid, makePointBuffer, planarSigmaPx, splatPlanar, splatSphere } from './saliency.js';

const DEG = Math.PI / 180;

/**
 * Scores observers frame by frame: call add(t) for each frame time, then result().
 * Each observer is compared with the others of the same condition only (in Salient360 the two conditions are
 * the left and right eye of the same people, so comparing across them would make everyone look alike).
 * For a frame, an observer counts only when they and the others both have gaze in it; frames where either map is
 * empty are skipped rather than scored 0, so someone with sparse data is not marked down for gaps.
 * The comparison is the correlation of the two maps; on 360° maps each pixel is weighted by its area on the sphere.
 */
export function agreementScorer(g, s, ctx, W = 128, H = 64) {
  const nObs = g.observers.length, nCond = g.conditions.length;
  // each observer's condition: the one most of their rows belong to
  const tally = new Map();
  for (let i = 0; i < g.n; i++) { const k = g.obs[i] * nCond + g.cond[i]; tally.set(k, (tally.get(k) || 0) + 1); }
  const obsCond = new Int32Array(nObs).fill(-1), best = new Float64Array(nObs);
  for (const [k, c] of tally) { const o = Math.floor(k / nCond); if (c > best[o]) { best[o] = c; obsCond[o] = k % nCond; } }
  const erp = s.projection !== 'planar';
  const weight = new Float64Array(W * H);
  for (let y = 0; y < H; y++) { const w = erp ? Math.cos((0.5 - (y + 0.5) / H) * Math.PI) : 1; weight.fill(w, y * W, (y + 1) * W); }
  const grid = erp ? makeGrid(W, H) : null;
  const sum = new Float64Array(nObs), count = new Uint32Array(nObs);
  const allCtx = { ...ctx, observerEnabled: null, conditionEnabled: null };
  let pts = makePointBuffer();
  const own = new Map(), total = new Map(), sub = makePointBuffer(), keep = [];

  function splat(map, p) {
    if (erp) splatSphere(map, grid, p, Math.max(0.05, s.sigmaDeg) * DEG, s.truncate);
    else splatPlanar(map, W, H, p, planarSigmaPx(s, W), s.truncate);
  }
  function add(t) {
    pts = collectPoints(g, t, s, allCtx, pts);
    if (!pts.n) return;
    // group this frame's points by observer
    const byObs = new Map();
    for (let i = 0; i < pts.n; i++) { const o = pts.obs[i]; if (obsCond[o] < 0) continue; (byObs.get(o) || byObs.set(o, []).get(o)).push(i); }
    for (const m of total.values()) m.fill(0);
    keep.length = 0;
    for (const [o, idx] of byObs) {
      let map = own.get(o);
      if (!map) own.set(o, map = new Float32Array(W * H)); else map.fill(0);
      sub.n = 0;
      for (const i of idx) {
        if (sub.n === sub.lon.length) for (const key of ['lon', 'lat', 'w', 'obs']) { const b = new sub[key].constructor(sub[key].length * 2); b.set(sub[key]); sub[key] = b; }
        sub.lon[sub.n] = pts.lon[i]; sub.lat[sub.n] = pts.lat[i]; sub.w[sub.n] = pts.w[i]; sub.obs[sub.n] = o; sub.n++;
      }
      splat(map, sub);
      const c = obsCond[o];
      let tm = total.get(c);
      if (!tm) total.set(c, tm = new Float32Array(W * H));
      for (let i = 0; i < map.length; i++) tm[i] += map[i];
      keep.push(o);
    }
    for (const o of keep) {
      const a = own.get(o), tm = total.get(obsCond[o]);
      const r = correlation(a, tm, weight, true);
      if (Number.isFinite(r)) { sum[o] += r; count[o]++; }
    }
  }
  function result() {
    const scores = Array.from(sum, (v, o) => count[o] ? v / count[o] : NaN);
    return { scores, frames: Array.from(count), conditionOf: Array.from(obsCond), low: lowScores(scores), W, H };
  }
  return { add, result };
}

/**
 * Weighted correlation of a with (b − a) when `leaveOut` (b is the total including a), or with b. NaN when either
 * map is flat (no gaze), so such frames are left out.
 */
export function correlation(a, b, w = null, leaveOut = false) {
  if (!w) w = new Float64Array(a.length).fill(1);
  let sw = 0, sa = 0, sb = 0;
  for (let i = 0; i < a.length; i++) { const bi = leaveOut ? b[i] - a[i] : b[i]; sw += w[i]; sa += w[i] * a[i]; sb += w[i] * bi; }
  const ma = sa / sw, mb = sb / sw;
  let cab = 0, caa = 0, cbb = 0;
  for (let i = 0; i < a.length; i++) {
    const da = a[i] - ma, db = (leaveOut ? b[i] - a[i] : b[i]) - mb;
    cab += w[i] * da * db; caa += w[i] * da * da; cbb += w[i] * db * db;
  }
  // differences in float32 leave tiny noise where the others have nothing: treat near-zero variance as empty
  if (!(caa > 1e-12 * sw) || !(cbb > 1e-12 * sw)) return NaN;
  return cab / Math.sqrt(caa * cbb);
}

/**
 * Which scores are low: below the lower quartile minus 1.5 × the spread between the quartiles (the usual box-plot
 * rule for outliers). Needs at least 4 scored observers; returns a Set of observer indexes and the cut-off.
 */
export function lowScores(scores) {
  const v = scores.filter(Number.isFinite).sort((a, b) => a - b);
  if (v.length < 4) return { set: new Set(), cutoff: NaN };
  const q = p => { const x = p * (v.length - 1), i = Math.floor(x); return v[i] + (v[Math.min(v.length - 1, i + 1)] - v[i]) * (x - i); };
  const q1 = q(0.25), q3 = q(0.75), cutoff = q1 - 1.5 * (q3 - q1);
  return { set: new Set(scores.map((s, i) => Number.isFinite(s) && s < cutoff ? i : -1).filter(i => i >= 0)), cutoff };
}
