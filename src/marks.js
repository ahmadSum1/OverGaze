// Gaze points to draw on the video: one mark per fixation or gaze sample within the pooling window (the maps count a
// fixation once per frame it covers; for drawing, once is enough). Each mark knows where it was recorded, where the
// motion moved it, how far it is in time from the current frame and how long its eye had stayed in place.
import { temporalWeight } from './saliency.js';
import { lowerBound } from './gaze.js';

const TAU = Math.PI * 2, DEG = Math.PI / 180;

/**
 * Raw gaze samples: for each sample, the time its eye arrived at the spot, i.e. the first sample of the run in
 * which the eye stayed within `radiusDeg` of where the run started. Cached on the gaze set.
 */
export function dwellStarts(g, radiusDeg = 1.5) {
  if (g._dwell && g._dwellRadius === radiusDeg) return g._dwell;
  const start = new Float64Array(g.n), last = new Map(), cosR = Math.cos(radiusDeg * DEG);
  for (let i = 0; i < g.n; i++) {
    const o = g.obs[i], run = last.get(o);
    if (run) {
      const a = run.i, dot = Math.sin(g.lat[a]) * Math.sin(g.lat[i]) + Math.cos(g.lat[a]) * Math.cos(g.lat[i]) * Math.cos(g.lon[a] - g.lon[i]);
      if (dot >= cosR) { start[i] = g.t0[a]; continue; }
    }
    last.set(o, { i }); start[i] = g.t0[i];
  }
  g._dwell = start; g._dwellRadius = radiusDeg;
  return start;
}

/**
 * Marks at time t. Returns an array of { lon, lat, lon0, lat0, obs, dt, w, current, dwell }:
 *   lon, lat   where to draw it (moved along the motion to time t when motion compensation is on)
 *   lon0, lat0 where it was recorded
 *   dt         its own moment minus t (negative = earlier); 0 for the current frame
 *   w          its weight in the pooling (1 for the current frame)
 *   current    active during the current frame
 *   dwell      seconds its eye had stayed in place by its own moment (fixations: time into the fixation)
 */
export function collectMarks(g, t, s, ctx = {}) {
  const out = [];
  if (!g || !g.n) return out;
  const fps = ctx.fps || 30, tol = ctx.tol ?? 0.5 / fps, reach = Math.max(0, s.windowSec) + tol;
  const start = lowerBound(g.t0, g.n, t - reach - g.maxDur - 1e-9);
  const offset = (s.lonOffsetDeg || 0) * DEG, obsOn = ctx.observerEnabled, condOn = ctx.conditionEnabled;
  const flow = s.motionComp ? ctx.flow : null;
  const dwell0 = g.hasDuration ? null : dwellStarts(g, ctx.dwellRadiusDeg ?? 1.5);
  for (let i = start; i < g.n; i++) {
    const a = g.t0[i];
    if (a > t + reach) break;
    if (s.causal && a > t + tol) continue;
    if (obsOn && !obsOn[g.obs[i]]) continue;
    if (condOn && !condOn[g.cond[i]]) continue;
    const b = g.t1[i], anchor = Math.min(b, Math.max(a, t)), d = Math.max(0, a - t, t - b);
    if (d > reach + 1e-9) continue;
    const current = a <= t + tol && b >= t - tol;
    let lon = s.flipX ? -g.lon[i] : g.lon[i], lat = s.flipY ? -g.lat[i] : g.lat[i];
    lon += offset; lon -= TAU * Math.floor((lon + Math.PI) / TAU);
    const lon0 = lon, lat0 = lat;
    if (flow && !current && Math.abs(anchor - t) > 1e-6) [lon, lat] = flow.propagate(lon, lat, anchor, t);
    out.push({ lon, lat, lon0, lat0, obs: g.obs[i], dt: current ? 0 : anchor - t, w: current ? 1 : temporalWeight(d, reach, s), current, dwell: dwell0 ? a - dwell0[i] : anchor - a });
  }
  return out;
}
