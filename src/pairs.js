// Left / right eye pairing: which two recordings are the same person's two eyes.
// Sides come from the names (a condition, folder or observer called L / R / left / right …) or, for Salient360
// head + eye data loaded as two files without their folders, from the file order (first file = left eye).
// Partners are found from the names when an observer ID is shared across the two eyes ("p01_L" and "p01_R"),
// otherwise by matching: two recordings are the same person when their fixations start at the same moments and
// land on the same spot (Salient360 numbers its recordings separately in the two files, so the numbers say nothing).

const EYE_TOKEN = /(^|[\/_\-\s.(])(left|right|l|r|le|re)(?=$|[\/_\-\s.)#])/gi;
const SIDE = { left: 'L', l: 'L', le: 'L', right: 'R', r: 'R', re: 'R' };

/** 'L', 'R' or null: the last eye word in a name. */
export function eyeOf(text) {
  let side = null;
  for (const m of String(text).matchAll(EYE_TOKEN)) side = SIDE[m[2].toLowerCase()];
  return side;
}
/** The name with its eye word taken out, for finding the partner with the same ID. */
export function withoutEye(text) {
  return String(text).replace(EYE_TOKEN, '$1').replace(/^[\/_\-\s.]+|[\/_\-\s.]+$/g, '').replace(/([\/_\-\s.])[\/_\-\s.]+/g, '$1');
}

/**
 * Pair the observers of a gaze set.
 *   g        gaze set (observers, conditions, obs, cond, t0, lon, lat); g.observerFile: file index of each observer
 *   context  { preset, observerFrom, fileCount }
 * Returns null when the data does not say which recordings are left and right eyes, else
 *   { people: [{ left, right, score }], sidesFrom, method, pairs, singleLeft, singleRight }
 * with observer indexes (−1 where an eye has no partner).
 */
export function pairEyes(g, context = {}) {
  const nObs = g.observers.length;
  if (nObs < 2) return null;
  const condOf = new Int32Array(nObs).fill(-1);
  for (let i = 0; i < g.n; i++) if (condOf[g.obs[i]] < 0) condOf[g.obs[i]] = g.cond[i];
  // which eye each recording is
  let side = g.observers.map((o, i) => eyeOf(o) || (condOf[i] >= 0 ? eyeOf(g.conditions[condOf[i]]) : null));
  let sidesFrom = 'names';
  if (side.some(s => !s)) {
    const files = g.observerFile;
    if (context.preset === 's360' && files && context.fileCount === 2 && new Set(files).size === 2) {
      side = Array.from(files, f => (f === 0 ? 'L' : 'R')); sidesFrom = 'file order';
    } else return null;
  }
  const L = [], R = [];
  side.forEach((s, i) => (s === 'L' ? L : R).push(i));
  if (!L.length || !R.length) return null;
  // partners from a shared ID, when the observer names are real IDs (not run numbers counted in each file)
  let pairs = [], method = 'names';
  if (context.observerFrom !== 'restart' && sidesFrom === 'names') {
    const key = i => withoutEye(g.observers[i]).toLowerCase(), rk = new Map();
    for (const j of R) { const k = key(j); rk.set(k, rk.has(k) ? -1 : j); }
    for (const i of L) { const j = rk.get(key(i)); if (j >= 0) pairs.push([i, j, 1]); }
    if (pairs.length < Math.min(L.length, R.length) / 2) pairs = [];
  }
  if (!pairs.length) { method = 'matching'; pairs = matchByFixations(g, L, R); }
  const usedL = new Set(pairs.map(p => p[0])), usedR = new Set(pairs.map(p => p[1]));
  const people = pairs.map(([l, r, s]) => ({ left: l, right: r, score: s }));
  for (const i of L) if (!usedL.has(i)) people.push({ left: i, right: -1, score: null });
  for (const j of R) if (!usedR.has(j)) people.push({ left: -1, right: j, score: null });
  people.sort((a, b) => Math.min(...[a.left, a.right].filter(x => x >= 0)) - Math.min(...[b.left, b.right].filter(x => x >= 0)));
  return { people, sidesFrom, method, pairs: pairs.length, singleLeft: L.length - pairs.length, singleRight: R.length - pairs.length };
}

/**
 * Share of a's fixations that have a fixation of b starting within 60 ms and within 3°, for every left × right
 * recording; then the best one-to-one assignment (Hungarian method). Assigned pairs below 0.1 are not pairs.
 */
export function matchByFixations(g, L, R, { dtSec = 0.06, maxDeg = 3, minScore = 0.1 } = {}) {
  const per = new Map();
  for (const i of [...L, ...R]) per.set(i, { t: [], x: [], y: [], z: [] });
  for (let k = 0; k < g.n; k++) {
    const p = per.get(g.obs[k]); if (!p) continue;
    const cl = Math.cos(g.lat[k]);
    p.t.push(g.t0[k]); p.x.push(cl * Math.sin(g.lon[k])); p.y.push(Math.sin(g.lat[k])); p.z.push(cl * Math.cos(g.lon[k]));
  }
  const cosMax = Math.cos(maxDeg * Math.PI / 180);
  const score = (a, b) => {
    if (!a.t.length || !b.t.length) return 0;
    let hit = 0, j = 0;
    for (let i = 0; i < a.t.length; i++) {
      while (j < b.t.length && b.t[j] < a.t[i] - dtSec) j++;
      for (let k = j; k < b.t.length && b.t[k] <= a.t[i] + dtSec; k++) {
        if (a.x[i] * b.x[k] + a.y[i] * b.y[k] + a.z[i] * b.z[k] > cosMax) { hit++; break; }
      }
    }
    return hit / a.t.length;
  };
  const M = L.map(i => R.map(j => score(per.get(i), per.get(j))));
  const assign = hungarianMax(M);
  return assign.map((j, a) => [L[a], j >= 0 ? R[j] : -1, j >= 0 ? M[a][j] : 0]).filter(p => p[1] >= 0 && p[2] >= minScore);
}

/** Best one-to-one assignment of rows to columns maximising the sum (rows without a column get −1). */
export function hungarianMax(M) {
  const n = M.length, m = M[0]?.length || 0;
  if (!n || !m) return new Array(n).fill(-1);
  const size = Math.max(n, m), INF = 1e18;
  const cost = (i, j) => (i < n && j < m ? -M[i][j] : 0);
  const u = new Float64Array(size + 1), v = new Float64Array(size + 1), p = new Int32Array(size + 1), way = new Int32Array(size + 1);
  for (let i = 1; i <= size; i++) {
    p[0] = i; let j0 = 0;
    const minv = new Float64Array(size + 1).fill(INF), used = new Uint8Array(size + 1);
    do {
      used[j0] = 1; const i0 = p[j0]; let delta = INF, j1 = 0;
      for (let j = 1; j <= size; j++) if (!used[j]) {
        const cur = cost(i0 - 1, j - 1) - u[i0] - v[j];
        if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
        if (minv[j] < delta) { delta = minv[j]; j1 = j; }
      }
      for (let j = 0; j <= size; j++) { if (used[j]) { u[p[j]] += delta; v[j] -= delta; } else minv[j] -= delta; }
      j0 = j1;
    } while (p[j0] !== 0);
    do { const j1 = way[j0]; p[j0] = p[j1]; j0 = j1; } while (j0);
  }
  const rowOf = new Array(n).fill(-1);
  for (let j = 1; j <= size; j++) if (p[j] - 1 < n && j - 1 < m) rowOf[p[j] - 1] = j - 1;
  return rowOf;
}
