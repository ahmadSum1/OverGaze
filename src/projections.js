// Video layouts for 360° footage: where a viewing direction lands in the video frame.
// The list follows the projection types of egjs-view360 (NAVER, MIT licence, https://github.com/naver/egjs-view360):
// equirectangular, stereo equirectangular (top-bottom / left-right), cubemap (3 × 2, cube strips 6 × 1 and 1 × 6)
// and the equi-angular cubemap of YouTube. The code here is our own; the face tables were fitted to ffmpeg's
// v360 filter (default face order "rludfb", no face rotation), which is how such files are usually made.
//
// Directions: x right, y up, z forward (centre of an equirectangular frame); lon = atan2(x, z), lat = asin(y).

export const LAYOUTS = {
  erp: { label: 'Equirectangular (2:1)' },
  'erp-tb': { label: 'Equirectangular, stereo top-bottom (top = left eye)' },
  'erp-lr': { label: 'Equirectangular, stereo side by side (left half)' },
  c3x2: { label: 'Cubemap 3 × 2', cols: 3, rows: 2 },
  c6x1: { label: 'Cube strip 6 × 1', cols: 6, rows: 1 },
  c1x6: { label: 'Cube strip 1 × 6', cols: 1, rows: 6 },
  eac: { label: 'Equi-angular cubemap 3 × 2 (YouTube)', cols: 3, rows: 2, equiAngular: true }
};

// Cube faces seen from inside: centre direction, and the directions of +s (right) and +t (up) on the face.
export const FACES = {
  r: { c: [1, 0, 0], s: [0, 0, -1], t: [0, 1, 0] },
  l: { c: [-1, 0, 0], s: [0, 0, 1], t: [0, 1, 0] },
  u: { c: [0, 1, 0], s: [1, 0, 0], t: [0, 0, -1] },
  d: { c: [0, -1, 0], s: [1, 0, 0], t: [0, 0, 1] },
  f: { c: [0, 0, 1], s: [1, 0, 0], t: [0, 1, 0] },
  b: { c: [0, 0, -1], s: [-1, 0, 0], t: [0, 1, 0] }
};
export const FACE_KEYS = ['r', 'l', 'u', 'd', 'f', 'b'];

// Per layout and face: tile index (row-major) and orientation of the face picture in its tile:
// rot = quarter turns, flip = mirrored left-right. Fitted to ffmpeg output: every tile correlates 0.92–0.997 with
// its prediction (0.73 for the low-contrast ground of one test frame); the plain cube faces need no turn at all.
export const FACE_TABLES = {
  c3x2: { r: [0, 0, 0], l: [1, 0, 0], u: [2, 0, 0], d: [3, 0, 0], f: [4, 0, 0], b: [5, 0, 0] },
  c6x1: { r: [0, 0, 0], l: [1, 0, 0], u: [2, 0, 0], d: [3, 0, 0], f: [4, 0, 0], b: [5, 0, 0] },
  c1x6: { r: [0, 0, 0], l: [1, 0, 0], u: [2, 0, 0], d: [3, 0, 0], f: [4, 0, 0], b: [5, 0, 0] },
  eac: { l: [0, 0, 0], f: [1, 0, 0], r: [2, 0, 0], d: [3, 1, 0], b: [4, 3, 0], u: [5, 1, 0] }   // top row l f r, bottom row d b u turned
};

/** Size of the equirectangular frame that a video of this layout holds. */
export function erpSize(layout, w, h) {
  const L = LAYOUTS[layout];
  if (!L || layout === 'erp') return [w, h];
  if (layout === 'erp-tb') return [w, Math.round(h / 2)];
  if (layout === 'erp-lr') return [Math.round(w / 2), h];
  const face = Math.min(w / L.cols, h / L.rows);
  return [Math.round(4 * face), Math.round(2 * face)];
}

/** Texture coordinates (u right, v down, 0–1) in a frame of this layout for direction d (need not be unit). */
export function dirToUV(layout, d, tables = FACE_TABLES, size = [3072, 2048]) {
  const [x, y, z] = d;
  if (layout === 'erp' || layout === 'erp-tb' || layout === 'erp-lr') {
    const r = Math.hypot(x, y, z);
    let u = Math.atan2(x, z) / (2 * Math.PI) + 0.5, v = 0.5 - Math.asin(y / r) / Math.PI;
    if (layout === 'erp-tb') v *= 0.5;
    if (layout === 'erp-lr') u *= 0.5;
    return [u, v];
  }
  const L = LAYOUTS[layout], ax = Math.abs(x), ay = Math.abs(y), az = Math.abs(z);
  const key = ax >= ay && ax >= az ? (x > 0 ? 'r' : 'l') : ay >= az ? (y > 0 ? 'u' : 'd') : (z > 0 ? 'f' : 'b');
  const F = FACES[key], dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2], m = dot(d, F.c);
  let s = dot(d, F.s) / m, t = dot(d, F.t) / m;                    // −1…1 on the face
  if (L.equiAngular) { s = Math.atan(s) * 4 / Math.PI; t = Math.atan(t) * 4 / Math.PI; }
  const [tile, rot, flip] = tables[layout][key];
  if (flip) s = -s;
  for (let k = 0; k < rot; k++) [s, t] = [-t, s];
  const col = tile % L.cols, row = Math.floor(tile / L.cols);
  if (L.equiAngular) {
    // 2 pixels of padding around the frame and between the rows, as in Google's spatial-media code and ffmpeg
    const pu = 2 / size[0], pv = 2 / size[1];
    return [(col + (s + 1) / 2) * (1 - 2 * pu) / 3 + pu, ((1 - t) / 2) * (0.5 - 2 * pv) + pv + 0.5 * row];
  }
  return [(col + (s + 1) / 2) / L.cols, (row + (1 - t) / 2) / L.rows];
}

/** Direction of an equirectangular pixel position (u, v in 0–1). */
export function uvToDir(u, v) {
  const lon = (u - 0.5) * 2 * Math.PI, lat = (0.5 - v) * Math.PI;
  return [Math.cos(lat) * Math.sin(lon), Math.sin(lat), Math.cos(lat) * Math.cos(lon)];
}

/** GLSL for the same mapping, with the face tables baked in. */
export function glslLayout(tables = FACE_TABLES) {
  const faceIdx = { r: 0, l: 1, u: 2, d: 3, f: 4, b: 5 };
  const arr = (layout, i) => FACE_KEYS.map(k => tables[layout][k][i]).join(', ');
  const ids = Object.keys(LAYOUTS);
  return `
const float PI = 3.141592653589793;
const int LAYOUT_ERP = ${ids.indexOf('erp')}, LAYOUT_TB = ${ids.indexOf('erp-tb')}, LAYOUT_LR = ${ids.indexOf('erp-lr')};
${['c3x2', 'c6x1', 'c1x6', 'eac'].map(l => `const int T_${l}[6] = int[6](${arr(l, 0)}); const int R_${l}[6] = int[6](${arr(l, 1)}); const int F_${l}[6] = int[6](${arr(l, 2)});`).join('\n')}
vec2 layoutUV(int lay, vec3 d, vec2 size) {
  if (lay == LAYOUT_ERP || lay == LAYOUT_TB || lay == LAYOUT_LR) {
    vec2 uv = vec2(atan(d.x, d.z) / (2.0 * PI) + 0.5, 0.5 - asin(clamp(normalize(d).y, -1.0, 1.0)) / PI);
    if (lay == LAYOUT_TB) uv.y *= 0.5;
    if (lay == LAYOUT_LR) uv.x *= 0.5;
    return uv;
  }
  vec3 a = abs(d); int f; vec3 c, sv, tv;
  if (a.x >= a.y && a.x >= a.z) { f = d.x > 0.0 ? ${faceIdx.r} : ${faceIdx.l}; }
  else if (a.y >= a.z) { f = d.y > 0.0 ? ${faceIdx.u} : ${faceIdx.d}; }
  else { f = d.z > 0.0 ? ${faceIdx.f} : ${faceIdx.b}; }
  ${FACE_KEYS.map((k, i) => `${i ? 'else ' : ''}if (f == ${i}) { c = vec3(${FACES[k].c}); sv = vec3(${FACES[k].s}); tv = vec3(${FACES[k].t}); }`).join('\n  ')}
  float m = dot(d, c), s = dot(d, sv) / m, t = dot(d, tv) / m;
  int tile, rot, flip; float cols, rows;
  ${['c3x2', 'c6x1', 'c1x6', 'eac'].map((l, i) => `${i ? 'else ' : ''}if (lay == ${ids.indexOf(l)}) { tile = T_${l}[f]; rot = R_${l}[f]; flip = F_${l}[f]; cols = ${LAYOUTS[l].cols}.0; rows = ${LAYOUTS[l].rows}.0; }`).join('\n  ')}
  if (lay == ${ids.indexOf('eac')}) { s = atan(s) * 4.0 / PI; t = atan(t) * 4.0 / PI; }
  if (flip == 1) s = -s;
  for (int k = 0; k < 3; k++) if (k < rot) { float q = s; s = -t; t = q; }
  float col = mod(float(tile), cols), row = floor(float(tile) / cols);
  if (lay == ${ids.indexOf('eac')}) {
    vec2 pad = 2.0 / size;
    return vec2((col + (s + 1.0) * 0.5) * (1.0 - 2.0 * pad.x) / 3.0 + pad.x, (1.0 - t) * 0.5 * (0.5 - 2.0 * pad.y) + pad.y + 0.5 * row);
  }
  return vec2((col + (s + 1.0) * 0.5) / cols, (row + (1.0 - t) * 0.5) / rows);
}`;
}
export const layoutIndex = layout => Object.keys(LAYOUTS).indexOf(layout);
