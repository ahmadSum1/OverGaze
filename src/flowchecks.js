// Checks applied to a motion field (u, v in pixels, from frame A to frame B, W × H, optionally wrapping left/right):
//   stillAreas      where A and B match within the noise level over a small neighbourhood, motion = 0
//   forwardBackward where the motion from A to B and back from B to A do not cancel out, motion = 0
//   poleBands       near the top and bottom edges of a 360° frame (beyond ±poleDeg latitude) the picture is
//                   stretched and motion is not trusted: the horizontal motion is taken from the last trusted row
//                   (a turning camera moves those rows sideways by the same amount), the vertical motion is 0

function boxMean(src, W, H, r, wrap) {
  const tmp = new Float32Array(W * H), out = new Float32Array(W * H), n = (2 * r + 1) ** 2;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    let s = 0;
    for (let k = -r; k <= r; k++) { const xx = wrap ? (x + k + W) % W : Math.min(W - 1, Math.max(0, x + k)); s += src[y * W + xx]; }
    tmp[y * W + x] = s;
  }
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    let s = 0;
    for (let k = -r; k <= r; k++) s += tmp[Math.min(H - 1, Math.max(0, y + k)) * W + x];
    out[y * W + x] = s / n;
  }
  return out;
}

/**
 * Motion 0 where the mean absolute difference of A and B over a (2r+1)² neighbourhood is below `noise` (0–1 scale)
 * and the neighbourhood has texture to tell (mean brightness step to the next pixel above `texture`): a textured
 * patch that did not change did not move. A flat patch (sky, a wall) looks the same whether it moved or not, so its
 * motion is left to the estimate, which takes it from the textured parts around it.
 */
export function stillAreas(A, B, W, H, u, v, { radius = 2, noise = 3 / 255, texture = 8 / 255, wrap = true } = {}) {
  const d = new Float32Array(W * H), g = new Float32Array(W * H);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const i = y * W + x, r = wrap ? y * W + (x + 1) % W : y * W + Math.min(W - 1, x + 1), dn = Math.min(H - 1, y + 1) * W + x;
    d[i] = Math.abs(A[i] - B[i]); g[i] = Math.abs(A[r] - A[i]) + Math.abs(A[dn] - A[i]);
  }
  const m = boxMean(d, W, H, radius, wrap), tex = boxMean(g, W, H, radius, wrap);
  let n = 0;
  for (let i = 0; i < d.length; i++) if (m[i] < noise && tex[i] > texture) { u[i] = 0; v[i] = 0; n++; }
  return n / d.length;
}

function sample(f, W, H, x, y, wrap) {
  if (wrap) x -= W * Math.floor(x / W); else x = Math.min(W - 1, Math.max(0, x));
  y = Math.min(H - 1, Math.max(0, y));
  const x0 = Math.floor(x), y0 = Math.floor(y), fx = x - x0, fy = y - y0;
  const x1 = wrap ? (x0 + 1) % W : Math.min(W - 1, x0 + 1), y1 = Math.min(H - 1, y0 + 1);
  return (f[y0 * W + x0] * (1 - fx) + f[y0 * W + x1] * fx) * (1 - fy) + (f[y1 * W + x0] * (1 - fx) + f[y1 * W + x1] * fx) * fy;
}

/** Motion 0 where forward (u, v) and backward (ub, vb) disagree by more than tol + rel × the motion's length. */
export function forwardBackward(u, v, ub, vb, W, H, { tol = 0.75, rel = 0.1, wrap = true } = {}) {
  let n = 0;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const i = y * W + x, xb = x + u[i], yb = y + v[i];
    const ex = u[i] + sample(ub, W, H, xb, yb, wrap), ey = v[i] + sample(vb, W, H, xb, yb, wrap);
    if (Math.hypot(ex, ey) > tol + rel * Math.hypot(u[i], v[i])) { u[i] = 0; v[i] = 0; n++; }
  }
  return n / (W * H);
}

/** Rows beyond ±poleDeg latitude: horizontal motion of the last trusted row, vertical motion 0. */
export function poleBands(u, v, W, H, { poleDeg = 70 } = {}) {
  const top = Math.max(0, Math.floor((90 - poleDeg) / 180 * H)), bottom = Math.min(H - 1, H - 1 - top);
  for (let y = 0; y < H; y++) {
    if (y >= top && y <= bottom) continue;
    const src = y < top ? top : bottom;
    for (let x = 0; x < W; x++) { u[y * W + x] = u[src * W + x]; v[y * W + x] = 0; }
  }
}
