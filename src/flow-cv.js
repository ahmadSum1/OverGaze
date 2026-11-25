// Dense optical flow with OpenCV's Farnebäck method (OpenCV.js, lib/opencv/opencv.js, loaded only when needed).
// For 360° video the picture is padded with its other side first, so motion across the left/right edge is found.

export const FARNEBACK = { pyrScale: 0.5, levels: 4, winsize: 15, iterations: 3, polyN: 5, polySigma: 1.1 };

/** Flow from A to B (Float32 greyscale 0–1, W × H): { u, v } in pixels. `cv` is the loaded OpenCV.js module. */
export function farneback(cv, A, B, W, H, wrap = true, p = FARNEBACK) {
  const pad = wrap ? Math.round(W / 8) : 0, PW = W + 2 * pad;
  const toMat = I => {
    const m = new cv.Mat(H, PW, cv.CV_8UC1), d = m.data;
    for (let y = 0; y < H; y++) for (let x = 0; x < PW; x++) {
      const sx = ((x - pad) % W + W) % W;
      d[y * PW + x] = Math.max(0, Math.min(255, Math.round(I[y * W + sx] * 255)));
    }
    return m;
  };
  const a = toMat(A), b = toMat(B), flow = new cv.Mat();
  try {
    cv.calcOpticalFlowFarneback(a, b, flow, p.pyrScale, p.levels, p.winsize, p.iterations, p.polyN, p.polySigma, 0);
    const f = flow.data32F, u = new Float32Array(W * H), v = new Float32Array(W * H);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) { const k = 2 * (y * PW + x + pad); u[y * W + x] = f[k]; v[y * W + x] = f[k + 1]; }
    return { u, v };
  } finally { a.delete(); b.delete(); flow.delete(); }
}
