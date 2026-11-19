// Frame-rate estimation from the gaps between decoded frame times.

export const COMMON_FPS = [23.976, 24, 25, 29.97, 30, 48, 50, 59.94, 60, 90, 100, 120];
export function snapFps(f) {
  const best = COMMON_FPS.reduce((a, c) => Math.abs(c - f) < Math.abs(a - f) ? c : a);
  return Math.abs(best - f) / best < 0.004 ? best : Math.round(f * 1000) / 1000;
}
export function fpsFromDeltas(deltas) {
  const d = deltas.filter(x => x > 0.002 && x < 0.2).sort((a, b) => a - b);
  if (d.length < 8) return null;
  // WebM stores times in whole milliseconds (29.97 fps alternates 33 and 34 ms), so average the typical gaps
  // instead of taking one of them; gaps far from the typical one (dropped or repeated frames) are ignored.
  const typical = d[d.length >> 1], inliers = d.filter(x => Math.abs(x - typical) < 0.5 * typical);
  return snapFps(inliers.length / inliers.reduce((a, b) => a + b, 0));
}
