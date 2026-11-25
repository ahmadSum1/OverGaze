// Motion between two greyscale frames, off the main thread.
//   estimate 'farneback': OpenCV's Farnebäck method (OpenCV.js from lib/opencv, loaded on the first request)
//   estimate 'builtin':   OverGaze's own estimate (flow.js)
// Then the checks of flowchecks.js: the motion back from B to A must cancel the motion from A to B, textured
// patches that did not change get no motion, and rows near the top and bottom of a 360° frame take the
// horizontal motion of the last trusted row.
import { opticalFlow } from './flow.js';
import { farneback } from './flow-cv.js';
import { forwardBackward, stillAreas, poleBands } from './flowchecks.js';

let cvReady = null;
function loadOpenCv() {
  if (!cvReady) cvReady = (async () => {
    const res = await fetch(new URL('../lib/opencv/opencv.js', import.meta.url));
    if (!res.ok) throw new Error(`OpenCV.js could not be loaded (${res.status})`);
    (0, eval)(await res.text());   // its loader sets self.cv to a module that finishes starting later
    const m = self.cv;
    return await new Promise((resolve, reject) => {
      if (m.Mat) { resolve(m); return; }
      const t = setTimeout(() => reject(new Error('OpenCV.js did not start')), 60000);
      m.then(c => { clearTimeout(t); delete c.then; resolve(c); });
    });
  })();
  return cvReady;
}

self.onmessage = async event => {
  const { id, a, b, W, H, wrap, estimate = 'farneback', checks = true, load } = event.data;
  try {
    if (load) { await loadOpenCv(); self.postMessage({ id, loaded: true }); return; }
    const cv = estimate === 'farneback' ? await loadOpenCv() : null;
    const flow = (p, q) => (cv ? farneback(cv, p, q, W, H, wrap) : opticalFlow(p, q, W, H, { wrap }));
    const { u, v } = flow(a, b);
    if (checks) {
      const back = flow(b, a);
      forwardBackward(u, v, back.u, back.v, W, H, { wrap });
      stillAreas(a, b, W, H, u, v, { wrap });
      if (wrap) poleBands(u, v, W, H);
    }
    self.postMessage({ id, u, v }, [u.buffer, v.buffer]);
  } catch (error) {
    self.postMessage({ id, error: String(error?.message || error) });
  }
};
