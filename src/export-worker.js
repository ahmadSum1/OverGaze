// Renders saliency maps for export off the main thread and encodes them (PNG) or quantises them (NPY).
import { coverage, renderMap } from './saliency.js';
import { FlowStore } from './flow.js';
import { encodeGrayPng, npyBody } from './export.js';

let job = null;

self.onmessage = async event => {
  const m = event.data;
  try {
    if (m.cmd === 'init') {
      job = {
        gaze: m.gaze, settings: m.settings, W: m.W, H: m.H, format: m.format, work: {},
        ctx: { fps: m.fps, observerEnabled: m.observerEnabled, conditionEnabled: m.conditionEnabled, flow: m.flow ? FlowStore.from(m.flow) : null }
      };
      self.postMessage({ cmd: 'ready' });
      return;
    }
    if (m.cmd === 'frame') {
      const { gaze, settings, W, H, format, ctx, work } = job;
      const r = renderMap(gaze, m.t, settings, W, H, ctx, work);
      const cov = coverage(r.map, W, H, settings.coverageThreshold ?? 0.1, settings.projection !== 'planar');
      const bytes = format === 'png8' || format === 'png16'
        ? await encodeGrayPng(r.map, W, H, format === 'png16' ? 16 : 8)
        : npyBody(r.map, format === 'npyu1' ? '|u1' : '<f4');
      self.postMessage({ cmd: 'frame', seq: m.seq, bytes, points: r.points.n, max: r.max, cov }, [bytes.buffer]);
    }
  } catch (error) {
    self.postMessage({ cmd: 'error', seq: m.seq, error: String(error?.message || error) });
  }
};
