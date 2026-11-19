import { opticalFlow } from './flow.js';

self.onmessage = event => {
  const { id, a, b, W, H, wrap } = event.data;
  try {
    const { u, v } = opticalFlow(a, b, W, H, { wrap });
    self.postMessage({ id, u, v }, [u.buffer, v.buffer]);
  } catch (error) {
    self.postMessage({ id, error: String(error?.message || error) });
  }
};
