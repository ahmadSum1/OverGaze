// Regenerates docs/screenshots/*.png from the bundled samples (no private data):  node tools/screenshots.mjs
import { mkdirSync } from 'node:fs';
import { join } from 'node:path';
import { launch } from './cdp.mjs';
import { ROOT, serve } from './serve.mjs';

const out = join(ROOT, 'docs', 'screenshots');
mkdirSync(out, { recursive: true });
const { server, url } = await serve();
const b = await launch({ width: 1500, height: 900 });
const sleep = ms => new Promise(r => setTimeout(r, ms));
const ready = async q => { await b.goto(url + q); await b.waitFor('!!overgaze.state.gaze && document.querySelector("#video").readyState >= 2', 20000); };
const seek = async t => { await b.eval(`document.querySelector('#video').currentTime = ${t}; 1`); await sleep(1200); };
const set = id_value => b.eval(`for (const [id, v] of ${JSON.stringify(id_value)}) { const e = document.querySelector('#' + id); if (e.type === 'checkbox') e.checked = v; else e.value = v; e.dispatchEvent(new Event('change')); } 1`);
const page = async name => { await b.eval(`window.scrollTo(0, 0); 1`); await b.screenshot(join(out, name)); };
// a region of the page: scroll it into view inside its scrolling panel, then clip to its box
async function region(name, selector, pad = 8) {
  await b.eval(`document.querySelector(${JSON.stringify(selector)}).scrollIntoView({ block: 'start' }); 1`);
  await sleep(200);
  const r = await b.eval(`(() => { const r = document.querySelector(${JSON.stringify(selector)}).getBoundingClientRect(); return { x: r.x + scrollX, y: r.y + scrollY, width: r.width, height: r.height }; })()`);
  await b.screenshot(join(out, name), { x: Math.max(0, r.x - pad), y: Math.max(0, r.y - pad), width: r.width + 2 * pad, height: Math.min(r.height + 2 * pad, 900) });
}

try {
  // 1. overview: 360° video, fixations of 8 observers, live heatmap
  await ready('?sample');
  await seek(4);
  await page('01-overview.png');

  // 2. temporal pooling smears the map along the ball's path; motion compensation pulls it back
  await b.eval(`overgaze.applySettings({ sigmaDeg: 6, windowSec: 1, kernel: 'gaussian' }); 1`);
  await seek(4);
  await page('02-pooling-without-motion-compensation.png');
  await set([['flow-method', 'seek']]);
  await b.click('#analyse');
  await b.waitFor('!!overgaze.state.flow', 120000, 'motion analysis');
  await seek(4);
  await page('03-motion-compensated.png');
  await set([['show-flow', true]]);
  await sleep(500);
  await page('04-flow-vectors.png');
  await set([['show-flow', false]]);

  // coverage readout under the video
  await region('10-coverage-readout.png', '.readouts');

  // 5. display options: spotlight, other colour map
  await set([['mode', 'spotlight']]);
  await page('05-spotlight.png');
  await set([['mode', 'heat']]);

  // 6. settings panels
  await region('06-pooling-settings.png', '.settings-section:has(#window)');
  await region('06a-motion-settings.png', '.settings-section:has(#motion)');
  await region('06b-display-settings.png', '.settings-section:has(#mode)');

  // 7. column mapping of the loaded gaze file
  await region('07-column-mapping.png', '#mapping-section');

  // 8. export of maps (NPY) with settings saved alongside
  await set([['ex-start', '2'], ['ex-end', '2.2'], ['ex-width', '256'], ['ex-format', 'npyf4'], ['ex-norm', 'none']]);
  await b.click('#export-maps');
  await b.waitFor('document.querySelector("#notice").textContent.startsWith("Exported")', 60000, 'export');
  await region('08-export.png', '.settings-section:has(#export-maps)');

  // 9. ordinary 2D video, gaze in pixels
  await ready('?sample=2d&sigma=3&window=0.5&kernel=gaussian');
  await seek(3);
  await page('09-2d-video.png');
  console.log('screenshots written to', out, b.logs.length ? '\nconsole: ' + b.logs.join('\n') : '');
} finally { b.close(); server.close(); }
