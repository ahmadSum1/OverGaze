import { FILE_TIME, buildGaze, columnValues, dropObservers, defaultRanges, detectMapping, gazeToCsv, parseTable, pathMatchesVideo, velocityFilter } from './gaze.js';
import { DEFAULTS, coverage, frameCentre, frameOf, mapSize, renderMap } from './saliency.js';
import { FlowStore, toGray } from './flow.js';
import { ZipWriter, encodeGrayPng, npyHeader } from './export.js';
import { fpsFromDeltas } from './timing.js';
import { compareMaps, openReference, resample } from './reference.js';
import { LayoutConverter, VIEW_MODES, View360 } from './view360.js';
import { CUBE_LAYOUTS, DEFAULT_OPTIONS, LAYOUTS, erpSize, faceTables } from './projections.js';
import { COLORMAPS, colorize, cssGradient } from './colormap.js';

const VERSION = '1.0.0';
const $ = id => document.getElementById(id);
const video = $('video');
const refCanvas = $('ref-canvas'), heatCanvas = $('heat-canvas'), marksCanvas = $('marks-canvas'), densityCanvas = $('density-canvas');
const GAZE_EXT = /\.(csv|tsv|txt|json|arff|dat)$/i, VIDEO_EXT = /\.(mp4|m4v|webm|mkv|mov|ogv)$/i;

const state = {
  video: null, fps: 30, fpsSource: 'default', fpsDeltas: [], lastFrameMeta: null, mediaTime: 0,
  files: [], mapping: null, gazeAll: null, gaze: null, observerEnabled: null, conditionEnabled: null,
  settings: { ...DEFAULTS, velocity: 0 },
  display: { mode: 'heat', colormap: 'jet', opacity: 0.65, gamma: 1, mapWidth: 512, showPoints: true, showFlow: false },
  flow: null, flowInfo: null, analysing: null, exporting: null, recording: null,
  work: {}, last: null, renderQueued: false, projectionTouched: false, density: null,
  ref: null, refShow: { heat: false, outline: true }, showGaze: true, removedObservers: new Set(), view: null,
  layout: 'erp', layoutTouched: false, layoutOpts: { ...DEFAULT_OPTIONS }, converter: null
};

// Handle for the tests and the console: mapAt(frame, width) renders the same map the export writes (raw, unnormalised).
window.overgaze = {
  state, debugReport: () => debugReport(), applySettings: o => applySettings(o),
  mapAt(k, width = 512, normalize = 'none') {
    const [vw, vh] = videoSize(), [W, H] = mapSize(width, state.settings, vw, vh);
    const r = renderMap(state.gaze, frameCentre(k, state.fps), settingsForRender(normalize), W, H, renderContext(), {});
    return { W, H, map: Array.from(r.map), points: r.points.n };
  }
};

// ---------- helpers ----------
const fmt = (x, d = 3) => Number.isFinite(x) ? x.toFixed(d) : '—';
const pct = x => Number.isFinite(x) ? `${(100 * x).toFixed(2)} %` : '—';
const bytesText = b => b > 1e9 ? `${(b / 1e9).toFixed(2)} GB` : b > 1e6 ? `${(b / 1e6).toFixed(1)} MB` : `${Math.max(1, Math.round(b / 1e3))} kB`;
const obsColor = i => `hsl(${(i * 137.508) % 360} 80% 62%)`;
function setStatus(text) { $('status').textContent = text; }
const debugLog = [];
function remember(kind, text) { if (text) { debugLog.push(`${new Date().toISOString().slice(11, 23)} ${kind}: ${text}`); if (debugLog.length > 200) debugLog.shift(); } }
window.addEventListener('error', e => remember('exception', `${e.message} (${(e.filename || '').split('/').pop()}:${e.lineno})`));
window.addEventListener('unhandledrejection', e => remember('unhandled promise', String(e.reason?.stack || e.reason)));
function notice(text, warn = false) {
  remember(warn ? 'warning' : 'notice', text);
  const n = $('notice');
  n.hidden = !text; $('notice-text').textContent = text || ''; n.classList.toggle('warn', warn);
}
function showError(text) { remember('error', text); const e = $('error-message'); e.hidden = !text; $('error-text').textContent = text || ''; }
function download(blob, name) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = name;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 60000);
}
function stem(name) { return (name || 'video').replace(/[?#].*$/, '').split('/').pop().replace(/\.[^.]+$/, '') || 'video'; }
function rawVideoSize() { return [video.videoWidth || 2, video.videoHeight || 1]; }
/** Size of the (equirectangular) frame the overlay works on: the video itself, or what its layout unfolds to. */
function videoSize() { const [w, h] = rawVideoSize(); return erpSize(state.layout, w, h, state.layoutOpts); }
const converting = () => state.layout !== 'erp';
const grabConverter = () => new LayoutConverter(undefined, state.layoutOpts);
/** The picture under the overlay: the video, or its layout converted to equirectangular (drawn on demand). */
function frameImage() {
  if (!converting()) return video;
  const [w, h] = rawVideoSize(), [W, H] = videoSize(), s = Math.min(1, 2048 / W);
  try {
    const key = JSON.stringify(state.layoutOpts);
    if (!state.converter || state.converterKey !== key) {
      // a new shader for other face options; the canvas keeps one WebGL context, so swap in a fresh canvas
      const old = $('erp-canvas'), c = old.cloneNode(false);
      if (state.converter) { old.replaceWith(c); }
      state.converter = new LayoutConverter(state.converter ? c : old, state.layoutOpts); state.converterKey = key;
    }
    return state.converter.convert(video, state.layout, w, h, Math.round(W * s), Math.round(H * s));
  } catch (e) { showError(`Converting the ${LAYOUTS[state.layout].label} video needs WebGL2 and readable pixels: ${e.message}`); return video; }
}
/** Guess the layout from the frame shape: only exact shapes, since 16:9 360° videos are usually stretched equirectangular. */
function guessLayout(w, h) {
  const r = w / h, near = x => Math.abs(r - x) / x < 0.01;
  return near(2) ? 'erp' : near(1) ? 'erp-tb' : near(4) ? 'erp-lr' : near(1.5) ? 'eac' : near(6) ? 'c6x1' : near(1 / 6) ? 'c1x6' : r > 2.2 ? 'cyl' : 'erp';
}
function setLayout(layout, touched = false) {
  state.layout = LAYOUTS[layout] ? layout : 'erp';
  if (touched) state.layoutTouched = true;
  $('layout').value = state.layout;
  $('erp-canvas').hidden = !converting();
  $('cube-options').hidden = !CUBE_LAYOUTS.includes(state.layout); $('cyl-options').hidden = state.layout !== 'cyl';
  const [vw, vh] = videoSize();
  $('stage').style.setProperty('--aspect', String(state.view ? 16 / 9 : vw / vh));
  if (state.flow) { state.flow = null; state.flowInfo = null; updateFlowUi(); }
  updateSubtitle(); updateButtons(); requestRender();
}
function settingsForRender(normalize = 'max') { return { ...state.settings, normalize }; }
function renderContext() {
  return { fps: state.fps, tol: 0.5 / state.fps, observerEnabled: state.observerEnabled, conditionEnabled: state.conditionEnabled, flow: state.settings.motionComp ? state.flow : null };
}
function currentFrame() {
  const k = frameOf(video.paused ? video.currentTime : state.mediaTime, state.fps);
  return [k, frameCentre(k, state.fps)];
}

// ---------- video ----------
function isYouTube(url) { return /(^|\.)(youtube\.com|youtu\.be|youtube-nocookie\.com)$/i.test(new URL(url, location.href).hostname); }

function resetVideoState() {
  state.flow = null; state.flowInfo = null; state.fpsDeltas = []; state.fpsSource = 'default'; state.mediaTime = 0;
  updateFlowUi();
}

function loadVideoFile(file) {
  showError('');
  if (state.video?.blobUrl) URL.revokeObjectURL(state.video.blobUrl);
  const src = URL.createObjectURL(file);
  state.video = { name: file.name, size: file.size, src, blobUrl: src, remote: false, tainted: false };
  resetVideoState();
  video.removeAttribute('crossorigin');
  video.src = src;
  setStatus('Loading video…');
}

function loadVideoUrl(url, { sameOrigin = false } = {}) {
  showError(''); notice('');
  try {
    if (!sameOrigin && isYouTube(url)) {
      showError('YouTube pages cannot be used directly: the browser gets no access to the frames, and YouTube shows a 360° viewport rather than the equirectangular frame. Download the video file (for example with yt-dlp) and load it from disk, or use a direct link to an .mp4 / .webm file.');
      return;
    }
  } catch { showError('That link is not a valid URL.'); return; }
  if (state.video?.blobUrl) URL.revokeObjectURL(state.video.blobUrl);
  state.video = { name: stem(url) + (url.match(/\.[a-z0-9]{2,4}(?=$|[?#])/i)?.[0] || ''), size: null, src: url, remote: !sameOrigin, tainted: false, triedPlain: false };
  resetVideoState();
  if (!sameOrigin) video.crossOrigin = 'anonymous'; else video.removeAttribute('crossorigin');
  video.src = url;
  setStatus('Loading video…');
}

video.addEventListener('error', () => {
  const v = state.video;
  if (!v) return;
  if (v.remote && !v.triedPlain && video.crossOrigin) {
    // The server does not send CORS headers: retry for display only.
    v.triedPlain = true; v.tainted = true;
    video.removeAttribute('crossorigin');
    video.src = v.src;
    return;
  }
  // Some embedded browsers (the VS Code browser, other Electron apps, Chromium builds) ship without H.264.
  const noH264 = /\.(mp4|m4v|mov)$/i.test(v.name) && !document.createElement('video').canPlayType('video/mp4; codecs="avc1.42E01E"');
  showError(`The video could not be loaded (${video.error?.message || 'unsupported format or unreachable link'}). ` + (noH264
    ? 'This browser cannot play H.264 (MP4) video at all; built-in browsers such as the one in VS Code lack it. Open the page in Chrome, Edge or Firefox, or convert the video to VP9: ffmpeg -i in.mp4 -c:v libvpx-vp9 -crf 32 -b:v 0 out.webm'
    : (v.size > 2e9
      ? `The file is large (${bytesText(v.size)}); built-in browsers such as the one in VS Code cannot open files this big, Chrome can. Or cut out the part you need (re-encoded, so it starts exactly at the start time): ffmpeg -ss <start s> -i in.webm -t 30 -vf scale=1920:960 -c:v libvpx-vp9 -crf 32 -b:v 0 clip.webm`
      : 'Browsers play H.264 / VP9 / AV1 in MP4 or WebM; H.265 works only in some browsers.')));
  setStatus('Video failed');
});

video.addEventListener('loadedmetadata', () => {
  const [rw, rh] = rawVideoSize();
  if (!state.layoutTouched) {
    const guess = guessLayout(rw, rh);
    if (guess !== state.layout) setLayout(guess);
    if (guess !== 'erp') notice(`The video is ${rw} × ${rh}, so it was taken as “${LAYOUTS[guess].label}”. Change “Video layout” if that is wrong${guess === 'eac' ? ' (3 × 2 frames are either a cubemap or YouTube’s equi-angular cubemap)' : ''}.`, true);
  }
  const [vw, vh] = videoSize();
  $('stage').style.setProperty('--aspect', String(state.view ? 16 / 9 : vw / vh));
  $('empty-state').hidden = true; $('stage-wrap').hidden = false;
  if (!state.projectionTouched) {
    const erp = Math.abs(vw / vh - 2) < 0.02 || converting();
    state.settings.projection = erp ? 'erp' : 'planar';
    $('projection').value = state.settings.projection; $('fov-row').hidden = erp;
  }
  $('video-name').textContent = state.video.name; $('video-name').title = state.video.name;
  $('video-size').textContent = `${rw} × ${rh}${state.video.size ? ' · ' + bytesText(state.video.size) : ''}`;
  $('video-duration').textContent = `${fmt(video.duration, 2)} s`;
  $('viewer-name').textContent = state.video.name;
  $('scrubber').max = String(video.duration || 1);
  $('ex-start').value = '0'; $('ex-end').value = fmt(video.duration, 2);
  ['play', 'prev-frame', 'next-frame', 'scrubber'].forEach(id => { $(id).disabled = false; });
  setStatus('Video ready');
  // Gaze in pixels that was loaded before the video: now the frame size is known, so are the value ranges.
  const m = state.mapping;
  if (m && m.coord === 'pixels' && !m.rangesTouched && state.files.length) {
    const t = state.files[0].table;
    Object.assign(m, defaultRanges('pixels', columnValues(t, m.x), columnValues(t, m.y), { width: vw, height: vh }));
    populateMappingForm(); rebuildGaze();
  }
  updateSubtitle(); updateButtons(); updateExportSummary(); drawDensity();
  detectFps();
});

video.addEventListener('loadeddata', () => {
  const v = state.video;
  if (!v.tainted) {
    try {
      const c = document.createElement('canvas'); c.width = c.height = 2;
      const g = c.getContext('2d'); g.drawImage(video, 0, 0, 2, 2); g.getImageData(0, 0, 1, 1);
    } catch { v.tainted = true; }
  }
  $('video-access').textContent = v.tainted ? 'display only (no CORS)' : 'readable';
  if (v.tainted) notice('This link is shown without access to its pixels (the server sends no CORS headers). The heatmap overlay and map export work; motion analysis, snapshots and overlay recording need a local file or a CORS-enabled link.', true);
  updateButtons(); requestRender();
});

function updateSubtitle() {
  const parts = [];
  if (state.video) { const [vw, vh] = rawVideoSize(); parts.push(`${vw}×${vh}${converting() ? ' ' + state.layout : ''}`, `${fmt(state.fps, 3)} fps (${state.fpsSource})`, state.settings.projection === 'erp' ? '360° ERP' : '2D'); }
  if (state.gaze) parts.push(`${state.gaze.n.toLocaleString()} gaze rows, ${state.gaze.observers.length} observers`);
  $('viewer-subtitle').textContent = parts.join(' · ') || 'Load a video and gaze data';
}

function setFps(f, source) {
  if (!(f > 0)) return;
  state.fps = f; state.fpsSource = source;
  $('fps').value = String(f);
  $('fps-note').textContent = source === 'manual' ? 'Set by hand.' : source === 'measured' ? `Measured from decoded frames: ${f} fps.` : 'Default 30. Measured from decoded frames once the video plays.';
  if (state.files.length && state.mapping && (/^frame/.test(state.mapping.timeUnit) || /^frame/.test(state.mapping.durationUnit))) rebuildGaze();
  updateSubtitle(); updateExportSummary(); drawDensity(); requestRender();
}

async function detectFps() {
  if (!('requestVideoFrameCallback' in HTMLVideoElement.prototype) || !state.video) return;
  const probe = hiddenVideo();
  try {
    await once(probe, 'loadeddata', 8000);
    probe.currentTime = Math.min(1, (probe.duration || 2) / 4);
    await once(probe, 'seeked', 8000);
    const deltas = [];
    let last = null;
    await probe.play();
    await new Promise(resolve => {
      const stop = setTimeout(resolve, 2500);
      const cb = (now, meta) => {
        if (last && meta.presentedFrames === last.presentedFrames + 1) deltas.push(meta.mediaTime - last.mediaTime);
        last = meta;
        if (deltas.length >= 40) { clearTimeout(stop); resolve(); } else probe.requestVideoFrameCallback(cb);
      };
      probe.requestVideoFrameCallback(cb);
    });
    const f = fpsFromDeltas(deltas);
    if (f && state.fpsSource !== 'manual') setFps(f, 'measured');
  } catch { /* fall back to measuring during playback */ }
  finally { disposeHidden(probe); }
}

function hiddenVideo() {
  const v = document.createElement('video');
  v.muted = true; v.playsInline = true; v.preload = 'auto';
  if (state.video.remote && !state.video.tainted) v.crossOrigin = 'anonymous';
  v.style.cssText = 'position:fixed;left:-10px;top:-10px;width:4px;height:4px;opacity:0;pointer-events:none';
  document.body.appendChild(v);
  v.src = state.video.src;
  return v;
}
function disposeHidden(v) { try { v.pause(); v.removeAttribute('src'); v.load(); v.remove(); } catch { /* ignore */ } }
function once(target, event, timeout = 10000) {
  return new Promise((resolve, reject) => {
    const t = setTimeout(() => { cleanup(); reject(new Error(`timeout waiting for ${event}`)); }, timeout);
    const ok = () => { cleanup(); resolve(); }, bad = () => { cleanup(); reject(new Error('media error')); };
    const cleanup = () => { clearTimeout(t); target.removeEventListener(event, ok); target.removeEventListener('error', bad); };
    target.addEventListener(event, ok); target.addEventListener('error', bad);
  });
}

// ---------- playback ----------
function onVideoFrame(now, meta) {
  state.mediaTime = meta.mediaTime;
  const last = state.lastFrameMeta;
  if (last && meta.presentedFrames === last.presentedFrames + 1 && video.playbackRate === 1) {
    state.fpsDeltas.push(meta.mediaTime - last.mediaTime);
    if (state.fpsDeltas.length === 60 && state.fpsSource === 'default') { const f = fpsFromDeltas(state.fpsDeltas); if (f) setFps(f, 'measured'); }
  }
  state.lastFrameMeta = meta;
  render();
  state.recording?.onFrame(meta);
  if (!video.paused && !video.ended) video.requestVideoFrameCallback(onVideoFrame);
}
function rafLoop() {
  if (video.paused || video.ended) return;
  state.mediaTime = video.currentTime; render();
  requestAnimationFrame(rafLoop);
}
video.addEventListener('play', () => {
  $('play').textContent = '❚❚'; $('play').setAttribute('aria-label', 'Pause');
  state.lastFrameMeta = null;
  if ('requestVideoFrameCallback' in video) video.requestVideoFrameCallback(onVideoFrame); else requestAnimationFrame(rafLoop);
});
video.addEventListener('pause', () => { $('play').textContent = '▶'; $('play').setAttribute('aria-label', 'Play'); state.mediaTime = video.currentTime; requestRender(); });
video.addEventListener('seeked', () => { state.mediaTime = video.currentTime; requestRender(); });
video.addEventListener('ended', () => { state.recording?.finish(); });

function togglePlay() { if (!state.video) return; if (video.paused) video.play().catch(e => showError(String(e.message || e))); else video.pause(); }
function stepFrames(n) {
  if (!state.video) return;
  video.pause();
  const k = frameOf(video.currentTime, state.fps) + n;
  const maxK = Math.max(0, Math.ceil(video.duration * state.fps) - 1);
  video.currentTime = frameCentre(Math.max(0, Math.min(maxK, k)), state.fps);
}
function seekSeconds(dt) { if (!state.video) return; video.currentTime = Math.max(0, Math.min(video.duration, video.currentTime + dt)); }

// ---------- rendering ----------
function requestRender() {
  if (state.renderQueued) return;
  state.renderQueued = true;
  requestAnimationFrame(() => { state.renderQueued = false; render(); });
}

function render() {
  if (!state.video || !video.videoWidth) return;
  const [k, t] = currentFrame();
  $('time-label').textContent = `${fmt(video.paused ? video.currentTime : state.mediaTime, 3)} s · frame ${k}`;
  if (document.activeElement !== $('scrubber')) $('scrubber').value = String(video.paused ? video.currentTime : state.mediaTime);
  if (converting()) frameImage();
  const [vw, vh] = videoSize();
  const [W, H] = mapSize(state.display.mapWidth, state.settings, vw, vh);
  if (heatCanvas.width !== W || heatCanvas.height !== H) { heatCanvas.width = W; heatCanvas.height = H; }
  const hctx = heatCanvas.getContext('2d');
  const thr = state.settings.coverageThreshold;
  let result = null;
  const ref = state.ref ? referenceAt(k, W, H) : null;
  if (!state.work.image || state.work.image.width !== W || state.work.image.height !== H) state.work.image = hctx.createImageData(W, H);
  const img = state.work.image, colorOpts = { colormap: state.display.colormap, opacity: state.display.opacity, gamma: state.display.gamma, mode: state.display.mode };
  img.data.fill(0);
  if (state.gaze) {
    const t0 = performance.now();
    result = renderMap(state.gaze, t, settingsForRender('max'), W, H, renderContext(), state.work);
    const ms = performance.now() - t0;
    const cov = coverage(result.map, W, H, thr, state.settings.projection === 'erp');
    const observers = new Set();
    for (let i = 0; i < result.points.n; i++) observers.add(result.points.obs[i]);
    $('ro-points').textContent = result.points.n.toLocaleString();
    $('ro-observers').textContent = `${observers.size} / ${state.observerEnabled ? state.observerEnabled.reduce((a, b) => a + b, 0) : 0}`;
    $('ro-coverage').textContent = result.max > 0 ? `${pct(cov.pixels)} / ${pct(cov.sphere)}` : '—';
    $('ro-map').textContent = `${W} × ${H}`;
    $('ro-ms').textContent = `${ms.toFixed(1)} ms per frame`;
    $('readout-badge').hidden = false;
    $('readout-badge').textContent = `σ ${state.settings.sigmaDeg}° · ±${state.settings.windowSec} s${state.settings.causal ? ' past' : ''}${state.settings.motionComp && state.flow ? ' · motion-comp.' : ''}`;
  }
  const wrapX = state.settings.projection === 'erp', refOutline = !!(state.ref && state.refShow.outline);
  if (state.showGaze && state.display.mode !== 'off' && result) colorize(result.map, img.data, colorOpts);
  if (refOutline && ref) outline(ref.map, W, H, thr, [255, 255, 255], img.data, wrapX);
  if (state.showGaze && (refOutline || state.display.outline) && result && result.max > 0) outline(result.map, W, H, thr, [200, 245, 106], img.data, wrapX);
  hctx.putImageData(img, 0, 0);
  // the reference heatmap has its own layer (under ours) and its own colours
  const rctx = refCanvas.getContext('2d');
  if (state.ref && state.refShow.heat && ref) {
    if (refCanvas.width !== W || refCanvas.height !== H) { refCanvas.width = W; refCanvas.height = H; }
    const rimg = rctx.createImageData(W, H);
    colorize(normalised(ref.map), rimg.data, { ...colorOpts, colormap: state.refColormap || 'ice', mode: state.display.mode === 'off' || state.display.mode === 'spotlight' ? 'heat' : state.display.mode });
    rctx.putImageData(rimg, 0, 0);
  } else rctx.clearRect(0, 0, refCanvas.width, refCanvas.height);
  updateReferenceReadout(ref, result);
  state.last = { t, k, result, W, H };
  drawMarks(t, result);
  drawPlayhead();
  if (state.view) draw360(result, W, H);
}

function normalised(map) {
  let mx = 0;
  for (const v of map) if (v > mx) mx = v;
  if (!(mx > 0)) return map;
  const out = new Float32Array(map.length);
  for (let i = 0; i < map.length; i++) out[i] = map[i] / mx;
  return out;
}
/** Paint the border of the area at or above frac × max (two pixels wide; wraps left/right for 360° maps). */
function outline(map, W, H, frac, rgb, data, wrap) {
  let mx = 0;
  for (const v of map) if (v > mx) mx = v;
  if (!(mx > 0)) return;
  const t = frac * mx, inside = (x, y) => {
    if (y < 0 || y >= H) return false;
    if (x < 0 || x >= W) { if (!wrap) return false; x = (x + W) % W; }
    return map[y * W + x] >= t;
  };
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const a = inside(x, y);
    if (a === inside(x + 1, y) && a === inside(x, y + 1) && a === inside(x - 1, y) && a === inside(x, y - 1)) continue;
    const j = 4 * (y * W + x);
    data[j] = rgb[0]; data[j + 1] = rgb[1]; data[j + 2] = rgb[2]; data[j + 3] = 255;
  }
}

// ---------- reference maps ----------
/** The reference map for video frame k at W×H, from the cache; starts loading it if needed (then redraws). */
function referenceAt(k, W, H) {
  const R = state.ref;
  const idx = R.N === 1 ? 0 : Math.round(k * R.scale) + R.offset;
  if (idx < 0 || idx >= R.N) { R.status = `video frame ${k} → reference frame ${idx}: outside 0–${R.N - 1}`; return null; }
  const key = `${idx}:${W}x${H}`;
  if (R.cache.has(key)) { const map = R.cache.get(key); R.status = ''; R.shown = { idx, map, W, H }; return { idx, map, fresh: true }; }
  if (!R.pending.has(idx)) {
    R.pending.add(idx);
    R.frame(idx).then(raw => {
      R.cache.set(key, resample(raw, R.W, R.H, W, H));
      while (R.cache.size > 24) R.cache.delete(R.cache.keys().next().value);
    }).catch(e => { R.error = `frame ${idx}: ${e.message}`; remember('reference error', R.error); })
      .finally(() => { R.pending.delete(idx); if (state.ref === R) requestRender(); });
  }
  R.status = `loading reference frame ${idx}…`;
  // meanwhile keep showing the last frame that was ready (marked as not current)
  return R.shown && R.shown.W === W && R.shown.H === H ? { idx: R.shown.idx, map: R.shown.map, fresh: false } : null;
}
function updateReferenceReadout(ref, result) {
  const R = state.ref;
  $('ro-ref-card').hidden = !R;
  if (!R) return;
  $('ref-status').textContent = R.error ? `Error: ${R.error}` : R.status || (ref ? `Showing reference frame ${ref.idx}.` : '');
  if (!ref) { $('ro-ref').textContent = '—'; $('ro-ref-detail').textContent = R.status || 'no reference for this frame'; return; }
  const thr = state.settings.coverageThreshold;
  if (result && result.max > 0) {
    const c = compareMaps(result.map, ref.map, thr);
    $('ro-ref').textContent = `CC ${fmt(c.cc, 3)} · overlap ${pct(c.iou)}`;
    $('ro-ref-detail').textContent = `reference frame ${ref.idx}${ref.fresh ? '' : ' (loading the current one)'}, its coverage ${pct(c.coverageB)}`;
  } else {
    const c = compareMaps(ref.map, ref.map, thr);
    $('ro-ref').textContent = `coverage ${pct(c.coverageA)}`;
    $('ro-ref-detail').textContent = `reference frame ${ref.idx}${ref.fresh ? '' : ' (loading)'}; no gaze map to compare`;
  }
}
async function loadReference(files) {
  try {
    const R = await openReference(files);
    Object.assign(R, { cache: new Map(), pending: new Set(), offset: R.suggestedOffset, scale: 1, status: '', error: '', shown: null });
    state.ref = R;
    $('ref-name').textContent = R.name; $('ref-name').title = R.name;
    $('ref-size').textContent = `${R.N} × ${R.W} × ${R.H}${R.kind === 'bin' ? ' float32' : ''}`;
    $('ref-offset').value = String(R.offset); $('ref-scale').value = '1'; $('clear-ref').disabled = false; syncLayerChips();
    const vFrames = video.duration ? Math.round(video.duration * state.fps) : 0;
    const msgs = [`Reference loaded: ${R.name}.`];
    if (R.kind === 'bin') msgs.push('Salient360 maps: frame offset +1 (bin frame 0 is empty, bin frame k belongs to video frame k − 1).');
    if (R.N > 1 && vFrames && Math.abs(R.N - vFrames) > 2) msgs.push(`It has ${R.N} frames and the video about ${vFrames}: set “Ref. frames per video frame” (e.g. ${fmt(R.N / vFrames, 3)}) or the offset if they do not line up.`);
    notice(msgs.join(' '), R.N > 1 && vFrames && Math.abs(R.N - vFrames) > 2);
    requestRender();
  } catch (e) { notice(`Could not open the reference: ${e.message}`, true); }
}
$('choose-ref').addEventListener('click', () => $('ref-input').click());
$('ref-input').addEventListener('change', e => { if (e.target.files.length) loadReference([...e.target.files]); e.target.value = ''; });
$('clear-ref').addEventListener('click', () => { state.ref = null; syncLayerChips(); $('ref-name').textContent = $('ref-size').textContent = '—'; $('ref-status').textContent = 'No reference loaded.'; $('clear-ref').disabled = true; requestRender(); });
$('ref-offset').addEventListener('change', e => { if (state.ref) { state.ref.offset = Math.round(Number(e.target.value) || 0); requestRender(); } });
$('ref-scale').addEventListener('change', e => { if (state.ref) { state.ref.scale = Math.max(0.001, Number(e.target.value) || 1); requestRender(); } });
$('ref-heat').addEventListener('change', e => { state.refShow.heat = e.target.checked; syncLayerChips(); requestRender(); });
$('ref-outline').addEventListener('change', e => { state.refShow.outline = e.target.checked; requestRender(); });
$('ref-colormap').innerHTML = COLORMAPS.map(c => `<option value="${c}">${c[0].toUpperCase() + c.slice(1)}</option>`).join('');
$('ref-colormap').value = 'ice';
$('ref-colormap').addEventListener('change', e => { state.refColormap = e.target.value; requestRender(); });
/** Toolbar chips: our gaze layer (heatmap, points, outline) and the reference heatmap, on and off. */
function syncLayerChips() {
  $('layer-gaze').classList.toggle('active', state.showGaze); $('layer-gaze').setAttribute('aria-pressed', String(state.showGaze));
  $('layer-ref').classList.toggle('active', state.refShow.heat); $('layer-ref').setAttribute('aria-pressed', String(state.refShow.heat));
  $('layer-ref').disabled = !state.ref; $('ref-heat').checked = state.refShow.heat;
}
$('layer-gaze').addEventListener('click', () => { state.showGaze = !state.showGaze; syncLayerChips(); requestRender(); });
$('layer-ref').addEventListener('click', () => { state.refShow.heat = !state.refShow.heat; syncLayerChips(); requestRender(); });
$('cov-thr').addEventListener('change', e => {
  state.settings.coverageThreshold = Math.min(0.99, Math.max(0.001, Number(e.target.value) || 0.1));
  $('ro-coverage-label').textContent = `COVERAGE ≥ ${state.settings.coverageThreshold} MAX`;
  onSettings();
});

// ---------- 360° view ----------
const glCanvas = $('view360-canvas');
function set360(on) {
  if (on && !state.view) {
    try { state.view = state.view360 || (state.view360 = new View360(glCanvas)); }
    catch (e) { showError(`The 360° view needs WebGL2: ${e.message}`); return; }
    state.viewSource ||= document.createElement('canvas');
  } else if (!on) state.view = null;
  glCanvas.hidden = !on; $('view360-bar').hidden = !on; $('view-360').classList.toggle('active', on);
  const [vw, vh] = videoSize();
  $('stage').style.setProperty('--aspect', String(on ? 16 / 9 : vw / vh));
  requestRender();
}
function draw360(result, W, H) {
  const v = state.view, src = state.viewSource;
  const [vw, vh] = videoSize(), sw = Math.min(vw, 2048), sh = Math.round(sw / 2);
  if (src.width !== sw || src.height !== sh) { src.width = sw; src.height = sh; }
  drawComposite(src);
  if ($('view360-follow').checked && result && result.max > 0) {
    let a = 0;
    for (let i = 1; i < result.map.length; i++) if (result.map[i] > result.map[a]) a = i;
    v.follow(((a % W + 0.5) / W - 0.5) * 360, (0.5 - (Math.floor(a / W) + 0.5) / H) * 180, video.paused ? 1 : 0.12);
  }
  v.setSource(src); v.draw();
  $('view360-info').textContent = `looking at ${Math.round(v.yaw)}°, ${Math.round(v.pitch)}° · field of view ${Math.round(v.fov)}°`;
}
function redraw360() { if (state.view) { state.view.draw(); $('view360-info').textContent = `looking at ${Math.round(state.view.yaw)}°, ${Math.round(state.view.pitch)}° · field of view ${Math.round(state.view.fov)}°`; } }
$('view-360').addEventListener('click', () => set360(!state.view));
{
  let drag = null;
  glCanvas.addEventListener('pointerdown', e => { drag = { x: e.clientX, y: e.clientY }; glCanvas.setPointerCapture(e.pointerId); });
  glCanvas.addEventListener('pointermove', e => {
    if (!drag || !state.view) return;
    state.view.drag(e.clientX - drag.x, e.clientY - drag.y); drag = { x: e.clientX, y: e.clientY };
    $('view360-follow').checked = false; redraw360();
  });
  glCanvas.addEventListener('pointerup', () => { drag = null; });
  glCanvas.addEventListener('wheel', e => { if (!state.view) return; e.preventDefault(); state.view.zoom(Math.exp(e.deltaY * 0.001)); redraw360(); }, { passive: false });
  glCanvas.addEventListener('dblclick', () => { if (!state.view) return; state.view.yaw = 0; state.view.setMode(state.view.mode); redraw360(); });
}

function drawMarks(t, result) {
  const rect = $('stage').getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
  const cw = Math.max(1, Math.round(rect.width * dpr)), ch = Math.max(1, Math.round(rect.height * dpr));
  if (marksCanvas.width !== cw || marksCanvas.height !== ch) { marksCanvas.width = cw; marksCanvas.height = ch; }
  const g = marksCanvas.getContext('2d');
  g.clearRect(0, 0, cw, ch);
  const TAU = Math.PI * 2;
  if (state.display.showFlow && state.flow) {
    const f = state.flow, stepPx = 26 * dpr, scale = 0.5; // arrow = displacement over 0.5 s
    g.strokeStyle = 'rgba(200,245,106,0.85)'; g.lineWidth = 1.2 * dpr;
    for (let y = stepPx / 2; y < ch; y += stepPx) for (let x = stepPx / 2; x < cw; x += stepPx) {
      const fx = x / cw * f.W - 0.5, fy = y / ch * f.H - 0.5;
      const [u, v] = f.velocity(t, fx, fy);
      const dx = u * scale * cw / f.W, dy = v * scale * ch / f.H;
      if (Math.hypot(dx, dy) < 0.8 * dpr) { g.fillStyle = 'rgba(200,245,106,0.35)'; g.fillRect(x - dpr, y - dpr, 2 * dpr, 2 * dpr); continue; }
      g.beginPath(); g.moveTo(x, y); g.lineTo(x + dx, y + dy);
      const a = Math.atan2(dy, dx), hl = 4 * dpr;
      g.lineTo(x + dx - hl * Math.cos(a - 0.5), y + dy - hl * Math.sin(a - 0.5));
      g.moveTo(x + dx, y + dy); g.lineTo(x + dx - hl * Math.cos(a + 0.5), y + dy - hl * Math.sin(a + 0.5));
      g.stroke();
    }
  }
  if (state.display.showPoints && state.showGaze && result) {
    const p = result.points;
    let wmax = 0;
    for (let i = 0; i < p.n; i++) wmax = Math.max(wmax, p.w[i]);
    for (let i = 0; i < p.n; i++) {
      const x = (p.lon[i] / TAU + 0.5) * cw, y = (0.5 - p.lat[i] / Math.PI) * ch;
      const r = (1.6 + 2.6 * (wmax > 0 ? p.w[i] / wmax : 1)) * dpr;
      g.beginPath(); g.arc(x, y, r, 0, TAU);
      g.fillStyle = obsColor(p.obs[i]); g.globalAlpha = 0.35 + 0.65 * (wmax > 0 ? p.w[i] / wmax : 1); g.fill();
      g.globalAlpha = 1; g.lineWidth = 1 * dpr; g.strokeStyle = 'rgba(0,0,0,0.7)'; g.stroke();
    }
  }
}

// ---------- density strip ----------
function computeDensity() {
  const g = state.gaze;
  if (!g || !state.video || !video.duration) { state.density = null; return; }
  const bins = Math.min(1200, Math.max(50, Math.round(video.duration * 10)));
  const counts = new Float32Array(bins), dt = video.duration / bins;
  for (let i = 0; i < g.n; i++) {
    if (state.observerEnabled && !state.observerEnabled[g.obs[i]]) continue;
    if (state.conditionEnabled && !state.conditionEnabled[g.cond[i]]) continue;
    const a = Math.max(0, Math.floor(g.t0[i] / dt)), b = Math.min(bins - 1, Math.floor(g.t1[i] / dt));
    for (let k = a; k <= b && k < bins; k++) counts[k] += 1;
  }
  state.density = counts;
}
function drawDensity() { computeDensity(); drawPlayhead(); }
function drawPlayhead() {
  const dpr = window.devicePixelRatio || 1, w = Math.max(1, Math.round(densityCanvas.clientWidth * dpr)), h = Math.max(1, Math.round(densityCanvas.clientHeight * dpr));
  if (densityCanvas.width !== w || densityCanvas.height !== h) { densityCanvas.width = w; densityCanvas.height = h; }
  const g = densityCanvas.getContext('2d');
  g.clearRect(0, 0, w, h);
  const d = state.density;
  if (d) {
    let max = 0; for (const v of d) max = Math.max(max, v);
    g.fillStyle = 'rgba(200,245,106,0.45)';
    const bw = w / d.length;
    for (let i = 0; i < d.length; i++) if (d[i] > 0) { const bh = Math.max(1, d[i] / max * (h - 2)); g.fillRect(i * bw, h - bh, Math.max(1, bw), bh); }
    const a = Number($('ex-start').value) || 0, b = Number($('ex-end').value) || 0;
    if (video.duration && b > a) { g.fillStyle = 'rgba(255,255,255,0.06)'; g.fillRect(a / video.duration * w, 0, (b - a) / video.duration * w, h); }
  }
  if (state.video && video.duration) {
    const x = (video.paused ? video.currentTime : state.mediaTime) / video.duration * w;
    g.fillStyle = '#ffffff'; g.fillRect(Math.round(x), 0, Math.max(1, dpr), h);
  }
}
densityCanvas.addEventListener('click', e => {
  if (!state.video) return;
  const r = densityCanvas.getBoundingClientRect();
  video.currentTime = Math.max(0, Math.min(video.duration, (e.clientX - r.left) / r.width * video.duration));
});

// ---------- gaze loading ----------
async function loadGazeFiles(list) {
  const chosen = list.length;   // a FileList empties when its input is reset
  let files = [...list].filter(f => GAZE_EXT.test(f.name) && !f.name.startsWith('.'));
  if (!files.length) { notice('No gaze files found (expected .csv, .tsv, .txt, .json, .arff or .dat).', true); return; }
  const relOf = f => f.webkitRelativePath || f.relativePath || f.name;
  const userFilter = $('gaze-filter').value.trim();
  let note = '';
  if (userFilter) {
    const kept = files.filter(f => relOf(f).includes(userFilter));
    note = `${kept.length} of ${files.length} files contain “${userFilter}” in their path.`;
    files = kept;
  } else if (files.length > 1 && state.video) {
    const key = stem(state.video.name), kept = files.filter(f => pathMatchesVideo(relOf(f), key));
    if (kept.length && kept.length < files.length) { note = `Used the ${kept.length} of ${files.length} files that belong to “${key}”.`; files = kept; }
  }
  if (!files.length) { notice(`${note} Nothing left to load.`, true); return; }
  setStatus(`Reading ${files.length} gaze file(s)…`);
  const parsed = [], failed = [];
  // read in batches: thousands of small files (one per frame) are slow one by one
  for (let i = 0; i < files.length; i += 64) {
    const texts = await Promise.all(files.slice(i, i + 64).map(f => f.text().then(t => [f, t], e => [f, null, e])));
    for (const [f, text, err] of texts) {
      try {
        if (err) throw err;
        const parts = relOf(f).split('/');
        parsed.push({ name: f.name, path: relOf(f), key: `${f.name}|${f.size}|${f.lastModified}`, folder: parts.length > 1 ? parts[parts.length - 2] : '', folder2: parts.length > 2 ? parts[parts.length - 3] : '', table: parseTable(text, f.name) });
      } catch (e) { failed.push(`${relOf(f)}: ${e.message}`); }
    }
    if (files.length > 200) setStatus(`Reading gaze files… ${Math.min(files.length, i + 64)} / ${files.length}`);
  }
  setStatus(state.video ? 'Video ready' : 'Gaze loaded');
  if (!parsed.length) { notice(`Could not read the gaze files. ${failed.slice(0, 3).join(' ')}`, true); return; }
  // Files with the same columns as those already loaded are added to them (one column mapping serves all);
  // a different format replaces them. Within a batch, files whose columns differ from the first one are skipped.
  const current = state.files.length ? state.files[0].table.columns.join('|') : null;
  const ref = current && parsed.some(p => p.table.columns.join('|') === current) ? current : parsed[0].table.columns.join('|');
  const same = parsed.filter(p => p.table.columns.join('|') === ref);
  const other = parsed.filter(p => p.table.columns.join('|') !== ref);
  const adding = current === ref;
  let added = same, replacedNote = '';
  if (adding) {
    // the same file (name, size and date) is not added twice, whichever way it was chosen
    const have = new Set(state.files.map(f => f.key));
    added = same.filter(f => !have.has(f.key));
    state.files = state.files.concat(added);
  } else {
    if (current) replacedNote = `These files have other columns than the ${state.files.length} loaded before, so they replace them.`;
    state.files = same; state.removedObservers = new Set();
  }
  // files with the same name (e.g. L/x.csv and R/x.csv chosen one at a time, without their folders) stay apart
  const seen = new Map();
  for (const f of state.files) {
    const id = `${f.folder}/${f.name}`, n = (seen.get(id) || 0) + 1;
    seen.set(id, n);
    f.label = n > 1 ? `${f.name.replace(/\.[^.]+$/, '')} (${n})` : undefined;
  }
  state.loadReport = { chosen, read: files.length, filter: note, failed, otherColumns: other.map(o => ({ path: o.path, columns: o.table.columns })) };
  let presetNote = '';
  if (!adding) {
    state.mapping = detectMapping(same[0].table, { duration: video.duration || 60, width: video.videoWidth || 0, height: video.videoHeight || 0 });
    if (same.length === 1 && state.mapping.observerFrom === 'file') { state.mapping.observerFrom = 'none'; state.mapping.observerAuto = true; }
    const folders = new Set(same.map(f => f.folder));
    if (state.mapping.conditionFrom === 'none' && folders.size > 1 && !state.mapping.preset) state.mapping.conditionFrom = 'folder';
    presetNote = applyDatasetDefaults(state.mapping);
  } else if (state.mapping.observerAuto && state.files.length > 1) { state.mapping.observerFrom = 'file'; state.mapping.observerAuto = false; }
  populateMappingForm();
  rebuildGaze();
  const msgs = [note, adding ? `Added ${added.length} file(s)${added.length < same.length ? ` (${same.length - added.length} were loaded already)` : ''}; ${state.files.length} in total.` : '', replacedNote, presetNote].filter(Boolean);
  if (other.length) msgs.push(`${other.length} file(s) with different columns were skipped (${other.slice(0, 2).map(o => o.name).join(', ')}…).`);
  if (failed.length) msgs.push(`${failed.length} file(s) could not be read.`);
  notice(msgs.join(' '), other.length > 0 || failed.length > 0);
}

/** Known datasets get their usual settings the first time their files are recognised. */
function applyDatasetDefaults(m) {
  if (!m.preset) return '';
  const msg = [];
  if (!state.projectionTouched) { state.settings.projection = 'erp'; $('projection').value = 'erp'; $('fov-row').hidden = true; }
  if (m.preset === 'vret') {
    applySettings({ velocity: 75 });
    msg.push('Recognised VR-EyeTracking gaze (eye position, y from the bottom, one sample per frame): speed filter set to 75°/s.');
  }
  if (m.preset === 'yt360frames') {
    applySettings({ velocity: 0 });
    msg.push('Recognised YT360-EyeTracking per-frame files (Fixation_per_video): the frame number comes from the file name and observers are not known, so all points of a condition count as one group. Fixation_per_user_per_video has observers and fixation durations and is the better choice.');
  }
  if (m.preset === 'yt360' || m.preset === 'yt360frames') {
    const start = /-(\d+)$/.exec(state.video ? stem(state.video.name) : '');
    if (start && video.duration > 35) {
      m.timeOffset = Number(start[1]);
      if (Number.isFinite(video.duration)) video.currentTime = m.timeOffset;
      msg.push(`The video is the full download, so gaze times are shifted by the clip start (${m.timeOffset} s).`);
    } else if (m.preset === 'yt360') msg.push('Recognised YT360-EyeTracking fixations (times in seconds from the clip start).');
  }
  return msg.join(' ');
}

async function loadGazeUrl(url) {
  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const name = stem(url) + (url.match(/\.[a-z]{3,4}(?=$|[?#])/i)?.[0] || '.csv');
    await loadGazeFiles([new File([await res.text()], name)]);
  } catch (e) { notice(`Could not fetch the gaze file (${e.message}). The server must allow cross-origin requests (CORS).`, true); }
}

function populateMappingForm() {
  const m = state.mapping, cols = state.files[0].table.columns;
  document.querySelectorAll('select[data-map]').forEach(sel => {
    const key = sel.dataset.map;
    if (['time', 'x', 'y', 'z', 'end', 'duration', 'observer', 'condition', 'weight'].includes(key)) {
      sel.innerHTML = `<option value="">— none —</option>` + (key === 'time' ? `<option>${FILE_TIME}</option>` : '') + cols.map(c => `<option>${escapeHtml(c)}</option>`).join('');
    }
    sel.value = m[key] ?? '';
  });
  document.querySelectorAll('input[data-map]').forEach(inp => { inp.value = m[inp.dataset.map] ?? ''; });
  $('map-z-wrap').hidden = m.coord !== 'vector';
  $('mapping-section').hidden = false;
  renderPreview();
}
function escapeHtml(s) { return String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c])); }
function renderPreview() {
  const t = state.files[0].table, m = state.mapping;
  const used = new Set([m.time, m.x, m.y, m.z, m.end, m.duration, m.observer, m.condition, m.weight].filter(Boolean));
  const rows = t.rows.slice(0, 6);
  $('preview-table').innerHTML = `<table><thead><tr>${t.columns.map(c => `<th class="${used.has(c) ? 'used' : ''}">${escapeHtml(c)}</th>`).join('')}</tr></thead><tbody>${rows.map(r => `<tr>${t.columns.map((_, i) => `<td>${escapeHtml(r[i] ?? '')}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
}

let rebuildTimer = 0;
function onMappingChange(e) {
  const key = e.target.dataset.map, m = state.mapping;
  const numeric = ['xLeft', 'xRight', 'yTop', 'yBottom', 'timeOffset'].includes(key);
  m[key] = numeric ? Number(e.target.value) : e.target.value;
  if (['xLeft', 'xRight', 'yTop', 'yBottom'].includes(key)) m.rangesTouched = true;
  if (key === 'coord') {
    const t = state.files[0].table;
    Object.assign(m, defaultRanges(m.coord, columnValues(t, m.x), columnValues(t, m.y), { width: video.videoWidth, height: video.videoHeight }));
    if (m.coord === 'vector' && !m.z) m.z = t.columns.find(c => /^z$/i.test(c)) || '';
    populateMappingForm();
  }
  if (key === 'timeUnit' && !m.duration) m.durationUnit = m.timeUnit;
  $('map-z-wrap').hidden = m.coord !== 'vector';
  renderPreview();
  clearTimeout(rebuildTimer);
  rebuildTimer = setTimeout(rebuildGaze, 120);
}
document.querySelectorAll('[data-map]').forEach(el => el.addEventListener('change', onMappingChange));

function rebuildGaze() {
  $('mapping-error').hidden = true;
  if (!state.files.length) return;
  try {
    const prevObs = state.gaze ? new Map(state.gaze.observers.map((o, i) => [o, state.observerEnabled[i]])) : null;
    const prevCond = state.gaze ? new Map(state.gaze.conditions.map((c, i) => [c, state.conditionEnabled[i]])) : null;
    const built = buildGaze(state.files, state.mapping, { fps: state.fps });
    const removed = new Set(built.observers.map((o, i) => state.removedObservers?.has(o) ? i : -1).filter(i => i >= 0));
    state.gazeAll = dropObservers(built, removed);
    state.gazeAll.skipped = built.skipped;
    state.gaze = velocityFilter(state.gazeAll, state.settings.velocity);
    state.observerEnabled = Uint8Array.from(state.gaze.observers, o => prevObs?.has(o) ? prevObs.get(o) : 1);
    state.conditionEnabled = Uint8Array.from(state.gaze.conditions, c => prevCond?.has(c) ? prevCond.get(c) : 1);
  } catch (e) {
    state.gaze = null;
    $('mapping-error').hidden = false; $('mapping-error').textContent = e.message; remember('mapping error', e.message);
  }
  renderGroups(); updateGazeSummary(); drawDensity(); updateSubtitle(); updateButtons(); updateExportSummary(); requestRender();
}

function updateGazeSummary() {
  const g = state.gaze, el = $('gaze-summary');
  if (!g) { el.textContent = state.files.length ? 'Check the column mapping.' : 'No gaze loaded. One file per observer, or one file with an observer column.'; return; }
  const kind = g.hasDuration ? 'fixations' : 'samples';
  let text = `${state.files.length} file(s), ${g.n.toLocaleString()} ${kind}, ${g.observers.length} observer(s), ${g.conditions.length} condition(s), ${fmt(g.tMin, 2)}–${fmt(g.tMax, 2)} s.`;
  if (state.gazeAll && state.gazeAll.skipped) text += ` ${state.gazeAll.skipped} row(s) skipped (no valid time or position, e.g. nan where the eye tracker lost the eye).`;
  if (g.filtered) text += ` Velocity filter removed ${g.filtered.toLocaleString()}.`;
  el.textContent = text;
  if (video.duration && (g.tMax > video.duration * 1.5 + 1 || g.tMax < video.duration * 0.2)) notice(`Gaze times run to ${fmt(g.tMax, 1)} s but the video lasts ${fmt(video.duration, 1)} s: check the time unit or the time offset in the column mapping.`, true);
}

function renderGroups() {
  const g = state.gaze;
  $('groups-section').hidden = !g;
  if (!g) return;
  const condCounts = new Array(g.conditions.length).fill(0), obsCounts = new Array(g.observers.length).fill(0);
  for (let i = 0; i < g.n; i++) { condCounts[g.cond[i]]++; obsCounts[g.obs[i]]++; }
  $('condition-list').innerHTML = g.conditions.length > 1
    ? g.conditions.map((c, i) => `<label class="chip"><input type="checkbox" data-cond="${i}" ${state.conditionEnabled[i] ? 'checked' : ''}>${escapeHtml(c)} <small>${condCounts[i]}</small></label>`).join('')
    : '';
  $('observer-count').textContent = `${g.observers.length} observer${g.observers.length === 1 ? '' : 's'}`;
  const removed = state.removedObservers || new Set();
  $('observer-count').textContent = `${g.observers.length - removed.size} observer${g.observers.length - removed.size === 1 ? '' : 's'}${removed.size ? ` (${removed.size} removed)` : ''}`;
  $('observer-list').innerHTML = g.observers.map((o, i) => removed.has(o) ? '' : `<div class="observer-row"><label><input type="checkbox" data-obs="${i}" ${state.observerEnabled[i] ? 'checked' : ''}><i style="background:${obsColor(i)}"></i><span title="${escapeHtml(o)}">${escapeHtml(o)}</span><small>${obsCounts[i]}</small></label><button class="row-remove" type="button" data-remove="${i}" title="Remove ${escapeHtml(o)}" aria-label="Remove ${escapeHtml(o)}">×</button></div>`).join('')
    + (removed.size ? `<button id="restore-observers" class="link-button" type="button">bring back the ${removed.size} removed</button>` : '');
}
$('observer-list').addEventListener('click', e => {
  const r = e.target.closest('[data-remove]');
  if (r) { (state.removedObservers ||= new Set()).add(state.gaze.observers[Number(r.dataset.remove)]); rebuildGaze(); return; }
  if (e.target.id === 'restore-observers') { state.removedObservers = new Set(); rebuildGaze(); }
});
function clearGaze() {
  Object.assign(state, { files: [], mapping: null, gazeAll: null, gaze: null, observerEnabled: null, conditionEnabled: null, removedObservers: new Set(), loadReport: null });
  $('mapping-section').hidden = true; $('mapping-error').hidden = true;
  renderGroups(); updateGazeSummary(); drawDensity(); updateSubtitle(); updateButtons(); updateExportSummary(); requestRender();
  notice('Gaze cleared.');
}
$('clear-gaze').addEventListener('click', clearGaze);
$('observer-list').addEventListener('change', e => { const i = e.target.dataset.obs; if (i == null) return; state.observerEnabled[i] = e.target.checked ? 1 : 0; drawDensity(); requestRender(); });
$('condition-list').addEventListener('change', e => { const i = e.target.dataset.cond; if (i == null) return; state.conditionEnabled[i] = e.target.checked ? 1 : 0; drawDensity(); requestRender(); });
$('obs-all').addEventListener('click', () => { if (!state.gaze) return; state.observerEnabled.fill(1); renderGroups(); drawDensity(); requestRender(); });
$('obs-none').addEventListener('click', () => { if (!state.gaze) return; state.observerEnabled.fill(0); renderGroups(); drawDensity(); requestRender(); });

// ---------- motion analysis ----------
function updateFlowUi() {
  const f = state.flow;
  $('motion').disabled = !f; $('show-flow').disabled = !f; $('clear-flow').disabled = !f;
  if (f && state.wantMotion && !state.settings.motionComp) { state.settings.motionComp = true; $('motion').checked = true; }
  if (!f) { $('motion').checked = false; state.settings.motionComp = false; $('show-flow').checked = false; state.display.showFlow = false; }
  $('flow-status').textContent = f
    ? `${f.size} flow fields at ${f.W} × ${f.H}, ${fmt(f.times[0] - f.step / 2, 2)}–${fmt(f.times[f.size - 1] + f.step / 2, 2)} s, one every ${fmt(f.step, 3)} s, ${bytesText(f.bytes)}.`
    : state.video?.tainted ? 'Needs pixel access: load the video from disk or a CORS-enabled link.' : 'Not analysed.';
  updateButtons();
}

async function analyseMotion() {
  if (state.analysing) { state.analysing.cancel = true; return; }
  if (!state.video || state.video.tainted) return;
  const [vw, vh] = videoSize();
  const rate = Number($('flow-rate').value), [W, H] = mapSize(Number($('flow-width').value), state.settings, vw, vh);
  const wrap = state.settings.projection === 'erp';
  let a = 0, b = video.duration;
  if ($('flow-range').value === 'export') { a = Math.max(0, Number($('ex-start').value) || 0); b = Math.min(video.duration, Number($('ex-end').value) || video.duration); }
  const store = state.flow && state.flow.W === W && state.flow.H === H && state.flow.wrap === wrap ? state.flow : new FlowStore(W, H, { wrap });
  const job = { cancel: false };
  state.analysing = job;
  $('analyse').textContent = 'Cancel analysis'; $('flow-progress').hidden = false;
  const bar = $('flow-progress').firstElementChild;
  const probe = hiddenVideo();
  const canvas = document.createElement('canvas'); canvas.width = W; canvas.height = H;
  const g = canvas.getContext('2d', { willReadFrequently: true });
  const worker = new Worker(new URL('./flow-worker.js', import.meta.url), { type: 'module' });
  const pending = new Map();
  let seq = 0, failures = 0;
  worker.onmessage = e => { const p = pending.get(e.data.id); pending.delete(e.data.id); p?.(e.data); };
  const submit = (A, B, tA, tB) => new Promise(resolve => {
    const id = ++seq;
    pending.set(id, msg => { if (msg.error) failures++; else store.add(tA, tB, msg.u, msg.v); resolve(); });
    worker.postMessage({ id, a: A, b: B, W, H, wrap });
  });
  const conv = converting() ? grabConverter() : null, [rw, rh] = rawVideoSize(), layout = state.layout;
  const grab = () => { g.drawImage(conv ? conv.convert(probe, layout, rw, rh, W, H) : probe, 0, 0, W, H); return toGray(g.getImageData(0, 0, W, H).data, W, H); };
  const started = performance.now();
  try {
    await once(probe, 'loadeddata', 15000);
    let prev = null, prevT = 0;
    const inflight = [];
    const push = async (img, t) => {
      if (prev) { inflight.push(submit(prev, img, prevT, t)); if (inflight.length > 3) await inflight.shift(); }
      prev = img; prevT = t;
      bar.style.width = `${Math.min(100, 100 * (t - a) / Math.max(1e-6, b - a))}%`;
      setStatus(`Analysing motion… ${fmt(t, 1)} / ${fmt(b, 1)} s`);
    };
    if ($('flow-method').value === 'seek' || !('requestVideoFrameCallback' in probe)) {
      for (let t = a; t <= b - 0.25 / state.fps && !job.cancel; t += 1 / rate) {
        const k = frameOf(t, state.fps), tc = frameCentre(k, state.fps);
        probe.currentTime = tc;
        await once(probe, 'seeked', 15000);
        await new Promise(r => ('requestVideoFrameCallback' in probe) ? (probe.requestVideoFrameCallback(() => r()), setTimeout(r, 300)) : setTimeout(r, 30));
        await push(grab(), tc);
      }
    } else {
      probe.currentTime = a;
      await once(probe, 'seeked', 15000);
      probe.playbackRate = 1;
      await new Promise((resolve, reject) => {
        let lastGrab = -Infinity, busy = false;
        const finish = () => { probe.pause(); resolve(); };
        probe.addEventListener('ended', finish, { once: true });
        const cb = async (now, meta) => {
          const t = meta.mediaTime;
          if (job.cancel || t >= b) { finish(); return; }
          if (!busy && t - lastGrab >= 1 / rate - 0.25 / state.fps) {
            busy = true; lastGrab = t;
            try { await push(grab(), t); } catch (e) { reject(e); return; } finally { busy = false; }
          }
          probe.requestVideoFrameCallback(cb);
        };
        probe.requestVideoFrameCallback(cb);
        probe.play().catch(reject);
      });
    }
    await Promise.all(inflight);
    while (pending.size) await new Promise(r => setTimeout(r, 20));
    if (store.size) {
      state.flow = store;
      state.flowInfo = { rate, method: $('flow-method').value, from: a, to: b, seconds: (performance.now() - started) / 1000, failures };
    }
    setStatus(job.cancel ? 'Motion analysis cancelled' : `Motion analysed in ${fmt((performance.now() - started) / 1000, 1)} s`);
    if (store.size && !job.cancel) { $('motion').checked = true; state.settings.motionComp = true; }
  } catch (e) {
    showError(`Motion analysis failed: ${e.message}`);
    setStatus('Motion analysis failed');
  } finally {
    worker.terminate(); disposeHidden(probe);
    state.analysing = null;
    $('analyse').textContent = 'Analyse motion'; $('flow-progress').hidden = true; bar.style.width = '0';
    updateFlowUi(); requestRender();
  }
}

// ---------- settings & display controls ----------
function bindPair(rangeId, numId, apply) {
  const r = $(rangeId), n = $(numId);
  r.addEventListener('input', () => { n.value = r.value; apply(Number(r.value)); });
  n.addEventListener('change', () => { const v = Number(n.value); if (!Number.isFinite(v)) return; r.value = String(v); apply(v); });
}
bindPair('sigma', 'sigma-num', v => { state.settings.sigmaDeg = Math.max(0.05, v); onSettings(); });
bindPair('window', 'window-num', v => { state.settings.windowSec = Math.max(0, v); onSettings(); });
bindPair('lon-offset', 'lon-offset-num', v => { state.settings.lonOffsetDeg = v; onSettings(); });
$('projection').addEventListener('change', e => { state.projectionTouched = true; state.settings.projection = e.target.value; $('fov-row').hidden = e.target.value !== 'planar'; updateSubtitle(); updateButtons(); onSettings(); });
$('fov').addEventListener('change', e => { state.settings.planarFovDeg = Math.max(1, Number(e.target.value) || 90); onSettings(); });
$('kernel').addEventListener('change', e => { state.settings.kernel = e.target.value; onSettings(); });
$('causal').addEventListener('change', e => { state.settings.causal = e.target.checked; onSettings(); });
$('duration-weight').addEventListener('change', e => { state.settings.durationWeight = e.target.checked; onSettings(); });
$('fps').addEventListener('change', e => { const f = Number(e.target.value); if (f > 0) { setFps(f, 'manual'); if (/^frame/.test(state.mapping?.timeUnit) || /^frame/.test(state.mapping?.durationUnit)) rebuildGaze(); } });
$('motion').addEventListener('change', e => { state.settings.motionComp = e.target.checked; onSettings(); });
$('show-flow').addEventListener('change', e => { state.display.showFlow = e.target.checked; requestRender(); });
$('vel').addEventListener('change', e => { state.settings.velocity = Math.max(0, Number(e.target.value) || 0); if (state.gazeAll) { state.gaze = velocityFilter(state.gazeAll, state.settings.velocity); updateGazeSummary(); drawDensity(); } onSettings(); });
$('flip-x').addEventListener('change', e => { state.settings.flipX = e.target.checked; onSettings(); });
$('flip-y').addEventListener('change', e => { state.settings.flipY = e.target.checked; onSettings(); });
$('mode').addEventListener('change', e => { state.display.mode = e.target.value; requestRender(); });
$('colormap').innerHTML = COLORMAPS.map(c => `<option value="${c}">${c[0].toUpperCase() + c.slice(1)}</option>`).join('');
$('colormap').addEventListener('change', e => { state.display.colormap = e.target.value; $('color-ramp').style.background = cssGradient(e.target.value); requestRender(); });
$('color-ramp').style.background = cssGradient('jet');
$('opacity').addEventListener('input', e => { state.display.opacity = Number(e.target.value) / 100; $('opacity-value').textContent = `${e.target.value}%`; requestRender(); });
$('gamma').addEventListener('input', e => { state.display.gamma = Number(e.target.value); $('gamma-value').textContent = Number(e.target.value).toFixed(2); requestRender(); });
$('map-width').addEventListener('change', e => { state.display.mapWidth = Number(e.target.value); requestRender(); });
$('show-outline').addEventListener('change', e => { state.display.outline = e.target.checked; requestRender(); });
$('show-points').addEventListener('change', e => { state.display.showPoints = e.target.checked; requestRender(); });
$('speed').addEventListener('change', e => { video.playbackRate = Number(e.target.value); });
function onSettings() { updateExportSummary(); requestRender(); }

/** Set several settings at once and keep the controls in step (used by presets, the address bar and the tests). */
function applySettings(o) {
  const S = state.settings;
  const pair = (key, rid, nid, min = 0) => { if (key in o) { S[key === 'sigmaDeg' ? 'sigmaDeg' : key] = Math.max(min, Number(o[key])); $(rid).value = $(nid).value = String(S[key]); } };
  pair('sigmaDeg', 'sigma', 'sigma-num', 0.05); pair('windowSec', 'window', 'window-num'); pair('lonOffsetDeg', 'lon-offset', 'lon-offset-num', -360);
  if ('kernel' in o) { S.kernel = o.kernel; $('kernel').value = o.kernel; }
  for (const [key, id] of [['causal', 'causal'], ['durationWeight', 'duration-weight'], ['flipX', 'flip-x'], ['flipY', 'flip-y']]) if (key in o) { S[key] = !!o[key]; $(id).checked = !!o[key]; }
  if ('motionComp' in o) { S.motionComp = !!o.motionComp && !!state.flow; $('motion').checked = S.motionComp; state.wantMotion = !!o.motionComp; }
  if ('velocity' in o) {
    S.velocity = Math.max(0, Number(o.velocity) || 0); $('vel').value = String(S.velocity);
    if (state.gazeAll) { state.gaze = velocityFilter(state.gazeAll, S.velocity); updateGazeSummary(); drawDensity(); }
  }
  onSettings();
}
$('preset-dataset').addEventListener('click', () => applySettings({ sigmaDeg: 3, windowSec: 1, kernel: 'gaussian', causal: false, motionComp: true, velocity: state.mapping?.preset === 'yt360' ? 0 : 75 }));

$('play').addEventListener('click', togglePlay);
$('prev-frame').addEventListener('click', () => stepFrames(-1));
$('next-frame').addEventListener('click', () => stepFrames(1));
$('scrubber').addEventListener('input', e => { video.currentTime = Number(e.target.value); });
$('analyse').addEventListener('click', analyseMotion);
$('clear-flow').addEventListener('click', () => { state.flow = null; state.flowInfo = null; updateFlowUi(); requestRender(); });

document.addEventListener('keydown', e => {
  if (e.target.closest('input, select, textarea') && e.target.type !== 'range' && e.target.type !== 'checkbox') return;
  if (e.key === ' ') { e.preventDefault(); togglePlay(); }
  else if (e.key === 'ArrowRight') { e.preventDefault(); e.shiftKey ? seekSeconds(1) : stepFrames(1); }
  else if (e.key === 'ArrowLeft') { e.preventDefault(); e.shiftKey ? seekSeconds(-1) : stepFrames(-1); }
  else if (e.key === 'm' || e.key === 'M') { if (!$('motion').disabled) { $('motion').checked = !$('motion').checked; $('motion').dispatchEvent(new Event('change')); } }
  else if (e.key === 'p' || e.key === 'P') { $('show-points').checked = !$('show-points').checked; $('show-points').dispatchEvent(new Event('change')); }
});
window.addEventListener('resize', () => requestRender());

// ---------- export ----------
function exportSize() {
  const [vw, vh] = videoSize(), sel = $('ex-width').value;
  return mapSize(sel === 'video' ? vw : Number(sel), state.settings, vw, vh);
}
function exportFrames() {
  const a = Math.max(0, Number($('ex-start').value) || 0), b = Math.min(video.duration || 0, Number($('ex-end').value) || 0);
  const step = Math.max(1, Math.round(Number($('ex-step').value) || 1));
  const k0 = frameOf(a, state.fps), k1 = Math.max(k0, Math.ceil(b * state.fps - 1e-6) - 1);
  const frames = [];
  for (let k = k0; k <= k1; k += step) frames.push(k);
  return { a, b, step, frames };
}
function updateExportSummary() {
  const fmtSel = $('ex-format').value, png = fmtSel.startsWith('png') || fmtSel === 'npyu1';
  $('ex-norm').disabled = png;
  if (png) $('ex-norm').value = 'max';
  if (!state.video) { $('ex-summary').textContent = '—'; return; }
  const [W, H] = exportSize(), { frames, step } = exportFrames();
  const per = fmtSel === 'npyf4' ? W * H * 4 : fmtSel === 'npyu1' ? W * H : W * H * (fmtSel === 'png16' ? 0.12 : 0.05);
  const est = frames.length * per;
  let text = `${frames.length.toLocaleString()} frames at ${W} × ${H}, ${fmt(state.fps / step, 3)} maps per second, about ${bytesText(est)}${fmtSel.startsWith('png') ? ' (PNG size varies)' : ''}.`;
  if (state.settings.motionComp && state.flow) {
    const { a, b } = exportFrames(), c = state.flow.coverage(a, b);
    if (c < 0.98) text += ` Motion flow covers only ${(c * 100).toFixed(0)} % of this range.`;
  }
  if (!window.showSaveFilePicker && est > 1.5e9) text += ' This browser builds the file in memory: use Chrome or Edge (writes straight to disk) or a smaller size or range.';
  $('ex-summary').textContent = text;
  drawPlayhead();
}
['ex-start', 'ex-end', 'ex-step', 'ex-width', 'ex-format', 'ex-norm'].forEach(id => $(id).addEventListener('change', updateExportSummary));
$('ex-start-here').addEventListener('click', () => { $('ex-start').value = fmt(video.currentTime, 3); updateExportSummary(); });
$('ex-end-here').addEventListener('click', () => { $('ex-end').value = fmt(video.currentTime, 3); updateExportSummary(); });

function settingsJson(extra = {}) {
  const s = state.settings, g = state.gaze, [vw, vh] = videoSize();
  return {
    tool: 'OverGaze', version: VERSION, created: new Date().toISOString(),
    video: state.video ? { name: state.video.name, width: vw, height: vh, duration_s: video.duration, fps: state.fps, fps_source: state.fpsSource } : null,
    gaze: g ? {
      files: state.files.map(f => (f.folder ? f.folder + '/' : '') + f.name), rows_read: state.gazeAll.n, rows_kept: g.n, kind: g.hasDuration ? 'fixations' : 'samples',
      observers_used: g.observers.filter((_, i) => state.observerEnabled[i]), conditions_used: g.conditions.filter((_, i) => state.conditionEnabled[i]),
      column_mapping: state.mapping
    } : null,
    saliency: {
      projection: s.projection, sigma_deg: s.sigmaDeg, kernel_cutoff_sigma: s.truncate, planar_fov_deg: s.projection === 'planar' ? s.planarFovDeg : undefined,
      pooling_window_s: s.windowSec, pooling_weight: s.kernel, causal: s.causal, weight_by_duration: s.durationWeight,
      motion_compensation: !!(s.motionComp && state.flow), lon_offset_deg: s.lonOffsetDeg, flip_left_right: s.flipX, flip_up_down: s.flipY,
      velocity_filter_deg_per_s: s.velocity || 0,
      frame_rule: 'frame k covers [k/fps, (k+1)/fps); a gaze sample or fixation counts if its time distance to that interval is <= pooling_window_s (0 = active during the frame).',
      gaussian: s.projection === 'erp' ? 'exp(-theta^2 / (2 sigma^2)), theta = great-circle angle to the gaze point, summed over points' : '2D Gaussian in the image plane, sigma_px = sigma_deg * width / planar_fov_deg'
    },
    motion: s.motionComp && state.flow ? { flow_width: state.flow.W, flow_height: state.flow.H, fields: state.flow.size, step_s: state.flow.step, ...state.flowInfo, method_note: 'pyramidal Lucas-Kanade on greyscale frames, wrap-around at the left/right edge for 360° video; each pooled point is moved along the flow from its own time to the frame time' } : null,
    ...extra
  };
}
$('copy-settings').addEventListener('click', async () => {
  const text = JSON.stringify(settingsJson(), null, 2);
  try { await navigator.clipboard.writeText(text); notice('Settings copied to the clipboard.'); }
  catch { download(new Blob([text], { type: 'application/json' }), 'overgaze_settings.json'); }
});

/** Plain-text report of what is loaded and what went wrong, to paste into a bug report or a chat. */
function debugReport() {
  const v = state.video, g = state.gaze, r = state.loadReport, [vw, vh] = videoSize(), L = [];
  const can = t => document.createElement('video').canPlayType(t) || 'no';
  L.push(`OverGaze ${VERSION} debug report, ${new Date().toISOString()}`, `page: ${location.href}`, `browser: ${navigator.userAgent}`,
    `can play: H.264 ${can('video/mp4; codecs="avc1.42E01E"')}, VP9 ${can('video/webm; codecs="vp9"')}, AV1 ${can('video/mp4; codecs="av01.0.05M.08"')}, H.265 ${can('video/mp4; codecs="hvc1.1.6.L93.B0"')}`, '');
  L.push('VIDEO');
  if (!v) L.push('  none');
  else L.push(`  ${v.name}${v.size ? ' (' + bytesText(v.size) + ')' : ''}${v.remote ? ' link: ' + v.src : ''}`, `  ${rawVideoSize().join(' × ')}, layout ${state.layout} (${LAYOUTS[state.layout].label})${converting() ? `, unfolded to ${vw} × ${vh}` : ''}, ${fmt(video.duration, 2)} s, ${fmt(state.fps, 3)} fps (${state.fpsSource}), ready state ${video.readyState}, pixels ${v.tainted ? 'not readable (no CORS)' : 'readable'}`,
    `  playhead ${fmt(video.currentTime, 3)} s${video.error ? `, error ${video.error.code}: ${video.error.message}` : ''}`);
  L.push('', 'GAZE');
  if (!state.files.length) L.push('  none');
  else {
    if (r) L.push(`  files chosen ${r.chosen}, read ${r.read}${r.filter ? ' — ' + r.filter : ''}`);
    L.push(`  used ${state.files.length} file(s) with columns: ${state.files[0].table.columns.join(', ')}${state.files[0].table.preset ? ' (format ' + state.files[0].table.preset + ')' : ''}`);
    const shown = state.files.slice(0, 40);
    for (const f of shown) L.push(`    ${f.path || f.name}: ${f.table.rows.length} rows`);
    if (state.files.length > shown.length) L.push(`    … and ${state.files.length - shown.length} more`);
    if (r?.failed.length) { L.push(`  could not read ${r.failed.length}:`); r.failed.slice(0, 20).forEach(x => L.push('    ' + x)); }
    if (r?.otherColumns.length) { L.push(`  skipped ${r.otherColumns.length} with other columns:`); r.otherColumns.slice(0, 10).forEach(o => L.push(`    ${o.path}: ${o.columns.join(', ')}`)); }
    L.push(`  mapping: ${JSON.stringify(state.mapping)}`);
    if (!$('mapping-error').hidden) L.push(`  MAPPING ERROR: ${$('mapping-error').textContent}`);
    if (g) L.push(`  ${state.gazeAll.n} rows read, ${g.n} kept${g.filtered ? ` (${g.filtered} dropped by the speed filter)` : ''}, ${state.gazeAll.skipped || 0} rows without valid time or position, ${g.observers.length} observers, ${g.conditions.length} conditions (${g.conditions.join(', ')}), time ${fmt(g.tMin, 3)}–${fmt(g.tMax, 3)} s`);
  }
  if (state.removedObservers?.size) L.push(`  removed observers: ${[...state.removedObservers].join(', ')}`);
  L.push('', 'REFERENCE');
  if (!state.ref) L.push('  none');
  else L.push(`  ${state.ref.name}: ${state.ref.kind}, ${state.ref.N} frame(s) of ${state.ref.W} × ${state.ref.H}, offset ${state.ref.offset}, rate ${state.ref.scale}, showing ${[state.refShow.heat && 'heatmap', state.refShow.outline && 'outline'].filter(Boolean).join(' + ') || 'numbers only'}`,
    `  ${$('ro-ref').textContent} — ${$('ro-ref-detail').textContent}${state.ref.error ? ' — ERROR ' + state.ref.error : ''}`);
  L.push('', '360° VIEW', state.view ? `  on, yaw ${fmt(state.view.yaw, 1)}°, pitch ${fmt(state.view.pitch, 1)}°, field of view ${fmt(state.view.fov, 0)}°` : '  off');
  L.push('', 'SETTINGS', '  ' + JSON.stringify(state.settings));
  L.push('', 'MOTION', `  ${state.flow ? $('flow-status').textContent : 'not analysed'}${state.flowInfo ? ' ' + JSON.stringify(state.flowInfo) : ''}`);
  if (state.last?.result) L.push('', 'CURRENT FRAME', `  frame ${state.last.k}, ${state.last.result.points.n} points, map ${state.last.W} × ${state.last.H}, coverage ${$('ro-coverage').textContent}`);
  L.push('', 'ON SCREEN', `  status: ${$('status').textContent}`);
  if (!$('error-message').hidden) L.push(`  error: ${$('error-text').textContent}`);
  if (!$('notice').hidden) L.push(`  notice: ${$('notice-text').textContent}`);
  L.push('', 'MESSAGES AND ERRORS (oldest first)', ...(debugLog.length ? debugLog.map(x => '  ' + x) : ['  none']));
  return L.join('\n') + '\n';
}
const copyDebug = async () => {
  $('help-menu').open = false;
  const text = debugReport();
  try { await navigator.clipboard.writeText(text); notice('Debug details copied to the clipboard: paste them into a message or a bug report.'); }
  catch { download(new Blob([text], { type: 'text/plain' }), 'overgaze_debug.txt'); }
};
$('copy-debug').addEventListener('click', copyDebug);
$('copy-debug-help').addEventListener('click', copyDebug);

function exportBaseName() {
  const s = state.settings;
  return `${stem(state.video?.name)}_sal_s${s.sigmaDeg}_w${s.windowSec}${s.causal ? 'past' : ''}${s.motionComp && state.flow ? '_mc' : ''}`;
}

async function exportMaps() {
  if (state.exporting) return;
  if (!state.gaze || !state.video) return;
  const format = $('ex-format').value, pngFormat = format.startsWith('png');
  const [W, H] = exportSize(), { a, b, step, frames } = exportFrames();
  if (!frames.length) { notice('The export range contains no frames.', true); return; }
  const name = `${exportBaseName()}_${W}x${H}_${pngFormat ? 'png' : 'npy'}.zip`;
  let writable = null;
  const parts = [];
  if (window.showSaveFilePicker) {
    try {
      const handle = await window.showSaveFilePicker({ suggestedName: name, types: [{ description: 'ZIP archive', accept: { 'application/zip': ['.zip'] } }] });
      writable = await handle.createWritable();
    } catch (e) { if (e.name === 'AbortError') return; writable = null; }
  }
  const write = writable ? bytes => writable.write(bytes) : async bytes => { parts.push(bytes); };
  const normalize = pngFormat || format === 'npyu1' ? 'max' : $('ex-norm').value;
  const settings = settingsForRender(normalize);
  const job = { cancel: false };
  state.exporting = job;
  $('export-maps').disabled = true; $('export-cancel').hidden = false; $('ex-progress').hidden = false;
  const bar = $('ex-progress').firstElementChild;
  const worker = new Worker(new URL('./export-worker.js', import.meta.url), { type: 'module' });
  const waiters = new Map();
  let ready;
  const readyP = new Promise(r => { ready = r; });
  worker.onmessage = e => { if (e.data.cmd === 'ready') ready(); else { const w = waiters.get(e.data.seq); waiters.delete(e.data.seq); w?.(e.data); } };
  const g = state.gaze;
  worker.postMessage({
    cmd: 'init', W, H, format, fps: state.fps, settings,
    gaze: { n: g.n, t0: g.t0, t1: g.t1, lon: g.lon, lat: g.lat, obs: g.obs, cond: g.cond, w: g.w, maxDur: g.maxDur },
    observerEnabled: state.observerEnabled, conditionEnabled: state.conditionEnabled,
    flow: settings.motionComp && state.flow ? state.flow.serialize() : null
  });
  const request = (seq, k) => new Promise(resolve => { waiters.set(seq, resolve); worker.postMessage({ cmd: 'frame', seq, t: frameCentre(k, state.fps) }); });
  const rows = ['index,frame,time_s,points,raw_max,coverage_pixels,coverage_sphere'];
  const started = performance.now();
  try {
    await readyP;
    const zip = new ZipWriter(write);
    const npy = pngFormat ? null : await zip.stream('maps.npy');
    if (npy) await npy.write(npyHeader(format === 'npyu1' ? '|u1' : '<f4', [frames.length, H, W]));
    const queue = [];
    let next = 0;
    const fill = () => { while (queue.length < 3 && next < frames.length) { queue.push(request(next, frames[next])); next++; } };
    fill();
    for (let i = 0; i < frames.length; i++) {
      if (job.cancel) throw new Error('cancelled');
      const res = await queue.shift();
      fill();
      if (res.cmd === 'error') throw new Error(res.error);
      const k = frames[i];
      if (npy) await npy.write(res.bytes); else await zip.add(`maps/${String(k).padStart(6, '0')}.png`, res.bytes);
      rows.push([i, k, frameCentre(k, state.fps).toFixed(5), res.points, res.max.toPrecision(6), res.cov.pixels.toFixed(6), res.cov.sphere.toFixed(6)].join(','));
      if (i % 5 === 0 || i === frames.length - 1) {
        bar.style.width = `${(100 * (i + 1) / frames.length).toFixed(1)}%`;
        const el = (performance.now() - started) / 1000;
        setStatus(`Exporting ${i + 1} / ${frames.length} (${fmt(el / (i + 1) * (frames.length - i - 1), 0)} s left)`);
      }
    }
    const meta = settingsJson({
      export: {
        format, width: W, height: H, from_s: a, to_s: b, every_nth_frame: step, frames: frames.length, normalize,
        encoding: format === 'png8' ? 'grey 8-bit PNG, value = round(255 * map), map divided by its frame maximum'
          : format === 'png16' ? 'grey 16-bit PNG, value = round(65535 * map), map divided by its frame maximum'
            : format === 'npyu1' ? 'uint8 (T, H, W) = round(255 * map), map divided by its frame maximum'
              : `float32 (T, H, W), normalisation: ${normalize}`,
        file_names: pngFormat ? 'maps/<frame index, 6 digits>.png; frame k covers [k/fps, (k+1)/fps)' : 'maps.npy, shape (T, H, W); array index i -> frame given in frames.csv'
      }
    });
    const framesCsv = rows.join('\n') + '\n';
    if (npy) await npy.end();
    await zip.add('settings.json', JSON.stringify(meta, null, 2));
    await zip.add('frames.csv', framesCsv);
    await zip.finish();
    if (writable) await writable.close(); else download(new Blob(parts, { type: 'application/zip' }), name);
    notice(`Exported ${frames.length} maps (${W} × ${H}) in ${fmt((performance.now() - started) / 1000, 1)} s, with settings.json and frames.csv in the ZIP.`);
    setStatus('Export finished');
  } catch (e) {
    if (writable) await writable.abort().catch(() => {});
    if (e.message === 'cancelled') { setStatus('Export cancelled'); notice('Export cancelled.'); }
    else { showError(`Export failed: ${e.message}`); setStatus('Export failed'); }
  } finally {
    worker.terminate();
    state.exporting = null;
    $('export-cancel').hidden = true; $('ex-progress').hidden = true; bar.style.width = '0';
    updateButtons();
  }
}
$('export-maps').addEventListener('click', exportMaps);
$('export-cancel').addEventListener('click', () => { if (state.exporting) state.exporting.cancel = true; });

$('export-frame').addEventListener('click', async () => {
  const [W, H] = exportSize(), [k, t] = currentFrame();
  const r = renderMap(state.gaze, t, settingsForRender('max'), W, H, renderContext(), {});
  download(new Blob([await encodeGrayPng(r.map, W, H, $('ex-format').value === 'png16' ? 16 : 8)], { type: 'image/png' }), `${exportBaseName()}_f${String(k).padStart(6, '0')}_${W}x${H}.png`);
});

function compositeCanvas(maxWidth = 3840) {
  const [vw, vh] = videoSize(), cw = Math.min(vw, maxWidth), ch = Math.round(cw * vh / vw);
  const c = document.createElement('canvas'); c.width = cw; c.height = ch;
  return c;
}
function drawComposite(c) {
  const g = c.getContext('2d');
  g.drawImage(converting() ? $('erp-canvas') : video, 0, 0, c.width, c.height);
  if (state.ref && state.refShow.heat) g.drawImage(refCanvas, 0, 0, c.width, c.height);
  { g.imageSmoothingEnabled = true; g.imageSmoothingQuality = 'high'; g.drawImage(heatCanvas, 0, 0, c.width, c.height); }
  g.drawImage(marksCanvas, 0, 0, c.width, c.height);
}
$('export-snapshot').addEventListener('click', () => {
  render();
  const c = compositeCanvas(); drawComposite(c);
  const [k] = currentFrame();
  c.toBlob(blob => download(blob, `${exportBaseName()}_overlay_f${String(k).padStart(6, '0')}.png`), 'image/png');
});

$('export-video').addEventListener('click', async () => {
  if (state.recording) { state.recording.finish(); return; }
  const types = ['video/mp4;codecs=avc1.640033', 'video/mp4;codecs=avc1', 'video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm'];
  const mimeType = types.find(t => window.MediaRecorder?.isTypeSupported?.(t));
  if (!mimeType) { showError('This browser cannot record canvas video (MediaRecorder).'); return; }
  const { a, b } = exportFrames();
  const c = compositeCanvas(1920), stream = c.captureStream(Math.min(60, Math.round(state.fps)));
  const rec = new MediaRecorder(stream, { mimeType, videoBitsPerSecond: 12e6 }), chunks = [];
  rec.ondataavailable = e => { if (e.data.size) chunks.push(e.data); };
  const ext = mimeType.startsWith('video/mp4') ? 'mp4' : 'webm';
  rec.onstop = () => { download(new Blob(chunks, { type: mimeType.split(';')[0] }), `${exportBaseName()}_overlay.${ext}`); setStatus('Recording saved'); };
  video.pause(); video.playbackRate = 1; $('speed').value = '1';
  video.currentTime = a;
  await once(video, 'seeked');
  render(); drawComposite(c);
  state.recording = {
    onFrame(meta) { if (meta.mediaTime >= b) { this.finish(); return; } drawComposite(c); },
    finish() { if (!state.recording) return; state.recording = null; video.pause(); if (rec.state !== 'inactive') rec.stop(); $('export-video').textContent = 'Record overlay video'; }
  };
  $('export-video').textContent = 'Stop recording';
  setStatus(`Recording ${fmt(a, 2)}–${fmt(b, 2)} s in real time…`);
  rec.start(500);
  video.play();
});

$('export-gaze').addEventListener('click', () => {
  const s = state.settings, DEG = Math.PI / 180;
  const csv = gazeToCsv(state.gaze, ([lon, lat]) => {
    let l = (s.flipX ? -lon : lon) + s.lonOffsetDeg * DEG;
    l -= 2 * Math.PI * Math.floor((l + Math.PI) / (2 * Math.PI));
    return [l, s.flipY ? -lat : lat];
  });
  download(new Blob([csv], { type: 'text/csv' }), `${stem(state.video?.name || 'gaze')}_gaze_processed.csv`);
});

function updateButtons() {
  const hasVideo = !!state.video && !!video.videoWidth, hasGaze = !!state.gaze, readable = hasVideo && !state.video.tainted;
  $('analyse').disabled = !readable && !state.analysing;
  $('export-maps').disabled = !(hasVideo && hasGaze) || !!state.exporting;
  $('export-frame').disabled = !(hasVideo && hasGaze);
  $('export-snapshot').disabled = !readable;
  $('export-video').disabled = !readable;
  $('export-gaze').disabled = !hasGaze;
  $('view-360').disabled = !readable || state.settings.projection !== 'erp';
  if (state.view && $('view-360').disabled) set360(false);
}

// ---------- file inputs, drag and drop, sample, URL parameters ----------
$('choose-video').addEventListener('click', () => $('video-input').click());
$('empty-video').addEventListener('click', () => $('video-input').click());
$('video-input').addEventListener('change', e => { if (e.target.files[0]) loadVideoFile(e.target.files[0]); e.target.value = ''; });
$('load-video-url').addEventListener('click', () => { const u = $('video-url').value.trim(); if (u) loadVideoUrl(u); });
$('video-url').addEventListener('keydown', e => { if (e.key === 'Enter') $('load-video-url').click(); });
$('choose-gaze').addEventListener('click', () => $('gaze-input').click());
$('choose-gaze-folder').addEventListener('click', () => $('gaze-folder-input').click());
$('gaze-input').addEventListener('change', e => { loadGazeFiles(e.target.files); e.target.value = ''; });
$('gaze-folder-input').addEventListener('change', e => { loadGazeFiles(e.target.files); e.target.value = ''; });
$('load-gaze-url').addEventListener('click', () => { const u = $('gaze-url').value.trim(); if (u) loadGazeUrl(u); });
$('gaze-url').addEventListener('keydown', e => { if (e.key === 'Enter') $('load-gaze-url').click(); });

const drop = $('drop-zone');
drop.addEventListener('dragover', e => { e.preventDefault(); drop.classList.add('dragging'); });
drop.addEventListener('dragleave', () => drop.classList.remove('dragging'));
drop.addEventListener('drop', async e => {
  e.preventDefault(); drop.classList.remove('dragging');
  const files = await droppedFiles(e.dataTransfer);
  const vid = files.find(f => VIDEO_EXT.test(f.name) || f.type.startsWith('video/'));
  if (vid) loadVideoFile(vid);
  const gaze = files.filter(f => GAZE_EXT.test(f.name));
  if (gaze.length) loadGazeFiles(gaze);
});
async function droppedFiles(dt) {
  const entries = [...dt.items].map(i => i.webkitGetAsEntry?.()).filter(Boolean);
  if (!entries.length) return [...dt.files];
  const out = [];
  const walk = async (entry, path) => {
    if (entry.isFile) { const f = await new Promise((res, rej) => entry.file(res, rej)); f.relativePath = path + f.name; out.push(f); return; }
    const reader = entry.createReader();
    let batch;
    do { batch = await new Promise((res, rej) => reader.readEntries(res, rej)); for (const child of batch) await walk(child, `${path}${entry.name}/`); } while (batch.length);
  };
  for (const entry of entries) await walk(entry, '');
  return out;
}

async function loadSample(kind = 'erp') {
  $('help-menu').open = false;
  const flat = kind === '2d', video_ = flat ? 'sample_2d.mp4' : 'sample_erp.mp4', gaze = flat ? 'sample_2d_gaze.csv' : 'sample_gaze.csv';
  try {
    // Fetched into memory so seeking works whatever the web server supports.
    const blob = await (await fetch(`./sample/${video_}`)).blob();
    loadVideoFile(new File([blob], video_, { type: 'video/mp4' }));
    const res = await fetch(`./sample/${gaze}`);
    await loadGazeFiles([new File([await res.text()], gaze)]);
    notice(flat
      ? 'Sample loaded: an ordinary 2D video, six simulated observers follow a red square (gaze in pixels). The Gaussian is drawn in the image plane; the field of view turns degrees into pixels.'
      : 'Sample loaded: 8 simulated observers follow a red ball across a scene whose camera pans. Try a ±1 s window with and without motion compensation (analyse motion first).');
  } catch (e) { notice(`Could not load the sample gaze (${e.message}).`, true); }
}
$('load-sample').addEventListener('click', () => loadSample('erp'));
$('load-sample-2d').addEventListener('click', () => loadSample('2d'));
$('empty-sample').addEventListener('click', () => loadSample('erp'));

(async function fromQuery() {
  const q = new URLSearchParams(location.search);
  const settingsFromQuery = () => {
    const o = {}, num = (k, to) => { if (q.get(k) != null && Number.isFinite(Number(q.get(k)))) o[to] = Number(q.get(k)); };
    num('sigma', 'sigmaDeg'); num('window', 'windowSec'); num('velocity', 'velocity'); num('rotate', 'lonOffsetDeg');
    if (['box', 'triangle', 'gaussian'].includes(q.get('kernel'))) o.kernel = q.get('kernel');
    if (q.has('past')) o.causal = true;
    if (Object.keys(o).length) applySettings(o);
  };
  if (q.get('sample') != null) { await loadSample(q.get('sample') === '2d' ? '2d' : 'erp'); settingsFromQuery(); return; }
  if (q.get('video')) loadVideoUrl(q.get('video'));
  if (q.get('gaze')) {
    try {
      const files = [];
      for (const u of q.get('gaze').split(',').filter(Boolean)) {
        const res = await fetch(u);
        if (!res.ok) throw new Error(`HTTP ${res.status} for ${u}`);
        files.push(new File([await res.text()], stem(u) + (u.match(/\.[a-z]{3,4}(?=$|[?#])/i)?.[0] || '.csv')));
      }
      await loadGazeFiles(files);
    } catch (e) { notice(`Could not fetch the gaze file (${e.message}). The server must allow cross-origin requests (CORS).`, true); }
  }
  settingsFromQuery();
})();
updateButtons(); updateExportSummary();
Object.assign(window.overgaze, { render, exportMaps });
document.addEventListener('click', e => { const d = e.target.closest('[data-dismiss]'); if (d) $(d.dataset.dismiss).hidden = true; });
$('layout').innerHTML = Object.entries(LAYOUTS).map(([k, v]) => `<option value="${k}">${v.label}</option>`).join('');
$('layout').addEventListener('change', e => setLayout(e.target.value, true));
$('view360-mode').innerHTML = Object.entries(VIEW_MODES).map(([k, v]) => `<option value="${k}">${v.label}</option>`).join('');
$('view360-mode').addEventListener('change', e => { if (state.view) { state.view.setMode(e.target.value); $('view360-follow').checked = false; redraw360(); } });

// layout options: cube face order and mirroring (like egjs-view360's cubemapOrder / cubemapFlipX), cylinder coverage
function setLayoutOption(o) {
  const next = { ...state.layoutOpts, ...o };
  try { faceTables(next); } catch (e) { notice(e.message, true); $('face-order').value = state.layoutOpts.order.toUpperCase(); return; }
  state.layoutOpts = next;
  setLayout(state.layout, true);
}
$('face-order').addEventListener('change', e => setLayoutOption({ order: e.target.value.trim().toLowerCase() }));
$('face-mirror').addEventListener('change', e => setLayoutOption({ mirror: e.target.checked }));
$('cyl-hfov').addEventListener('change', e => setLayoutOption({ hfov: Math.max(1, Math.min(360, Number(e.target.value) || 360)) }));

// click on the video: play / pause (in the 360° view only a click, not a drag)
{
  let down = null;
  $('stage').addEventListener('pointerdown', e => { down = { x: e.clientX, y: e.clientY }; });
  let timer = 0;
  $('stage').addEventListener('click', e => {
    if (down && Math.hypot(e.clientX - down.x, e.clientY - down.y) > 4) return;
    clearTimeout(timer);
    if (e.detail > 1) return;   // a double-click resets the 360° view and does not play / pause
    timer = setTimeout(togglePlay, state.view ? 250 : 0);
  });
}
