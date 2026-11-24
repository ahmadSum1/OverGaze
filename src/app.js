import { FILE_TIME, RULE_OPS, buildGaze, columnValues, dropObservers, defaultRanges, detectMapping, gazeToCsv, minDurationFilter, parseTable, pathMatchesVideo, ruleText, velocityFilter } from './gaze.js';
import { agreementScorer, lowScores } from './agreement.js';
import { DEFAULTS, coverage, frameCentre, frameOf, mapSize, renderMap } from './saliency.js';
import { FlowStore, toGray } from './flow.js';
import { openMotionFile } from './motionfile.js';
import { ZipWriter, encodeGrayPng, npyHeader } from './export.js';
import { fpsFromDeltas } from './timing.js';
import { fpsFromMp4File } from './mp4fps.js';
import { pairEyes, withoutEye } from './pairs.js';
import { collectMarks } from './marks.js';
import { compareMaps, openReference, referencesFor, resample } from './reference.js';
import { PRESETS, pickSettings, presetDifferences, settingsFromJson } from './presets.js';
import { LayoutConverter, VIEW_MODES, View360 } from './view360.js';
import { CUBE_LAYOUTS, DEFAULT_OPTIONS, LAYOUTS, erpSize, faceTables } from './projections.js';
import { COLORMAPS, colorize, cssGradient } from './colormap.js';
import { contourSegments, strokeSegments } from './contour.js';

const VERSION = '1.0.0';
const $ = id => document.getElementById(id);
const video = $('video');
const refCanvas = $('ref-canvas'), heatCanvas = $('heat-canvas'), outlineCanvas = $('outline-canvas'), marksCanvas = $('marks-canvas'), densityCanvas = $('density-canvas');
const GAZE_EXT = /\.(csv|tsv|txt|json|arff|dat)$/i, VIDEO_EXT = /\.(mp4|m4v|webm|mkv|mov|ogv)$/i;

const state = {
  video: null, fps: 30, fpsSource: 'default', fpsDeltas: [], lastFrameMeta: null, mediaTime: 0,
  files: [], mapping: null, gazeAll: null, gaze: null, observerEnabled: null, conditionEnabled: null,
  settings: { ...DEFAULTS, velocity: 0, minDurationMs: 0, eyePairing: 'auto' }, people: null,
  rowRules: [], agreement: null, observerSort: 'name',
  display: { outline: false, outlineColor: '#c8f56a', outlineTouched: false, outlineWidth: 2, mode: 'heat', colormap: 'jet', opacity: 0.65, gamma: 1, mapWidth: 512, showPoints: true, showFlow: false },
  points: { size: 6, dwell: false, dwellCap: 3, eyeLines: false },
  flow: null, flowInfo: null, analysing: null, exporting: null, recording: null,
  work: {}, last: null, renderQueued: false, projectionTouched: false, density: null,
  library: { videos: [], gaze: [], refs: [] }, tiedTo: null,
  ref: null, refShow: { heat: false, outline: true, outlineColor: '#ffffff', outlineWidth: 2 }, showGaze: true, showRef: true, removedObservers: new Set(), view: null,
  layout: 'erp', layoutTouched: false, layoutOpts: { ...DEFAULT_OPTIONS }, converter: null
};

// Handle for the tests and the console: mapAt(frame, width) renders the same map the export writes (raw, unnormalised).
window.overgaze = {
  state, debugReport: () => debugReport(), applySettings: o => applySettings(o), settingsJson: () => settingsJson(),
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
  state.video = { name: file.name, size: file.size, src, blobUrl: src, remote: false, tainted: false, file };
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
    // 2:1, or close to it: many 360° videos are full-sphere equirectangular with a few rows more or less (3840 × 2048)
    const erp = (vw / vh > 1.86 && vw / vh < 2.15) || converting();
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
  tieToVideo();
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
  $('fps').value = String(Math.round(f * 1000) / 1000);
  $('fps-note').textContent = { manual: 'Set by hand.', file: `Read from the video file: ${Math.round(f * 1000) / 1000} frames per second.`, measured: `Measured from decoded frames: ${Math.round(f * 1000) / 1000} frames per second.` }[source]
    || 'Default 30 until the frame rate is known: it is read from MP4 / MOV files, otherwise measured from decoded frames.';
  if (state.files.length && state.mapping && (/^frame/.test(state.mapping.timeUnit) || /^frame/.test(state.mapping.durationUnit))) rebuildGaze();
  updateSubtitle(); updateExportSummary(); drawDensity(); requestRender();
}

// The frame rate, as soon as the video has loaded: from the file header for MP4 / MOV files (exact), otherwise
// by playing a hidden muted copy and timing the decoded frames. On a busy computer the decoder can skip frames, so
// the measurement is retried with a longer wait before giving up (playback then measures it as before).
async function detectFps() {
  const v = state.video;
  if (!v) return;
  if (v.file && /\.(mp4|m4v|mov|3gp)$/i.test(v.name)) {
    try {
      const r = await fpsFromMp4File(v.file);
      if (r && state.video === v) { if (state.fpsSource !== 'manual') setFps(r.fps, 'file'); return; }
    } catch { /* not readable as MP4: measure instead */ }
  }
  if (!('requestVideoFrameCallback' in HTMLVideoElement.prototype)) return;
  for (const waitMs of [3000, 8000, 20000]) {
    if (state.video !== v || state.fpsSource !== 'default') return;
    const f = await measureFps(waitMs).catch(() => null);
    if (f && state.video === v) { if (state.fpsSource === 'default') setFps(f, 'measured'); return; }
  }
  if (state.video === v && state.fpsSource === 'default') notice('The frame rate could not be measured yet, so 30 frames per second is used until the video plays. If the gaze is in frame numbers, set “Video frame rate” by hand.', true);
}

async function measureFps(waitMs) {
  const probe = hiddenVideo();
  try {
    await once(probe, 'loadeddata', waitMs + 5000);
    probe.currentTime = Math.min(1, (probe.duration || 2) / 4);
    await once(probe, 'seeked', waitMs + 5000);
    const deltas = [];
    let last = null;
    await probe.play();
    await new Promise(resolve => {
      const stop = setTimeout(resolve, waitMs);
      const cb = (now, meta) => {
        if (last && meta.presentedFrames === last.presentedFrames + 1) deltas.push(meta.mediaTime - last.mediaTime);
        last = meta;
        if (deltas.length >= 40 || probe.ended) { clearTimeout(stop); resolve(); } else probe.requestVideoFrameCallback(cb);
      };
      probe.requestVideoFrameCallback(cb);
    });
    return fpsFromDeltas(deltas);
  } finally { disposeHidden(probe); }
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

function togglePlay() { if (!state.video) return; if (state.ref) state.ref.resume = false; if (video.paused) video.play().catch(e => showError(String(e.message || e))); else video.pause(); }
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
  refreshPresetLabel();
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
  const wrapX = state.settings.projection === 'erp', refOutline = !!(state.ref && state.showRef && state.refShow.outline);
  if (state.showGaze && state.display.mode !== 'off' && result) colorize(result.map, img.data, colorOpts);
  hctx.putImageData(img, 0, 0);
  // the two outlines are independent (each has its own switch and colour); they are traced here as lines and drawn
  // on their own screen-sized layer, so their thickness is in screen pixels whatever the map size or zoom
  state.outlines = { W, H, lines: [] };
  if (refOutline && ref) state.outlines.lines.push({ segs: outlineOf(ref.map, W, H, thr, wrapX), color: state.refShow.outlineColor, width: state.refShow.outlineWidth });
  if (state.showGaze && state.display.outline && result && result.max > 0) state.outlines.lines.push({ segs: outlineOf(result.map, W, H, thr, wrapX), color: state.display.outlineColor, width: state.display.outlineWidth });
  // the reference heatmap has its own layer (under ours) and its own colours
  const rctx = refCanvas.getContext('2d');
  if (state.ref && state.showRef && state.refShow.heat && ref) {
    if (refCanvas.width !== W || refCanvas.height !== H) { refCanvas.width = W; refCanvas.height = H; }
    const rimg = rctx.createImageData(W, H);
    colorize(normalised(ref.map), rimg.data, { ...colorOpts, colormap: state.refColormap || 'ice', mode: state.display.mode === 'off' || state.display.mode === 'spotlight' ? 'heat' : state.display.mode });
    rctx.putImageData(rimg, 0, 0);
  } else rctx.clearRect(0, 0, refCanvas.width, refCanvas.height);
  updateReferenceReadout(ref, result);
  state.last = { t, k, result, W, H };
  drawOutlineLayer();
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
/** The border of the area at or above frac × max, as line segments (wraps left/right for 360° maps). */
function outlineOf(map, W, H, frac, wrap) {
  let mx = 0;
  for (const v of map) if (v > mx) mx = v;
  return mx > 0 ? contourSegments(map, W, H, frac * mx, wrap) : new Float32Array(0);
}
/**
 * Draw the current outlines onto a context of cw × ch pixels. `perScreenPixel` is how many of that context's pixels
 * one screen pixel covers; each outline has its own thickness in screen pixels.
 */
function drawOutlines(g, cw, ch, perScreenPixel) {
  const o = state.outlines;
  if (!o) return;
  for (const l of o.lines) strokeSegments(g, l.segs, o.W, o.H, cw, ch, l.width * perScreenPixel, l.color);
}
/** The on-screen outline layer: as big as the stage in device pixels. */
function drawOutlineLayer() {
  const rect = $('stage').getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
  const cw = Math.max(1, Math.round(rect.width * dpr)), ch = Math.max(1, Math.round(rect.height * dpr));
  if (outlineCanvas.width !== cw || outlineCanvas.height !== ch) { outlineCanvas.width = cw; outlineCanvas.height = ch; }
  const g = outlineCanvas.getContext('2d');
  g.clearRect(0, 0, cw, ch);
  if (!state.view) drawOutlines(g, cw, ch, dpr);   // the 360° view draws its own
}

// ---------- reference maps ----------
/** The reference map for video frame k at W×H, from the cache; starts loading it if needed (then redraws). */
function referenceAt(k, W, H) {
  const R = state.ref;
  const idx = R.N === 1 ? 0 : Math.round(k * R.scale) + R.offset;
  if (idx < 0 || idx >= R.N) { R.status = `video frame ${k} → reference frame ${idx}: outside 0–${R.N - 1}`; return null; }
  const load = j => {
    const key = `${j}:${W}x${H}`;
    if (j < 0 || j >= R.N || R.cache.has(key) || R.pending.has(j)) return;
    R.pending.add(j);
    R.frame(j).then(raw => {
      const map = resample(raw, R.W, R.H, W, H);
      R.cache.set(key, map);
      while (R.cache.size > 32) R.cache.delete(R.cache.keys().next().value);
      // while playing, show the newest frame that is ready, so the reference keeps moving with the video
      if (!R.shown || R.shown.W !== W || Math.abs(j - R.want) < Math.abs(R.shown.idx - R.want)) R.shown = { idx: j, map, W, H };
    }).catch(e => { R.error = `frame ${j}: ${e.message}`; remember('reference error', R.error); })
      .finally(() => { R.pending.delete(j); if (state.ref === R && (video.paused || j === R.want)) { if (j === R.want) resumeAfterReference(); requestRender(); } });
  };
  R.want = idx;
  const key = `${idx}:${W}x${H}`;
  const P = R.pre;
  if (P && P.done[idx]) {                       // prepared in the background after loading: no disk read needed
    let map = R.cache.get(key);
    if (!map) {
      const n = P.W * P.H, q = P.data.subarray(idx * n, (idx + 1) * n), f = new Float32Array(n), m = P.max[idx] / 255;
      for (let i = 0; i < n; i++) f[i] = q[i] * m;
      map = resample(f, P.W, P.H, W, H);
      R.cache.set(key, map);
      while (R.cache.size > 32) R.cache.delete(R.cache.keys().next().value);
    }
    R.status = ''; R.shown = { idx, map, W, H };
    return { idx, map, fresh: true };
  }
  // read ahead while playing (a few frames, so slow disks are not swamped)
  if (!video.paused && R.N > 1) for (let j = idx + 1; j <= idx + 6 && R.pending.size < 4; j++) load(j);
  if (R.cache.has(key)) { const map = R.cache.get(key); R.status = ''; R.shown = { idx, map, W, H }; return { idx, map, fresh: true }; }
  load(idx);
  R.status = `loading reference frame ${idx}…`;
  if (R.hold && !video.paused) { video.pause(); R.resume = true; }   // keep video and reference together
  // meanwhile keep showing the nearest frame that was ready (marked as not current)
  return R.shown && R.shown.W === W && R.shown.H === H ? { idx: R.shown.idx, map: R.shown.map, fresh: false } : null;
}
function updateReferenceReadout(ref, result) {
  const R = state.ref;
  $('ro-ref-card').hidden = !R;
  if (!R) return;
  $('ref-status').textContent = (R.error ? `Error: ${R.error}. ` : '') + (R.prepStatus ? R.prepStatus[0].toUpperCase() + R.prepStatus.slice(1) + '. ' : '') + (R.status || (ref ? `Showing reference frame ${ref.idx}.` : ''));
  if (arguments[2]) return;   // progress only
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
    $('ref-offset').value = String(R.offset); $('ref-scale').value = '1'; $('clear-ref').disabled = false; state.showRef = true; syncLayerChips();
    const vFrames = video.duration ? Math.round(video.duration * state.fps) : 0;
    const msgs = [`Reference loaded: ${R.name}.`];
    if (R.kind === 'bin' && !state.projectionTouched && state.settings.projection !== 'erp') { state.settings.projection = 'erp'; $('projection').value = 'erp'; $('fov-row').hidden = true; updateSubtitle(); msgs.push('Projection set to 360° (Salient360 maps are equirectangular).'); }
    if (R.kind === 'bin') msgs.push('Salient360 maps: frame offset +1 (bin frame 0 is empty, bin frame k belongs to video frame k − 1).');
    if (R.N > 1 && vFrames && Math.abs(R.N - vFrames) > 2) msgs.push(`It has ${R.N} frames and the video about ${vFrames}: set “Ref. frames per video frame” (e.g. ${fmt(R.N / vFrames, 3)}) or the offset if they do not line up.`);
    notice(msgs.join(' '), R.N > 1 && vFrames && Math.abs(R.N - vFrames) > 2);
    R.hold = $('ref-hold').checked;
    if (!state.display.outlineTouched) { state.display.outline = true; $('show-outline').checked = true; }   // ours next to the reference's, to compare
    requestRender();
    prepareReference(R);
  } catch (e) { notice(`Could not open the reference: ${e.message}`, true); }
}
function resumeAfterReference() { const R = state.ref; if (R?.resume) { R.resume = false; video.play().catch(() => {}); } }

/**
 * Read every reference frame once, right after loading, and keep it small in memory (at most 512 × 256, one byte
 * per value plus the frame maximum: 79 MB for 600 frames), starting at the playhead. Playback then never waits for
 * the disk. Frames not prepared yet are still read on demand.
 */
async function prepareReference(R) {
  const PW = Math.min(R.W, 512), PH = Math.max(1, Math.round(PW * R.H / R.W)), n = PW * PH;
  if (R.N * n > 800e6) { R.prepStatus = 'too many frames to keep in memory: read on demand'; return; }
  const P = R.pre = { W: PW, H: PH, data: new Uint8Array(R.N * n), max: new Float32Array(R.N), done: new Uint8Array(R.N), count: 0 };
  const start = Math.max(0, Math.min(R.N - 1, R.want ?? 0)), order = [];
  for (let i = 0; i < R.N; i++) order.push((start + i) % R.N);
  const t0 = performance.now();
  let next = 0;
  const worker = async () => {
    while (next < order.length && state.ref === R) {
      const j = order[next++];
      if (P.done[j]) continue;
      try {
        const small = resample(await R.frame(j), R.W, R.H, PW, PH);
        let mx = 0;
        for (const v of small) if (v > mx) mx = v;
        const q = P.data.subarray(j * n, (j + 1) * n), k = mx > 0 ? 255 / mx : 0;
        for (let i = 0; i < n; i++) q[i] = Math.round(small[i] * k);
        P.max[j] = mx; P.done[j] = 1; P.count++;
      } catch (e) { R.error = `frame ${j}: ${e.message}`; }
      if (P.count % 10 === 0 || P.count === R.N) {
        const el = (performance.now() - t0) / 1000;
        R.prepStatus = P.count === R.N ? `all ${R.N} frames ready in memory (${fmt(el, 1)} s)` : `preparing frames: ${P.count} / ${R.N} (${fmt(el / P.count * (R.N - P.count), 0)} s left)`;
        if (video.paused) requestRender(); else updateReferenceReadout(null, null, true);
        if (j === R.want) resumeAfterReference();
      }
    }
  };
  await Promise.all([worker(), worker()]);
}
$('choose-ref').addEventListener('click', () => $('ref-input').click());
$('ref-input').addEventListener('change', e => { if (e.target.files.length) chooseReferences([...e.target.files]); e.target.value = ''; });
$('choose-ref-folder').addEventListener('click', () => $('ref-folder-input').click());
$('ref-folder-input').addEventListener('change', e => { if (e.target.files.length) chooseReferences([...e.target.files], true); e.target.value = ''; });
const REF_EXT = /\.(bin|npy|png|jpe?g|webp|bmp)$/i;
/** One file or one image sequence opens directly; several (a folder of maps for many videos) become a library. */
function chooseReferences(files, folder = false) {
  files = files.filter(f => REF_EXT.test(f.name) && !f.name.startsWith('.'));
  if (!files.length) { notice('No reference maps found (.bin, .npy or images).', true); return; }
  const arrays = files.filter(f => /\.(bin|npy)$/i.test(f.name));
  if (!folder && (files.length === 1 || !arrays.length)) {
    const key = state.video ? stem(state.video.name) : '', mine = key && files.length > 1 ? referencesFor(files, key) : [];
    loadReference(mine.length ? mine : files);
    return;
  }
  state.library.refs = files;
  if (!state.video) { notice(`${files.length} reference files kept; open a video to use its own.`); return; }
  tieReference();
}
function tieReference() {
  if (!state.library.refs.length || !state.video) return;
  const key = stem(state.video.name), mine = referencesFor(state.library.refs, key);
  if (mine.length) loadReference(mine);
  else { state.ref = null; syncLayerChips(); requestRender(); notice(`None of the ${state.library.refs.length} reference files belongs to “${key}”.`, true); }
}
/** After another video is opened: take its own fixation files and reference from what was loaded before. */
async function tieToVideo() {
  if (!state.video || state.tiedTo === state.video) return;
  const first = !state.tiedTo;
  state.tiedTo = state.video;
  if (state.library.gaze.length && !first) { clearGaze(true); await loadGazeFiles(state.library.gaze, { fromLibrary: true }); }
  else if (state.library.gaze.length && first && !state.files.length) await loadGazeFiles(state.library.gaze, { fromLibrary: true });
  if (state.library.refs.length) tieReference();
}
$('clear-ref').addEventListener('click', () => { state.ref = null; state.library.refs = []; syncLayerChips(); $('ref-name').textContent = $('ref-size').textContent = '—'; $('ref-status').textContent = 'No reference loaded.'; $('clear-ref').disabled = true; requestRender(); });
$('ref-hold').addEventListener('change', e => { if (state.ref) state.ref.hold = e.target.checked; if (!e.target.checked) resumeAfterReference(); });
$('ref-offset').addEventListener('change', e => { if (state.ref) { state.ref.offset = Math.round(Number(e.target.value) || 0); requestRender(); } });
$('ref-scale').addEventListener('change', e => { if (state.ref) { state.ref.scale = Math.max(0.001, Number(e.target.value) || 1); requestRender(); } });
// switching one of a side's layers on also shows that side, so the switch does what it says
$('ref-heat').addEventListener('change', e => { state.refShow.heat = e.target.checked; if (e.target.checked) state.showRef = true; syncLayerChips(); requestRender(); });
$('ref-outline').addEventListener('change', e => { state.refShow.outline = e.target.checked; if (e.target.checked) state.showRef = true; syncLayerChips(); requestRender(); });
$('ref-outline-color').addEventListener('input', e => { state.refShow.outlineColor = e.target.value; requestRender(); });
$('outline-color').addEventListener('input', e => { state.display.outlineColor = e.target.value; requestRender(); });
const widthInput = (id, set) => $(id).addEventListener('change', e => { const w = Math.min(20, Math.max(0.5, Number(e.target.value) || 2)); e.target.value = String(w); set(w); requestRender(); });
widthInput('outline-width', w => { state.display.outlineWidth = w; });
widthInput('ref-outline-width', w => { state.refShow.outlineWidth = w; });
$('ref-colormap').innerHTML = COLORMAPS.map(c => `<option value="${c}">${c[0].toUpperCase() + c.slice(1)}</option>`).join('');
$('ref-colormap').value = 'ice';
$('ref-colormap').addEventListener('change', e => { state.refColormap = e.target.value; requestRender(); });
/**
 * Toolbar buttons: one per side. "Gaze" hides or shows everything of ours (heatmap, gaze points, outline),
 * "Reference" everything of the reference (heatmap and outline). The switches in the panels say which layers a
 * side has; the button hides the whole side without touching them.
 */
function syncLayerChips() {
  const on = (id, v) => { $(id).classList.toggle('active', v); $(id).setAttribute('aria-pressed', String(v)); };
  on('layer-gaze', state.showGaze); on('layer-ref', !!state.ref && state.showRef);
  $('layer-ref').disabled = !state.ref;
}
$('layer-gaze').addEventListener('click', () => { state.showGaze = !state.showGaze; syncLayerChips(); requestRender(); });
$('layer-ref').addEventListener('click', () => { state.showRef = !state.showRef; syncLayerChips(); requestRender(); });
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
  if (on && state.settings.projection !== 'erp') notice('The gaze is set to “2D video”: the 360° view treats the frame as equirectangular anyway. Set Projection to 360° in the settings if the video is a 360° video.', true);
  glCanvas.hidden = !on; $('view360-bar').hidden = !on; $('view-360').classList.toggle('active', on);
  const [vw, vh] = videoSize();
  $('stage').style.setProperty('--aspect', String(on ? 16 / 9 : vw / vh));
  requestRender();
}
function draw360(result, W, H) {
  const v = state.view, src = state.viewSource;
  const sw = 2048, sh = 1024;   // fixed, even for small videos, so thin outlines stay sharp when zoomed in
  if (src.width !== sw || src.height !== sh) { src.width = sw; src.height = sh; }
  // outline thickness in screen pixels at the centre of the view: source pixels per radian over screen pixels per radian
  drawComposite(src, (window.devicePixelRatio || 1) * (sw / (2 * Math.PI)) / v.pixelsPerRadian());
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
  glCanvas.addEventListener('wheel', e => { if (!state.view) return; e.preventDefault(); state.view.zoom(Math.exp(e.deltaY * 0.001)); requestRender(); }, { passive: false });   // redraws the outlines for the new zoom
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
  state.marksT = t; state.marksResult = !!result;
  // in the 360° view the points are drawn into the view's source at the view's zoom (draw360), so their size stays
  // in screen pixels
  if (state.display.showPoints && state.showGaze && result && !state.view) drawGazePoints(g, t, cw, ch, dpr);
}

/** Gaze points: one per fixation or sample in the pooling window (see marks.js). */
function drawGazePoints(g, t, cw, ch, dpr) {
  const P = state.points, TAU = Math.PI * 2;
  const marks = collectMarks(state.gaze, t, settingsForRender(), renderContext());
  const X = m => (m.lon / TAU + 0.5) * cw, Y = m => (0.5 - m.lat / Math.PI) * ch;
  const base = P.size / 2 * dpr;
  const radius = m => base * (P.dwell ? Math.min(P.dwellCap, 1 + Math.max(0, m.dwell) / 0.5) : 1);
  if (P.eyeLines && state.people) {
    // each person's left and right eye at the same moment: their marks in the current frame
    const now = new Map(); for (const m of marks) if (m.current) now.set(m.obs, m);
    g.save(); g.setLineDash([4 * dpr, 3 * dpr]); g.lineWidth = 1 * dpr; g.strokeStyle = 'rgba(255,255,255,0.8)';
    for (const p of peopleNow()) {
      const a = now.get(p.left), b = now.get(p.right);
      if (!a || !b || Math.abs(X(a) - X(b)) > cw / 2) continue;
      g.beginPath(); g.moveTo(X(a), Y(a)); g.lineTo(X(b), Y(b)); g.stroke();
    }
    g.restore();
  }
  // largest first, so small dots stay visible on top
  marks.sort((a, b) => radius(b) - radius(a));
  for (const m of marks) {
    g.beginPath(); g.arc(X(m), Y(m), radius(m), 0, TAU);
    g.fillStyle = obsColor(m.obs); g.globalAlpha = 0.35 + 0.65 * m.w; g.fill();
    g.globalAlpha = 1; g.lineWidth = 1 * dpr; g.strokeStyle = 'rgba(0,0,0,0.7)'; g.stroke();
  }
}
function pointsJson() { const P = state.points; return { show: state.display.showPoints, dot_size_px: P.size, grow_while_in_place: P.dwell, grow_cap_times_size: P.dwellCap, grow_rule: 'radius × (1 + time in place / 0.5 s), up to the cap; fixations: time into the fixation; raw samples: time within 1.5° of where the eye arrived', join_eyes: P.eyeLines }; }
function applyPoints(o = {}) {
  const P = state.points;
  if ('show' in o) { state.display.showPoints = !!o.show; $('show-points').checked = !!o.show; }
  if (Number(o.dot_size_px) > 0) P.size = Math.min(30, Math.max(2, Number(o.dot_size_px)));
  if ('grow_while_in_place' in o) P.dwell = !!o.grow_while_in_place;
  if (Number(o.grow_cap_times_size) >= 1) P.dwellCap = Math.min(10, Number(o.grow_cap_times_size));
  if ('join_eyes' in o) P.eyeLines = !!o.join_eyes;
  $('point-size').value = String(P.size); $('point-dwell').checked = P.dwell; $('point-dwell-cap').value = String(P.dwellCap); $('point-eye-lines').checked = P.eyeLines;
  requestRender();
}
$('point-size').addEventListener('change', e => applyPoints({ dot_size_px: e.target.value }));
$('point-dwell').addEventListener('change', e => applyPoints({ grow_while_in_place: e.target.checked }));
$('point-dwell-cap').addEventListener('change', e => applyPoints({ grow_cap_times_size: e.target.value }));
$('point-eye-lines').addEventListener('change', e => applyPoints({ join_eyes: e.target.checked }));

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
async function loadGazeFiles(list, { fromLibrary = false } = {}) {
  const chosen = list.length;   // a FileList empties when its input is reset
  let files = [...list].filter(f => GAZE_EXT.test(f.name) && !f.name.startsWith('.'));
  if (!fromLibrary && files.length > 1) {
    // keep every file chosen: when another video is opened, its own files are taken from these
    const key = f => `${f.webkitRelativePath || f.name}|${f.size}|${f.lastModified}`, have = new Set(state.library.gaze.map(key));
    state.library.gaze = state.library.gaze.concat(files.filter(f => !have.has(key(f))));
  }
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
  if (m.preset === 's360') {
    const p = PRESETS.find(x => x.id === 'salient360-he');
    applySettings(p.settings); renderPresetList('p:salient360-he'); $('preset-about').textContent = p.about;
    msg.push('Recognised Salient360 fixations: the Salient360 head + eye preset is set (σ 2°, per frame, both eyes).');
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
    const built = buildGaze(state.files, state.mapping, { fps: state.fps, rules: state.rowRules });
    const removed = new Set(built.observers.map((o, i) => state.removedObservers?.has(o) ? i : -1).filter(i => i >= 0));
    state.gazeBuilt = built;
    state.gazeAll = dropObservers(built, removed);
    state.gazeAll.skipped = built.skipped; state.gazeAll.ruleRemoved = built.ruleRemoved; state.gazeAll.rules = built.rules;
    updatePairs();
    applyGazeFilters(false);
    state.observerEnabled = Uint8Array.from(state.gaze.observers, o => prevObs?.has(o) ? prevObs.get(o) : 1);
    state.conditionEnabled = Uint8Array.from(state.gaze.conditions, c => prevCond?.has(c) ? prevCond.get(c) : 1);
  } catch (e) {
    state.gaze = null;
    $('mapping-error').hidden = false; $('mapping-error').textContent = e.message; remember('mapping error', e.message);
  }
  renderRules(); scheduleAgreement();
  renderGroups(); updateGazeSummary(); drawDensity(); updateSubtitle(); updateButtons(); updateExportSummary(); requestRender();
}

/**
 * The filters after reading, in a fixed order: rows without a valid time or position are skipped and the drop
 * rules applied while reading (buildGaze); then fixations shorter than the minimum duration; then the speed filter.
 */
/**
 * Pair each person's left and right eye (see pairs.js). Made on the gaze before the duration and speed filters, so
 * the pairs do not change with them, nor with removing a recording (its partner then stays as a one-eye person).
 */
function updatePairs() {
  const g = state.gazeBuilt, m = state.mapping;
  state.people = g && state.settings.eyePairing !== 'off' ? pairEyes(g, { preset: m?.preset, observerFrom: m?.observerFrom, fileCount: state.files.length }) : null;
  const P = state.people, el = $('pair-status');
  if (!el) return;
  if (!g) { el.textContent = ''; return; }
  if (state.settings.eyePairing === 'off') { el.textContent = 'Every recording is its own observer.'; return; }
  if (!P) { el.textContent = 'No left and right eyes to pair: the names do not say which recording is which eye.'; return; }
  const single = P.singleLeft + P.singleRight;
  el.textContent = `${P.pairs} people with both eyes, ${single} recording${single === 1 ? '' : 's'} with no partner (${P.singleLeft} left, ${P.singleRight} right). `
    + (P.method === 'names' ? 'Partners found from the shared names. ' : 'Partners found by matching: fixations that start at the same moments (within 60 ms) at the same spot (within 3°). ')
    + (P.sidesFrom === 'file order' ? 'The file names do not say which eye is which: the first file is taken as the left eye.' : 'Left and right taken from the names (folders, conditions or observer names).');
}
$('eye-pairing').addEventListener('change', e => { applySettings({ eyePairing: e.target.value }); });

function applyGazeFilters(refresh = true) {
  if (!state.gazeAll) return;
  const S = state.settings;
  try {
    const d = minDurationFilter(state.gazeAll, S.minDurationMs);
    const g = velocityFilter(d, S.velocity);
    g.durationRemoved = d.durationRemoved || 0;
    state.gaze = g;
    $('mapping-error').hidden = true;
  } catch (e) {
    state.gaze = null; $('mapping-error').hidden = false; $('mapping-error').textContent = e.message; remember('filter error', e.message);
  }
  $('min-dur').disabled = !state.gazeAll.hasDuration;
  $('min-dur-hint').textContent = state.gazeAll.hasDuration ? '0 = off. Drops fixations shorter than this, e.g. 100 ms.' : 'Only for fixation files (with a duration or end column): these gaze rows have no durations.';
  scheduleAgreement();
  if (refresh) { renderGroups(); updateGazeSummary(); drawDensity(); updateButtons(); updateExportSummary(); requestRender(); }
}

function updateGazeSummary() {
  const g = state.gaze, el = $('gaze-summary');
  if (!g) { el.textContent = state.files.length ? 'Check the column mapping.' : 'No gaze loaded. One file per observer, or one file with an observer column.'; return; }
  const kind = g.hasDuration ? 'fixations' : 'samples';
  let text = `${state.files.length} file(s), ${g.n.toLocaleString()} ${kind}, ${g.observers.length} observer(s), ${g.conditions.length} condition(s), ${fmt(g.tMin, 2)}–${fmt(g.tMax, 2)} s.`;
  if (state.gazeAll && state.gazeAll.skipped) text += ` ${state.gazeAll.skipped} row(s) skipped (no valid time or position, e.g. nan where the eye tracker lost the eye).`;
  (state.gazeAll?.rules || []).forEach((r, i) => { text += ` Rule “${ruleText(r)}” removed ${(state.gazeAll.ruleRemoved[i] || 0).toLocaleString()}.`; });
  if (state.settings.minDurationMs > 0 && g.hasDuration) text += ` Minimum duration (${state.settings.minDurationMs} ms) removed ${(g.durationRemoved || 0).toLocaleString()}.`;
  if (state.settings.velocity > 0) text += ` Speed filter (${state.settings.velocity}°/s) removed ${(g.filtered || 0).toLocaleString()}.`;
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
  const A = state.agreement?.done && state.agreement.gaze === g ? state.agreement : null;
  const order = g.observers.map((_, i) => i).filter(i => !removed.has(g.observers[i]));
  if (state.observerSort === 'agreement' && A) order.sort((a, b) => (Number.isFinite(A.scores[a]) ? A.scores[a] : Infinity) - (Number.isFinite(A.scores[b]) ? A.scores[b] : Infinity));
  const scoreCell = i => {
    if (!A) return '';
    const v = A.scores[i], low = A.low.set.has(i);
    const tip = Number.isFinite(v)
      ? `Agreement with the other observers${g.conditions.length > 1 ? ' of the same condition' : ''}: ${v.toFixed(3)} (correlation of the maps, averaged over ${A.frames[i]} frames).` + (low ? ` Far below the others: under ${A.low.cutoff.toFixed(3)}, the lower quartile minus 1.5 × the spread between the quartiles (the box-plot rule). Not removed: use × if you decide to.` : '')
      : 'No frame where this observer and the others both have gaze.';
    return `<b class="agree${low ? ' low' : ''}" title="${escapeHtml(tip)}">${low ? '⚠ ' : ''}${Number.isFinite(v) ? (Math.abs(v) < 0.005 ? 0 : v).toFixed(2) : '—'}</b>`;
  };
  if (state.people) { renderPeople(g, obsCounts, A, removed); return; }
  $('observer-list').innerHTML = order.map(i => { const o = g.observers[i]; return `<div class="observer-row${A?.low.set.has(i) ? ' flagged' : ''}"><label><input type="checkbox" data-obs="${i}" ${state.observerEnabled[i] ? 'checked' : ''}><i style="background:${obsColor(i)}"></i><span title="${escapeHtml(o)}">${escapeHtml(o)}</span><small>${obsCounts[i]}</small>${scoreCell(i)}</label><button class="row-remove" type="button" data-remove="${i}" title="Remove ${escapeHtml(o)}" aria-label="Remove ${escapeHtml(o)}">×</button></div>`; }).join('')
    + (removed.size ? `<button id="restore-observers" class="link-button" type="button">bring back the ${removed.size} removed</button>` : '');
}
/** The people of the current gaze set: observer indexes of state.gaze (the pairs are made on state.gazeAll). */
function peopleNow() {
  const P = state.people, g = state.gaze, all = state.gazeBuilt;
  if (!P || !g || !all) return [];
  const removed = state.removedObservers || new Set();
  const idx = new Map(g.observers.map((o, i) => [o, i])), at = i => (i >= 0 && !removed.has(all.observers[i]) ? idx.get(all.observers[i]) ?? -1 : -1);
  return P.people.map((p, k) => ({ k, left: at(p.left), right: at(p.right), match: p.score })).filter(p => p.left >= 0 || p.right >= 0);
}
const shortName = o => o.split('/').pop();
// a recording numbered within its file ("…#07") is shown as L#07 / R#07, otherwise by its own name
const recName = (o, side) => (/#\d+$/.test(o) ? side + o.match(/#\d+$/)[0] : shortName(o));
function personName(g, p) {
  const l = p.left >= 0 ? g.observers[p.left] : null, r = p.right >= 0 ? g.observers[p.right] : null;
  if (state.people.method === 'names') return withoutEye(shortName(l || r));
  return [l && recName(l, 'L'), r && recName(r, 'R')].filter(Boolean).join(' + ');
}
/** A person's agreement: the mean of their eyes' scores (each eye is scored against the other observers of its side). */
function personScore(A, p) {
  const v = [p.left, p.right].filter(i => i >= 0).map(i => A.scores[i]).filter(Number.isFinite);
  return v.length ? v.reduce((a, b) => a + b, 0) / v.length : NaN;
}
/** One row per person, their left and right eye underneath (collapsible). */
function renderPeople(g, counts, A, removed) {
  let people = peopleNow();
  const scores = A ? people.map(p => personScore(A, p)) : null, low = A ? lowScores(scores) : null;
  people = people.map((p, i) => ({ ...p, score: scores?.[i] ?? NaN, low: !!low?.set.has(i) }));
  if (state.observerSort === 'agreement' && A) people.sort((a, b) => (Number.isFinite(a.score) ? a.score : Infinity) - (Number.isFinite(b.score) ? b.score : Infinity));
  const both = people.filter(p => p.left >= 0 && p.right >= 0).length;
  $('observer-count').textContent = `${people.length} people (${both} with both eyes)${removed.size ? `, ${removed.size} recording${removed.size === 1 ? '' : 's'} removed` : ''}`;
  const score = (v, low, tip) => A ? `<b class="agree${low ? ' low' : ''}" title="${escapeHtml(tip)}">${low ? '⚠ ' : ''}${Number.isFinite(v) ? (Math.abs(v) < 0.005 ? 0 : v).toFixed(2) : '—'}</b>` : '';
  const eyeRow = (i, side) => {
    const o = g.observers[i], v = A?.scores[i];
    return `<div class="observer-row eye-row${A?.low.set.has(i) ? ' flagged' : ''}"><label><input type="checkbox" data-obs="${i}" ${state.observerEnabled[i] ? 'checked' : ''}><i class="eye-mark ${side}" style="--c:${obsColor(i)}"></i><span title="${escapeHtml(o)}">${side === 'L' ? 'left' : 'right'} eye · ${escapeHtml(recName(o, side))}</span><small>${counts[i]}</small>${score(v, A?.low.set.has(i), Number.isFinite(v) ? `This eye's agreement with the other observers of its side: ${v.toFixed(3)}.` : 'No frame with gaze of this eye and the others.')}</label><button class="row-remove" type="button" data-remove="${i}" title="Remove this eye's recording" aria-label="Remove ${escapeHtml(o)}">×</button></div>`;
  };
  $('observer-list').innerHTML = people.map(p => {
    const eyes = [p.left, p.right].filter(i => i >= 0), on = eyes.filter(i => state.observerEnabled[i]).length;
    const name = personName(g, p), n = eyes.reduce((a, i) => a + counts[i], 0);
    const tip = Number.isFinite(p.score) ? `Mean of the eyes' agreement scores: ${p.score.toFixed(3)}.` + (p.low ? ` Far below the other people: under ${low.cutoff.toFixed(3)} (box-plot rule). Not removed: use × if you decide to.` : '') : 'No score.';
    return `<details class="person${p.low ? ' flagged' : ''}" data-person="${p.k}"${state.openPeople?.has(p.k) ? ' open' : ''}><summary class="observer-row person-row"><input type="checkbox" data-person-check="${eyes.join(',')}" ${on === eyes.length ? 'checked' : ''} ${on && on < eyes.length ? 'data-partial' : ''}><i class="eye-mark L" style="--c:${obsColor(eyes[0])}"></i><span title="${escapeHtml(eyes.map(i => g.observers[i]).join(' + '))}">${escapeHtml(name)}${eyes.length < 2 ? ` <em>one eye</em>` : ''}</span><small>${n}</small>${score(p.score, p.low, tip)}<button class="row-remove" type="button" data-remove="${eyes.join(',')}" title="Remove this person (both eyes)" aria-label="Remove ${escapeHtml(name)}">×</button></summary>`
      + (p.left >= 0 ? eyeRow(p.left, 'L') : '') + (p.right >= 0 ? eyeRow(p.right, 'R') : '') + '</details>';
  }).join('') + (removed.size ? `<button id="restore-observers" class="link-button" type="button">bring back the ${removed.size} removed</button>` : '');
  for (const c of $('observer-list').querySelectorAll('[data-partial]')) c.indeterminate = true;
}
$('observer-list').addEventListener('toggle', e => { const d = e.target.closest?.('details[data-person]'); if (!d) return; (state.openPeople ||= new Set())[d.open ? 'add' : 'delete'](Number(d.dataset.person)); }, true);
$('observer-list').addEventListener('click', e => {
  const r = e.target.closest('[data-remove]');
  if (r && r.dataset.remove.includes(',')) { e.preventDefault(); for (const i of r.dataset.remove.split(',')) (state.removedObservers ||= new Set()).add(state.gaze.observers[Number(i)]); rebuildGaze(); return; }
  if (r) e.preventDefault();
  if (r) { (state.removedObservers ||= new Set()).add(state.gaze.observers[Number(r.dataset.remove)]); rebuildGaze(); return; }
  if (e.target.id === 'restore-observers') { state.removedObservers = new Set(); rebuildGaze(); }
});
function clearGaze(quiet = false) {
  if (!quiet) state.library.gaze = [];
  Object.assign(state, { files: [], mapping: null, gazeAll: null, gazeBuilt: null, people: null, gaze: null, observerEnabled: null, conditionEnabled: null, removedObservers: new Set(), loadReport: null });
  $('mapping-section').hidden = true; $('mapping-error').hidden = true;
  renderGroups(); updateGazeSummary(); drawDensity(); updateSubtitle(); updateButtons(); updateExportSummary(); requestRender();
  if (!quiet) notice('Gaze cleared.');
}
$('clear-gaze').addEventListener('click', () => clearGaze());
$('observer-list').addEventListener('change', e => {
  const both = e.target.dataset.personCheck;
  if (both != null) { for (const i of both.split(',')) state.observerEnabled[Number(i)] = e.target.checked ? 1 : 0; renderGroups(); drawDensity(); requestRender(); return; }
  const i = e.target.dataset.obs; if (i == null) return; state.observerEnabled[i] = e.target.checked ? 1 : 0;
  if (state.people) renderGroups();
  drawDensity(); requestRender();
});
$('condition-list').addEventListener('change', e => { const i = e.target.dataset.cond; if (i == null) return; state.conditionEnabled[i] = e.target.checked ? 1 : 0; drawDensity(); requestRender(); });
// ---------- observer agreement (a flag, never a filter) ----------
const AGREE_W = 128, AGREE_H = 64, AGREE_MAX_FRAMES = 400;
/** What the scores depend on: the gaze after filters and the settings that shape the maps. */
function agreementKey() {
  const s = state.settings;
  return JSON.stringify([s.projection, s.sigmaDeg, s.truncate, s.planarFovDeg, s.windowSec, s.kernel, s.peakTauSec, s.causal, s.durationWeight, s.motionComp && state.flow ? state.flowStamp : 0, s.lonOffsetDeg, s.flipX, s.flipY, state.fps, video.duration || 0]);
}
let agreementTimer = 0;
function scheduleAgreement() {
  const g = state.gaze;
  if (state.agreement && state.agreement.gaze === g && state.agreement.key === agreementKey()) return;   // up to date or on its way
  clearTimeout(agreementTimer);
  if (!g) { if (state.agreement) state.agreement.cancel = true; state.agreement = null; updateAgreementStatus(); return; }
  if (state.agreement) state.agreement.cancel = true;
  state.agreement = { gaze: g, key: agreementKey(), done: false, stale: !!state.agreement?.done, progress: 0, scores: state.agreement?.scores };
  updateAgreementStatus();
  agreementTimer = setTimeout(runAgreement, 600);
}
async function runAgreement() {
  const job = state.agreement, g = job?.gaze;
  if (!g || job.cancel) return;
  const s = { ...settingsForRender('none') }, fps = state.fps;
  const t0 = Math.max(0, g.tMin), t1 = video.duration ? Math.min(video.duration, g.tMax) : g.tMax;
  const k0 = frameOf(t0, fps), k1 = Math.max(k0, frameOf(t1, fps));
  const step = Math.max(1, Math.ceil((k1 - k0 + 1) / AGREE_MAX_FRAMES));
  const ks = []; for (let k = k0; k <= k1; k += step) ks.push(k);
  const scorer = agreementScorer(g, s, { ...renderContext() }, AGREE_W, AGREE_H);
  let i = 0;
  while (i < ks.length) {
    const until = performance.now() + 12;
    while (i < ks.length && performance.now() < until) scorer.add(frameCentre(ks[i++], fps));
    job.progress = i / ks.length; updateAgreementStatus();
    await new Promise(r => setTimeout(r, 0));
    if (job.cancel || state.agreement !== job) return;
  }
  Object.assign(job, scorer.result(), { done: true, stale: false, frameCount: ks.length, step });
  updateAgreementStatus(); renderGroups();
}
function updateAgreementStatus() {
  const A = state.agreement, el = $('agreement-status');
  if (!el) return;
  $('obs-sort').hidden = $('obs-sort-sep').hidden = !(A && A.done);
  if (!A) { el.textContent = ''; return; }
  if (!A.done) { el.textContent = `Scoring agreement between observers… ${Math.round(100 * (A.progress || 0))} %`; return; }
  const n = A.low.set.size;
  el.textContent = `Agreement: each observer's map against the map of all the others${state.gaze.conditions.length > 1 ? ' in the same condition' : ''} (correlation, ${A.W} × ${A.H}, ${A.frameCount} frames over the clip). `
    + (n ? `${n} far below the rest (marked ⚠). Nobody is removed automatically.` : Number.isFinite(A.low.cutoff) ? 'None stands out.' : 'Too few observers to mark low ones.');
}
$('obs-sort').addEventListener('click', () => { state.observerSort = state.observerSort === 'agreement' ? 'name' : 'agreement'; $('obs-sort').textContent = state.observerSort === 'agreement' ? 'sort by name' : 'sort by agreement'; renderGroups(); });

$('obs-all').addEventListener('click', () => { if (!state.gaze) return; state.observerEnabled.fill(1); renderGroups(); drawDensity(); requestRender(); });
$('obs-none').addEventListener('click', () => { if (!state.gaze) return; state.observerEnabled.fill(0); renderGroups(); drawDensity(); requestRender(); });

// ---------- motion analysis ----------
function updateFlowUi() {
  const f = state.flow;
  $('motion').disabled = !f; $('show-flow').disabled = !f; $('clear-flow').disabled = !f;
  if (f && state.wantMotion && !state.settings.motionComp) { state.settings.motionComp = true; $('motion').checked = true; }
  if (!f) { $('motion').checked = false; state.settings.motionComp = false; $('show-flow').checked = false; state.display.showFlow = false; }
  if (f !== state.flowSeen) { state.flowSeen = f; state.flowStamp = (state.flowStamp || 0) + 1; }
  const from = state.flowInfo?.method === 'file' ? `From ${state.flowInfo.file}${state.flowInfo.about ? ' (' + state.flowInfo.about + ')' : ''}: ` : '';
  $('flow-status').textContent = f
    ? `${from}${f.size} flow fields at ${f.W} × ${f.H}, ${fmt(f.times[0] - f.step / 2, 2)}–${fmt(f.times[f.size - 1] + f.step / 2, 2)} s, one every ${fmt(f.step, 3)} s, ${bytesText(f.bytes)}.`
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
$('kernel').addEventListener('change', e => { state.settings.kernel = e.target.value; $('tau-row').hidden = e.target.value !== 'peaked'; onSettings(); });
$('peak-tau').addEventListener('change', e => { const v = Math.min(5, Math.max(0.01, Number(e.target.value) || 0.25)); e.target.value = String(v); state.settings.peakTauSec = v; onSettings(); });
$('causal').addEventListener('change', e => { state.settings.causal = e.target.checked; onSettings(); });
$('duration-weight').addEventListener('change', e => { state.settings.durationWeight = e.target.checked; onSettings(); });
$('fps').addEventListener('change', e => { const f = Number(e.target.value); if (f > 0) { setFps(f, 'manual'); if (/^frame/.test(state.mapping?.timeUnit) || /^frame/.test(state.mapping?.durationUnit)) rebuildGaze(); } });
$('motion').addEventListener('change', e => { state.settings.motionComp = e.target.checked; state.wantMotion = e.target.checked; onSettings(); });
$('show-flow').addEventListener('change', e => { state.display.showFlow = e.target.checked; requestRender(); });
$('vel').addEventListener('change', e => { state.settings.velocity = Math.max(0, Number(e.target.value) || 0); applyGazeFilters(); onSettings(); });
$('min-dur').addEventListener('change', e => { state.settings.minDurationMs = Math.max(0, Number(e.target.value) || 0); e.target.value = String(state.settings.minDurationMs); applyGazeFilters(); onSettings(); });
$('flip-x').addEventListener('change', e => { state.settings.flipX = e.target.checked; onSettings(); });
$('flip-y').addEventListener('change', e => { state.settings.flipY = e.target.checked; onSettings(); });
$('mode').addEventListener('change', e => { state.display.mode = e.target.value; requestRender(); });
$('colormap').innerHTML = COLORMAPS.map(c => `<option value="${c}">${c[0].toUpperCase() + c.slice(1)}</option>`).join('');
$('colormap').addEventListener('change', e => { state.display.colormap = e.target.value; $('color-ramp').style.background = cssGradient(e.target.value); requestRender(); });
$('color-ramp').style.background = cssGradient('jet');
$('opacity').addEventListener('input', e => { state.display.opacity = Number(e.target.value) / 100; $('opacity-value').textContent = `${e.target.value}%`; requestRender(); });
$('gamma').addEventListener('input', e => { state.display.gamma = Number(e.target.value); $('gamma-value').textContent = Number(e.target.value).toFixed(2); requestRender(); });
$('map-width').addEventListener('change', e => { state.display.mapWidth = Number(e.target.value); requestRender(); });
$('show-outline').addEventListener('change', e => { state.display.outline = e.target.checked; state.display.outlineTouched = true; if (e.target.checked) state.showGaze = true; syncLayerChips(); requestRender(); });
$('show-points').addEventListener('change', e => { state.display.showPoints = e.target.checked; if (e.target.checked) state.showGaze = true; syncLayerChips(); requestRender(); });
$('speed').addEventListener('change', e => { video.playbackRate = Number(e.target.value); });
function onSettings() { updateExportSummary(); requestRender(); scheduleAgreement(); }

/** Set several settings at once and keep the controls in step (used by presets, the address bar and the tests). */
function applySettings(o) {
  const S = state.settings;
  const pair = (key, rid, nid, min = 0) => { if (key in o) { S[key === 'sigmaDeg' ? 'sigmaDeg' : key] = Math.max(min, Number(o[key])); $(rid).value = $(nid).value = String(S[key]); } };
  pair('sigmaDeg', 'sigma', 'sigma-num', 0.05); pair('windowSec', 'window', 'window-num'); pair('lonOffsetDeg', 'lon-offset', 'lon-offset-num', -360);
  if ('kernel' in o) { S.kernel = o.kernel; $('kernel').value = o.kernel; $('tau-row').hidden = o.kernel !== 'peaked'; }
  if ('peakTauSec' in o && o.peakTauSec > 0) { S.peakTauSec = o.peakTauSec; $('peak-tau').value = String(o.peakTauSec); }
  for (const [key, id] of [['causal', 'causal'], ['durationWeight', 'duration-weight'], ['flipX', 'flip-x'], ['flipY', 'flip-y']]) if (key in o) { S[key] = !!o[key]; $(id).checked = !!o[key]; }
  if ('motionComp' in o) { S.motionComp = !!o.motionComp && !!state.flow; $('motion').checked = S.motionComp; state.wantMotion = !!o.motionComp; }
  if ('velocity' in o) { S.velocity = Math.max(0, Number(o.velocity) || 0); $('vel').value = String(S.velocity); }
  if ('eyePairing' in o && ['auto', 'off'].includes(o.eyePairing) && o.eyePairing !== S.eyePairing) { S.eyePairing = o.eyePairing; $('eye-pairing').value = o.eyePairing; updatePairs(); renderGroups(); }
  if ('minDurationMs' in o) { S.minDurationMs = Math.max(0, Number(o.minDurationMs) || 0); $('min-dur').value = String(S.minDurationMs); }
  if (('velocity' in o || 'minDurationMs' in o) && state.gazeAll) applyGazeFilters();
  if (o.projection === 'erp' || o.projection === 'planar') { S.projection = o.projection; $('projection').value = o.projection; $('fov-row').hidden = o.projection !== 'planar'; state.projectionTouched = true; updateSubtitle(); updateButtons(); }
  if ('planarFovDeg' in o) { S.planarFovDeg = Math.max(1, Number(o.planarFovDeg) || 90); $('fov').value = String(S.planarFovDeg); }
  if ('truncate' in o) S.truncate = Math.max(1, Number(o.truncate) || 3);
  if ('coverageThreshold' in o) { S.coverageThreshold = Math.min(0.99, Math.max(0.001, Number(o.coverageThreshold) || 0.1)); $('cov-thr').value = String(S.coverageThreshold); $('ro-coverage-label').textContent = `COVERAGE ≥ ${S.coverageThreshold} MAX`; }
  onSettings();
}
// ---------- drop rules: "drop rows where <column> <, >, = value [and …]" ----------
function ruleColumns() { const set = new Set(); for (const f of state.files) for (const c of f.table.columns) set.add(c); return [...set]; }
function renderRules() {
  const cols = ruleColumns(), esc = escapeHtml;
  const colOpts = sel => (cols.includes(sel) || !sel ? cols : [sel, ...cols]).map(c => `<option${c === sel ? ' selected' : ''}>${esc(c)}</option>`).join('');
  $('rule-list').innerHTML = state.rowRules.map((r, i) => `<div class="rule" data-rule="${i}">${r.clauses.map((c, j) => `<span class="rule-part" data-part="${j}"><span class="rule-and">${j ? 'and' : 'drop if'}</span><select data-k="column" aria-label="Column">${colOpts(c.column)}</select><select data-k="op" aria-label="Comparison">${RULE_OPS.map(o => `<option${o === c.op ? ' selected' : ''}>${o}</option>`).join('')}</select><input data-k="value" type="text" value="${esc(c.value)}" aria-label="Value" placeholder="value"></span>`).join('')}<span class="rule-buttons"><button class="link-button" type="button" data-and title="Add another part: the row is dropped only when all parts hold">＋ and</button><button class="link-button" type="button" data-drop-rule title="Remove this rule">remove rule</button></span></div>`).join('');
  $('rule-add').disabled = !cols.length;
}
let rulesTimer = 0;
function rulesChanged() { clearTimeout(rulesTimer); rulesTimer = setTimeout(() => { if (state.files.length) rebuildGaze(); }, 250); }
$('rule-add').addEventListener('click', () => {
  const cols = ruleColumns(), pick = cols.find(c => /conf|valid|quality/i.test(c)) || cols[0];
  state.rowRules.push({ clauses: [{ column: pick, op: '<', value: '' }] }); renderRules();
  $('rule-list').lastElementChild?.querySelector('input')?.focus();
});
$('rule-list').addEventListener('click', e => {
  const r = e.target.closest('[data-rule]'); if (!r) return;
  const rule = state.rowRules[Number(r.dataset.rule)];
  if (e.target.closest('[data-drop-rule]')) { state.rowRules.splice(Number(r.dataset.rule), 1); renderRules(); rulesChanged(); }
  else if (e.target.closest('[data-and]')) { const last = rule.clauses[rule.clauses.length - 1]; rule.clauses.push({ column: ruleColumns().find(c => c !== last.column) || last.column, op: '=', value: '' }); renderRules(); }
});
$('rule-list').addEventListener('change', e => {
  const r = e.target.closest('[data-rule]'), part = e.target.closest('[data-part]'); if (!r || !part) return;
  state.rowRules[Number(r.dataset.rule)].clauses[Number(part.dataset.part)][e.target.dataset.k] = e.target.value.trim();
  rulesChanged();
});
function setRowRules(rules) { state.rowRules = (rules || []).map(r => ({ clauses: r.clauses.map(c => ({ ...c })) })); renderRules(); if (state.files.length) rebuildGaze(); }

// ---------- presets and saved templates ----------
const TEMPLATE_KEY = 'overgaze.templates';
function loadTemplates() { try { return JSON.parse(localStorage.getItem(TEMPLATE_KEY) || '[]'); } catch { return []; } }
function storeTemplates(list) { try { localStorage.setItem(TEMPLATE_KEY, JSON.stringify(list)); return true; } catch { notice('Templates cannot be saved in this browser (storage blocked); use Export instead.', true); return false; } }
function renderPresetList(selected = '') {
  const t = loadTemplates(), esc = escapeHtml;
  $('preset').innerHTML = `<option value="">— choose a preset or template —</option><optgroup label="Presets">${PRESETS.map(p => `<option value="p:${p.id}">${esc(p.name)}</option>`).join('')}</optgroup>`
    + (t.length ? `<optgroup label="My templates">${t.map((x, i) => `<option value="t:${i}">${esc(x.name)}</option>`).join('')}</optgroup>` : '');
  state.presetBase = selected || null; state.presetCustom = null;
  $('preset').value = selected;
  $('template-delete').disabled = !selected.startsWith('t:');
  refreshPresetLabel();
}
function presetByValue(v) {
  if (!v) return null;
  if (v.startsWith('p:')) return PRESETS.find(p => p.id === v.slice(2));
  if (v.startsWith('t:')) return loadTemplates()[Number(v.slice(2))];
  return null;
}
/** The preset or template the settings came from (also while they have been changed since). */
function chosenPreset() { return presetByValue(state.presetBase); }
const SETTING_LABELS = { projection: 'projection', sigmaDeg: 'σ', truncate: 'kernel cut-off', planarFovDeg: 'field of view', windowSec: 'pooling window',
  kernel: 'pooling weight', peakTauSec: 'τ', causal: 'past only', durationWeight: 'weight by duration', motionComp: 'motion compensation', lonOffsetDeg: 'rotation',
  flipX: 'left–right flip', flipY: 'up–down flip', velocity: 'speed filter', minDurationMs: 'minimum duration', coverageThreshold: 'coverage threshold' };
/**
 * Once any setting differs from the chosen preset, the menu shows "(custom <preset name>)" as a separate entry,
 * so that choosing the preset itself again is a change of the menu and restores its values.
 */
function refreshPresetLabel() {
  const sel = $('preset'), p = presetByValue(state.presetBase);
  // motion compensation that waits for "Analyse motion" still counts as on
  const diff = p ? presetDifferences(p.settings, { ...state.settings, motionComp: state.settings.motionComp || !!state.wantMotion }) : [];
  const key = diff.length ? `${state.presetBase}|${diff.join()}` : null;
  if (key === state.presetCustom) return;
  state.presetCustom = key;
  sel.querySelector('option[data-custom]')?.remove();
  if (!diff.length) { sel.value = state.presetBase || ''; sel.title = ''; $('preset-changes').hidden = true; return; }
  const o = document.createElement('option');
  o.value = `c:${state.presetBase}`; o.dataset.custom = '1'; o.textContent = `(custom ${p.name})`;
  sel.insertBefore(o, sel.children[1] || null);   // right after the placeholder, outside the groups
  sel.value = o.value;
  sel.title = `Changed from “${p.name}”: ${diff.map(k => SETTING_LABELS[k] || k).join(', ')}. Choose the preset again to restore its values.`;
  $('preset-changes').hidden = false; $('preset-changes').textContent = `Changed: ${diff.map(k => SETTING_LABELS[k] || k).join(', ')}. Choose the preset again to restore its values.`;
}
$('preset').addEventListener('change', () => {
  const v = $('preset').value;
  if (v.startsWith('c:')) return;          // the "(custom …)" entry itself: nothing to apply
  state.presetBase = v || null;
  const p = chosenPreset();
  $('template-delete').disabled = !v.startsWith('t:');
  $('preset-about').textContent = p ? (p.about || 'Saved template.') + (p.settings.motionComp && !state.flow ? ' Motion compensation starts once “Analyse motion” has run.' : '') : '';
  if (p) applySettings(p.settings);
  refreshPresetLabel();
});
$('template-save').addEventListener('click', () => {
  const name = prompt('Name of the template:', `σ ${state.settings.sigmaDeg}° · ±${state.settings.windowSec} s`);
  if (!name) return;
  const list = loadTemplates().filter(x => x.name !== name);
  list.push({ name, settings: pickSettings({ ...state.settings, motionComp: state.settings.motionComp || !!state.wantMotion }) });
  if (storeTemplates(list)) { renderPresetList(`t:${list.length - 1}`); notice(`Template “${name}” saved in this browser.`); }
});
$('template-delete').addEventListener('click', () => {
  const v = state.presetBase || '';
  if (!v.startsWith('t:')) return;
  const list = loadTemplates(), [gone] = list.splice(Number(v.slice(2)), 1);
  if (storeTemplates(list)) { renderPresetList(); $('preset-about').textContent = ''; notice(`Template “${gone?.name}” deleted.`); }
});
$('template-export').addEventListener('click', () => {
  const p = chosenPreset(), name = (p && state.presetCustom ? `custom ${p.name}` : p?.name) || `OverGaze settings ${new Date().toISOString().slice(0, 10)}`;
  const body = { overgaze_template: 1, name, settings: pickSettings({ ...state.settings, motionComp: state.settings.motionComp || !!state.wantMotion }), row_rules: state.rowRules };
  download(new Blob([JSON.stringify(body, null, 2)], { type: 'application/json' }), `${name.replace(/[^\w.-]+/g, '_').slice(0, 60)}.overgaze.json`);
});
$('template-import').addEventListener('click', () => $('template-input').click());
$('template-input').addEventListener('change', async e => {
  const f = e.target.files[0]; e.target.value = '';
  if (!f) return;
  try {
    const raw = JSON.parse(await f.text()), { name, settings, rowRules } = settingsFromJson(raw);
    applySettings(settings);
    if (raw.display?.gaze_points) applyPoints(raw.display.gaze_points);
    if (rowRules) setRowRules(rowRules);
    const list = loadTemplates(), label = name || f.name.replace(/\.(overgaze\.)?json$/i, '');
    list.push({ name: label, settings });
    storeTemplates(list); renderPresetList(`t:${list.length - 1}`);
    notice(`Settings imported from ${f.name} and kept as template “${label}”.`);
  } catch (err) { notice(`Could not import ${f.name}: ${err.message}`, true); }
});
renderPresetList();

$('play').addEventListener('click', togglePlay);
$('prev-frame').addEventListener('click', () => stepFrames(-1));
$('next-frame').addEventListener('click', () => stepFrames(1));
$('scrubber').addEventListener('input', e => { video.currentTime = Number(e.target.value); });
$('analyse').addEventListener('click', analyseMotion);
$('motion-file').addEventListener('click', () => $('motion-file-input').click());
$('motion-file-input').addEventListener('change', async e => {
  const file = e.target.files[0];
  e.target.value = '';
  if (!file || state.analysing) return;
  setStatus(`Reading ${file.name}…`);
  try {
    const m = await openMotionFile(file);
    state.flow = m.store;
    state.flowInfo = { method: 'file', file: file.name, about: m.method, fields: m.fields, file_fps: m.fps };
    if (state.video && m.fps && Math.abs(m.fps - state.fps) / state.fps > 0.001) notice(`${file.name} was made at ${fmt(m.fps, 3)} frames per second; this video is at ${fmt(state.fps, 3)}. Its times are in seconds, so it still lines up if it was made from this video.`, true);
    $('motion').checked = true; state.settings.motionComp = true;
    setStatus(`Motion loaded from ${file.name}`);
  } catch (err) {
    showError(`Motion file: ${err.message}`); setStatus('Motion file not loaded');
  }
  updateFlowUi(); requestRender();
});
$('clear-flow').addEventListener('click', () => { state.flow = null; state.flowInfo = null; updateFlowUi(); requestRender(); });

document.addEventListener('keydown', e => {
  if (e.target.closest('input, select, textarea') && e.target.type !== 'range' && e.target.type !== 'checkbox') return;
  if (e.key === ' ') { e.preventDefault(); togglePlay(); }
  else if (e.key === 'ArrowRight') { e.preventDefault(); e.shiftKey ? seekSeconds(1) : stepFrames(1); }
  else if (e.key === 'ArrowLeft') { e.preventDefault(); e.shiftKey ? seekSeconds(-1) : stepFrames(-1); }
  else if (e.key === 'm' || e.key === 'M') { if (!$('motion').disabled) { $('motion').checked = !$('motion').checked; $('motion').dispatchEvent(new Event('change')); } }
  else if (e.key === 'l' || e.key === 'L') $('loop').click();
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
    display: { gaze_points: pointsJson() },
    saliency: {
      projection: s.projection, sigma_deg: s.sigmaDeg, kernel_cutoff_sigma: s.truncate, planar_fov_deg: s.projection === 'planar' ? s.planarFovDeg : undefined,
      pooling_window_s: s.windowSec, pooling_weight: s.kernel, peaked_tau_s: s.kernel === 'peaked' ? (s.peakTauSec ?? 0.25) : undefined, causal: s.causal, weight_by_duration: s.durationWeight,
      motion_compensation: !!(s.motionComp && state.flow), lon_offset_deg: s.lonOffsetDeg, flip_left_right: s.flipX, flip_up_down: s.flipY,
      velocity_filter_deg_per_s: s.velocity || 0, coverage_threshold: s.coverageThreshold,
      frame_rule: 'frame k covers [k/fps, (k+1)/fps); a gaze sample or fixation counts if its time distance to that interval is <= pooling_window_s (0 = active during the frame).',
      gaussian: s.projection === 'erp' ? 'exp(-theta^2 / (2 sigma^2)), theta = great-circle angle to the gaze point, summed over points' : '2D Gaussian in the image plane, sigma_px = sigma_deg * width / planar_fov_deg'
    },
    gaze_processing: g ? gazeProcessingJson() : { min_fixation_duration_ms: s.minDurationMs || 0, speed_filter_deg_per_s: s.velocity || 0, drop_row_rules: state.rowRules.map(r => ({ all_of: r.clauses })) },
    motion: s.motionComp && state.flow ? { flow_width: state.flow.W, flow_height: state.flow.H, fields: state.flow.size, step_s: state.flow.step, ...state.flowInfo,
      method_note: (state.flowInfo?.method === 'file' ? `loaded from ${state.flowInfo.file}: ${state.flowInfo.about || 'no method given'}` : 'pyramidal Lucas-Kanade on greyscale frames (each pixel then keeps its own movement, none, or a neighbour\'s, whichever matches the next frame best), wrap-around at the left/right edge for 360° video')
        + '; each pooled point is moved along the flow from its own time to the frame time' } : null,
    ...extra
  };
}
/** The gaze filters with what each removed, the removed observers and the agreement scores, for settings.json. */
function gazeProcessingJson() {
  const s = state.settings, g = state.gaze, all = state.gazeAll, A = state.agreement?.done && state.agreement.gaze === g ? state.agreement : null;
  return {
    order: 'rows without a valid time or position are skipped, then the drop rules, then the minimum fixation duration, then the speed filter; removed observers are left out first',
    rows_skipped_invalid: all.skipped || 0,
    drop_row_rules: (all.rules || []).map((r, i) => ({ all_of: r.clauses, text: ruleText(r), rows_removed: all.ruleRemoved[i] || 0 })),
    min_fixation_duration_ms: s.minDurationMs || 0, rows_removed_by_min_duration: g.durationRemoved || 0,
    speed_filter_deg_per_s: s.velocity || 0, rows_removed_by_speed_filter: g.filtered || 0,
    removed_observers: [...(state.removedObservers || [])],
    observers_switched_off: state.observerEnabled ? g.observers.filter((o, i) => !state.observerEnabled[i]) : [],
    eye_pairing: { setting: s.eyePairing, ...(state.people ? {
      method: state.people.method === 'names' ? 'partners from shared names' : 'matching: share of one eye\'s fixations with a fixation of the other starting within 60 ms and within 3°, best one-to-one assignment, pairs below 0.1 dropped',
      sides_from: state.people.sidesFrom, pairs: state.people.pairs, single_left: state.people.singleLeft, single_right: state.people.singleRight,
      people: state.people.people.map(p => ({ left: p.left >= 0 ? state.gazeBuilt.observers[p.left] : null, right: p.right >= 0 ? state.gazeBuilt.observers[p.right] : null, match: p.score == null ? null : +p.score.toFixed(3) }))
    } : { pairs: 0 }) },
    observer_agreement: A ? {
      method: 'per frame, the correlation between one observer\'s map and the summed map of all the other observers of the same condition (leave-one-out), each pixel weighted by its area on the sphere for 360° video; averaged over the frames where both have gaze. The current saliency settings shape the maps.',
      map_width: A.W, map_height: A.H, frames_sampled: A.frameCount, frame_step: A.step,
      low_rule: 'below the lower quartile minus 1.5 × the spread between the quartiles (box-plot rule), with at least 4 scored observers',
      low_cutoff: Number.isFinite(A.low.cutoff) ? A.low.cutoff : null,
      scores: Object.fromEntries(g.observers.map((o, i) => [o, Number.isFinite(A.scores[i]) ? +A.scores[i].toFixed(4) : null])),
      frames_scored: Object.fromEntries(g.observers.map((o, i) => [o, A.frames[i]])),
      low: [...A.low.set].map(i => g.observers[i]),
      ...(state.people ? { person_scores: Object.fromEntries(peopleNow().map(p => [personName(g, p), +(personScore(A, p) || 0).toFixed(4)])), person_score_rule: 'a person\'s score is the mean of their eyes\' scores' } : {}),
      note: 'a flag only: nobody is removed automatically'
    } : null
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
  else L.push(`  ${state.ref.name}: ${state.ref.kind}, ${state.ref.N} frame(s) of ${state.ref.W} × ${state.ref.H}, offset ${state.ref.offset}, rate ${state.ref.scale}, showing ${(state.showRef && [state.refShow.heat && 'heatmap', state.refShow.outline && 'outline'].filter(Boolean).join(' + ')) || 'numbers only'}`,
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
/**
 * Video with all overlays at c's size. `perScreenPixel`: c's pixels per screen pixel for the outlines; by default the
 * outlines keep the thickness they have on screen relative to the picture.
 */
function drawComposite(c, perScreenPixel) {
  const g = c.getContext('2d');
  g.drawImage(converting() ? $('erp-canvas') : video, 0, 0, c.width, c.height);
  if (state.ref && state.showRef && state.refShow.heat) g.drawImage(refCanvas, 0, 0, c.width, c.height);
  { g.imageSmoothingEnabled = true; g.imageSmoothingQuality = 'high'; g.drawImage(heatCanvas, 0, 0, c.width, c.height); }
  const stageWidth = $('stage').getBoundingClientRect().width || c.width;
  drawOutlines(g, c.width, c.height, perScreenPixel ?? c.width / stageWidth);
  g.drawImage(marksCanvas, 0, 0, c.width, c.height);
  if (state.view && c === state.viewSource && state.display.showPoints && state.showGaze && state.marksResult && state.gaze) drawGazePoints(g, state.marksT, c.width, c.height, perScreenPixel);
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
  $('view-360').disabled = !readable;
  if (state.view && $('view-360').disabled) set360(false);
}

// ---------- file inputs, drag and drop, sample, URL parameters ----------
$('choose-video').addEventListener('click', () => $('video-input').click());
$('empty-video').addEventListener('click', () => $('video-input').click());
$('video-input').addEventListener('change', e => { if (e.target.files[0]) loadVideoFile(e.target.files[0]); e.target.value = ''; });
$('choose-video-folder').addEventListener('click', () => $('video-folder-input').click());
$('video-folder-input').addEventListener('change', e => {
  const vids = [...e.target.files].filter(f => VIDEO_EXT.test(f.name) && !f.name.startsWith('.')).sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
  e.target.value = '';
  if (!vids.length) { notice('No videos in that folder (.mp4, .webm, .mkv, .mov …).', true); return; }
  state.library.videos = vids;
  $('video-list').innerHTML = vids.map((f, i) => `<option value="${i}">${escapeHtml(f.name)}</option>`).join('');
  $('video-list-row').hidden = false;
  loadVideoFile(vids[0]);
  notice(`${vids.length} videos in the folder. Pick one in the list (or with the arrows); its fixation files and reference are taken from what you load.`);
});
$('video-list').addEventListener('change', e => { const f = state.library.videos[Number(e.target.value)]; if (f) loadVideoFile(f); });
for (const [id, d] of [['video-prev', -1], ['video-next', 1]]) $(id).addEventListener('click', () => {
  const n = state.library.videos.length; if (!n) return;
  const i = (Number($('video-list').value) + d + n) % n; $('video-list').value = String(i); $('video-list').dispatchEvent(new Event('change'));
});
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
  if (kind === 'eyes') return loadSampleEyes();
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
$('load-sample-eyes').addEventListener('click', () => loadSample('eyes'));
/** The 360° sample with a left- and a right-eye file in the Salient360 head + eye format (folders L and R). */
async function loadSampleEyes() {
  try {
    const blob = await (await fetch('./sample/sample_erp.mp4')).blob();
    loadVideoFile(new File([blob], 'sample_erp.mp4', { type: 'video/mp4' }));
    const files = [];
    for (const side of ['L', 'R']) {
      const f = new File([await (await fetch(`./sample/eyes/${side}/sample_fixations.csv`)).text()], 'sample_fixations.csv');
      f.relativePath = `eyes/${side}/sample_fixations.csv`; files.push(f);
    }
    await loadGazeFiles(files);
    notice('Sample loaded with both eyes: the 8 simulated observers as a left- and a right-eye file (Salient360 format, folders L and R), plus one recording in each file with no partner.');
  } catch (e) { notice(`Could not load the sample gaze (${e.message}).`, true); }
}
$('empty-sample').addEventListener('click', () => loadSample('erp'));

(async function fromQuery() {
  const q = new URLSearchParams(location.search);
  const settingsFromQuery = () => {
    const o = {}, num = (k, to) => { if (q.get(k) != null && Number.isFinite(Number(q.get(k)))) o[to] = Number(q.get(k)); };
    num('sigma', 'sigmaDeg'); num('window', 'windowSec'); num('velocity', 'velocity'); num('rotate', 'lonOffsetDeg');
    if (['box', 'triangle', 'gaussian', 'peaked'].includes(q.get('kernel'))) o.kernel = q.get('kernel');
    num('tau', 'peakTauSec');
    if (q.has('past')) o.causal = true;
    if (Object.keys(o).length) applySettings(o);
  };
  if (q.get('sample') != null) { await loadSample(['2d', 'eyes'].includes(q.get('sample')) ? q.get('sample') : 'erp'); settingsFromQuery(); return; }
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
{
  const opt = ([k, v]) => `<option value="${k}">${v.label}</option>`, all = Object.entries(VIEW_MODES);
  $('view360-mode').innerHTML = `<optgroup label="Look around">${all.filter(([, v]) => !v.group).map(opt).join('')}</optgroup><optgroup label="Unfolded (whole sphere)">${all.filter(([, v]) => v.group).map(opt).join('')}</optgroup>`;
}
$('view360-mode').addEventListener('change', e => { if (state.view) { state.view.setMode(e.target.value); $('view360-follow').checked = false; requestRender(); } });

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

// ⟳: start again at the end of the video
$('loop').addEventListener('click', () => {
  video.loop = !video.loop;
  $('loop').classList.toggle('active', video.loop); $('loop').setAttribute('aria-pressed', String(video.loop));
});
