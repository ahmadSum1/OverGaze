import test from 'node:test';
import assert from 'node:assert/strict';
import { inflateSync } from 'node:zlib';
import { buildGaze, detectMapping, parseTable, toLonLat, vectorToLonLat, velocityFilter, packRecords } from '../src/gaze.js';
import { collectPoints, coverage, makeGrid, renderMap, splatSphere, makePointBuffer, DEFAULTS, frameCentre } from '../src/saliency.js';
import { FlowStore, opticalFlow } from '../src/flow.js';
import { crc32, encodeGrayPng, npyHeader, ZipWriter } from '../src/export.js';

const DEG = Math.PI / 180;
const close = (a, b, eps, msg) => assert.ok(Math.abs(a - b) <= eps, `${msg ?? ''} ${a} vs ${b}`);

test('parses CSV with header and detects fixation columns', () => {
  const t = parseTable('user,start_time,duration,longitude,latitude\n1,0.10,0.25,0.5,0.5\n1,0.40,0.30,0.75,0.25\n2,0.05,0.20,0.1,0.9\n', 'a.csv');
  assert.deepEqual(t.columns, ['user', 'start_time', 'duration', 'longitude', 'latitude']);
  const m = detectMapping(t, { duration: 30 });
  assert.equal(m.time, 'start_time'); assert.equal(m.duration, 'duration');
  assert.equal(m.x, 'longitude'); assert.equal(m.y, 'latitude');
  assert.equal(m.coord, 'normalized'); assert.equal(m.observer, 'user'); assert.equal(m.timeUnit, 's');
  const g = buildGaze([{ table: t, name: 'a.csv' }], m);
  assert.equal(g.n, 3); assert.equal(g.observers.length, 2);
  assert.equal(g.t0[0], 0.05);
  close(g.t1[1], 0.35, 1e-9);
  close(g.lon[1], 0, 1e-9); close(g.lat[1], 0, 1e-9);
});

test('parses headerless TSV and ARFF', () => {
  const t = parseTable('0\t1920\t960\n33\t1900\t950\n66\t1910\t940\n', 'g.tsv');
  assert.equal(t.columns.length, 3);
  const m = detectMapping(t, { duration: 1 });
  assert.equal(m.time, 'col0'); assert.equal(m.coord, 'pixels'); assert.equal(m.timeUnit, 'ms');
  const a = parseTable('@RELATION gaze_360\n@ATTRIBUTE time NUMERIC\n@ATTRIBUTE x NUMERIC\n@ATTRIBUTE y NUMERIC\n@DATA\n0,10,20\n1000,11,21\n', 'x.arff');
  assert.deepEqual(a.columns, ['time', 'x', 'y']); assert.equal(a.rows.length, 2);
});

test('coordinate conversions', () => {
  const deg = { xLeft: -180, xRight: 180, yTop: 90, yBottom: -90 };
  const [lon, lat] = toLonLat(90, 45, deg);
  close(lon, 90 * DEG, 1e-9); close(lat, 45 * DEG, 1e-9);
  const [lon2] = toLonLat(270, 0, { xLeft: 0, xRight: 360, yTop: 90, yBottom: -90 });
  close(lon2, 90 * DEG, 1e-9);
  const [vl, va] = vectorToLonLat(1, 0, 0);
  close(vl, 90 * DEG, 1e-9); close(va, 0, 1e-9);
});

test('sphere Gaussian is centred, wraps across the seam and is wider in pixels near the poles', () => {
  const W = 256, H = 128, grid = makeGrid(W, H);
  const pts = makePointBuffer();
  pts.n = 1; pts.lon[0] = Math.PI - 0.001; pts.lat[0] = 0; pts.w[0] = 1;
  const map = new Float32Array(W * H);
  splatSphere(map, grid, pts, 4 * DEG);
  const row = 64 * W;
  assert.ok(map[row + W - 1] > 0.9 && map[row] > 0.9, 'both sides of the seam');
  const equatorWidth = map.slice(row, row + W).filter(v => v > 0.5).length;
  map.fill(0); pts.lat[0] = 75 * DEG; pts.lon[0] = 0;
  splatSphere(map, grid, pts, 4 * DEG);
  const y = Math.round((0.5 - 75 / 180) * H - 0.5);
  const poleWidth = map.slice(y * W, y * W + W).filter(v => v > 0.5).length;
  assert.ok(poleWidth > 2.5 * equatorWidth, `${poleWidth} vs ${equatorWidth}`);
});

test('temporal pooling window, causal mode and coverage', () => {
  const recs = [0.05, 0.5, 1.0, 1.5].map((t, i) => [t, t, i * 0.5 - 1, 0, 0, 0, 1]);
  const g = packRecords(recs, ['o'], ['c']);
  const s = { ...DEFAULTS };
  assert.equal(collectPoints(g, frameCentre(0, 10), s, { fps: 10 }).n, 1);
  assert.equal(collectPoints(g, 1.0, { ...s, windowSec: 0.5 }, { fps: 10 }).n, 3);
  assert.equal(collectPoints(g, 1.0, { ...s, windowSec: 0.5, causal: true }, { fps: 10 }).n, 2);
  const tri = collectPoints(g, 1.0, { ...s, windowSec: 0.5, kernel: 'triangle' }, { fps: 10 });
  assert.ok(tri.w[0] < 1 && tri.w[1] === 1);
  const r = renderMap(g, 1.0, { ...s, windowSec: 0.5, sigmaDeg: 3 }, 128, 64, { fps: 10 });
  const c = coverage(r.map, 128, 64);
  assert.ok(c.pixels > 0 && c.pixels < 0.2 && c.sphere > c.pixels * 0.9);
  close(Math.max(...r.map), 1, 1e-6);
});

test('velocity filter drops fast samples', () => {
  const recs = [[0, 0, 0, 0, 0, 0, 1], [0.1, 0.1, 2 * DEG, 0, 0, 0, 1], [0.2, 0.2, 60 * DEG, 0, 0, 0, 1]];
  const g = velocityFilter(packRecords(recs, ['o'], ['c']), 75);
  assert.equal(g.n, 2); assert.equal(g.filtered, 1);
});

test('optical flow recovers a horizontal shift with wrap-around', () => {
  const W = 128, H = 64, a = new Float32Array(W * H), b = new Float32Array(W * H);
  const f = (x, y) => 0.5 + 0.25 * Math.sin(x * 2 * Math.PI / W * 5) * Math.cos(y * 0.3) + 0.2 * Math.sin((x * 2 * Math.PI / W) * 3 + y * 0.17);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) { a[y * W + x] = f(x, y); b[y * W + x] = f(x - 3, y); }
  const { u, v } = opticalFlow(a, b, W, H, { wrap: true });
  let su = 0, sv = 0;
  for (let y = 10; y < H - 10; y++) for (let x = 0; x < W; x++) { su += u[y * W + x]; sv += v[y * W + x]; }
  const n = (H - 20) * W;
  close(su / n, 3, 0.3, 'u'); close(sv / n, 0, 0.3, 'v');
});

test('flow store moves points with the measured velocity', () => {
  const s = new FlowStore(64, 32);
  const u = new Float32Array(64 * 32).fill(2), v = new Float32Array(64 * 32).fill(0);
  for (let k = 0; k < 20; k++) s.add(k * 0.1, (k + 1) * 0.1, u, v); // 2 px per 0.1 s = 20 px/s
  const [lon] = s.propagate(0, 0, 0.5, 1.0);
  close(lon, 10 / 64 * 2 * Math.PI, 1e-3);
  const [back] = s.propagate(lon, 0, 1.0, 0.5);
  close(back, 0, 1e-3);
  close(s.coverage(0, 2), 1, 1e-9);
});

test('motion-compensated pooling collapses a moving track onto the current position', () => {
  const W = 64, H = 32, flow = new FlowStore(W, H);
  const u = new Float32Array(W * H).fill(1), v = new Float32Array(W * H);
  for (let k = 0; k < 40; k++) flow.add(k * 0.1, (k + 1) * 0.1, u, v);  // 10 px/s
  const pxToLon = x => ((x + 0.5) / W - 0.5) * 2 * Math.PI;
  const recs = [];
  for (let k = 0; k <= 20; k++) recs.push([k * 0.1, k * 0.1, pxToLon(10 + k), 0, 0, 0, 1]); // object at 10 px/s
  const g = packRecords(recs, ['o'], ['c']);
  const s = { ...DEFAULTS, windowSec: 1, motionComp: true };
  const p = collectPoints(g, 1.0, s, { fps: 10, flow });
  const target = pxToLon(20);
  for (let i = 0; i < p.n; i++) close(p.lon[i], target, 0.02);
  const plain = collectPoints(g, 1.0, { ...s, motionComp: false }, { fps: 10, flow });
  const spread = Math.max(...plain.lon.slice(0, plain.n)) - Math.min(...plain.lon.slice(0, plain.n));
  assert.ok(spread > 1.5);
});

test('zip, npy and png writers produce valid files', async () => {
  assert.equal(crc32(new TextEncoder().encode('123456789')), 0xCBF43926);
  const parts = [];
  const zip = new ZipWriter(async b => parts.push(b));
  await zip.add('a.txt', 'hello'); await zip.finish();
  const bytes = Buffer.concat(parts);
  assert.equal(bytes.readUInt32LE(0), 0x04034b50);
  assert.equal(bytes.readUInt32LE(bytes.length - 22), 0x06054b50);
  const h = npyHeader('<f4', [3, 64, 128]);
  assert.equal(h.length % 64, 0);
  assert.match(new TextDecoder().decode(h.slice(10)), /'shape': \(3, 64, 128\)/);
  const vals = new Float32Array(16 * 8).map((_, i) => (i % 16) / 15);
  const png = Buffer.from(await encodeGrayPng(vals, 16, 8));
  assert.equal(png.readUInt32BE(16), 16); assert.equal(png[24], 8); assert.equal(png[25], 0);
  const idatLen = png.readUInt32BE(33);
  const raw = inflateSync(png.subarray(41, 41 + idatLen));
  assert.equal(raw.length, 8 * 17);
  assert.equal(raw[1], 0); assert.equal(raw[16], 255); // first row, Up filter against zeros
});

test('streamed ZIP entries carry a data descriptor and ZIP64 records', async () => {
  const parts = [];
  const zip = new ZipWriter(async b => parts.push(b));
  const s = await zip.stream('maps.npy');
  await s.write(npyHeader('|u1', [2, 2, 2])); await s.write(new Uint8Array(8).fill(7)); await s.end();
  await zip.add('settings.json', '{}'); await zip.finish();
  const bytes = Buffer.concat(parts);
  assert.ok(bytes.indexOf(Buffer.from([0x50, 0x4b, 0x07, 0x08])) > 0, 'data descriptor');
  assert.ok(bytes.indexOf(Buffer.from([0x50, 0x4b, 0x06, 0x06])) > 0, 'ZIP64 end record');
  assert.equal(bytes.readUInt32LE(bytes.length - 22), 0x06054b50);
});

// ---------- real dataset formats ----------
import { pathMatchesVideo, FILE_TIME } from '../src/gaze.js';
import { fpsFromDeltas } from '../src/timing.js';

const VRET = 'frame,1,forward,0.4979051,0.51714,eye,0.25,0.25\nframe,2,forward,0.4974872,0.5171432,eye,0.26,0.26\nframe,3,forward,0.49,0.5,eye,0.27,0.27\n';

test('VR-EyeTracking rows are recognised, y counts from the bottom and frame numbers give frame centres', () => {
  const t = parseTable(VRET, '001._2017-10-03-18-55_ori_0.txt');
  assert.equal(t.preset, 'vret'); assert.equal(t.rows.length, 3);
  const m = detectMapping(t, {});
  assert.equal(m.timeUnit, 'frame1'); assert.equal(m.x, 'gaze_x'); assert.equal(m.y, 'gaze_y');
  const g = buildGaze([{ table: t, name: 'a.txt', folder: 'p003' }], m, { fps: 25 });
  close(g.t0[0], 0.5 / 25, 1e-9, 'frame 1 is centred at half a frame'); close(g.t0[2], 2.5 / 25, 1e-9);
  close(g.lon[0], (0.25 - 0.5) * 2 * Math.PI, 1e-9, 'longitude from the left edge');
  close(g.lat[0], -Math.PI / 4, 1e-9, 'y = 0.25 from the bottom is below the equator');
  assert.deepEqual(g.observers, ['p003'], 'observer is the folder name');
});

test('YT360 per-user fixation files map eye_rect_x / eye_rect_y (y from the top) and use the condition folder', () => {
  const csv = 'eye_rect_x,eye_rect_y,head_rect_x,head_rect_y,eye_x,eye_y,eye_z,head_x,head_y,head_z,start_time,end_time,duration\n0.25,0.25,0.5,0.5,0,0,1,0,0,1,0.5,0.8,0.3\n';
  const t = parseTable(csv, '0b4b.txt'), m = detectMapping(t, { duration: 30 });
  assert.equal(m.preset, 'yt360'); assert.equal(m.conditionFrom, 'folder2'); assert.equal(m.time, 'start_time'); assert.equal(m.duration, 'duration');
  const g = buildGaze([{ table: t, name: '0b4b.txt', folder: 'clip-28', folder2: 'videos-audios' }, { table: t, name: '0b4b.txt', folder: 'clip-28', folder2: 'videos_mono-audios' }], m, { fps: 30 });
  assert.equal(g.n, 2); assert.equal(g.conditions.length, 2); assert.equal(g.observers.length, 2, 'same observer id in two conditions stays two people');
  close(g.lat[0], Math.PI / 4, 1e-9, 'y = 0.25 from the top is above the equator');
  close(g.t1[0] - g.t0[0], 0.3, 1e-9);
});

test('a gaze sample pools with Gaussian weights over +-1 s exactly like the dataset maps (whole frames, sigma = fps / 2 frames)', () => {
  const fps = 25, rec = [];
  for (let f = -30; f <= 30; f++) rec.push([(f + 0.5) / fps, (f + 0.5) / fps, 0, 0, 0, 0, 1]);
  const g = packRecords(rec, ['a'], ['c']);
  const s = { ...DEFAULTS, windowSec: 1, kernel: 'gaussian' };
  const pts = collectPoints(g, 0.5 / fps, s, { fps });
  assert.equal(pts.n, 51, 'frames -25 … +25');
  const byFrame = new Map();
  for (let i = 0; i < pts.n; i++) byFrame.set(Math.round(pts.w[i] * 1e6), 1);
  const want = d => Math.exp(-d * d / (2 * 12.5 * 12.5));
  const ws = Array.from({ length: pts.n }, (_, i) => pts.w[i]).sort((a, b) => a - b);
  close(ws[0], want(25), 1e-9, 'weight at ±1 s'); close(ws[ws.length - 1], 1, 1e-12);
});

test('a fixation counts once for every frame it covers, a short window gives its frames inside the window', () => {
  const fps = 25;
  const g = packRecords([[0.0, 0.2, 0, 0, 0, 0, 1]], ['a'], ['c']);   // frames 0…4 (centres 0.02 … 0.18)
  const at = (k, win) => collectPoints(g, (k + 0.5) / fps, { ...DEFAULTS, windowSec: win }, { fps }).n;
  assert.equal(at(2, 0), 1); assert.equal(at(5, 0), 0);
  assert.equal(at(2, 0.08), 5, '±2 frames around frame 2 all lie inside the fixation');
  assert.equal(at(5, 0.08), 2, 'frames 3 and 4 only');
});

test('files of a whole dataset folder are matched to the loaded video', () => {
  assert.ok(pathMatchesVideo('Gaze_txt_files/p003/001._2017-10-03-18-55_ori_0.txt', '001'));
  assert.ok(!pathMatchesVideo('Gaze_txt_files/p003/0010._2017-10-03_ori_0.txt', '001'));
  assert.ok(pathMatchesVideo('Fixation_per_user_per_video/videos-audios/-ZrvrKMC7Tc-28/0b4b.txt', '-ZrvrKMC7Tc-28'));
  assert.ok(!pathMatchesVideo('Fixation_per_user_per_video/videos-audios/-ZrvrKMC7Tc-280/0b4b.txt', '-ZrvrKMC7Tc-28'));
});

test('frame rate from decoded frame gaps survives whole-millisecond timestamps (29.97 fps in WebM)', () => {
  const t = []; let x = 0.5;
  for (let i = 0; i < 40; i++) { const n = x + 1 / 29.97; t.push(Math.round(n * 1000) / 1000 - Math.round(x * 1000) / 1000); x = n; }
  assert.equal(fpsFromDeltas(t), 29.97);
  assert.equal(fpsFromDeltas(Array(40).fill(0.04)), 25);
});

test('YT360 per-frame files (Fixation_per_video) take the frame from the file name', () => {
  const t = parseTable('eye_rect_x,eye_rect_y,head_rect_x,head_rect_y\n0.25,0.5,0.3,0.5\n0.75,0.5,0.7,0.5\n', '00000037.txt');
  const m = detectMapping(t, { duration: 30 });
  assert.equal(m.preset, 'yt360frames'); assert.equal(m.time, FILE_TIME); assert.equal(m.timeUnit, 'frame');
  const g = buildGaze([{ table: t, name: '00000037.txt', folder: 'clip-28', folder2: 'videos-audios' }], m, { fps: 25 });
  assert.equal(g.n, 2); close(g.t0[0], 37.5 / 25, 1e-9, 'centre of frame 37');
  assert.equal(g.observers.length, 1, 'no observer ids in these files');
  assert.throws(() => buildGaze([{ table: t, name: 'frame.txt' }], m, { fps: 25 }), /no number/);
});

// ---------- Salient360, observer removal, reference maps ----------
import { dropObservers } from '../src/gaze.js';
import { compareMaps, parseNpyHeader, resample } from '../src/reference.js';

test('Salient360 fixation files: comment header, observers where the index restarts, end frame included', () => {
  const csv = '# Idx, longitude, latitude, start timestamp, duration, start frame, end frame\n0,0.25,0.5,0,300,1,8\n1,0.5,0.5,400,200,11,15\n0,0.75,0.25,0,100,1,3\n';
  const t = parseTable(csv, '1_PortoRiverside_fixations.csv');
  assert.equal(t.preset, 's360');
  const m = detectMapping(t, {});
  const g = buildGaze([{ table: t, name: '1_PortoRiverside_fixations.csv', folder: 'L' }], m, { fps: 25 });
  assert.equal(g.observers.length, 2, 'index 0 again = next observer');
  close(g.t0[0], 0.5 / 25, 1e-9, 'frame 1 = first video frame');
  close(g.t1[0], 8 / 25, 1e-9, 'end frame 8 included: runs to the end of video frame 7');
  const i = [...g.lon].findIndex(l => Math.abs(l - Math.PI / 2) < 1e-9);
  close(g.lat[i], Math.PI / 4, 1e-9, 'latitude 0.25 from the top is above the equator');
});

test('removing an observer drops its rows only', () => {
  const g = packRecords([[0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 1, 0, 1], [1, 1, 0, 0, 1, 0, 1]], ['a', 'b'], ['c']);
  const h = dropObservers(g, new Set([1]));
  assert.equal(h.n, 1); assert.equal(h.removed, 2); assert.deepEqual(h.observers, ['a', 'b']);
  assert.throws(() => dropObservers(g, new Set([0, 1])), /All observers/);
});

test('reference helpers: npy header, resampling and map comparison', () => {
  const hdr = "{'descr': '<f2', 'fortran_order': False, 'shape': (500, 64, 128), }";
  const pad = 128 - 10 - hdr.length - 1, text = hdr + ' '.repeat(pad) + '\n';
  const buf = new Uint8Array(10 + text.length);
  buf.set([0x93, 78, 85, 77, 80, 89, 1, 0]); new DataView(buf.buffer).setUint16(8, text.length, true);
  buf.set(new TextEncoder().encode(text), 10);
  const h = parseNpyHeader(buf.buffer);
  assert.equal(h.descr, '<f2'); assert.deepEqual(h.shape, [500, 64, 128]); assert.equal(h.offset, 128);
  const src = Float32Array.from({ length: 16 }, (_, i) => i);           // 4 × 4
  assert.deepEqual([...resample(src, 4, 4, 2, 2)], [2.5, 4.5, 10.5, 12.5], 'area average when shrinking');
  assert.equal(resample(src, 4, 4, 8, 8).length, 64);
  const a = Float32Array.from([0, 1, 2, 3]);
  const c = compareMaps(a, a, 0.5);
  close(c.cc, 1, 1e-9); assert.equal(c.iou, 1); assert.equal(c.coverageA, 0.5);
});

test('video layouts: cube faces and the padded equi-angular cubemap land where ffmpeg puts them', async () => {
  const { dirToUV, erpSize, uvToDir } = await import('../src/projections.js');
  const front = [0, 0, 1];
  assert.deepEqual(dirToUV('c3x2', front), [0.5, 0.75], 'cubemap 3 × 2: front face is tile 4 (middle of the bottom row)');
  const [u, v] = dirToUV('eac', front, undefined, [768, 512]);
  close(u, 0.5, 1e-9); close(v, 0.25, 1e-9, 'equi-angular cubemap: front face in the middle of the top row');
  assert.deepEqual(dirToUV('erp-tb', front), [0.5, 0.25]); assert.deepEqual(dirToUV('erp-lr', front), [0.25, 0.5]);
  assert.deepEqual(erpSize('eac', 3072, 2048), [4096, 2048]); assert.deepEqual(erpSize('erp-tb', 3840, 3840), [3840, 1920]);
  const back = dirToUV('erp', uvToDir(0.3, 0.7)); close(back[0], 0.3, 1e-9); close(back[1], 0.7, 1e-9);
});

test('cylindrical panoramas and cube face orders', async () => {
  const { dirToUV, erpSize, faceTables } = await import('../src/projections.js');
  const o = { hfov: 360 };
  assert.deepEqual(dirToUV('cyl', [0, 0, 1], undefined, [1024, 512], o), [0.5, 0.5]);
  // the top edge of a full 360° panorama of 1024 × 512 is at elevation atan(π·512/1024 / 2·…) = atan(π/2)
  const top = Math.atan(Math.PI / 2);
  const p = dirToUV('cyl', [0, Math.sin(top) * 0.999, Math.cos(top)], undefined, [1024, 512], o);
  assert.ok(p && p[1] < 0.01, `just inside the top edge: ${p}`);
  assert.equal(dirToUV('cyl', [0, 1, 0.1], undefined, [1024, 512], o), null, 'straight up is outside the cylinder');
  assert.equal(dirToUV('cyl', [0, 0, -1], undefined, [1024, 512], { hfov: 180 }), null, 'behind a 180° panorama');
  assert.deepEqual(erpSize('cyl', 1800, 400, { hfov: 180 }), [3600, 1800]);
  const t = faceTables({ order: 'LRDUBF', mirror: true });
  assert.deepEqual(t.c3x2.l, [0, 0, 1]); assert.deepEqual(t.c3x2.f, [5, 0, 1]);
  assert.throws(() => faceTables({ order: 'RRUDFB' }), /each of R L U D F B/);
});

test('presets, imported settings and references matched to a video', async () => {
  const { PRESETS, settingsFromJson } = await import('../src/presets.js');
  const { referencesFor } = await import('../src/reference.js');
  const he = PRESETS.find(p => p.id === 'salient360-he');
  assert.equal(he.settings.sigmaDeg, 2); assert.equal(he.settings.windowSec, 0); assert.equal(he.settings.velocity, 0);
  const fromExport = settingsFromJson({ video: { name: 'a.mp4' }, saliency: { projection: 'erp', sigma_deg: 3, pooling_window_s: 1, pooling_weight: 'gaussian', motion_compensation: true, coverage_threshold: 0.2 } });
  assert.deepEqual(fromExport.settings, { projection: 'erp', sigmaDeg: 3, windowSec: 1, kernel: 'gaussian', motionComp: true, coverageThreshold: 0.2 });
  assert.deepEqual(settingsFromJson({ overgaze_template: 1, name: 'x', settings: { sigmaDeg: 4, junk: 1 } }), { name: 'x', settings: { sigmaDeg: 4 } });
  assert.throws(() => settingsFromJson({ hello: 1 }), /no OverGaze settings/);
  const f = path => ({ name: path.split('/').pop(), webkitRelativePath: path });
  const files = ['SalMaps/2_Diner_2048x1024x600_32b.bin', 'SalMaps/2_Diner_saliency.png', 'SalMaps/12_TeatroRegioTorino_2048x1024x600_32b.bin', 'maps_mc3_vret/vret_001/sal.npy', 'maps_mc3_vret/vret_0010/sal.npy'].map(f);
  assert.deepEqual(referencesFor(files, '2_Diner').map(x => x.name), ['2_Diner_2048x1024x600_32b.bin'], 'the .bin is preferred to the static image');
  assert.deepEqual(referencesFor(files, '001').map(x => x.webkitRelativePath), ['maps_mc3_vret/vret_001/sal.npy']);
  assert.deepEqual(referencesFor(files, '7_GazaFishermen'), []);
});
