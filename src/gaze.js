// Gaze / fixation table parsing and conversion to an internal, time-sorted representation.
// Internal coordinates: lon in radians (−π … π, 0 = image centre column, positive = right),
// lat in radians (−π/2 … π/2, positive = up / top of the frame). Times in seconds.

const TAU = Math.PI * 2;

/** Parse CSV / TSV / semicolon / whitespace tables, ARFF, or JSON (array of objects or arrays). */
export function parseTable(text, name = '') {
  const trimmed = text.replace(/^﻿/, '').trim();
  if (!trimmed) throw new Error(`${name || 'File'} is empty.`);
  if (/\.json$/i.test(name) || trimmed[0] === '[' || trimmed[0] === '{') return parseJsonTable(trimmed, name);
  if (/^@relation/im.test(trimmed.slice(0, 4000)) || /\.arff$/i.test(name)) return parseArff(trimmed);
  const lines = trimmed.split(/\r?\n/).filter(line => line.trim() && !/^\s*(#|%|\/\/)/.test(line));
  // VR-EyeTracking raw gaze: "frame,<n>,forward,<head x>,<head y>,eye,<gaze x>,<gaze y>" with label words in the rows.
  if (/^frame\s*,\s*\d+\s*,\s*forward\s*,/i.test(lines[0])) {
    const rows = lines.map(l => l.split(',').map(c => c.trim())).filter(r => r.length >= 8).map(r => [r[1], r[3], r[4], r[6], r[7]]);
    return { columns: ['frame_number', 'head_x', 'head_y', 'gaze_x', 'gaze_y'], rows, preset: 'vret' };
  }
  const delimiter = detectDelimiter(lines.slice(0, 20));
  const split = line => splitLine(line, delimiter);
  const first = split(lines[0]);
  const header = first.some(cell => cell !== '' && !isNumeric(cell));
  const width = Math.max(...lines.slice(0, 50).map(line => split(line).length));
  const columns = header ? uniqueNames(first.map((c, i) => c || `col${i}`)) : Array.from({ length: width }, (_, i) => `col${i}`);
  const rows = (header ? lines.slice(1) : lines).map(split);
  return { columns, rows };
}

function parseJsonTable(text, name) {
  let data = JSON.parse(text);
  if (!Array.isArray(data)) {
    const key = Object.keys(data).find(k => Array.isArray(data[k]));
    if (!key) throw new Error(`${name || 'JSON'}: expected an array of records.`);
    data = data[key];
  }
  if (!data.length) throw new Error(`${name || 'JSON'} contains no records.`);
  if (Array.isArray(data[0])) {
    const width = Math.max(...data.map(r => r.length));
    return { columns: Array.from({ length: width }, (_, i) => `col${i}`), rows: data.map(r => r.map(v => String(v ?? ''))) };
  }
  const columns = [...new Set(data.flatMap(r => Object.keys(r)))];
  return { columns, rows: data.map(r => columns.map(c => String(r[c] ?? ''))) };
}

function parseArff(text) {
  const lines = text.split(/\r?\n/);
  const columns = [];
  let i = 0;
  for (; i < lines.length; i++) {
    const line = lines[i].trim();
    const attribute = line.match(/^@attribute\s+('[^']+'|"[^"]+"|\S+)/i);
    if (attribute) columns.push(attribute[1].replace(/^['"]|['"]$/g, ''));
    if (/^@data/i.test(line)) { i++; break; }
  }
  const rows = lines.slice(i).filter(l => l.trim() && !/^\s*%/.test(l)).map(l => splitLine(l, ','));
  return { columns: uniqueNames(columns), rows };
}

function detectDelimiter(lines) {
  const candidates = ['\t', ',', ';', '|'];
  let best = null, bestScore = 0;
  for (const d of candidates) {
    const counts = lines.map(l => splitLine(l, d).length);
    const min = Math.min(...counts);
    if (min > 1 && counts.every(c => c === counts[0]) && min > bestScore) { best = d; bestScore = min; }
  }
  if (best) return best;
  for (const d of candidates) if (lines.every(l => splitLine(l, d).length > 1)) return d;
  return 'whitespace';
}

function splitLine(line, delimiter) {
  if (delimiter === 'whitespace') return line.trim().split(/\s+/);
  const out = [];
  let cell = '', quoted = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (quoted) {
      if (ch === '"' && line[i + 1] === '"') { cell += '"'; i++; }
      else if (ch === '"') quoted = false;
      else cell += ch;
    } else if (ch === '"' && cell.trim() === '') { quoted = true; cell = ''; }
    else if (ch === delimiter) { out.push(cell.trim()); cell = ''; }
    else cell += ch;
  }
  out.push(cell.trim());
  return out;
}

function uniqueNames(names) {
  const seen = new Map();
  return names.map(n => {
    const count = seen.get(n) || 0;
    seen.set(n, count + 1);
    return count ? `${n}_${count + 1}` : n;
  });
}

export function isNumeric(value) {
  return value !== '' && value != null && Number.isFinite(Number(value));
}

const PATTERNS = {
  time: [/^(t|time|times|timestamp|ts|sec|secs|seconds|start|start_?time|start_?timestamp|starttime|onset|t_?start|time_?s|time_?ms|frame|frame_?idx|frame_?index|frame_?no|frame_?number)$/i, /time|start|onset|frame/i],
  end: [/^(end|end_?time|stop|t_?end|offset|end_?timestamp)$/i],
  duration: [/^(dur|duration|duration_?s|duration_?ms|fix_?dur|fixation_?duration)$/i, /dur/i],
  x: [/^(x|u|lon|long|longitude|yaw|theta|azimuth|gaze_?x|x_?norm|x_?px|norm_?x|fix_?x|pos_?x|lon_?deg|longitude_?deg)$/i, /(^|_)(x|lon|longitude|yaw|azimuth)($|_)/i],
  y: [/^(y|v|lat|latitude|pitch|phi|elevation|gaze_?y|y_?norm|y_?px|norm_?y|fix_?y|pos_?y|lat_?deg|latitude_?deg)$/i, /(^|_)(y|lat|latitude|pitch|elevation)($|_)/i],
  z: [/^(z|gaze_?z|dir_?z)$/i],
  observer: [/^(user|users|user_?id|subject|subject_?id|subj|observer|observer_?id|participant|participant_?id|viewer|pid|sid)$/i, /user|subj|observer|particip|viewer/i],
  condition: [/^(cond|condition|audio|color|colour|modality|group|session_?type)$/i, /cond/i]
};

/** Guess a column mapping from header names and value ranges. Everything can be edited in the UI. */
export function detectMapping(table, context = {}) {
  if (table.preset === 'vret') {
    // Positions are fractions of the panorama from the BOTTOM-LEFT corner; one gaze sample per video frame, numbered from 1.
    return {
      time: 'frame_number', timeUnit: 'frame1', end: '', duration: '', durationUnit: 'frame1', x: 'gaze_x', y: 'gaze_y', z: '',
      coord: 'normalized', xLeft: 0, xRight: 1, yTop: 1, yBottom: 0,
      observerFrom: 'folder', observer: '', conditionFrom: 'none', condition: '', weight: '', timeOffset: 0, preset: 'vret'
    };
  }
  const pick = (key, exclude = []) => {
    for (const re of PATTERNS[key]) {
      const hit = table.columns.find(c => re.test(c) && !exclude.includes(c));
      if (hit) return hit;
    }
    return '';
  };
  const numericColumns = table.columns.filter((c, i) => sample(table, i).filter(isNumeric).length > 0.8 * Math.max(1, sample(table, i).length));
  let time = pick('time');
  let x = pick('x', [time]);
  let y = pick('y', [time, x]);
  const z = pick('z', [time, x, y]);
  if (!x || !y) {
    // No header names: assume (time, x, y) or (x, y, time) by value ranges.
    const rest = numericColumns.filter(c => c !== time);
    if (!time && rest.length >= 3) {
      const monotonic = rest.find(c => isMonotonic(columnValues(table, c)));
      time = monotonic || rest[0];
    }
    const remaining = numericColumns.filter(c => c !== time);
    x = x || remaining[0] || '';
    y = y || remaining.find(c => c !== x) || '';
  }
  const duration = pick('duration', [time, x, y]);
  const end = duration ? '' : pick('end', [time, x, y]);
  const observer = pick('observer', [time, x, y, duration, end]);
  const condition = pick('condition', [time, x, y, duration, end, observer]);
  const xs = columnValues(table, x), ys = columnValues(table, y), ts = columnValues(table, time);
  const coord = z && x && y ? 'vector' : guessCoord(xs, ys, x, y, context);
  const ranges = defaultRanges(coord, xs, ys, context);
  const timeUnit = guessTimeUnit(time, ts, context);
  const yt360 = table.columns.includes('eye_rect_x') && table.columns.includes('start_time');
  // YT360 Fixation_per_video: one file per frame (00000037.txt = frame 37), rows = points of that frame, no observer.
  const yt360frames = table.columns.includes('eye_rect_x') && !table.columns.includes('start_time');
  const durationUnit = /(^|_)ms$|msec|millis/i.test(duration) ? 'ms' : /(^|_)s$|sec/i.test(duration) ? 's' : timeUnit;
  return {
    time, timeUnit, end, duration, durationUnit, x, y, z: coord === 'vector' ? z : '',
    coord, ...ranges,
    observerFrom: observer ? 'column' : 'file', observer,
    conditionFrom: condition ? 'column' : 'none', condition,
    weight: '', timeOffset: 0,
    ...(yt360 ? { preset: 'yt360', x: 'eye_rect_x', y: 'eye_rect_y', time: 'start_time', duration: 'duration', end: '', timeUnit: 's', durationUnit: 's', coord: 'normalized', xLeft: 0, xRight: 1, yTop: 0, yBottom: 1, z: '', conditionFrom: 'folder2' } : {}),
    ...(yt360frames ? { preset: 'yt360frames', x: 'eye_rect_x', y: 'eye_rect_y', time: FILE_TIME, timeUnit: 'frame', duration: '', end: '', durationUnit: 'frame', coord: 'normalized', xLeft: 0, xRight: 1, yTop: 0, yBottom: 1, z: '', observerFrom: 'none', observer: '', conditionFrom: 'folder2' } : {})
  };
}

function sample(table, index, n = 200) {
  const step = Math.max(1, Math.floor(table.rows.length / n));
  const out = [];
  for (let i = 0; i < table.rows.length; i += step) out.push(table.rows[i][index]);
  return out;
}

export function columnValues(table, name) {
  const i = table.columns.indexOf(name);
  if (i < 0) return [];
  return table.rows.map(r => Number(r[i])).filter(Number.isFinite);
}

function isMonotonic(values) {
  let ups = 0;
  for (let i = 1; i < values.length; i++) if (values[i] >= values[i - 1]) ups++;
  return values.length > 3 && ups >= 0.95 * (values.length - 1);
}

function range(values) {
  let min = Infinity, max = -Infinity;
  for (const v of values) { if (v < min) min = v; if (v > max) max = v; }
  return [min, max];
}

function guessCoord(xs, ys, xName, yName, context) {
  const [xMin, xMax] = range(xs), [yMin, yMax] = range(ys);
  const name = `${xName} ${yName}`;
  if (/deg/i.test(name)) return 'degrees';
  if (/rad/i.test(name)) return 'radians';
  if (/px|pix/i.test(name)) return 'pixels';
  if (xMin >= -0.001 && xMax <= 1.001 && yMin >= -0.001 && yMax <= 1.001) return 'normalized';
  if (Math.abs(xMin) <= TAU + 0.01 && Math.abs(xMax) <= TAU + 0.01 && Math.abs(yMin) <= Math.PI + 0.01 && Math.abs(yMax) <= Math.PI + 0.01) return 'radians';
  if (xMin >= -360 && xMax <= 360 && yMin >= -180 && yMax <= 180) return 'degrees';
  return 'pixels';
}

export function defaultRanges(coord, xs = [], ys = [], context = {}) {
  const [xMin, xMax] = range(xs), [yMin, yMax] = range(ys);
  switch (coord) {
    case 'normalized': return { xLeft: 0, xRight: 1, yTop: 0, yBottom: 1 };
    case 'degrees':
      return {
        xLeft: xMin >= 0 && xMax > 180 ? 0 : -180, xRight: xMin >= 0 && xMax > 180 ? 360 : 180,
        yTop: yMin >= 0 && yMax > 90 ? 0 : 90, yBottom: yMin >= 0 && yMax > 90 ? 180 : -90
      };
    case 'radians':
      return {
        xLeft: xMin >= 0 && xMax > Math.PI ? 0 : -Math.PI, xRight: xMin >= 0 && xMax > Math.PI ? TAU : Math.PI,
        yTop: yMin >= 0 && yMax > Math.PI / 2 ? 0 : Math.PI / 2, yBottom: yMin >= 0 && yMax > Math.PI / 2 ? Math.PI : -Math.PI / 2
      };
    case 'pixels': {
      const w = context.width || (Number.isFinite(xMax) ? niceSize(xMax) : 3840);
      const h = context.height || (Number.isFinite(yMax) ? niceSize(yMax) : w / 2);
      return { xLeft: 0, xRight: w, yTop: 0, yBottom: h };
    }
    default: return { xLeft: 0, xRight: 1, yTop: 0, yBottom: 1 };
  }
}

function niceSize(max) {
  const sizes = [256, 512, 640, 720, 1024, 1080, 1280, 1440, 1920, 2048, 2160, 2560, 2880, 3840, 4096, 5760, 7680, 8192];
  return sizes.find(s => s >= max) || Math.ceil(max);
}

function guessTimeUnit(name, values, context) {
  if (/frame/i.test(name)) return 'frame';
  if (/(^|_)ms$|msec|millis/i.test(name)) return 'ms';
  if (/(^|_)us$|usec|micro/i.test(name)) return 'us';
  const [, max] = range(values);
  const duration = context.duration || 60;
  if (max > duration * 50000) return 'us';
  if (max > duration * 20) return 'ms';
  return 's';
}

/** Mapping value meaning "the time is the number in the file name" (one file per frame). */
export const FILE_TIME = '(number in file name)';

const UNIT = { s: 1, ms: 1e-3, us: 1e-6 };
/** Seconds of a time value. Frame numbers give the frame CENTRE (frame k covers [k/fps, (k+1)/fps)); 'frame1' counts from 1. */
export function toSeconds(value, unit, fps = 30) {
  if (unit === 'frame') return (value + 0.5) / fps;
  if (unit === 'frame1') return (value - 0.5) / fps;
  return value * (UNIT[unit] ?? 1);
}

/**
 * Convert parsed files into one time-sorted gaze set.
 * files: [{ table, name, folder }]; mapping: see detectMapping; context: { fps, width, height }.
 */
export function buildGaze(files, mapping, context = {}) {
  const fps = context.fps || 30;
  const secs = (value, unit) => toSeconds(value, unit, fps);
  const observers = [], conditions = [];
  const observerIndex = new Map(), conditionIndex = new Map();
  const index = (list, map, key) => {
    if (!map.has(key)) { map.set(key, list.length); list.push(key); }
    return map.get(key);
  };
  const records = [];
  let skipped = 0;
  for (const file of files) {
    const cols = file.table.columns;
    const col = name => (name ? cols.indexOf(name) : -1);
    const it = col(mapping.time), ix = col(mapping.x), iy = col(mapping.y), iz = col(mapping.z);
    const ie = col(mapping.end), id = col(mapping.duration), io = col(mapping.observer), ic = col(mapping.condition), iw = col(mapping.weight);
    const fileTime = mapping.time === FILE_TIME ? (m => m ? Number(m[0]) : NaN)(file.name.match(/\d+(?=\.[^.]*$|$)/)) : NaN;
    if (mapping.time === FILE_TIME && !Number.isFinite(fileTime)) throw new Error(`${file.name}: the time is taken from the file name, but the name holds no number.`);
    if ((it < 0 && mapping.time !== FILE_TIME) || ix < 0 || iy < 0) throw new Error(`${file.name}: time, x and y columns are required (missing ${[it < 0 && 'time', ix < 0 && 'x', iy < 0 && 'y'].filter(Boolean).join(', ')}).`);
    if (mapping.coord === 'vector' && iz < 0) throw new Error(`${file.name}: 3D direction needs a z column.`);
    const stem = file.name.replace(/\.[^.]+$/, '');
    for (const row of file.table.rows) {
      const t = mapping.time === FILE_TIME ? fileTime : Number(row[it]), x = Number(row[ix]), y = Number(row[iy]);
      if (!Number.isFinite(t) || !Number.isFinite(x) || !Number.isFinite(y)) { skipped++; continue; }
      const t0 = secs(t, mapping.timeUnit) + (Number(mapping.timeOffset) || 0);
      let t1 = t0;
      if (id >= 0 && Number.isFinite(Number(row[id]))) t1 = t0 + Math.max(0, durSecs(Number(row[id]), mapping.durationUnit || mapping.timeUnit, fps));
      else if (ie >= 0 && Number.isFinite(Number(row[ie]))) t1 = Math.max(t0, secs(Number(row[ie]), mapping.timeUnit) + (Number(mapping.timeOffset) || 0));
      let lonlat;
      if (mapping.coord === 'vector') {
        const z = Number(row[iz]);
        if (!Number.isFinite(z)) { skipped++; continue; }
        lonlat = vectorToLonLat(x, y, z);
      } else lonlat = toLonLat(x, y, mapping);
      if (!lonlat) { skipped++; continue; }
      const observerKey = mapping.observerFrom === 'column' && io >= 0 ? String(row[io]) : mapping.observerFrom === 'file' ? stem
        : mapping.observerFrom === 'folder' ? (file.folder || stem) : 'all';
      const conditionKey = mapping.conditionFrom === 'column' && ic >= 0 ? String(row[ic])
        : mapping.conditionFrom === 'folder' ? (file.folder || '(root)')
          : mapping.conditionFrom === 'folder2' ? (file.folder2 || file.folder || '(root)')
            : mapping.conditionFrom === 'file' ? stem : 'all';
      // Observers that share a name across conditions stay separate people.
      const observer = index(observers, observerIndex, mapping.conditionFrom === 'folder' || mapping.conditionFrom === 'folder2' ? `${conditionKey}/${observerKey}` : observerKey);
      const condition = index(conditions, conditionIndex, conditionKey);
      const weight = iw >= 0 && Number.isFinite(Number(row[iw])) ? Number(row[iw]) : 1;
      records.push([t0, t1, lonlat[0], lonlat[1], observer, condition, weight]);
    }
  }
  if (!records.length) throw new Error('No valid gaze rows were found with this column mapping.');
  return packRecords(records, observers, conditions, skipped);
}

/** A duration in seconds (frame units are plain frame counts here). */
function durSecs(value, unit, fps) { return unit === 'frame' || unit === 'frame1' ? value / fps : value * (UNIT[unit] ?? 1); }

export function packRecords(records, observers, conditions, skipped = 0) {
  records.sort((a, b) => a[0] - b[0]);
  const n = records.length;
  const g = {
    n, t0: new Float64Array(n), t1: new Float64Array(n), lon: new Float64Array(n), lat: new Float64Array(n),
    obs: new Uint16Array(n), cond: new Uint16Array(n), w: new Float32Array(n),
    observers, conditions, maxDur: 0, tMin: Infinity, tMax: -Infinity, skipped, hasDuration: false
  };
  records.forEach((r, i) => {
    g.t0[i] = r[0]; g.t1[i] = r[1]; g.lon[i] = r[2]; g.lat[i] = r[3]; g.obs[i] = r[4]; g.cond[i] = r[5]; g.w[i] = r[6];
    g.maxDur = Math.max(g.maxDur, r[1] - r[0]);
    g.tMin = Math.min(g.tMin, r[0]); g.tMax = Math.max(g.tMax, r[1]);
  });
  g.hasDuration = g.maxDur > 0;
  return g;
}

/** Map table x, y to (lon, lat) radians using the linear ranges in the mapping. */
export function toLonLat(x, y, m) {
  const xr = Number(m.xRight) - Number(m.xLeft), yr = Number(m.yBottom) - Number(m.yTop);
  if (!xr || !yr) return null;
  let u = (x - Number(m.xLeft)) / xr;
  let v = (y - Number(m.yTop)) / yr;
  u -= Math.floor(u);
  v = Math.min(1, Math.max(0, v));
  return [(u - 0.5) * TAU, (0.5 - v) * Math.PI];
}

/** Direction vector with x right, y up, z forward (Unity-style) → (lon, lat). */
export function vectorToLonLat(x, y, z) {
  const r = Math.hypot(x, y, z);
  if (!r) return null;
  return [Math.atan2(x, z), Math.asin(Math.max(-1, Math.min(1, y / r)))];
}

export function angularDistance(lon1, lat1, lon2, lat2) {
  const dot = Math.sin(lat1) * Math.sin(lat2) + Math.cos(lat1) * Math.cos(lat2) * Math.cos(lon1 - lon2);
  return Math.acos(Math.max(-1, Math.min(1, dot)));
}

/**
 * Velocity-threshold filter for raw gaze samples (I-VT): drop samples whose angular speed to the previous
 * sample of the same observer exceeds maxDegPerSec. Returns a new gaze set (or the same one when disabled).
 */
export function velocityFilter(g, maxDegPerSec) {
  if (!maxDegPerSec || maxDegPerSec <= 0) return g;
  const maxRad = maxDegPerSec * Math.PI / 180;
  const last = new Map();
  const keep = new Uint8Array(g.n);
  let kept = 0;
  for (let i = 0; i < g.n; i++) {
    const key = g.obs[i];
    const prev = last.get(key);
    let ok = true;
    if (prev !== undefined) {
      const dt = g.t0[i] - g.t0[prev];
      if (dt > 0) ok = angularDistance(g.lon[prev], g.lat[prev], g.lon[i], g.lat[i]) / dt <= maxRad;
    }
    last.set(key, i);
    if (ok) { keep[i] = 1; kept++; }
  }
  const records = [];
  for (let i = 0; i < g.n; i++) if (keep[i]) records.push([g.t0[i], g.t1[i], g.lon[i], g.lat[i], g.obs[i], g.cond[i], g.w[i]]);
  const out = packRecords(records, g.observers, g.conditions, g.skipped);
  out.filtered = g.n - kept;
  return out;
}

/** Index of the first record with t0 >= t (binary search). */
export function lowerBound(arr, n, t) {
  let lo = 0, hi = n;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (arr[mid] < t) lo = mid + 1; else hi = mid;
  }
  return lo;
}

/** Processed fixations as CSV text (after mapping and filtering). */
export function gazeToCsv(g, transform = p => p) {
  const lines = ['t_start_s,t_end_s,lon_deg,lat_deg,u,v,observer,condition,weight'];
  for (let i = 0; i < g.n; i++) {
    const [lon, lat] = transform([g.lon[i], g.lat[i]]);
    const u = lon / TAU + 0.5, v = 0.5 - lat / Math.PI;
    lines.push([g.t0[i].toFixed(4), g.t1[i].toFixed(4), (lon * 180 / Math.PI).toFixed(4), (lat * 180 / Math.PI).toFixed(4), u.toFixed(6), v.toFixed(6),
      csvCell(g.observers[g.obs[i]]), csvCell(g.conditions[g.cond[i]]), g.w[i]].join(','));
  }
  return lines.join('\n') + '\n';
}

function csvCell(s) { return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s; }

/**
 * Which gaze files belong to the loaded video when a whole dataset folder is chosen: a folder named like the video
 * (YT360: <condition>/<video id>-<start>/<observer>.txt) or a file name that starts with the video name
 * (VR-EyeTracking: <observer>/<video number>._<date>_ori_0.txt).
 */
export function pathMatchesVideo(path, key) {
  const parts = path.split('/'), file = parts[parts.length - 1];
  return parts.slice(0, -1).includes(key) || file.startsWith(key + '.') || file.startsWith(key + '_');
}
