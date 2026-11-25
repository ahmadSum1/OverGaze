// Built-in settings presets, with the source of each value, and helpers for saved templates.

export const PRESETS = [
  {
    id: 'salient360-he', name: 'Salient360: head + eye maps (σ 2°, per frame)',
    about: 'David et al., “A Dataset of Head and Eye Movements for 360° Videos”, MMSys 2018, and the dataset readme: ' +
      'each eye fixation adds an isotropic 2° Gaussian on the sphere; fixations of both eyes are combined; one map per frame. ' +
      'Fixations were already parsed by the authors (velocity below 80°/s, at least 80 ms), so no speed filter. ' +
      'Load the L and R fixation files (both eyes = the two conditions) and the video; frame offset +1 for the .bin maps.',
    settings: { projection: 'erp', sigmaDeg: 2, windowSec: 0, kernel: 'box', causal: false, durationWeight: false, motionComp: false, velocity: 0, minDurationMs: 0 }
  },
  {
    id: 'salient360-200ms', name: 'Salient360: 200 ms pooling (paper’s comparisons)',
    about: 'Same maps, pooled as in the paper’s comparisons (section 4.2): frames in intervals of 200 ms (4–6 frames) added ' +
      'together and divided by the total sum. Approximated here by a flat window of ±0.1 s around each frame.',
    settings: { projection: 'erp', sigmaDeg: 2, windowSec: 0.1, kernel: 'box', causal: false, durationWeight: false, motionComp: false, velocity: 0, minDurationMs: 0 }
  },
  {
    id: 'salient360-h', name: 'Salient360: head-only maps (σ 3.34°)',
    about: 'Head-only maps of the dataset: the viewport centre of each head sample (head trajectories resampled to 100 ' +
      'samples of 200 ms) with an isotropic 3.34° Gaussian, chosen in the paper to match the image dataset.',
    settings: { projection: 'erp', sigmaDeg: 3.34, windowSec: 0, kernel: 'box', causal: false, durationWeight: false, motionComp: false, velocity: 0, minDurationMs: 0 }
  },
  {
    id: 'video-sal-mc3', name: 'Video-saliency training maps “mc3” (σ 3°, ±1 s, motion comp.)',
    about: 'The pooled training maps of the VR-EyeTracking / YT360 data (make_pooled_maps.py): 3° sphere Gaussians, ' +
      '±1 s with Gaussian weights (σ 0.5 s), motion-compensated, raw gaze filtered at 75°/s (not for fixation files). ' +
      'Motion compensation needs “Analyse motion”.',
    settings: { projection: 'erp', sigmaDeg: 3, windowSec: 1, kernel: 'gaussian', causal: false, durationWeight: false, motionComp: true, velocity: 75, minDurationMs: 0 }
  },
  {
    id: 'live-mask', name: 'Live mask (σ 3°, past 1 s only)',
    about: 'Only gaze up to the current frame, fading over the last second: what a real-time, deployed mask could use.',
    settings: { projection: 'erp', sigmaDeg: 3, windowSec: 1, kernel: 'triangle', causal: true, durationWeight: false, motionComp: false, velocity: 0, minDurationMs: 0 }
  },
  {
    id: 'default', name: 'OverGaze defaults (σ 2°, this frame only)', about: 'The settings OverGaze starts with.',
    settings: { sigmaDeg: 2, windowSec: 0, kernel: 'box', causal: false, durationWeight: false, motionComp: false, velocity: 0, minDurationMs: 0, coverageThreshold: 0.1 }
  }
];

/** The settings that a preset or template holds (everything that changes the maps or the coverage). */
export const SETTING_KEYS = ['projection', 'sigmaDeg', 'truncate', 'planarFovDeg', 'windowSec', 'kernel', 'peakTauSec', 'causal', 'durationWeight',
  'motionComp', 'lonOffsetDeg', 'flipX', 'flipY', 'velocity', 'minDurationMs', 'coverageThreshold', 'eyePairing'];

export function pickSettings(s) { return Object.fromEntries(SETTING_KEYS.filter(k => s[k] !== undefined && s[k] !== null).map(k => [k, s[k]])); }

/** Drop rules from a settings file: [{ clauses: [{ column, op, value }] }] or settings.json's [{ all_of: [...] }]. */
function rowRulesFromJson(list) {
  if (!Array.isArray(list)) return null;
  return list.map(r => ({ clauses: (r.clauses || r.all_of || []).filter(c => c && c.column).map(c => ({ column: String(c.column), op: String(c.op), value: String(c.value) })) }))
    .filter(r => r.clauses.length);
}

/**
 * Settings from an imported JSON: a template file written by OverGaze ({ overgaze_template, settings }), an exported
 * settings.json (its "saliency" section), or a plain object of settings.
 */
export function settingsFromJson(obj) {
  if (!obj || typeof obj !== 'object') throw new Error('not a JSON object');
  const rules = rowRulesFromJson(obj.row_rules ?? obj.gaze_processing?.drop_row_rules);
  const withRules = r => rules ? { ...r, rowRules: rules } : r;
  if (obj.settings && typeof obj.settings === 'object') return withRules({ name: obj.name, settings: pickSettings(obj.settings) });
  const sal = obj.saliency;
  if (sal) {
    const gp = obj.gaze_processing || {};
    return withRules({
      name: obj.video?.name ? `from ${obj.video.name}` : undefined,
      settings: pickSettings({
        projection: sal.projection, sigmaDeg: sal.sigma_deg, truncate: sal.kernel_cutoff_sigma, planarFovDeg: sal.planar_fov_deg,
        windowSec: sal.pooling_window_s, kernel: sal.pooling_weight, peakTauSec: sal.peaked_tau_s, causal: sal.causal, durationWeight: sal.weight_by_duration,
        motionComp: sal.motion_compensation, lonOffsetDeg: sal.lon_offset_deg, flipX: sal.flip_left_right, flipY: sal.flip_up_down,
        velocity: gp.speed_filter_deg_per_s ?? sal.velocity_filter_deg_per_s, minDurationMs: gp.min_fixation_duration_ms, eyePairing: gp.eye_pairing?.setting,
        coverageThreshold: sal.coverage_threshold
      })
    });
  }
  const plain = pickSettings(obj);
  if (!Object.keys(plain).length) throw new Error('no OverGaze settings found in it');
  return { settings: plain };
}

/**
 * The settings in which `current` differs from a preset or template. Only the keys the preset sets are compared:
 * a preset that leaves out the projection (the defaults) fits 360° and 2D video alike. Numbers are compared as
 * numbers, so 2 and "2" are the same value.
 */
export function presetDifferences(presetSettings, current) {
  const out = [];
  for (const key of SETTING_KEYS) {
    if (!presetSettings || !(key in presetSettings)) continue;
    const a = presetSettings[key], b = current[key];
    const numeric = v => (typeof v === 'number' || (typeof v === 'string' && v.trim() !== '')) && Number.isFinite(Number(v));
    const same = typeof a === 'boolean' ? !!a === !!b
      : numeric(a) && numeric(b) ? Math.abs(Number(a) - Number(b)) < 1e-9 : a === b;
    if (!same) out.push(key);
  }
  return out;
}
