// Colour lookup tables (256 × RGB) interpolated from a few stops of the usual palettes.
const STOPS = {
  jet: ['#000080', '#0000ff', '#0080ff', '#00ffff', '#80ff80', '#ffff00', '#ff8000', '#ff0000', '#800000'],
  turbo: ['#30123b', '#4662d7', '#36aaf9', '#1ae4b6', '#72fe5e', '#c8ef34', '#faba39', '#f66b19', '#ca2a04', '#7a0403'],
  inferno: ['#000004', '#320a5e', '#781c6d', '#bc3754', '#ed6925', '#fbb61a', '#fcffa4'],
  magma: ['#000004', '#2c115f', '#721f81', '#b73779', '#f1605d', '#feb078', '#fcfdbf'],
  viridis: ['#440154', '#414487', '#2a788e', '#22a884', '#7ad151', '#fde725'],
  hot: ['#000000', '#800000', '#ff0000', '#ff8000', '#ffff00', '#ffffff'],
  gray: ['#000000', '#ffffff'],
  ice: ['#04121f', '#0b3c5d', '#1170a8', '#2fb3e8', '#9fe6ff', '#ffffff']
};

export const COLORMAPS = Object.keys(STOPS);

const cache = new Map();

export function lut(name) {
  if (cache.has(name)) return cache.get(name);
  const stops = (STOPS[name] || STOPS.jet).map(h => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16)));
  const out = new Uint8Array(256 * 3);
  for (let i = 0; i < 256; i++) {
    const pos = i / 255 * (stops.length - 1), k = Math.min(stops.length - 2, Math.floor(pos)), f = pos - k;
    for (let c = 0; c < 3; c++) out[3 * i + c] = Math.round(stops[k][c] * (1 - f) + stops[k + 1][c] * f);
  }
  cache.set(name, out);
  return out;
}

export function cssGradient(name) {
  const t = lut(name), parts = [];
  for (let i = 0; i <= 8; i++) {
    const k = Math.round(i / 8 * 255);
    parts.push(`rgb(${t[3 * k]},${t[3 * k + 1]},${t[3 * k + 2]}) ${i * 12.5}%`);
  }
  return `linear-gradient(90deg, ${parts.join(', ')})`;
}

/**
 * Colourise a normalised map (values in [0, 1]) into RGBA.
 * mode 'heat': colour with alpha rising with the value; 'solid': colour at constant alpha;
 * 'spotlight': black veil that hides low-saliency regions; 'gray': opaque grey map.
 */
export function colorize(map, rgba, { colormap = 'jet', opacity = 0.65, gamma = 1, floor = 0.02, mode = 'heat' } = {}) {
  const t = lut(colormap), n = map.length;
  for (let i = 0, j = 0; i < n; i++, j += 4) {
    let v = map[i];
    v = v > 1 ? 1 : v < 0 ? 0 : v;
    if (gamma !== 1) v = Math.pow(v, gamma);
    if (mode === 'spotlight') {
      rgba[j] = rgba[j + 1] = rgba[j + 2] = 0;
      rgba[j + 3] = Math.round(255 * opacity * (1 - v));
      continue;
    }
    if (mode === 'gray') {
      const g = Math.round(v * 255);
      rgba[j] = rgba[j + 1] = rgba[j + 2] = g; rgba[j + 3] = 255;
      continue;
    }
    const k = Math.round(v * 255) * 3;
    rgba[j] = t[k]; rgba[j + 1] = t[k + 1]; rgba[j + 2] = t[k + 2];
    if (v < floor) rgba[j + 3] = 0;
    else rgba[j + 3] = Math.round(255 * opacity * (mode === 'solid' ? 1 : Math.min(1, 0.25 + 0.75 * v)));
  }
  return rgba;
}
