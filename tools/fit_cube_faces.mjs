// Fits the cube-face tables in src/projections.js to ffmpeg's v360 output. Run in a folder holding erp.gray (1024 × 512 grey
// frame) and its conversions c3x2.gray, c6x1.gray, c1x6.gray, eac.gray made with
//   ffmpeg -f rawvideo -pix_fmt gray -s 1024x512 -i erp.gray -vf v360=input=e:output=<layout>:w=..:h=.. -f rawvideo <layout>.gray
// Prints, for every tile, the best face, quarter turns and mirroring, with the correlation.
import { readFileSync } from 'node:fs';
import { FACES, FACE_KEYS, LAYOUTS } from '../src/projections.js';
const erp = readFileSync('erp.gray'), EW = 1024, EH = 512;
const sample = d => { const r = Math.hypot(...d); const u = Math.atan2(d[0], d[2]) / (2 * Math.PI) + 0.5, v = 0.5 - Math.asin(d[1] / r) / Math.PI;
  const x = Math.min(EW - 1, Math.max(0, u * EW - 0.5)), y = Math.min(EH - 1, Math.max(0, v * EH - 0.5)); return erp[Math.round(y) * EW + Math.round(x) % EW]; };
const sizes = { c3x2: [768, 512], c6x1: [1536, 256], c1x6: [256, 1536], eac: [768, 512] };
const out = {};
for (const layout of ['c3x2', 'c6x1', 'c1x6', 'eac']) {
  const img = readFileSync(layout + '.gray'), [W, H] = sizes[layout], L = LAYOUTS[layout], T = W / L.cols;
  out[layout] = {};
  for (let tile = 0; tile < 6; tile++) {
    const col = tile % L.cols, row = Math.floor(tile / L.cols);
    let best = null;
    for (const fk of FACE_KEYS) for (let rot = 0; rot < 4; rot++) for (let flip = 0; flip < 2; flip++) {
      let sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0, n = 0;
      for (let j = 2; j < T - 2; j += 3) for (let i = 2; i < T - 2; i += 3) {
        let s = (i + 0.5) / T * 2 - 1, t = 1 - (j + 0.5) / T * 2;
        for (let k = 0; k < rot; k++) [s, t] = [t, -s];
        if (flip) s = -s;
        if (L.equiAngular) { s = Math.tan(s * Math.PI / 4); t = Math.tan(t * Math.PI / 4); }
        const F = FACES[fk], d = [0, 1, 2].map(q => F.c[q] + s * F.s[q] + t * F.t[q]);
        const a = sample(d), b2 = img[(row * T + j) * W + col * T + i];
        sa += a; sb += b2; saa += a * a; sbb += b2 * b2; sab += a * b2; n++;
      }
      const cc = (sab - sa * sb / n) / Math.sqrt((saa - sa * sa / n) * (sbb - sb * sb / n));
      if (!best || cc > best.cc) best = { fk, rot, flip, cc };
    }
    out[layout][best.fk] = [tile, best.rot, best.flip];
    console.log(layout, 'tile', tile, best.fk, 'rot', best.rot, 'flip', best.flip, 'cc', best.cc.toFixed(3));
  }
}
console.log(JSON.stringify(out));
