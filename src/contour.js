// Outlines as lines: the border of the area at or above a level, traced between pixel centres (marching squares),
// so it can be drawn at any size with a line width chosen in screen pixels.

/**
 * Line segments [x1, y1, x2, y2, …] along the border of { map ≥ level } on a W×H grid, in map pixel units with
 * pixel i's centre at i + 0.5 (so x / W and y / H are fractions of the frame). Areas that touch the frame edge are
 * closed along the edge pixels. `wrap` joins the left and right edges (360° video): a border crossing that seam
 * comes out on both sides.
 */
export function contourSegments(map, W, H, level, wrap = false) {
  const out = [];
  if (!Number.isFinite(level)) return new Float32Array(0);
  // outside the frame counts as "below", except across the left/right seam of a 360° frame
  const at = (x, y) => {
    if (y < 0 || y >= H) return -Infinity;
    if (x < 0 || x >= W) { if (!wrap) return -Infinity; x = (x + W) % W; }
    return map[y * W + x];
  };
  // where the level is crossed between two neighbouring values a (at 0) and b (at 1)
  const cross = (a, b) => a === -Infinity ? 1 : b === -Infinity ? 0 : (level - a) / (b - a);
  // cell (x, y) joins the pixel centres of (x, y) and (x + 1, y + 1); with wrap-around the last cell crosses the seam
  for (let y = -1; y < H; y++) {
    for (let x = wrap ? 0 : -1; x < W; x++) {
      const tl = at(x, y), tr = at(x + 1, y), br = at(x + 1, y + 1), bl = at(x, y + 1);
      const c = (tl >= level ? 8 : 0) | (tr >= level ? 4 : 0) | (br >= level ? 2 : 0) | (bl >= level ? 1 : 0);
      if (c === 0 || c === 15) continue;
      const X = x + 0.5, Y = y + 0.5;
      const top = [X + cross(tl, tr), Y], right = [X + 1, Y + cross(tr, br)];
      const bottom = [X + cross(bl, br), Y + 1], left = [X, Y + cross(tl, bl)];
      const seg = (p, q) => {
        out.push(p[0], p[1], q[0], q[1]);
        if (wrap && x === W - 1) out.push(p[0] - W, p[1], q[0] - W, q[1]);   // the seam cell shows on the left too
      };
      switch (c) {
        case 1: case 14: seg(left, bottom); break;
        case 2: case 13: seg(bottom, right); break;
        case 3: case 12: seg(left, right); break;
        case 4: case 11: seg(top, right); break;
        case 6: case 9: seg(top, bottom); break;
        case 7: case 8: seg(left, top); break;
        case 5: case 10: {
          // a saddle: the mean of the four corners says whether the two "inside" corners are joined
          const finite = [tl, tr, br, bl].filter(Number.isFinite);
          const joined = finite.reduce((s, v) => s + v, 0) / Math.max(1, finite.length) >= level;
          if ((c === 5) === joined) { seg(left, top); seg(bottom, right); } else { seg(left, bottom); seg(top, right); }
          break;
        }
      }
    }
  }
  return Float32Array.from(out);
}

/** Stroke segments from contourSegments onto a 2D context of cw × ch pixels, for a W×H map. */
export function strokeSegments(g, segs, W, H, cw, ch, lineWidth, color) {
  if (!segs.length) return;
  const sx = cw / W, sy = ch / H;
  g.save();
  g.strokeStyle = color; g.lineWidth = lineWidth; g.lineCap = 'round'; g.lineJoin = 'round';
  g.beginPath();
  for (let i = 0; i < segs.length; i += 4) { g.moveTo(segs[i] * sx, segs[i + 1] * sy); g.lineTo(segs[i + 2] * sx, segs[i + 3] * sy); }
  g.stroke();
  g.restore();
}
