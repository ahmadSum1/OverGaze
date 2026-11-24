// Frame rate read from an MP4 / MOV file's header: the video track's time scale (ticks per second) and its table of
// frame durations ('stts'). Exact and instant, so frame-numbered gaze is timed right before the video ever plays.
// Works on anything with slice(a, b).arrayBuffer() (a File or Blob). Returns null when the file is not MP4 / MOV,
// has no video track, or stores its frames in fragments (then the frame rate is measured from decoded frames).

function* boxes(view, start, end) {
  let p = start;
  while (p + 8 <= end) {
    let size = view.getUint32(p), head = 8;
    const type = String.fromCharCode(view.getUint8(p + 4), view.getUint8(p + 5), view.getUint8(p + 6), view.getUint8(p + 7));
    if (size === 1) { if (p + 16 > end) return; size = Number(view.getBigUint64(p + 8)); head = 16; }
    else if (size === 0) size = end - p;
    if (size < head || p + size > end) return;
    yield { type, start: p + head, end: p + size };
    p += size;
  }
}

/** Parse a 'moov' box (bytes of the whole box or just its contents). Returns { fps, frames, timescale } or null. */
export function fpsFromMoov(buffer, offset = 0, length = buffer.byteLength - offset) {
  const view = new DataView(buffer, offset, length);
  let top = { start: 0, end: length };
  const first = boxes(view, 0, length).next().value;
  if (first?.type === 'moov') top = first;
  for (const trak of boxes(view, top.start, top.end)) {
    if (trak.type !== 'trak') continue;
    const mdia = [...boxes(view, trak.start, trak.end)].find(b => b.type === 'mdia');
    if (!mdia) continue;
    let timescale = 0, isVideo = false, stts = null;
    for (const b of boxes(view, mdia.start, mdia.end)) {
      if (b.type === 'mdhd') timescale = view.getUint8(b.start) === 1 ? view.getUint32(b.start + 20) : view.getUint32(b.start + 12);
      else if (b.type === 'hdlr') isVideo = String.fromCharCode(...new Uint8Array(buffer, offset + b.start + 8, 4)) === 'vide';
      else if (b.type === 'minf') {
        const stbl = [...boxes(view, b.start, b.end)].find(c => c.type === 'stbl');
        stts = stbl && [...boxes(view, stbl.start, stbl.end)].find(c => c.type === 'stts');
      }
    }
    if (!isVideo || !timescale || !stts) continue;
    const n = view.getUint32(stts.start + 4);
    let frames = 0, ticks = 0;
    for (let i = 0; i < n && stts.start + 16 + 8 * i <= stts.end; i++) {
      const count = view.getUint32(stts.start + 8 + 8 * i), delta = view.getUint32(stts.start + 12 + 8 * i);
      frames += count; ticks += count * delta;
    }
    if (frames < 2 || !ticks) return null;
    return { fps: timescale * frames / ticks, frames, timescale };
  }
  return null;
}

/** Find the 'moov' box among the top-level boxes of a File / Blob and read the frame rate from it. */
export async function fpsFromMp4File(file) {
  const read = async (a, b) => file.slice(a, b).arrayBuffer();
  let p = 0;
  for (let guard = 0; guard < 64 && p + 8 <= file.size; guard++) {
    const head = new DataView(await read(p, p + 16));
    let size = head.getUint32(0);
    const type = String.fromCharCode(head.getUint8(4), head.getUint8(5), head.getUint8(6), head.getUint8(7));
    if (guard === 0 && type !== 'ftyp' && type !== 'moov' && type !== 'wide' && type !== 'free' && type !== 'mdat') return null;
    if (size === 1) size = Number(head.getBigUint64(8));
    else if (size === 0) size = file.size - p;
    if (size < 8) return null;
    if (type === 'moov') {
      if (size > 256 * 1024 * 1024) return null;
      return fpsFromMoov(await read(p, p + size));
    }
    p += size;
  }
  return null;
}
