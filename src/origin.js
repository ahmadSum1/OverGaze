// Where a downloaded video came from, read from its file name, so the original can be opened on its site.
//   YouTube: "…youtube.com/watch?v=ID…", "…youtu.be/ID…", yt-dlp's "Title [ID].webm", or the YT360 dataset's
//            "ID-START.webm" / "ID.webm" (START = where the dataset's clip starts, in seconds)
//   Vimeo:   "…vimeo.com/123456…" or "vimeo_123456…"

const YT_ID = '[A-Za-z0-9_-]{11}';
// a bare 11-character name is taken as a YouTube ID only if it looks random: a digit or both upper and lower case
const looksLikeId = id => /\d/.test(id) || (/[a-z]/.test(id) && /[A-Z]/.test(id));

/** { site, id, start, url(t) } or null. url(t) opens the original at t seconds (rounded down). */
export function originOf(name) {
  const stem = String(name || '').replace(/\.[a-z0-9]{2,4}$/i, '');
  const yt = (id, start = null) => ({ site: 'YouTube', id, start, url: t => `https://www.youtube.com/watch?v=${id}${t > 0 ? `&t=${Math.floor(t)}` : ''}` });
  let m;
  if ((m = new RegExp(`(?:youtube\\.com[^A-Za-z0-9]+watch[^A-Za-z0-9]+v[=_](${YT_ID})|youtu\\.be[^A-Za-z0-9]+(${YT_ID}))`, 'i').exec(stem))) return yt(m[1] || m[2]);
  if ((m = new RegExp(`\\[(${YT_ID})\\]$`).exec(stem))) return yt(m[1]);
  if ((m = new RegExp(`^(${YT_ID})(?:-(\\d+))?$`).exec(stem)) && looksLikeId(m[1])) return yt(m[1], m[2] != null ? Number(m[2]) : null);
  if ((m = /vimeo(?:\.com)?[^A-Za-z0-9]+(\d{6,})/i.exec(stem))) { const id = m[1]; return { site: 'Vimeo', id, start: null, url: t => `https://vimeo.com/${id}${t > 0 ? `#t=${Math.floor(t)}s` : ''}` }; }
  return null;
}

/**
 * The moment on the original that matches the playhead. A full download plays on the original's own clock; a file
 * cut from it at the dataset's clip start (shorter than that start) is shifted by the start.
 */
export function originalTime(origin, playhead, duration) {
  if (!origin) return 0;
  const cut = origin.start != null && !(duration > origin.start + 1);
  return (cut ? origin.start : 0) + playhead;
}
