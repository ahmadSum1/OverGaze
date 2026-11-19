// Tiny static file server with range requests (Chrome seeks in video through them). Used by the browser tests
// and the screenshot script; for ordinary use any static server works (python3 -m http.server).
import { createServer } from 'node:http';
import { readFileSync, existsSync, statSync } from 'node:fs';
import { extname, join, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';

export const ROOT = fileURLToPath(new URL('..', import.meta.url));
const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.mp4': 'video/mp4', '.csv': 'text/csv', '.json': 'application/json' };
export function serve() {
  const server = createServer((req, res) => {
    const path = normalize(decodeURIComponent(new URL(req.url, 'http://x').pathname)).replace(/^(\.\.[/\\])+/, '');
    const file = join(ROOT, path.endsWith('/') ? path + 'index.html' : path);
    if (!file.startsWith(ROOT) || !existsSync(file) || statSync(file).isDirectory()) { res.writeHead(404); res.end(); return; }
    const body = readFileSync(file), range = /bytes=(\d+)-(\d*)/.exec(req.headers.range || '');
    const type = MIME[extname(file)] || 'application/octet-stream';
    if (range) {   // Chrome seeks in video through range requests
      const a = Number(range[1]), b = range[2] ? Number(range[2]) : body.length - 1;
      res.writeHead(206, { 'content-type': type, 'accept-ranges': 'bytes', 'content-range': `bytes ${a}-${b}/${body.length}`, 'content-length': b - a + 1 });
      res.end(body.subarray(a, b + 1));
    } else { res.writeHead(200, { 'content-type': type, 'accept-ranges': 'bytes', 'content-length': body.length }); res.end(body); }
  });
  return new Promise(r => server.listen(0, '127.0.0.1', () => r({ server, url: `http://127.0.0.1:${server.address().port}/` })));
}
