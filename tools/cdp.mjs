// Minimal Chrome DevTools Protocol driver (no dependencies; needs Node 22+ for the built-in WebSocket).
// Used for the browser checks and the feature screenshots (see tools/browser-check.mjs).
import { spawn } from 'node:child_process';
import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

export async function launch({ width = 1500, height = 900, args = [] } = {}) {
  const dir = mkdtempSync(join(tmpdir(), 'overgaze-chrome-'));
  // Port 0: Chrome picks a free port and writes it to DevToolsActivePort in its own profile folder. A port picked
  // at random here could already be taken by another test browser started at the same time; this browser would
  // then fail to listen and the driver would connect to the other one, so two scripts would drive one page.
  const chrome = spawn(process.env.CHROME || 'google-chrome', [
    '--headless=new', '--disable-gpu', '--hide-scrollbars', '--remote-debugging-port=0',
    `--user-data-dir=${dir}`, `--window-size=${width},${height}`, '--autoplay-policy=no-user-gesture-required',
    ...args, 'about:blank'], { stdio: 'ignore' });
  let port = 0, targets;
  for (let i = 0; i < 300 && !targets?.length; i++) {
    try {
      port ||= Number(readFileSync(join(dir, 'DevToolsActivePort'), 'utf8').split('\n')[0]);
      if (port) targets = await (await fetch(`http://127.0.0.1:${port}/json`)).json();
    } catch {}
    if (!targets?.length) await new Promise(r => setTimeout(r, 100));
  }
  if (!targets?.length) { chrome.kill(); throw new Error('Chrome did not start (no DevToolsActivePort)'); }
  const page = targets.find(t => t.type === 'page');
  const ws = new WebSocket(page.webSocketDebuggerUrl);
  await new Promise(r => ws.addEventListener('open', r));
  let id = 0; const pending = new Map(); const handlers = [];
  ws.addEventListener('message', ev => {
    const m = JSON.parse(ev.data);
    if (m.id && pending.has(m.id)) { const { res, rej } = pending.get(m.id); pending.delete(m.id); m.error ? rej(new Error(m.error.message)) : res(m.result); }
    else if (m.method) handlers.forEach(h => h(m));
  });
  const send = (method, params = {}) => new Promise((res, rej) => { const i = ++id; pending.set(i, { res, rej }); ws.send(JSON.stringify({ id: i, method, params })); });
  const logs = [];
  handlers.push(m => {
    if (m.method === 'Runtime.exceptionThrown') { const d = m.params.exceptionDetails; logs.push(`EXCEPTION ${d.exception?.description || d.text} @${d.url || ''}:${d.lineNumber}`); }
    if (m.method === 'Runtime.consoleAPICalled') logs.push(`console.${m.params.type}: ${m.params.args.map(a => a.value ?? a.description).join(' ')}`);
    if (m.method === 'Log.entryAdded') logs.push(`log.${m.params.entry.level}: ${m.params.entry.text} ${m.params.entry.url || ''}`);
  });
  await send('Runtime.enable'); await send('Page.enable'); await send('DOM.enable'); await send('Log.enable');
  const api = {
    logs, send,
    async goto(url) {
      const loaded = new Promise(r => { const h = m => { if (m.method === 'Page.loadEventFired') { handlers.splice(handlers.indexOf(h), 1); r(); } }; handlers.push(h); });
      await send('Page.navigate', { url });
      await Promise.race([loaded, new Promise(r => setTimeout(r, 30000))]);
    },
    async eval(expr) {
      const r = await send('Runtime.evaluate', { expression: expr, awaitPromise: true, returnByValue: true });
      if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description || r.exceptionDetails.text);
      return r.result.value;
    },
    async setFiles(selector, files) {
      const { root } = await send('DOM.getDocument');
      const { nodeId } = await send('DOM.querySelector', { nodeId: root.nodeId, selector });
      await send('DOM.setFileInputFiles', { files, nodeId });
    },
    async click(selector) { await api.eval(`document.querySelector(${JSON.stringify(selector)}).click()`); },
    async screenshot(path, clip) {
      const r = await send('Page.captureScreenshot', { format: 'png', ...(clip ? { clip: { ...clip, scale: 1 } } : {}) });
      writeFileSync(path, Buffer.from(r.data, 'base64'));
    },
    async waitFor(expr, ms = 30000, what = expr) {
      const t0 = Date.now();
      while (Date.now() - t0 < ms) { try { if (await api.eval(expr)) return true; } catch {} await new Promise(r => setTimeout(r, 200)); }
      throw new Error('timeout waiting for ' + what + '\n' + logs.slice(-10).join('\n'));
    },
    close() { try { ws.close(); } catch {} chrome.kill(); },
  };
  return api;
}
