// End-to-end checks in a real Chrome (headless), driven through the DevTools protocol (tools/cdp.mjs).
// Skipped when Chrome or Node's built-in WebSocket is missing. Set CHROME=/path/to/chrome to choose a binary.
import test from 'node:test';
import assert from 'node:assert/strict';
import { copyFileSync, mkdirSync, mkdtempSync, readdirSync, writeFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { launch } from '../tools/cdp.mjs';
import { ROOT, serve } from '../tools/serve.mjs';

const chromeOk = typeof WebSocket !== 'undefined' && spawnSync(process.env.CHROME || 'google-chrome', ['--version']).status === 0;
const opts = { skip: chromeOk ? false : 'Chrome (or Node 22+) not available' };

test('sample loads without console errors, draws the heatmap and reports coverage', opts, async () => {
  const { server, url } = await serve(), b = await launch();
  try {
    await b.goto(url + '?sample');
    await b.waitFor('!!overgaze.state.gaze && document.querySelector("#video").readyState >= 2', 20000, 'sample ready');
    await b.eval('overgaze.applySettings({ sigmaDeg: 3, windowSec: 1, kernel: "gaussian" }); document.querySelector("#video").currentTime = 4; 1');
    await b.waitFor('document.querySelector("#ro-coverage").textContent.includes("%")', 10000, 'coverage readout');
    assert.deepEqual(b.logs, [], 'no console errors or warnings');
    assert.equal(await b.eval('overgaze.state.fps'), 30);
    assert.equal(await b.eval('overgaze.state.gaze.n'), 169);
    const alpha = await b.eval('(()=>{const c=document.querySelector("#heat-canvas");const d=c.getContext("2d").getImageData(0,0,c.width,c.height).data;let m=0;for(let i=3;i<d.length;i+=4)m=Math.max(m,d[i]);return m})()');
    assert.ok(alpha > 100, `heatmap is drawn (max alpha ${alpha})`);
    const report = await b.eval('overgaze.debugReport()');
    for (const want of ['sample_erp.mp4', 'sample_gaze.csv: 169 rows', 'SETTINGS', 'MESSAGES AND ERRORS']) assert.ok(report.includes(want), `debug report mentions ${want}`);
  } finally { b.close(); server.close(); }
});

test('opening index.html as a file shows what to do instead of a dead page', opts, async () => {
  const b = await launch();
  try {
    await b.goto('file://' + join(ROOT, 'index.html'));
    assert.equal(await b.eval('document.querySelector("#file-warning").hidden'), false);
    assert.match(await b.eval('document.querySelector("#file-warning").textContent'), /http\.server/);
  } finally { b.close(); }
});

test('motion compensation pulls the pooled map back onto the moving ball; export writes maps and settings', opts, async () => {
  const { server, url } = await serve(), b = await launch();
  const dir = mkdtempSync(join(tmpdir(), 'overgaze-dl-'));
  try {
    await b.goto(url + '?sample');
    await b.waitFor('!!overgaze.state.gaze && document.querySelector("#video").readyState >= 2', 20000);
    await b.eval('overgaze.applySettings({ sigmaDeg: 3, windowSec: 1, kernel: "gaussian" }); 1');
    // mean coverage over six frames: the share of the frame at or above 0.1 of its maximum
    const covOf = () => b.eval('(()=>{let s=0;for(const k of [45,75,105,135,165,195]){const r=overgaze.mapAt(k,256);let mx=0;for(const v of r.map)if(v>mx)mx=v;let c=0;for(const v of r.map)if(v>=0.1*mx)c++;s+=c/r.map.length}return s/6})()');
    const smeared = await covOf();
    await b.eval("document.querySelector('#flow-method').value = 'seek'; 1");   // exact frame times, independent of decoding speed
    await b.click('#analyse');
    await b.waitFor('!!overgaze.state.flow', 120000, 'motion analysis');
    assert.equal(await b.eval('overgaze.state.settings.motionComp'), true, 'switched on after the analysis');
    const compensated = await covOf();
    assert.ok(compensated < 0.6 * smeared, `coverage ${smeared} without vs ${compensated} with motion compensation`);

    await b.send('Browser.setDownloadBehavior', { behavior: 'allow', downloadPath: dir });
    await b.eval(`for (const [id, v] of [['ex-start','2'],['ex-end','2.2'],['ex-width','256'],['ex-format','npyf4'],['ex-norm','none']]) { const e = document.querySelector('#'+id); e.value = v; e.dispatchEvent(new Event('change')); } 1`);
    await b.click('#export-maps');
    await b.waitFor('document.querySelector("#notice").textContent.startsWith("Exported")', 60000, 'export');
    for (let i = 0; i < 50 && !readdirSync(dir).some(f => f.endsWith('.zip')); i++) await new Promise(r => setTimeout(r, 100));
    const zip = join(dir, readdirSync(dir).find(f => f.endsWith('.zip')));
    const py = `import zipfile,json,sys,struct;z=zipfile.ZipFile(sys.argv[1]);n=z.namelist();s=json.loads(z.read('settings.json'));h=z.read('maps.npy')[:128];print(json.dumps({'names':n,'sigma':s['saliency']['sigma_deg'],'mc':s['saliency']['motion_compensation'],'frames':s['export']['frames'],'header':h[:10].decode('latin1'),'csv':z.read('frames.csv').decode().count('\\n')}))`;
    const r = spawnSync('python3', ['-c', py, zip], { encoding: 'utf8' });
    assert.equal(r.status, 0, r.stderr);
    const info = JSON.parse(r.stdout);
    assert.deepEqual(info.names.sort(), ['frames.csv', 'maps.npy', 'settings.json']);
    assert.equal(info.sigma, 3); assert.equal(info.mc, true); assert.ok(info.frames >= 5);
    assert.ok(info.header.startsWith('\x93NUMPY'));
    assert.equal(info.csv, info.frames + 1, 'one line per map plus the header');
  } finally { b.close(); server.close(); }
});

test('choosing a whole VR-EyeTracking-style folder keeps only the files of the loaded video and recognises the format', opts, async () => {
  const { server, url } = await serve(), b = await launch();
  const dir = mkdtempSync(join(tmpdir(), 'overgaze-vret-'));
  try {
    const rows = y => Array.from({ length: 240 }, (_, i) => `frame,${i + 1},forward,0.5,0.5,eye,0.5,${y}`).join('\n') + '\n';
    for (const [p, file, y] of [['p001', '001._2017-10-01-10-00_ori_0.txt', 0.5], ['p002', '001._2017-10-01-11-00_ori_0.txt', 0.5], ['p003', '002._2017-10-01-12-00_ori_0.txt', 0.5]]) {
      mkdirSync(join(dir, 'Gaze_txt_files', p), { recursive: true });
      writeFileSync(join(dir, 'Gaze_txt_files', p, file), rows(y));
    }
    copyFileSync(join(ROOT, 'sample', 'sample_erp.mp4'), join(dir, '001.mp4'));
    await b.goto(url);
    await b.setFiles('#video-input', [join(dir, '001.mp4')]);
    await b.waitFor('document.querySelector("#video-name").textContent === "001.mp4" && overgaze.state.fpsSource === "measured"', 20000, 'video');
    await b.setFiles('#gaze-folder-input', [join(dir, 'Gaze_txt_files')]);
    await b.waitFor('!!overgaze.state.gaze', 20000, 'gaze');
    assert.deepEqual(await b.eval('overgaze.state.gaze.observers'), ['p001', 'p002'], 'observers come from the folder names; video 002 is left out');
    assert.equal(await b.eval('overgaze.state.mapping.preset'), 'vret');
    assert.equal(await b.eval('overgaze.state.settings.velocity'), 75);
    // gaze at (0.5, 0.5) is the middle of the frame; frame 1 is centred at half a frame
    assert.ok(Math.abs(await b.eval('overgaze.state.gaze.t0[0] * overgaze.state.fps - 0.5')) < 1e-6);
    await b.eval('overgaze.applySettings({ sigmaDeg: 3 }); 1');
    const peak = await b.eval('(()=>{const r=overgaze.mapAt(100,256);let a=0;r.map.forEach((v,i)=>{if(v>r.map[a])a=i});return [a%r.W/r.W,Math.floor(a/r.W)/r.H]})()');
    assert.ok(Math.abs(peak[0] - 0.5) < 0.02 && Math.abs(peak[1] - 0.5) < 0.02, `map peaks in the middle: ${peak}`);
    assert.deepEqual(b.logs, []);
  } finally { b.close(); server.close(); }
});
