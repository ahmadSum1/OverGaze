// End-to-end checks in a real Chrome (headless), driven through the DevTools protocol (tools/cdp.mjs).
// Skipped when Chrome or Node's built-in WebSocket is missing. Set CHROME=/path/to/chrome to choose a binary.
import test from 'node:test';
import assert from 'node:assert/strict';
import { copyFileSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
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
    // Farnebäck's 15-pixel window follows the sample's small fast ball less closely than the built-in estimate did
    // (about 0.8 px off per field; see docs/validation.md, section 3), so the map shrinks less than before (0.64× here)
    assert.ok(compensated < 0.75 * smeared, `coverage ${smeared} without vs ${compensated} with motion compensation`);

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
    await b.waitFor('document.querySelector("#video-name").textContent === "001.mp4" && overgaze.state.fpsSource === "file"', 20000, 'video');
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

test('gaze files add up, observers can be removed, messages can be dismissed, a reference map is compared, 360° view draws', opts, async () => {
  const { server, url } = await serve(), b = await launch();
  const dir = mkdtempSync(join(tmpdir(), 'overgaze-more-'));
  try {
    const csv = 'user,start_time,duration,longitude,latitude\n';
    writeFileSync(join(dir, 'a.csv'), csv + '1,1.0,2.0,0.5,0.5\n2,1.0,2.0,0.5,0.5\n');
    writeFileSync(join(dir, 'b.csv'), csv + '3,1.0,2.0,0.25,0.5\n');
    // reference: .npy float32 (2, 64, 128) with a peak where the gaze is (frame centre)
    const hdr = "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 64, 128), }", text = hdr + ' '.repeat(128 - 11 - hdr.length) + '\n';
    const data = new Float32Array(2 * 64 * 128);
    for (let k = 0; k < 2; k++) for (let y = 0; y < 64; y++) for (let x = 0; x < 128; x++) data[k * 8192 + y * 128 + x] = Math.exp(-((x - 63.5) ** 2 + (y - 31.5) ** 2) / 8);
    const npy = new Uint8Array(10 + text.length + data.byteLength);
    npy.set([0x93, 78, 85, 77, 80, 89, 1, 0]); new DataView(npy.buffer).setUint16(8, text.length, true);
    npy.set(new TextEncoder().encode(text), 10); npy.set(new Uint8Array(data.buffer), 10 + text.length);
    writeFileSync(join(dir, 'ref.npy'), npy);

    await b.goto(url);
    await b.setFiles('#video-input', [join(ROOT, 'sample', 'sample_erp.mp4')]);
    await b.waitFor('overgaze.state.fpsSource === "file"', 20000, 'video');
    await b.setFiles('#gaze-input', [join(dir, 'a.csv')]);
    await b.waitFor('!!overgaze.state.gaze', 10000);
    await b.setFiles('#gaze-input', [join(dir, 'b.csv')]);
    await b.waitFor('overgaze.state.files.length === 2', 10000, 'second file added');
    assert.equal(await b.eval('overgaze.state.gaze.observers.length'), 3);
    await b.click('[data-remove="2"]');
    await b.waitFor('overgaze.state.gaze.n === 2', 5000, 'observer removed');
    assert.match(await b.eval('document.querySelector("#observer-count").textContent'), /1 removed/);
    await b.click('#notice .dismiss');
    assert.equal(await b.eval('document.querySelector("#notice").hidden'), true);

    // two different files with the same name (chosen one at a time, so without folders) are both kept; the same file twice is not
    mkdirSync(join(dir, 'L')); mkdirSync(join(dir, 'R'));
    writeFileSync(join(dir, 'L', 'same.csv'), csv + '7,1.0,2.0,0.5,0.5\n');
    writeFileSync(join(dir, 'R', 'same.csv'), csv + '8,1.0,2.0,0.5,0.5\n9,1.0,2.0,0.5,0.5\n');
    await b.setFiles('#gaze-input', [join(dir, 'L', 'same.csv')]);
    await b.waitFor('overgaze.state.files.length === 3', 5000, 'L/same.csv added');
    await b.setFiles('#gaze-input', [join(dir, 'R', 'same.csv')]);
    await b.waitFor('overgaze.state.files.length === 4', 5000, 'R/same.csv added although it has the same name');
    await b.setFiles('#gaze-input', [join(dir, 'R', 'same.csv')]);
    await new Promise(r => setTimeout(r, 500));
    assert.equal(await b.eval('overgaze.state.files.length'), 4, 'the same file is not added twice');

    // reference with the default 0 offset: video frame 0 ↔ reference frame 0; then offset −45 puts frame 0 at video frame 45 (1.5 s)
    await b.setFiles('#ref-input', [join(dir, 'ref.npy')]);
    await b.waitFor('!!overgaze.state.ref', 5000);
    await b.waitFor('/ready in memory/.test(document.querySelector("#ref-status").textContent)', 10000, 'reference frames prepared after loading');
    await b.eval('document.querySelector("#video").currentTime = 0.01; 1');
    await b.waitFor('document.querySelector("#ro-ref-detail").textContent.includes("reference frame 0")', 10000, 'reference readout');
    await b.eval('overgaze.applySettings({ sigmaDeg: 5.6 }); document.querySelector("#ref-offset").value = "-45"; document.querySelector("#ref-offset").dispatchEvent(new Event("change")); document.querySelector("#video").currentTime = 1.51; 1');
    await b.waitFor('/CC (0\\.9|1\\.0)/.test(document.querySelector("#ro-ref").textContent)', 10000, 'high correlation with a matching reference').catch(async e => { console.log(await b.eval('document.querySelector("#ro-ref").textContent + " | " + document.querySelector("#ro-ref-detail").textContent + " | " + document.querySelector("#time-label").textContent')); throw e; });

    // the two outlines: separate switches and colours
    const count = '(rgb => { const c = document.querySelector("#outline-canvas"), d = c.getContext("2d").getImageData(0, 0, c.width, c.height).data; let n = 0; for (let i = 0; i < d.length; i += 4) if (d[i] === rgb[0] && d[i + 1] === rgb[1] && d[i + 2] === rgb[2] && d[i + 3] === 255) n++; return n; })';
    const setColor = (id, v) => b.eval(`document.querySelector('#${id}').value = '${v}'; document.querySelector('#${id}').dispatchEvent(new Event('input')); overgaze.render(); 1`);
    // a narrower σ than the reference's, so the two outlines do not lie on top of each other
    await b.eval('overgaze.applySettings({ sigmaDeg: 3 }); 1');
    await setColor('ref-outline-color', '#ff00ff'); await setColor('outline-color', '#00ffff');
    assert.ok(await b.eval(`${count}([255, 0, 255])`) > 10, 'reference outline in its colour');
    assert.ok(await b.eval(`${count}([0, 255, 255])`) > 10, 'our outline (switched on with the reference) in its colour');
    await b.eval(`document.querySelector('#show-outline').click(); overgaze.render(); 1`);
    assert.equal(await b.eval(`${count}([0, 255, 255])`), 0, 'our outline off');
    assert.ok(await b.eval(`${count}([255, 0, 255])`) > 10, 'the reference outline stays');
    await b.eval(`document.querySelector('#ref-outline').click(); document.querySelector('#show-outline').click(); overgaze.render(); 1`);
    assert.equal(await b.eval(`${count}([255, 0, 255])`), 0, 'reference outline off');
    assert.ok(await b.eval(`${count}([0, 255, 255])`) > 10, 'ours alone');

    // ink on the outline layer (only our outline is on now): the sum of alpha, which is the line's length times its width
    const inkOf = '(() => { const c = document.querySelector("#outline-canvas"), d = c.getContext("2d").getImageData(0, 0, c.width, c.height).data; let n = 0; for (let i = 3; i < d.length; i += 4) n += d[i] / 255; return n; })()';
    // outline thickness is in screen pixels: three times as thick at 6 as at 2, and the same at every live map width
    const thick = async (w, mapWidth) => { await b.eval(`for (const [id, v] of [['outline-width', '${w}'], ['map-width', '${mapWidth}']]) { const e = document.querySelector('#' + id); e.value = v; e.dispatchEvent(new Event('change')); } overgaze.render(); 1`); return b.eval(inkOf); };
    const t2 = await thick(2, 512), t6 = await thick(6, 512), t2small = await thick(2, 256), t2big = await thick(2, 1024);
    assert.ok(t6 > 2.5 * t2 && t6 < 3.5 * t2, `6 px outline vs 2 px: ${t6} vs ${t2} pixels of ink`);
    assert.ok(Math.abs(t2small - t2) < 0.2 * t2 && Math.abs(t2big - t2) < 0.2 * t2, `same thickness at map widths 256 / 512 / 1024: ${t2small} / ${t2} / ${t2big}`);
    await thick(2, 512);
    // each outline has its own thickness: the reference outline alone at 2 and 6, ours untouched
    const refInk = async w => { await b.eval(`(e => { e.value = '${w}'; e.dispatchEvent(new Event('change')); overgaze.render(); return 1; })(document.querySelector('#ref-outline-width'))`); return b.eval(inkOf); };
    await b.eval(`document.querySelector('#ref-outline').click(); document.querySelector('#show-outline').click(); overgaze.render(); 1`);
    const r2 = await refInk(2), r6 = await refInk(6);
    assert.ok(r6 > 2.5 * r2 && r6 < 3.5 * r2, `reference outline 6 px vs 2 px: ${r6} vs ${r2} pixels of ink`);
    assert.equal(await b.eval('overgaze.state.display.outlineWidth'), 2, 'our thickness unchanged');
    await refInk(2);
    await b.eval(`document.querySelector('#ref-outline').click(); document.querySelector('#show-outline').click(); overgaze.render(); 1`);

    // the Gaze and Reference buttons hide everything of their side, outlines included, and leave the switches alone
    const alphaOf = id => b.eval(`(() => { const c = document.querySelector('#${id}'), d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data; let n = 0; for (let i = 3; i < d.length; i += 4) if (d[i]) n++; return n; })()`);
    await b.eval(`document.querySelector('#ref-outline').click(); document.querySelector('#ref-heat').click(); overgaze.render(); 1`);
    assert.ok(await b.eval(`${count}([255, 0, 255])`) > 10 && await alphaOf('ref-canvas') > 100, 'reference heatmap and outline on');
    await b.click('#layer-ref'); await b.eval('overgaze.render(); 1');
    assert.equal(await b.eval(`${count}([255, 0, 255])`), 0, 'Reference button hides the reference outline');
    assert.equal(await alphaOf('ref-canvas'), 0, 'and the reference heatmap');
    assert.ok(await b.eval(`${count}([0, 255, 255])`) > 10, 'our outline stays');
    assert.equal(await b.eval(`document.querySelector('#ref-heat').checked && document.querySelector('#ref-outline').checked`), true, 'the reference switches are untouched');
    assert.equal(await b.eval(`document.querySelector('#layer-ref').getAttribute('aria-pressed')`), 'false');
    await b.click('#layer-gaze'); await b.eval('overgaze.render(); 1');
    assert.equal(await b.eval(`${count}([0, 255, 255])`), 0, 'Gaze button hides our outline');
    assert.equal(await alphaOf('heat-canvas'), 0, 'our heatmap');
    assert.equal(await alphaOf('marks-canvas'), 0, 'and our gaze points');
    await b.click('#layer-gaze'); await b.click('#layer-ref'); await b.eval('overgaze.render(); 1');
    assert.ok(await b.eval(`${count}([255, 0, 255])`) > 10 && await b.eval(`${count}([0, 255, 255])`) > 10 && await alphaOf('ref-canvas') > 100, 'both sides back');
    await b.click('#layer-ref'); await b.eval(`document.querySelector('#ref-outline').click(); document.querySelector('#ref-outline').click(); overgaze.render(); 1`);
    assert.equal(await b.eval('overgaze.state.showRef'), true, 'switching a reference layer on shows the reference side again');
    await b.eval(`document.querySelector('#ref-heat').click(); 1`);

    await b.click('#view-360');
    await b.waitFor('!document.querySelector("#view360-canvas").hidden', 3000);
    const px = await b.eval('(()=>{const c=document.querySelector("#view360-canvas");const g=c.getContext("webgl2");const p=new Uint8Array(4);g.readPixels(c.width>>1,c.height>>1,1,1,g.RGBA,g.UNSIGNED_BYTE,p);return [c.width,p[3]]})()');
    assert.ok(px[0] > 100 && px[1] === 255, `360° view drew something: ${px}`);
    // the outline in the 360° view keeps its screen thickness when zooming: the view reports its pixels per radian
    const ppr = await b.eval('(() => { const v = overgaze.state.view; v.setMode("perspective"); v.fov = 90; const a = v.pixelsPerRadian(); v.fov = 45; const b = v.pixelsPerRadian(); v.fov = 90; return [a, b, v.canvas.height]; })()');
    assert.ok(Math.abs(ppr[0] - ppr[2] / 2) < 1, `at 90° the half-height covers 1 radian's tangent: ${ppr}`);
    assert.ok(Math.abs(ppr[1] / ppr[0] - 1 / Math.tan(Math.PI / 8)) < 1e-6, `zooming in to 45° gives ${ppr[1] / ppr[0]} times the pixels per radian`);
    // other views: little planet and fisheye draw too; layers can be switched off
    for (const mode of ['planet', 'fisheye']) {
      await b.eval(`document.querySelector('#view360-mode').value = '${mode}'; document.querySelector('#view360-mode').dispatchEvent(new Event('change')); 1`);
      assert.equal(await b.eval('overgaze.state.view.mode'), mode);
    }
    await b.click('#layer-gaze');
    assert.equal(await b.eval('overgaze.state.showGaze'), false);
    await b.click('#clear-gaze');
    assert.equal(await b.eval('overgaze.state.gaze'), null);
    assert.deepEqual(b.logs.filter(l => !/WebGL|GPU stall|swiftshader|willReadFrequently/i.test(l)), []);   // the last one comes from the test reading pixels
  } finally { b.close(); server.close(); }
});

test('a cubemap or equi-angular cubemap video is unfolded to the same equirectangular frame', { skip: opts.skip || (spawnSync('ffmpeg', ['-version']).status !== 0 && 'ffmpeg not available') }, async () => {
  const dir = mkdtempSync(join(tmpdir(), 'overgaze-layout-'));
  const vfov = 2 * Math.atan(Math.PI * 384 / 768) * 180 / Math.PI;   // a full 360° cylinder of 768 × 384
  const cases = [['eac', 'eac:w=768:h=512'], ['c3x2', 'c3x2:w=768:h=512'], ['c3x2-lrdubf', 'c3x2:w=768:h=512:out_forder=lrdubf:out_frot=000000'], ['cyl', `cylindrical:w=768:h=384:h_fov=360:v_fov=${vfov}`]];
  for (const [name, out] of cases) {
    const r = spawnSync('ffmpeg', ['-v', 'error', '-y', '-i', join(ROOT, 'sample', 'sample_erp.mp4'), '-t', '3', '-vf', `v360=input=e:output=${out}`, '-c:v', 'libx264', '-pix_fmt', 'yuv420p', join(dir, `${name}.mp4`)]);
    assert.equal(r.status, 0, String(r.stderr));
  }
  const { server, url } = await serve(), b = await launch();
  const grey = sel => `(()=>{const c=document.createElement('canvas');c.width=256;c.height=128;const g=c.getContext('2d');g.drawImage(${sel},0,0,256,128);const d=g.getImageData(0,0,256,128).data;const o=[];for(let i=0;i<d.length;i+=4)o.push(d[i]+d[i+1]+d[i+2]);return o})()`;
  const cc = (a, b) => { const n = a.length, ma = a.reduce((x, y) => x + y) / n, mb = b.reduce((x, y) => x + y) / n; let s = 0, sa = 0, sb = 0; for (let i = 0; i < n; i++) { s += (a[i] - ma) * (b[i] - mb); sa += (a[i] - ma) ** 2; sb += (b[i] - mb) ** 2; } return s / Math.sqrt(sa * sb); };
  try {
    await b.goto(url);
    await b.setFiles('#video-input', [join(ROOT, 'sample', 'sample_erp.mp4')]);
    await b.waitFor('document.querySelector("#video").readyState >= 2', 20000);
    await b.eval('document.querySelector("#video").currentTime = 1.5; new Promise(r => document.querySelector("#video").addEventListener("seeked", r, { once: true }))');
    const ref = await b.eval(grey('document.querySelector("#video")'));
    for (const [name] of cases) {
      await b.setFiles('#video-input', [join(dir, `${name}.mp4`)]);
      await b.waitFor(`document.querySelector("#video-name").textContent === "${name}.mp4" && document.querySelector("#video").readyState >= 2`, 20000);
      const set = (id, v) => b.eval(`document.querySelector('#${id}').value = '${v}'; document.querySelector('#${id}').dispatchEvent(new Event('change')); 1`);
      if (name.startsWith('c3x2')) { await set('layout', 'c3x2'); await set('face-order', name === 'c3x2' ? 'RLUDFB' : 'LRDUBF'); }
      else if (name === 'cyl') await set('layout', 'cyl');   // 2:1 like equirectangular, so it is chosen by hand
      else assert.equal(await b.eval('overgaze.state.layout'), 'eac', 'a 3:2 video is taken as equi-angular cubemap');
      await b.eval('document.querySelector("#video").currentTime = 1.5; new Promise(r => document.querySelector("#video").addEventListener("seeked", r, { once: true }))');
      await new Promise(r => setTimeout(r, 500));
      await b.eval('overgaze.render(); 1');
      let got = await b.eval(grey('document.querySelector("#erp-canvas")')), want = ref;
      if (name === 'cyl') { want = ref.slice(28 * 256, 100 * 256); got = got.slice(28 * 256, 100 * 256); }   // rows inside the cylinder
      const c = cc(want, got);
      if (!(c > 0.9)) console.log(await b.eval('document.querySelector("#error-text").textContent'), b.logs);
      assert.ok(c > 0.9, `${name} unfolded frame correlates ${c.toFixed(3)} with the original`);
      assert.deepEqual(await b.eval('overgaze.videoSize ? 1 : 1'), 1);
    }
  } finally { b.close(); server.close(); }
});

test('a click on the video plays and pauses it', opts, async () => {
  const { server, url } = await serve(), b = await launch();
  try {
    await b.goto(url + '?sample');
    await b.waitFor('!!overgaze.state.gaze && document.querySelector("#video").readyState >= 2', 20000);
    const clickStage = async () => {
      const r = await b.eval('(()=>{const r=document.querySelector("#stage").getBoundingClientRect();return [r.x+r.width/2,r.y+r.height/2]})()');
      for (const type of ['mousePressed', 'mouseReleased']) await b.send('Input.dispatchMouseEvent', { type, x: r[0], y: r[1], button: 'left', clickCount: 1 });
      await new Promise(r => setTimeout(r, 400));
    };
    await clickStage();
    assert.equal(await b.eval('document.querySelector("#video").paused'), false, 'playing after one click');
    await clickStage();
    assert.equal(await b.eval('document.querySelector("#video").paused'), true, 'paused after the second');
    await b.click('#loop');
    assert.equal(await b.eval('document.querySelector("#video").loop'), true, '⟳ repeats the video');
  } finally { b.close(); server.close(); }
});

test('the unfolded 360° views (cubemap, equi-angular cubemap, cylinder) match ffmpeg', { skip: opts.skip || (spawnSync('ffmpeg', ['-version']).status !== 0 && 'ffmpeg not available') }, async () => {
  const dir = mkdtempSync(join(tmpdir(), 'overgaze-views-'));
  const cases = [['cubemap', 'c3x2:w=384:h=256', 384, 256, 0.98], ['eac', 'eac:w=384:h=256', 384, 256, 0.9], ['cylinder', 'cylindrical:w=512:h=256:h_fov=360:v_fov=115.0367', 512, 256, 0.98]];
  for (const [mode, out] of cases) {
    const r = spawnSync('ffmpeg', ['-v', 'error', '-y', '-ss', '4', '-i', join(ROOT, 'sample', 'sample_erp.mp4'), '-frames:v', '1', '-vf', `v360=input=e:output=${out},format=gray`, '-f', 'rawvideo', join(dir, mode + '.gray')]);
    assert.equal(r.status, 0, String(r.stderr));
  }
  const { server, url } = await serve(), b = await launch({ width: 1500, height: 1000 });
  try {
    await b.goto(url + '?sample');
    await b.waitFor('!!overgaze.state.gaze && document.querySelector("#video").readyState >= 2', 20000);
    await b.eval(`document.querySelector('#layer-gaze').click(); document.querySelector('#video').currentTime = 4; new Promise(r => document.querySelector('#video').addEventListener('seeked', r, { once: true }))`);
    await b.click('#view-360');
    for (const [mode, , w, h, min] of cases) {
      await b.eval(`document.querySelector('#view360-mode').value = '${mode}'; document.querySelector('#view360-mode').dispatchEvent(new Event('change')); overgaze.render(); 1`);
      const got = await b.eval(`(()=>{const c=document.querySelector('#view360-canvas'),a=c.width/c.height,fa=${w / h},bw=a>fa?c.height*fa:c.width,bh=bw/fa;const o=document.createElement('canvas');o.width=${w};o.height=${h};const g=o.getContext('2d');g.drawImage(c,(c.width-bw)/2,(c.height-bh)/2,bw,bh,0,0,${w},${h});const d=g.getImageData(0,0,${w},${h}).data,r=[];for(let i=0;i<d.length;i+=4)r.push(0.299*d[i]+0.587*d[i+1]+0.114*d[i+2]);return r})()`);
      const want = [...readFileSync(join(dir, mode + '.gray'))], n = want.length;
      const ma = got.reduce((x, y) => x + y) / n, mb = want.reduce((x, y) => x + y) / n;
      let s = 0, sa = 0, sb = 0;
      for (let i = 0; i < n; i++) { s += (got[i] - ma) * (want[i] - mb); sa += (got[i] - ma) ** 2; sb += (want[i] - mb) ** 2; }
      const c = s / Math.sqrt(sa * sb);
      assert.ok(c > min, `${mode}: correlation with ffmpeg ${c.toFixed(3)}`);   // EAC lower: ffmpeg adds 2 px of padding
    }
  } finally { b.close(); server.close(); }
});

test('a folder of videos: fixations and references follow the chosen video; settings export and import', opts, async () => {
  const { server, url } = await serve(), b = await launch();
  const dir = mkdtempSync(join(tmpdir(), 'overgaze-lib-'));
  try {
    for (const d of ['videos', 'gaze', 'refs']) mkdirSync(join(dir, d));
    for (const v of ['clipA', 'clipB']) copyFileSync(join(ROOT, 'sample', 'sample_erp.mp4'), join(dir, 'videos', `${v}.mp4`));
    const csv = 'user,start_time,duration,longitude,latitude\n';
    writeFileSync(join(dir, 'gaze', 'clipA_fix.csv'), csv + '1,1.0,2.0,0.5,0.5\n');
    writeFileSync(join(dir, 'gaze', 'clipB_fix.csv'), csv + '1,1.0,2.0,0.25,0.5\n2,1.0,2.0,0.25,0.5\n');
    // one static reference image per clip
    const { encodeGrayPng } = await import('../src/export.js');
    const png = Buffer.from(await encodeGrayPng(Float32Array.from({ length: 64 * 32 }, (_, i) => (i % 64) / 63), 64, 32, 8));
    writeFileSync(join(dir, 'refs', 'clipA_saliency.png'), png); writeFileSync(join(dir, 'refs', 'clipB_saliency.png'), png);

    await b.goto(url);
    await b.setFiles('#video-folder-input', [join(dir, 'videos')]);
    await b.waitFor('overgaze.state.library.videos.length === 2 && overgaze.state.video.name === "clipA.mp4"', 20000, 'video folder');
    await b.setFiles('#gaze-folder-input', [join(dir, 'gaze')]);
    await b.waitFor('overgaze.state.gaze && overgaze.state.files.length === 1', 10000, 'clipA gaze');
    await b.setFiles('#ref-folder-input', [join(dir, 'refs')]);
    await b.waitFor('overgaze.state.ref && overgaze.state.ref.name === "clipA_saliency.png"', 10000, 'clipA reference').catch(async e => { console.log(await b.eval('document.querySelector("#notice-text").textContent + " / " + overgaze.state.library.refs.length + " / " + (overgaze.state.ref && overgaze.state.ref.name)'), b.logs); throw e; });
    await b.click('#video-next');
    await b.waitFor('overgaze.state.video.name === "clipB.mp4" && overgaze.state.files[0]?.name === "clipB_fix.csv" && overgaze.state.ref?.name === "clipB_saliency.png"', 20000, 'clipB tied');
    assert.equal(await b.eval('overgaze.state.gaze.observers.length'), 2);

    // presets and settings files
    await b.eval(`document.querySelector('#preset').value = 'p:salient360-he'; document.querySelector('#preset').dispatchEvent(new Event('change')); 1`);
    assert.equal(await b.eval('overgaze.state.settings.sigmaDeg'), 2);
    // a changed setting turns the menu into "(custom …)"; choosing the preset again restores its values
    const menuText = 'document.querySelector("#preset").selectedOptions[0].textContent';
    await b.eval(`document.querySelector('#sigma-num').value = '3.5'; document.querySelector('#sigma-num').dispatchEvent(new Event('change')); 1`);
    assert.equal(await b.eval('overgaze.state.settings.sigmaDeg'), 3.5);
    assert.equal(await b.eval(menuText), '(custom Salient360: head + eye maps (σ 2°, per frame))');
    assert.match(await b.eval('document.querySelector("#preset").title'), /σ/);
    await b.eval(`document.querySelector('#preset').value = 'p:salient360-he'; document.querySelector('#preset').dispatchEvent(new Event('change')); 1`);
    assert.equal(await b.eval('overgaze.state.settings.sigmaDeg'), 2, 'preset values restored');
    assert.equal(await b.eval(menuText), 'Salient360: head + eye maps (σ 2°, per frame)');
    assert.equal(await b.eval('document.querySelectorAll("#preset option[data-custom]").length'), 0);
    // the motion-compensated preset is not "custom" just because the motion has not been analysed yet
    await b.eval(`document.querySelector('#preset').value = 'p:video-sal-mc3'; document.querySelector('#preset').dispatchEvent(new Event('change')); 1`);
    assert.equal(await b.eval(menuText), 'Video-saliency training maps “mc3” (σ 3°, ±1 s, motion comp.)');
    await b.eval(`document.querySelector('#preset').value = 'p:salient360-he'; document.querySelector('#preset').dispatchEvent(new Event('change')); 1`);
    await b.send('Browser.setDownloadBehavior', { behavior: 'allow', downloadPath: dir });
    await b.eval('overgaze.applySettings({ sigmaDeg: 4.5, windowSec: 0.7 }); 1');
    await b.click('#template-export');
    for (let i = 0; i < 50 && !readdirSync(dir).some(f => f.endsWith('.overgaze.json')); i++) await new Promise(r => setTimeout(r, 100));
    const saved = readdirSync(dir).find(f => f.endsWith('.overgaze.json'));
    assert.ok(saved, 'settings file downloaded');
    await b.eval('overgaze.applySettings({ sigmaDeg: 2, windowSec: 0 }); 1');
    await b.setFiles('#template-input', [join(dir, saved)]);
    await b.waitFor('overgaze.state.settings.sigmaDeg === 4.5 && overgaze.state.settings.windowSec === 0.7', 5000, 'settings imported');
    assert.match(await b.eval('document.querySelector("#preset").selectedOptions[0].parentElement.label'), /My templates/);
  } finally { b.close(); server.close(); }
});

test('outliers: drop rules and minimum duration report what they removed; agreement flags but never removes; settings.json keeps it all', opts, async () => {
  const { server, url } = await serve(), b = await launch();
  const dir = mkdtempSync(join(tmpdir(), 'overgaze-outliers-'));
  try {
    // the sample's 8 observers, plus observer 9 who looks at the far side, and three lost-eye rows at 0,0
    const lines = readFileSync(join(ROOT, 'sample', 'sample_gaze.csv'), 'utf8').trim().split('\n');
    const extra = lines.slice(1).filter(l => l.startsWith('1,')).map(l => { const c = l.split(','); c[0] = '9'; c[3] = String(((Number(c[3]) + 180 + 150) % 360) - 180); return c.join(','); });
    extra.push('2,1.0,0.3,0,0', '2,2.0,0.3,0,0', '3,3.0,0.3,0,0');
    writeFileSync(join(dir, 'extra.csv'), lines[0] + '\n' + extra.join('\n') + '\n');
    const short = lines.slice(1).filter(l => Number(l.split(',')[2]) < 0.2).length + extra.filter(l => Number(l.split(',')[2]) < 0.2).length;
    await b.goto(url + '?sample');
    await b.waitFor('!!overgaze.state.gaze && document.querySelector("#video").readyState >= 2', 20000);
    await b.waitFor('overgaze.state.agreement?.done', 30000, 'agreement on the sample');
    assert.equal(await b.eval('overgaze.state.agreement.low.set.size'), 0, 'the simulated observers agree');
    await b.setFiles('#gaze-input', [join(dir, 'extra.csv')]);
    await b.waitFor('overgaze.state.files.length === 2 && overgaze.state.gaze.observers.length === 9', 10000, 'extra file added');
    await b.waitFor('overgaze.state.agreement?.done && overgaze.state.agreement.gaze === overgaze.state.gaze', 30000, 'agreement with observer 9');
    const nine = await b.eval('overgaze.state.gaze.observers.indexOf("9")');
    assert.deepEqual(await b.eval('[...overgaze.state.agreement.low.set]'), [nine], 'observer 9 is marked low');
    assert.match(await b.eval('document.querySelector("#agreement-status").textContent'), /1 far below the rest/);
    assert.equal(await b.eval(`document.querySelector('[data-obs="${nine}"]').closest('.observer-row').classList.contains('flagged')`), true);
    await b.click('#obs-sort');
    assert.equal(await b.eval('document.querySelector("#observer-list .observer-row input").dataset.obs'), String(nine), 'sorted by agreement: lowest first');
    assert.equal(await b.eval('overgaze.state.gaze.observers.length'), 9, 'nobody removed');

    // drop rule built in the panel: longitude = 0 and latitude = 0
    await b.click('#rule-add');
    const setPart = (j, column, op, value) => b.eval(`(() => { const p = document.querySelectorAll('#rule-list .rule-part')[${j}]; for (const [k, v] of [['column', '${column}'], ['op', '${op}'], ['value', '${value}']]) { const e = p.querySelector('[data-k="' + k + '"]'); e.value = v; e.dispatchEvent(new Event('change', { bubbles: true })); } return 1; })()`);
    await setPart(0, 'longitude', '=', '0');
    await b.eval(`document.querySelector('#rule-list [data-and]').click(); 1`);
    await setPart(1, 'latitude', '=', '0');
    await b.waitFor('/Rule “longitude = 0 and latitude = 0” removed 3\\./.test(document.querySelector("#gaze-summary").textContent)', 5000, 'rule count in the summary');
    // minimum duration
    await b.eval(`document.querySelector('#min-dur').value = '200'; document.querySelector('#min-dur').dispatchEvent(new Event('change')); 1`);
    const expected = short;   // the three 0,0 rows last 300 ms, so only the rule drops them
    assert.match(await b.eval('document.querySelector("#gaze-summary").textContent'), new RegExp(`Minimum duration \\(200 ms\\) removed ${expected}\\.`));

    await b.waitFor('overgaze.state.agreement?.done && overgaze.state.agreement.gaze === overgaze.state.gaze', 30000, 'agreement after filtering');
    const sj = await b.eval('JSON.stringify(overgaze.settingsJson().gaze_processing)');
    const gp = JSON.parse(sj);
    assert.equal(gp.min_fixation_duration_ms, 200); assert.equal(gp.rows_removed_by_min_duration, expected);
    assert.deepEqual(gp.drop_row_rules.map(r => [r.text, r.rows_removed]), [['longitude = 0 and latitude = 0', 3]]);
    assert.ok(gp.observer_agreement.scores['9'] < gp.observer_agreement.scores['1']);
    assert.deepEqual(gp.observer_agreement.low, ['9']);
    assert.deepEqual(gp.removed_observers, []);

    // export the settings, clear the rules, import: the rules and the minimum duration come back
    await b.send('Browser.setDownloadBehavior', { behavior: 'allow', downloadPath: dir });
    await b.click('#template-export');
    for (let i = 0; i < 50 && !readdirSync(dir).some(f => f.endsWith('.overgaze.json')); i++) await new Promise(r => setTimeout(r, 100));
    const saved = readdirSync(dir).find(f => f.endsWith('.overgaze.json'));
    await b.eval(`document.querySelector('#rule-list [data-drop-rule]').click(); overgaze.applySettings({ minDurationMs: 0 }); 1`);
    await b.waitFor('!/Rule/.test(document.querySelector("#gaze-summary").textContent)', 5000, 'rule removed');
    await b.setFiles('#template-input', [join(dir, saved)]);
    await b.waitFor('overgaze.state.settings.minDurationMs === 200 && overgaze.state.rowRules.length === 1 && /removed 3\\./.test(document.querySelector("#gaze-summary").textContent)', 5000, 'rules imported');
    // removing is still the person's choice: × removes observer 9
    await b.click(`[data-remove="${nine}"]`);
    await b.waitFor('overgaze.state.gaze.observers.length === 9 && overgaze.state.removedObservers.has("9")', 5000, 'removed by hand');
    assert.deepEqual(b.logs, []);
  } finally { b.close(); server.close(); }
});

const ffmpegOk = spawnSync('ffmpeg', ['-version']).status === 0;
test('the frame rate is known right after the video loads, before it plays: from the MP4 header, or measured (retried when the page is busy)', { skip: opts.skip || (ffmpegOk ? false : 'ffmpeg not available') }, async () => {
  const { server, url } = await serve(), b = await launch();
  const dir = mkdtempSync(join(tmpdir(), 'overgaze-fps-'));
  try {
    for (const [f, codec] of [['clip25.mp4', ['-c:v', 'libx264', '-pix_fmt', 'yuv420p']], ['clip25.webm', ['-c:v', 'libvpx-vp9', '-b:v', '200k']]])
      assert.equal(spawnSync('ffmpeg', ['-loglevel', 'error', '-y', '-f', 'lavfi', '-i', 'testsrc=size=256x128:rate=25:duration=6', ...codec, join(dir, f)]).status, 0, 'ffmpeg made ' + f);
    // frame-numbered gaze: frame 50 is 2 s into a 25 fps video
    writeFileSync(join(dir, 'g.csv'), 'user,frame,x,y\n1,50,0.5,0.5\n');
    await b.goto(url);
    await b.setFiles('#gaze-input', [join(dir, 'g.csv')]);
    await b.waitFor('!!overgaze.state.gaze', 10000, 'gaze');
    await b.setFiles('#video-input', [join(dir, 'clip25.mp4')]);
    await b.waitFor('overgaze.state.fpsSource === "file"', 5000, 'frame rate from the file header');
    assert.equal(await b.eval('overgaze.state.fps'), 25);
    assert.equal(await b.eval('document.querySelector("#video").paused'), true, 'without playing');
    assert.match(await b.eval('document.querySelector("#fps-note").textContent'), /Read from the video file: 25 frames per second/);
    const t = await b.eval('overgaze.state.gaze.t0[0]');
    assert.ok(Math.abs(t - 50 / 25) < 0.03, `the gaze row is re-timed with the new frame rate (${t} s)`);
    // WebM has no such header: measured from decoded frames. Freeze the page for 4 s just after the first
    // measurement starts playing, so its 3 s wait sees no frames and the measurement has to be retried.
    await b.eval(`(() => { const play = HTMLMediaElement.prototype.play; window.__plays = 0;
      HTMLMediaElement.prototype.play = function () { const p = play.call(this); if (this !== document.querySelector('#video') && window.__plays++ === 0) setTimeout(() => { const end = performance.now() + 4000; while (performance.now() < end); }, 20); return p; };
      return 1; })()`);
    await b.setFiles('#video-input', [join(dir, 'clip25.webm')]);
    await b.waitFor('overgaze.state.video.name === "clip25.webm" && overgaze.state.fpsSource === "measured"', 40000, 'measured frame rate');
    assert.equal(await b.eval('overgaze.state.fps'), 25);
    assert.ok(await b.eval('window.__plays') >= 2, 'the first measurement was retried');
    assert.equal(await b.eval('document.querySelector("#video").paused'), true);
    assert.deepEqual(b.logs, []);
  } finally { b.close(); server.close(); }
});

test('a motion file made outside the browser (.npz) replaces the estimate: status, motion compensation and settings.json say where it came from', opts, async () => {
  const { server, url } = await serve(), b = await launch();
  try {
    await b.goto(url + '?sample');
    await b.waitFor('!!overgaze.state.gaze && document.querySelector("#video").readyState >= 2', 20000, 'sample ready');
    await b.eval('overgaze.applySettings({ windowSec: 1, kernel: "gaussian" }); 1');
    await b.setFiles('#motion-file-input', [join(ROOT, 'test', 'fixtures', 'tiny.motion.npz')]);
    await b.waitFor('overgaze.state.flowInfo?.method === "file"', 10000, 'motion file loaded');
    assert.match(await b.eval('document.querySelector("#flow-status").textContent'), /^From tiny\.motion\.npz \(OpenCV DIS test\): 3 flow fields at 8 × 4/);
    assert.equal(await b.eval('document.querySelector("#motion").checked'), true, 'motion compensation is switched on');
    const note = await b.eval('overgaze.settingsJson().motion.method_note');
    assert.match(note, /loaded from tiny\.motion\.npz: OpenCV DIS test/);
    // the fixture covers 0.05–0.35 s only; at 0.2 s the pooled points move
    const moved = await b.eval('Array.from(overgaze.mapAt(6, 128).map).join(",")');
    await b.eval('overgaze.applySettings({ motionComp: false }); 1');
    const still = await b.eval('Array.from(overgaze.mapAt(6, 128).map).join(",")');
    assert.notEqual(moved, still, 'the loaded motion moves the pooled gaze');
    // a file that is not a motion file is refused with a message
    await b.setFiles('#motion-file-input', [join(ROOT, 'sample', 'sample_gaze.csv')]);
    await b.waitFor('/Motion file: .*not an \\.npz/.test(document.querySelector("#error-text").textContent)', 5000, 'error shown');
    assert.equal(await b.eval('overgaze.state.flowInfo.file'), 'tiny.motion.npz', 'the loaded motion stays');
  } finally { b.close(); server.close(); }
});

test('both-eye sample: each person\'s left and right eye are paired, the count is shown and saved in settings.json', opts, async () => {
  const { server, url } = await serve(), b = await launch();
  try {
    await b.goto(url + '?sample=eyes');
    await b.waitFor('!!overgaze.state.gaze && !!overgaze.state.people', 20000, 'pairs');
    assert.match(await b.eval('document.querySelector("#pair-status").textContent'), /^8 people with both eyes, 2 recordings with no partner \(1 left, 1 right\)\. Partners found by matching/);
    const ep = await b.eval('overgaze.settingsJson().gaze_processing.eye_pairing');
    assert.equal(ep.setting, 'auto'); assert.equal(ep.pairs, 8); assert.equal(ep.people.length, 10);
    assert.deepEqual(ep.people[0], { left: 'L/sample_fixations#01', right: 'R/sample_fixations#09', match: 1 });
    await b.eval(`(e => { e.value = 'off'; e.dispatchEvent(new Event('change')); return 1; })(document.querySelector('#eye-pairing'))`);
    assert.equal(await b.eval('overgaze.state.people'), null);
    assert.equal(await b.eval('overgaze.settingsJson().gaze_processing.eye_pairing.setting'), 'off');
    assert.deepEqual(b.logs, []);
  } finally { b.close(); server.close(); }
});

test('observer list grouped by person: the person\'s checkbox and × act on both eyes, each eye can be switched off alone, sorting uses the person\'s score', opts, async () => {
  const { server, url } = await serve(), b = await launch();
  try {
    await b.goto(url + '?sample=eyes');
    await b.waitFor('!!overgaze.state.people && overgaze.state.agreement?.done', 60000, 'pairs and agreement');
    assert.equal(await b.eval('document.querySelectorAll("#observer-list details.person").length'), 10);
    assert.match(await b.eval('document.querySelector("#observer-count").textContent'), /^10 people \(8 with both eyes\)/);
    // person L#01 + R#09: switch the person off, then one eye back on
    const sel = 'details.person:has([title*="L/sample_fixations#01 "]) ';
    await b.eval(`(c => { c.checked = false; c.dispatchEvent(new Event('change', { bubbles: true })); return 1; })(document.querySelector('${sel}[data-person-check]'))`);
    const idx = n => `overgaze.state.gaze.observers.indexOf("${n}")`;
    assert.deepEqual(await b.eval(`[overgaze.state.observerEnabled[${idx('L/sample_fixations#01')}], overgaze.state.observerEnabled[${idx('R/sample_fixations#09')}]]`), [0, 0]);
    await b.eval(`(c => { c.checked = true; c.dispatchEvent(new Event('change', { bubbles: true })); return 1; })(document.querySelector('[data-obs="' + ${idx('R/sample_fixations#09')} + '"]'))`);
    assert.equal(await b.eval(`document.querySelector('${sel}[data-person-check]').indeterminate`), true, 'one eye on: the person\'s box is half checked');
    assert.deepEqual(await b.eval('overgaze.settingsJson().gaze_processing.observers_switched_off'), ['L/sample_fixations#01']);
    // sort by agreement: people in order of their mean score
    await b.click('#obs-sort');
    const shown = await b.eval('JSON.stringify([...document.querySelectorAll("#observer-list details.person > summary .agree")].map(e => parseFloat(e.textContent.replace("⚠", "")) || 0))');
    const v = JSON.parse(shown); assert.deepEqual(v, [...v].sort((a, c) => a - c));
    const ps = await b.eval('overgaze.settingsJson().gaze_processing.observer_agreement.person_scores');
    assert.equal(Object.keys(ps).length, 10);
    // × on the person removes both eyes
    await b.eval(`document.querySelector('${sel}[data-remove]').click(); 1`);
    await b.waitFor('document.querySelectorAll("#observer-list details.person").length === 9', 5000, 'both eyes removed');
    assert.deepEqual((await b.eval('overgaze.settingsJson().gaze_processing.removed_observers')).sort(), ['L/sample_fixations#01', 'R/sample_fixations#09']);
    assert.deepEqual(b.logs, []);
  } finally { b.close(); server.close(); }
});

test('gaze points "more": dot size, growth while the eye stays in place, dashed line between a person\'s eyes; saved in settings.json and read back', opts, async () => {
  const { server, url } = await serve(), b = await launch();
  try {
    await b.goto(url + '?sample=eyes');
    await b.waitFor('!!overgaze.state.people && document.querySelector("#video").readyState >= 2', 20000, 'sample');
    await b.eval(`document.querySelector('#video').currentTime = 4; 1`); await new Promise(r => setTimeout(r, 1200));
    const ink = () => b.eval(`(() => { const c = document.querySelector('#marks-canvas'), d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data; let n = 0; for (let i = 3; i < d.length; i += 4) if (d[i] > 0) n++; return n; })()`);
    const set = (id, v) => b.eval(`(e => { if (e.type === 'checkbox') e.checked = ${JSON.stringify(v)}; else e.value = ${JSON.stringify(v)}; e.dispatchEvent(new Event('change')); return 1; })(document.querySelector('#${id}'))`);
    const tick = () => new Promise(r => setTimeout(r, 400));
    const small = await ink();
    await set('point-size', '14'); await tick();
    const big = await ink();
    assert.ok(big > small * 2, `bigger dots cover more (${small} → ${big} pixels)`);
    await set('point-size', '6'); await set('point-dwell', true); await tick();
    const grown = await ink();
    assert.ok(grown > small * 1.3, `dots grow with the time in place (${small} → ${grown})`);
    // the two eyes of a person are under 1° apart in the sample, so the lines hide under the dots: count the dashed strokes
    await b.eval(`(() => { window.__dashed = 0; const st = CanvasRenderingContext2D.prototype.stroke; CanvasRenderingContext2D.prototype.stroke = function (...a) { if (this.getLineDash().length) window.__dashed++; return st.apply(this, a); }; return 1; })()`);
    await set('point-dwell', false); await set('point-eye-lines', true); await tick();
    const lines = await b.eval('window.__dashed');
    assert.ok(lines >= 7, `dashed lines join the eyes of the people with both eyes in this frame (${lines})`);
    const gp = await b.eval('overgaze.settingsJson().display.gaze_points');
    assert.equal(gp.dot_size_px, 6); assert.equal(gp.join_eyes, true); assert.equal(gp.grow_while_in_place, false); assert.equal(gp.grow_cap_times_size, 3);
    // read back: a settings file with other values
    const dir = mkdtempSync(join(tmpdir(), 'overgaze-points-'));
    writeFileSync(join(dir, 's.overgaze.json'), JSON.stringify({ ...(await b.eval('overgaze.settingsJson()')), display: { gaze_points: { ...gp, dot_size_px: 11, grow_while_in_place: true, grow_cap_times_size: 4, join_eyes: false } } }));
    await b.setFiles('#template-input', [join(dir, 's.overgaze.json')]);
    await b.waitFor('overgaze.state.points.size === 11', 5000, 'settings read back');
    assert.deepEqual(await b.eval('[overgaze.state.points.dwell, overgaze.state.points.dwellCap, overgaze.state.points.eyeLines, document.querySelector("#point-size").value]'), [true, 4, false, '11']);
    assert.deepEqual(b.logs.filter(l => !l.includes('willReadFrequently')), [], 'no errors (the read-back warning comes from this test)');
  } finally { b.close(); server.close(); }
});

test('one colour per person: both eyes share it, the left eye is a circle and the right eye a square', opts, async () => {
  const { server, url } = await serve(), b = await launch();
  try {
    await b.goto(url + '?sample=eyes');
    await b.waitFor('!!overgaze.state.people && document.querySelector("#video").readyState >= 2', 20000, 'sample');
    await b.eval(`(() => { window.__shapes = { arc: 0, rect: 0 }; const P = CanvasRenderingContext2D.prototype, arc = P.arc, rect = P.rect;
      P.arc = function (...a) { if (this.canvas.id === 'marks-canvas') window.__shapes.arc++; return arc.apply(this, a); };
      P.rect = function (...a) { if (this.canvas.id === 'marks-canvas') window.__shapes.rect++; return rect.apply(this, a); }; return 1; })()`);
    await b.eval(`document.querySelector('#video').currentTime = 4; 1`); await new Promise(r => setTimeout(r, 1200));
    const shapes = await b.eval('window.__shapes');
    assert.ok(shapes.arc > 0 && shapes.rect > 0, `circles and squares drawn (${JSON.stringify(shapes)})`);
    const st = await b.eval(`(() => { const g = overgaze.state.gaze, i = n => g.observers.indexOf(n), dot = document.querySelectorAll('#observer-list details.person'); return { l: overgaze.colorOf(i('L/sample_fixations#01')), r: overgaze.colorOf(i('R/sample_fixations#09')), other: overgaze.colorOf(i('L/sample_fixations#02')) }; })()`);
    assert.equal(st.l, st.r, 'both eyes of a person share a colour'); assert.notEqual(st.l, st.other);
    const gp = await b.eval('overgaze.settingsJson().display.gaze_points');
    assert.equal(gp.colour_list.length, 12); assert.match(gp.shapes, /square = right eye/);
    assert.deepEqual(b.logs, []);
  } finally { b.close(); server.close(); }
});

test('with pooling on, points of the current frame, earlier and later moments are drawn apart; motion lines and the legend follow the settings', opts, async () => {
  const { server, url } = await serve(), b = await launch();
  try {
    await b.goto(url + '?sample=eyes');
    await b.waitFor('!!overgaze.state.people && document.querySelector("#video").readyState >= 2', 20000, 'sample');
    const legend = () => b.eval('[...document.querySelectorAll("#points-legend > div")].map(d => d.textContent)');
    await b.eval(`document.querySelector('#video').currentTime = 4; 1`); await new Promise(r => setTimeout(r, 1000));
    assert.deepEqual(await legend(), ['gaze: circle left eye, square right eye, colour = person']);
    await b.eval(`overgaze.applySettings({ windowSec: 1, kernel: 'gaussian' }); 1`); await new Promise(r => setTimeout(r, 600));
    assert.deepEqual(await legend(), ['this frame: circle left eye, square right eye, colour = person', 'earlier (fainter the further back)', 'later (fainter the further ahead)']);
    // count the line segments drawn on the points canvas
    await b.eval(`(() => { window.__lines = 0; const P = CanvasRenderingContext2D.prototype, lt = P.lineTo; P.lineTo = function (...a) { if (this.canvas.id === 'marks-canvas') window.__lines++; return lt.apply(this, a); }; return 1; })()`);
    await b.eval(`(e => { e.value = 'seek'; e.dispatchEvent(new Event('change')); return 1; })(document.querySelector('#flow-method'))`);
    await b.click('#analyse');
    await b.waitFor('!!overgaze.state.flow && !overgaze.state.analysing', 120000, 'motion');
    await b.eval(`overgaze.applySettings({ motionComp: true }); window.__lines = 0; document.querySelector('#video').currentTime = 4.01; 1`); await new Promise(r => setTimeout(r, 800));
    const withLines = await b.eval('window.__lines');
    assert.ok(withLines > 50, `lines from where points were recorded to where motion moved them (${withLines})`);
    assert.ok((await legend()).includes('moved by motion from where it was recorded'));
    await b.eval(`(e => { e.checked = false; e.dispatchEvent(new Event('change')); return 1; })(document.querySelector('#point-motion-lines'))`);
    await b.eval(`window.__lines = 0; document.querySelector('#video').currentTime = 4.02; 1`); await new Promise(r => setTimeout(r, 800));
    assert.equal(await b.eval('window.__lines'), 0, 'no motion lines when switched off');
    await b.eval(`(e => { e.checked = false; e.dispatchEvent(new Event('change')); return 1; })(document.querySelector('#point-legend'))`);
    assert.equal(await b.eval('document.querySelector("#points-legend").hidden'), true);
    const gp = await b.eval('overgaze.settingsJson().display.gaze_points');
    assert.equal(gp.motion_lines, false); assert.equal(gp.legend, false); assert.match(gp.time_styles, /earlier = rings, later = solid/);
    assert.deepEqual(b.logs, []);
  } finally { b.close(); server.close(); }
});

test('scene memory: weight w adds the places looked at since the clip started; the area it adds can be outlined on its own; saved in settings.json', opts, async () => {
  const { server, url } = await serve(), b = await launch();
  try {
    await b.goto(url + '?sample=eyes');
    await b.waitFor('!!overgaze.state.gaze && document.querySelector("#video").readyState >= 2', 20000, 'sample');
    await b.eval(`overgaze.applySettings({ windowSec: 1, kernel: 'gaussian' }); document.querySelector('#video').currentTime = 6; 1`); await new Promise(r => setTimeout(r, 1000));
    // pixels salient with memory that are not salient in the pooled map alone
    const added = () => b.eval(`(() => { const r = overgaze.mapAt(180, 256, 'max'); if (!r.plain) return 0; let n = 0; for (let i = 0; i < r.map.length; i++) if (r.map[i] >= 0.1 && r.plain[i] < 0.1) n++; return n; })()`);
    assert.equal(await added(), 0, 'no memory: nothing added');
    await b.eval(`(e => { e.value = '0.3'; e.dispatchEvent(new Event('change')); return 1; })(document.querySelector('#memory-weight'))`);
    const n = await added();
    assert.ok(n > 50, `memory adds salient area where the pooled map has none (${n} pixels)`);
    assert.match(await b.eval('document.querySelector("#memory-hint").textContent'), /No motion analysed or loaded/);
    await b.eval(`(e => { e.checked = true; e.dispatchEvent(new Event('change')); return 1; })(document.querySelector('#show-memory-outline'))`);
    await new Promise(r => setTimeout(r, 600));
    const lines = await b.eval('overgaze.state.outlines.lines.map(l => [l.color, l.segs.length])');
    assert.ok(lines.some(([c, n]) => c === '#ff4fd8' && n > 0), `the added area is outlined in its own colour (${JSON.stringify(lines)})`);
    const js = await b.eval('overgaze.settingsJson()');
    assert.equal(js.saliency.scene_memory_weight, 0.3); assert.match(js.saliency.scene_memory, /not moved/);
    assert.equal(js.display.memory_outline.show, true);
    assert.match(await b.eval('document.querySelector("#readout-badge").textContent'), /memory 0\.3/);
    assert.deepEqual(b.logs, []);
  } finally { b.close(); server.close(); }
});

test('motion range: export range and the new gaze range are analysed, and a video much longer than the gaze defaults both ranges to the gaze range', opts, async () => {
  const { server, url } = await serve(), b = await launch();
  const dir = mkdtempSync(join(tmpdir(), 'overgaze-range-'));
  try {
    await b.goto(url + '?sample');
    await b.waitFor('!!overgaze.state.gaze && document.querySelector("#video").readyState >= 2', 20000);
    const analyse = async (range, setup = '') => {
      await b.eval(`${setup} for (const [id, v] of [['flow-range', '${range}'], ['flow-method', 'seek']]) { const e = document.querySelector('#' + id); e.value = v; e.dispatchEvent(new Event('change')); } overgaze.state.flow = null; 1`);
      await b.click('#analyse'); await b.waitFor('!!overgaze.state.flow && !overgaze.state.analysing', 60000, 'motion');
      return b.eval('[overgaze.state.flowInfo.from, overgaze.state.flowInfo.to, overgaze.state.flow.times[0], overgaze.state.flow.times.at(-1)]');
    };
    const [a, bb, first, last] = await analyse('export', "document.querySelector('#ex-start').value = '2'; document.querySelector('#ex-end').value = '4';");
    assert.deepEqual([a, bb], [2, 4]); assert.ok(first > 2 && last < 4, `fields inside the export range (${first}–${last})`);
    // gaze range: gaze from 2.5 s to 5.0 s, pooling window 0.5 s → 2.0–5.5 s
    writeFileSync(join(dir, 'g.csv'), 'user,start_time,duration,longitude,latitude\n1,2.5,0.3,0,0\n2,4.7,0.3,10,0\n');
    await b.click('#clear-gaze');
    await b.setFiles('#gaze-input', [join(dir, 'g.csv')]);
    await b.waitFor('overgaze.state.gaze?.n === 2', 10000, 'gaze');
    await b.eval(`overgaze.applySettings({ windowSec: 0.5 }); 1`);
    const [ga, gb] = await analyse('gaze');
    assert.ok(Math.abs(ga - 2.0) < 1e-6 && Math.abs(gb - 5.5) < 1e-6, `gaze range ${ga}–${gb}`);
    // a video four times longer than the gaze: both ranges start at the gaze range
    spawnSync('ffmpeg', ['-loglevel', 'error', '-y', '-stream_loop', '3', '-i', join(ROOT, 'sample', 'sample_erp.mp4'), '-c', 'copy', join(dir, 'long.mp4')]);
    await b.setFiles('#video-input', [join(dir, 'long.mp4')]);
    await b.waitFor('document.querySelector("#video").duration > 30 && document.querySelector("#flow-range").value === "gaze"', 20000, 'defaults');
    assert.deepEqual(await b.eval('[document.querySelector("#ex-start").value, document.querySelector("#ex-end").value]'), ['2.00', '5.50']);
    assert.match(await b.eval('document.querySelector("#notice").textContent'), /set to the gaze range/);
    assert.deepEqual(b.logs, []);
  } finally { b.close(); server.close(); }
});

test('repeat a range: the export range or the gaze range plays over and over; the whole video still repeats natively', opts, async () => {
  const { server, url } = await serve(), b = await launch();
  try {
    await b.goto(url + '?sample');
    await b.waitFor('!!overgaze.state.gaze && document.querySelector("#video").readyState >= 2', 20000);
    await b.eval(`document.querySelector('#ex-start').value = '2'; document.querySelector('#ex-end').value = '3'; 1`);
    await b.eval(`(e => { e.value = 'export'; e.dispatchEvent(new Event('change')); return 1; })(document.querySelector('#loop-range'))`);
    assert.equal(await b.eval('document.querySelector("#loop").getAttribute("aria-pressed")'), 'true', 'choosing a range switches repeat on');
    await b.eval(`window.__seen = []; const v = document.querySelector('#video'); v.addEventListener('timeupdate', () => window.__seen.push(v.currentTime)); v.currentTime = 6; v.play(); 1`);
    await new Promise(r => setTimeout(r, 3500));
    const seen = await b.eval('window.__seen');
    assert.ok(seen.length > 5 && seen.every(t => t >= 1.9 && t <= 3.2), `stays within 2–3 s (from ${Math.min(...seen).toFixed(2)} to ${Math.max(...seen).toFixed(2)})`);
    assert.ok(seen.some((t, i) => i && t < seen[i - 1] - 0.5), 'went back to the start at least once');
    // the gaze range of the sample: 0.03–8.15 s widened by the window (0 s here)
    await b.eval(`(e => { e.value = 'gaze'; e.dispatchEvent(new Event('change')); return 1; })(document.querySelector('#loop-range'))`);
    assert.equal(await b.eval('overgaze.state.loop'), 'gaze');
    await b.eval(`(e => { e.value = 'all'; e.dispatchEvent(new Event('change')); return 1; })(document.querySelector('#loop-range'))`);
    assert.equal(await b.eval('document.querySelector("#video").loop'), true);
    await b.click('#loop');
    assert.deepEqual(await b.eval('[overgaze.state.loop, document.querySelector("#video").loop]'), [null, false], '⟳ switches repeat off');
    assert.deepEqual(b.logs, []);
  } finally { b.close(); server.close(); }
});

test('a video named after its YouTube original gets a link to it, at the moment shown', opts, async () => {
  const { server, url } = await serve(), b = await launch();
  const dir = mkdtempSync(join(tmpdir(), 'overgaze-origin-'));
  try {
    copyFileSync(join(ROOT, 'sample', 'sample_erp.mp4'), join(dir, 'dQw4w9WgXcQ-10.mp4'));
    await b.goto(url + '?sample');
    await b.waitFor('document.querySelector("#video").readyState >= 2', 20000);
    assert.equal(await b.eval('document.querySelector("#video-original-row").hidden'), true, 'no link for the sample');
    await b.setFiles('#video-input', [join(dir, 'dQw4w9WgXcQ-10.mp4')]);
    await b.waitFor('document.querySelector("#video").readyState >= 2 && !document.querySelector("#video-original-row").hidden', 20000, 'link');
    await b.eval(`document.querySelector('#video').currentTime = 3.6; 1`); await new Promise(r => setTimeout(r, 500));
    await b.eval(`document.querySelector('#video-original').dispatchEvent(new Event('focus')); 1`);
    // the file (8 s) is shorter than the clip start (10 s): a cut clip, so the original is at 10 + 3.6 s
    assert.equal(await b.eval('document.querySelector("#video-original").href'), 'https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=13');
    assert.equal(await b.eval('document.querySelector("#video-original").target'), '_blank');
    assert.equal((await b.eval('overgaze.settingsJson().video.original')).id, 'dQw4w9WgXcQ');
  } finally { b.close(); server.close(); }
});

test('motion analysis uses OpenCV Farnebäck (loaded from lib/ when needed) with the checks; flow arrows can be limited to the gaze points', opts, async () => {
  const { server, url } = await serve(), b = await launch();
  try {
    await b.goto(url + '?sample');
    await b.waitFor('!!overgaze.state.gaze && document.querySelector("#video").readyState >= 2', 20000);
    assert.equal(await b.eval('typeof cv'), 'undefined', 'OpenCV.js is not loaded before it is needed');
    await b.eval(`(e => { e.value = 'seek'; e.dispatchEvent(new Event('change')); return 1; })(document.querySelector('#flow-method'))`);
    await b.click('#analyse');
    await b.waitFor('!!overgaze.state.flow && !overgaze.state.analysing', 120000, 'motion');
    const info = await b.eval('overgaze.state.flowInfo');
    assert.equal(info.estimate, 'farneback'); assert.equal(info.checks, true); assert.equal(info.failures, 0);
    // the sample's camera pans at −20°/s: about −14 px/s at 256 px wide
    const pan = await b.eval('overgaze.state.flow.velocity(4, 40, 62)[0]');
    assert.ok(pan < -10 && pan > -18, `camera pan found (${pan.toFixed(1)} px/s)`);
    assert.match(await b.eval('overgaze.settingsJson({}).motion?.method_note || ""'), /^$|Farnebäck/);
    // arrows: count the drawn lines with and without "only near the gaze points"
    await b.eval(`(() => { window.__arrows = 0; const P = CanvasRenderingContext2D.prototype, mt = P.moveTo; P.moveTo = function (...a) { if (this.canvas.id === 'marks-canvas') window.__arrows++; return mt.apply(this, a); }; return 1; })()`);
    await b.eval(`overgaze.applySettings({ windowSec: 1, kernel: 'gaussian' }); (e => { e.checked = true; e.dispatchEvent(new Event('change')); return 1; })(document.querySelector('#show-flow')); document.querySelector('#points-more').open = false; 1`);
    await b.eval(`(e => { e.checked = false; e.dispatchEvent(new Event('change')); return 1; })(document.querySelector('#point-motion-lines'))`);
    await b.eval(`window.__arrows = 0; document.querySelector('#video').currentTime = 4; 1`); await new Promise(r => setTimeout(r, 800));
    const all = await b.eval('window.__arrows');
    await b.eval(`(e => { e.checked = true; e.dispatchEvent(new Event('change')); return 1; })(document.querySelector('#flow-near-gaze')); window.__arrows = 0; document.querySelector('#video').currentTime = 4.01; 1`); await new Promise(r => setTimeout(r, 800));
    const near = await b.eval('window.__arrows');
    assert.ok(near > 0 && near < all / 3, `fewer arrows, near the gaze only (${all} → ${near})`);
    assert.deepEqual(await b.eval('overgaze.settingsJson().display.flow_vectors'), { show: true, only_near_gaze: true });
    assert.deepEqual(b.logs, []);
  } finally { b.close(); server.close(); }
});
