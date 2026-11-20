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
    await b.waitFor('overgaze.state.fpsSource === "measured"', 20000, 'video');
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

    await b.click('#view-360');
    await b.waitFor('!document.querySelector("#view360-canvas").hidden', 3000);
    const px = await b.eval('(()=>{const c=document.querySelector("#view360-canvas");const g=c.getContext("webgl2");const p=new Uint8Array(4);g.readPixels(c.width>>1,c.height>>1,1,1,g.RGBA,g.UNSIGNED_BYTE,p);return [c.width,p[3]]})()');
    assert.ok(px[0] > 100 && px[1] === 255, `360° view drew something: ${px}`);
    // other views: little planet and fisheye draw too; layers can be switched off
    for (const mode of ['planet', 'fisheye']) {
      await b.eval(`document.querySelector('#view360-mode').value = '${mode}'; document.querySelector('#view360-mode').dispatchEvent(new Event('change')); 1`);
      assert.equal(await b.eval('overgaze.state.view.mode'), mode);
    }
    await b.click('#layer-gaze');
    assert.equal(await b.eval('overgaze.state.showGaze'), false);
    await b.click('#clear-gaze');
    assert.equal(await b.eval('overgaze.state.gaze'), null);
    assert.deepEqual(b.logs.filter(l => !/WebGL|GPU stall|swiftshader/i.test(l)), []);
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
