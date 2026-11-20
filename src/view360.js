// WebGL2 rendering for 360° video:
//  - LayoutConverter turns a frame of any supported layout (cubemap, equi-angular cubemap, stereo …, see
//    projections.js) into an equirectangular frame, on which the gaze overlay is drawn;
//  - View360 shows an equirectangular frame from inside the sphere: perspective (like the 360° players of YouTube
//    or VLC), little planet (stereographic) or fisheye (equidistant). These views follow egjs-view360 (NAVER,
//    MIT licence, https://github.com/naver/egjs-view360); the shaders are our own.
import { DEFAULT_OPTIONS, faceTables, glslLayout, layoutIndex } from './projections.js';

const VS = `#version 300 es
in vec2 p; out vec2 ndc;
void main() { ndc = p; gl_Position = vec4(p, 0.0, 1.0); }`;

function program(gl, fs) {
  const sh = (type, src) => { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s); if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s)); return s; };
  const prog = gl.createProgram();
  gl.attachShader(prog, sh(gl.VERTEX_SHADER, VS)); gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, fs)); gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
  gl.useProgram(prog);
  gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
  const loc = gl.getAttribLocation(prog, 'p');
  gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  return { prog, tex, u: name => gl.getUniformLocation(prog, name) };
}
function webgl(canvas) {
  const gl = canvas.getContext('webgl2', { preserveDrawingBuffer: true, antialias: false });
  if (!gl) throw new Error('WebGL2 is not available in this browser.');
  return gl;
}

export class LayoutConverter {
  constructor(canvas = document.createElement('canvas'), opts = DEFAULT_OPTIONS) {
    this.canvas = canvas; this.gl = webgl(canvas); this.opts = { ...DEFAULT_OPTIONS, ...opts };
    this.p = program(this.gl, `#version 300 es
precision highp float;
in vec2 ndc; out vec4 color;
uniform sampler2D src; uniform int lay; uniform vec2 size; uniform float hfov;  // ("layout" is a reserved word in GLSL)
${glslLayout(faceTables(this.opts))}
void main() {
  vec2 uv = vec2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);                 // output pixel, v down
  float lon = (uv.x - 0.5) * 2.0 * PI, lat = (0.5 - uv.y) * PI;
  vec3 d = vec3(cos(lat) * sin(lon), sin(lat), cos(lat) * cos(lon));
  vec2 p = layoutUV(lay, d, size, hfov);
  color = p.x < 0.0 ? vec4(0.0, 0.0, 0.0, 1.0) : texture(src, p);
}`);
  }
  /** Draw `source` (video, image, canvas) of the given layout as an equirectangular frame of W × H. */
  convert(source, layout, srcW, srcH, W, H) {
    const gl = this.gl, c = this.canvas;
    if (c.width !== W || c.height !== H) { c.width = W; c.height = H; }
    gl.viewport(0, 0, W, H);
    gl.bindTexture(gl.TEXTURE_2D, this.p.tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source);
    gl.uniform1i(this.p.u('lay'), layoutIndex(layout));
    gl.uniform2f(this.p.u('size'), srcW, srcH);
    gl.uniform1f(this.p.u('hfov'), Math.max(1, Math.min(360, this.opts.hfov || 360)) * Math.PI / 180);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    return c;
  }
}

export const VIEW_MODES = {
  perspective: { label: 'Perspective', fov: [25, 130], start: { fov: 90 } },
  planet: { label: 'Little planet', fov: [60, 330], start: { fov: 260, pitch: -90 } },
  fisheye: { label: 'Fisheye', fov: [60, 360], start: { fov: 180 } }
};

export class View360 {
  constructor(canvas) {
    this.canvas = canvas; this.gl = webgl(canvas);
    this.yaw = 0; this.pitch = 0; this.fov = 90; this.mode = 'perspective';   // degrees; yaw 0 = centre of the frame
    this.p = program(this.gl, `#version 300 es
precision highp float;
in vec2 ndc; out vec4 color;
uniform sampler2D erp; uniform float yaw, pitch, fov, aspect; uniform int mode;
const float PI = 3.141592653589793;
void main() {
  vec2 q = vec2(ndc.x * aspect, ndc.y);
  vec3 d;
  if (mode == 0) d = normalize(vec3(q * tan(fov / 2.0), 1.0));                         // perspective
  else if (mode == 1) { q *= tan(fov / 4.0); float r2 = dot(q, q);                     // stereographic
    d = vec3(2.0 * q, 1.0 - r2) / (1.0 + r2); }
  else { float th = length(q) * fov / 2.0;                                             // equidistant fisheye
    if (th > PI) { color = vec4(0.0, 0.0, 0.0, 1.0); return; }
    vec2 dir = length(q) > 0.0 ? normalize(q) : vec2(0.0); d = vec3(sin(th) * dir, cos(th)); }
  float cp = cos(pitch), sp = sin(pitch), cy = cos(yaw), sy = sin(yaw);
  d = vec3(d.x, cp * d.y + sp * d.z, -sp * d.y + cp * d.z);       // look up / down
  d = vec3(cy * d.x + sy * d.z, d.y, -sy * d.x + cy * d.z);         // turn left / right
  float lon = atan(d.x, d.z), lat = asin(clamp(d.y, -1.0, 1.0));
  color = texture(erp, vec2(fract(lon / (2.0 * PI) + 0.5), 0.5 - lat / PI));
}`);
  }
  setMode(mode) {
    this.mode = VIEW_MODES[mode] ? mode : 'perspective';
    Object.assign(this, { pitch: 0, ...VIEW_MODES[this.mode].start });
  }
  /** Upload the equirectangular frame (canvas, image or video). */
  setSource(src) {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.p.tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, src);
  }
  draw() {
    const gl = this.gl, c = this.canvas, dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.round(c.clientWidth * dpr)), h = Math.max(1, Math.round(c.clientHeight * dpr));
    if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
    gl.viewport(0, 0, w, h);
    const D = Math.PI / 180;
    gl.uniform1f(this.p.u('yaw'), this.yaw * D); gl.uniform1f(this.p.u('pitch'), this.pitch * D);
    gl.uniform1f(this.p.u('fov'), this.fov * D); gl.uniform1f(this.p.u('aspect'), w / h);
    gl.uniform1i(this.p.u('mode'), Object.keys(VIEW_MODES).indexOf(this.mode));
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }
  /** Turn by a drag of (dx, dy) screen pixels: the content follows the pointer. */
  drag(dx, dy) {
    const h = this.canvas.clientHeight || 1, perPx = Math.min(this.fov, 130) / h;
    this.yaw = wrap180(this.yaw - dx * perPx);
    this.pitch = Math.max(-90, Math.min(90, this.pitch + dy * perPx));
  }
  zoom(factor) { const [a, b] = VIEW_MODES[this.mode].fov; this.fov = Math.max(a, Math.min(b, this.fov * factor)); }
  /** Turn a step of the way towards (lon, lat) degrees. */
  follow(lon, lat, rate = 0.15) {
    this.yaw = wrap180(this.yaw + wrap180(lon - this.yaw) * rate);
    this.pitch += (Math.max(-89, Math.min(89, lat)) - this.pitch) * rate;
  }
}

export const wrap180 = a => a - 360 * Math.floor((a + 180) / 360);
