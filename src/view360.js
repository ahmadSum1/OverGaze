// Perspective "look around" view of an equirectangular frame (like the 360° players of YouTube or VLC), in WebGL2.
// The source is a canvas holding the ERP frame with the heatmap and gaze points already drawn on it.

const VS = `#version 300 es
in vec2 p; out vec2 ndc;
void main() { ndc = p; gl_Position = vec4(p, 0.0, 1.0); }`;

const FS = `#version 300 es
precision highp float;
in vec2 ndc; out vec4 color;
uniform sampler2D erp; uniform float yaw, pitch, tanHalf, aspect;
const float PI = 3.141592653589793;
void main() {
  vec3 d = normalize(vec3(ndc.x * tanHalf * aspect, ndc.y * tanHalf, 1.0));
  float cp = cos(pitch), sp = sin(pitch), cy = cos(yaw), sy = sin(yaw);
  d = vec3(d.x, cp * d.y + sp * d.z, -sp * d.y + cp * d.z);       // look up / down
  d = vec3(cy * d.x + sy * d.z, d.y, -sy * d.x + cy * d.z);         // turn left / right
  float lon = atan(d.x, d.z), lat = asin(clamp(d.y, -1.0, 1.0));
  color = texture(erp, vec2(fract(lon / (2.0 * PI) + 0.5), 0.5 - lat / PI));
}`;

export class View360 {
  constructor(canvas) {
    const gl = canvas.getContext('webgl2', { preserveDrawingBuffer: true, antialias: false });
    if (!gl) throw new Error('WebGL2 is not available in this browser.');
    this.canvas = canvas; this.gl = gl;
    this.yaw = 0; this.pitch = 0; this.fov = 90;   // degrees; yaw 0 looks at the centre of the frame
    const sh = (type, src) => { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s); if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s)); return s; };
    const prog = gl.createProgram();
    gl.attachShader(prog, sh(gl.VERTEX_SHADER, VS)); gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, FS)); gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
    this.prog = prog;
    gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    const loc = gl.getAttribLocation(prog, 'p');
    gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
    this.tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    this.u = Object.fromEntries(['yaw', 'pitch', 'tanHalf', 'aspect'].map(n => [n, gl.getUniformLocation(prog, n)]));
  }
  /** Upload the ERP frame (canvas, image or video). */
  setSource(src) {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, src);
  }
  draw() {
    const gl = this.gl, c = this.canvas, dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.round(c.clientWidth * dpr)), h = Math.max(1, Math.round(c.clientHeight * dpr));
    if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
    gl.viewport(0, 0, w, h);
    gl.useProgram(this.prog);
    const D = Math.PI / 180;
    gl.uniform1f(this.u.yaw, this.yaw * D); gl.uniform1f(this.u.pitch, this.pitch * D);
    gl.uniform1f(this.u.tanHalf, Math.tan(this.fov * D / 2)); gl.uniform1f(this.u.aspect, w / h);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }
  /** Turn by a drag of (dx, dy) screen pixels: the content follows the pointer. */
  drag(dx, dy) {
    const h = this.canvas.clientHeight || 1, perPx = this.fov / h;
    this.yaw = wrap180(this.yaw - dx * perPx);
    this.pitch = Math.max(-89, Math.min(89, this.pitch + dy * perPx));
  }
  zoom(factor) { this.fov = Math.max(25, Math.min(130, this.fov * factor)); }
  /** Turn a step of the way towards (lon, lat) degrees. */
  follow(lon, lat, rate = 0.15) {
    this.yaw = wrap180(this.yaw + wrap180(lon - this.yaw) * rate);
    this.pitch += (Math.max(-89, Math.min(89, lat)) - this.pitch) * rate;
  }
}

export const wrap180 = a => a - 360 * Math.floor((a + 180) / 360);
