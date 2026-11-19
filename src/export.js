// Dependency-free writers: CRC-32, ZIP (stored, streamed file by file), NumPy .npy headers and greyscale PNG.

const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xEDB88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();

export function crc32(bytes, crc = 0) {
  let c = ~crc >>> 0;
  for (let i = 0; i < bytes.length; i++) c = CRC_TABLE[(c ^ bytes[i]) & 0xFF] ^ (c >>> 8);
  return ~c >>> 0;
}

const encoder = new TextEncoder();

/**
 * ZIP writer without compression (PNGs are already compressed; NPY stays directly readable).
 * `write(Uint8Array)` may be async, e.g. a FileSystemWritableFileStream or an array collector.
 * Small entries go through add(); large ones through stream(), which writes a data descriptor afterwards.
 * ZIP64 records are written when sizes, offsets or the entry count need them.
 */
export class ZipWriter {
  constructor(write) { this.write = write; this.offset = 0; this.entries = []; }
  async add(name, data) {
    const bytes = typeof data === 'string' ? encoder.encode(data) : data;
    const nameBytes = encoder.encode(name), crc = crc32(bytes), { time, date } = dosTime(new Date());
    const header = new Uint8Array(30 + nameBytes.length), v = new DataView(header.buffer);
    v.setUint32(0, 0x04034b50, true); v.setUint16(4, 20, true); v.setUint16(6, 0x0800, true); v.setUint16(8, 0, true);
    v.setUint16(10, time, true); v.setUint16(12, date, true); v.setUint32(14, crc, true);
    v.setUint32(18, bytes.length, true); v.setUint32(22, bytes.length, true); v.setUint16(26, nameBytes.length, true); v.setUint16(28, 0, true);
    header.set(nameBytes, 30);
    this.entries.push({ nameBytes, crc, size: bytes.length, offset: this.offset, time, date, zip64: false });
    await this.write(header); await this.write(bytes);
    this.offset += header.length + bytes.length;
  }
  /** Start a streamed entry; returns { write(bytes), end() }. */
  async stream(name) {
    const nameBytes = encoder.encode(name), { time, date } = dosTime(new Date());
    const header = new Uint8Array(30 + nameBytes.length + 20), v = new DataView(header.buffer);
    v.setUint32(0, 0x04034b50, true); v.setUint16(4, 45, true); v.setUint16(6, 0x0808, true); v.setUint16(8, 0, true);
    v.setUint16(10, time, true); v.setUint16(12, date, true); v.setUint32(14, 0, true);
    v.setUint32(18, 0xFFFFFFFF, true); v.setUint32(22, 0xFFFFFFFF, true); v.setUint16(26, nameBytes.length, true); v.setUint16(28, 20, true);
    header.set(nameBytes, 30);
    v.setUint16(30 + nameBytes.length, 0x0001, true); v.setUint16(32 + nameBytes.length, 16, true); // sizes follow in the descriptor
    const entry = { nameBytes, crc: 0, size: 0, offset: this.offset, time, date, zip64: true };
    await this.write(header); this.offset += header.length;
    return {
      write: async bytes => { entry.crc = crc32(bytes, entry.crc); entry.size += bytes.length; await this.write(bytes); this.offset += bytes.length; },
      end: async () => {
        const d = new Uint8Array(24), dv = new DataView(d.buffer);
        dv.setUint32(0, 0x08074b50, true); dv.setUint32(4, entry.crc, true);
        dv.setBigUint64(8, BigInt(entry.size), true); dv.setBigUint64(16, BigInt(entry.size), true);
        await this.write(d); this.offset += d.length;
        this.entries.push(entry);
      }
    };
  }
  async finish() {
    const start = this.offset;
    for (const e of this.entries) {
      const big = e.zip64 || e.size >= 0xFFFFFFFF || e.offset >= 0xFFFFFFFF;
      const extra = big ? 28 : 0;
      const h = new Uint8Array(46 + e.nameBytes.length + extra), v = new DataView(h.buffer);
      v.setUint32(0, 0x02014b50, true); v.setUint16(4, big ? 45 : 20, true); v.setUint16(6, big ? 45 : 20, true); v.setUint16(8, e.zip64 ? 0x0808 : 0x0800, true);
      v.setUint16(10, 0, true); v.setUint16(12, e.time, true); v.setUint16(14, e.date, true); v.setUint32(16, e.crc, true);
      v.setUint32(20, big ? 0xFFFFFFFF : e.size, true); v.setUint32(24, big ? 0xFFFFFFFF : e.size, true); v.setUint16(28, e.nameBytes.length, true);
      v.setUint16(30, extra, true); v.setUint32(42, big ? 0xFFFFFFFF : e.offset, true);
      h.set(e.nameBytes, 46);
      if (big) {
        const o = 46 + e.nameBytes.length;
        v.setUint16(o, 0x0001, true); v.setUint16(o + 2, 24, true);
        v.setBigUint64(o + 4, BigInt(e.size), true); v.setBigUint64(o + 12, BigInt(e.size), true); v.setBigUint64(o + 20, BigInt(e.offset), true);
      }
      await this.write(h); this.offset += h.length;
    }
    const cdSize = this.offset - start, n = this.entries.length;
    if (start >= 0xFFFFFFFF || cdSize >= 0xFFFFFFFF || n >= 0xFFFF || this.entries.some(e => e.zip64)) {
      const z = new Uint8Array(56 + 20), v = new DataView(z.buffer), eocd64 = this.offset;
      v.setUint32(0, 0x06064b50, true); v.setBigUint64(4, 44n, true); v.setUint16(12, 45, true); v.setUint16(14, 45, true);
      v.setBigUint64(24, BigInt(n), true); v.setBigUint64(32, BigInt(n), true); v.setBigUint64(40, BigInt(cdSize), true); v.setBigUint64(48, BigInt(start), true);
      v.setUint32(56, 0x07064b50, true); v.setBigUint64(64, BigInt(eocd64), true); v.setUint32(72, 1, true);
      await this.write(z); this.offset += z.length;
    }
    const end = new Uint8Array(22), v = new DataView(end.buffer);
    v.setUint32(0, 0x06054b50, true); v.setUint16(8, Math.min(n, 0xFFFF), true); v.setUint16(10, Math.min(n, 0xFFFF), true);
    v.setUint32(12, Math.min(cdSize, 0xFFFFFFFF), true); v.setUint32(16, Math.min(start, 0xFFFFFFFF), true);
    await this.write(end); this.offset += end.length;
  }
}

function dosTime(d) {
  return {
    time: (d.getHours() << 11) | (d.getMinutes() << 5) | (d.getSeconds() >> 1),
    date: ((d.getFullYear() - 1980) << 9) | ((d.getMonth() + 1) << 5) | d.getDate()
  };
}

/** NumPy format 1.0 header for a C-ordered array; dtype e.g. '<f4', '|u1', '<u2'. Data starts 64-byte aligned. */
export function npyHeader(dtype, shape) {
  const dict = `{'descr': '${dtype}', 'fortran_order': False, 'shape': (${shape.join(', ')}${shape.length === 1 ? ',' : ''}), }`;
  const total = Math.ceil((10 + dict.length + 1) / 64) * 64;
  const text = dict + ' '.repeat(total - 10 - dict.length - 1) + '\n';
  const out = new Uint8Array(total);
  out.set([0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59, 1, 0]);
  new DataView(out.buffer).setUint16(8, text.length, true);
  out.set(encoder.encode(text), 10);
  return out;
}

async function zlib(bytes) {
  const stream = new Blob([bytes]).stream().pipeThrough(new CompressionStream('deflate'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

function chunk(type, data) {
  const out = new Uint8Array(12 + data.length), v = new DataView(out.buffer);
  v.setUint32(0, data.length);
  out.set(encoder.encode(type), 4); out.set(data, 8);
  v.setUint32(8 + data.length, crc32(out.subarray(4, 8 + data.length)));
  return out;
}

/** Greyscale PNG from values in [0, 1] (Float32Array, W×H). bitDepth 8 stores round(255·v), 16 stores round(65535·v). */
export async function encodeGrayPng(values, W, H, bitDepth = 8) {
  const bpp = bitDepth === 16 ? 2 : 1, stride = W * bpp;
  const raw = new Uint8Array(H * (stride + 1));
  const max = bitDepth === 16 ? 65535 : 255;
  let prev = new Uint8Array(stride), line = new Uint8Array(stride);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const q = Math.round(Math.min(1, Math.max(0, values[y * W + x])) * max);
      if (bpp === 2) { line[2 * x] = q >> 8; line[2 * x + 1] = q & 0xFF; } else line[x] = q;
    }
    const o = y * (stride + 1);
    raw[o] = 2; // "Up" filter: smooth maps compress well
    for (let i = 0; i < stride; i++) raw[o + 1 + i] = (line[i] - prev[i]) & 0xFF;
    [prev, line] = [line, prev];
  }
  const ihdr = new Uint8Array(13), v = new DataView(ihdr.buffer);
  v.setUint32(0, W); v.setUint32(4, H); ihdr[8] = bitDepth; ihdr[9] = 0; ihdr[10] = 0; ihdr[11] = 0; ihdr[12] = 0;
  const parts = [new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]), chunk('IHDR', ihdr), chunk('IDAT', await zlib(raw)), chunk('IEND', new Uint8Array(0))];
  const out = new Uint8Array(parts.reduce((s, p) => s + p.length, 0));
  let o = 0;
  for (const p of parts) { out.set(p, o); o += p.length; }
  return out;
}

/** Quantise [0, 1] floats for an NPY body. */
export function npyBody(values, dtype) {
  if (dtype === '|u1') {
    const out = new Uint8Array(values.length);
    for (let i = 0; i < values.length; i++) out[i] = Math.round(Math.min(1, Math.max(0, values[i])) * 255);
    return out;
  }
  return new Uint8Array(Float32Array.from(values).buffer);
}
