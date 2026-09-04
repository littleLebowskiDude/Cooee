/* Generates the Cooee app icon. No dependencies — writes a PNG directly.
 *
 *   node tools/make-icon.cjs src-tauri/icons/icon.png [size]
 *
 * The mark: a source point with arcs travelling outward. Directional, because a
 * cooee is a call that carries — concentric rings would read as a target. */
const zlib = require("zlib"), fs = require("fs");

const SS = 3; // supersampling factor

function render(S) {
const hex = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
const BG_TOP = hex("#26262e"), BG_BOT = hex("#15151a");
// Copilot ramp: the core is the blue, the arcs travel through violet to pink.
const RAMP = [hex("#4cc2ff"), hex("#a06cff"), hex("#ff6bb5")];

const M = S * 0.055, R = S * 0.235, lo = M, hi = S - M;
const inSquircle = (x, y) => {
  if (x < lo || x > hi || y < lo || y > hi) return false;
  const cx = Math.min(Math.max(x, lo + R), hi - R);
  const cy = Math.min(Math.max(y, lo + R), hi - R);
  return (x - cx) ** 2 + (y - cy) ** 2 <= R * R;
};

// Source sits left of centre so the arcs have room to travel.
const CX = S * 0.33, CY = S / 2, CORE = S * 0.062;
const RINGS = [
  { r: S * 0.140, w: S * 0.034, a: 1.00 },
  { r: S * 0.245, w: S * 0.029, a: 0.70 },
  { r: S * 0.350, w: S * 0.025, a: 0.42 },
];
const APERTURE = (Math.PI * 62) / 180; // arcs span +/-62 deg of straight ahead

const px = Buffer.alloc(S * S * 4);
for (let y = 0; y < S; y++) {
  for (let x = 0; x < S; x++) {
    let ar = 0, ag = 0, ab = 0, hits = 0;
    for (let sy = 0; sy < SS; sy++) {
      for (let sx = 0; sx < SS; sx++) {
        const fx = x + (sx + 0.5) / SS, fy = y + (sy + 0.5) / SS;
        if (!inSquircle(fx, fy)) continue;

        const t = fy / S;
        let r = BG_TOP[0] + (BG_BOT[0] - BG_TOP[0]) * t;
        let g = BG_TOP[1] + (BG_BOT[1] - BG_TOP[1]) * t;
        let b = BG_TOP[2] + (BG_BOT[2] - BG_TOP[2]) * t;

        const d = Math.hypot(fx - CX, fy - CY);
        if (d <= CORE) {
          [r, g, b] = RAMP[0];
        } else {
          for (let k = 0; k < RINGS.length; k++) {
            const ring = RINGS[k];
            if (Math.abs(d - ring.r) > ring.w / 2) continue;
            const ang = Math.atan2(fy - CY, fx - CX);
            if (Math.abs(ang) > APERTURE) break;
            // Taper toward the tips so each arc dissolves instead of stopping dead.
            const taper = Math.min(1, (APERTURE - Math.abs(ang)) / (APERTURE * 0.45));
            // Each arc takes the next stop of the ramp as it travels.
            const [cr, cg, cb] = RAMP[Math.min(k, RAMP.length - 1)];
            const A = ring.a * taper;
            r += (cr - r) * A; g += (cg - g) * A; b += (cb - b) * A;
            break;
          }
        }
        ar += r; ag += g; ab += b; hits++;
      }
    }
    const i = (y * S + x) * 4;
    if (hits > 0) {
      px[i] = Math.round(ar / hits);
      px[i + 1] = Math.round(ag / hits);
      px[i + 2] = Math.round(ab / hits);
      px[i + 3] = Math.round((255 * hits) / (SS * SS));
    }
  }
}

// --- PNG encoding ---
const raw = Buffer.alloc(S * (S * 4 + 1));
for (let y = 0; y < S; y++) {
  raw[y * (S * 4 + 1)] = 0; // filter: none
  px.copy(raw, y * (S * 4 + 1) + 1, y * S * 4, (y + 1) * S * 4);
}
let TBL = null;
function crc32(buf) {
  if (!TBL) {
    TBL = [];
    for (let n = 0; n < 256; n++) {
      let c = n;
      for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
      TBL[n] = c;
    }
  }
  let c = 0xffffffff;
  for (const v of buf) c = TBL[(c ^ v) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}
function chunk(type, data) {
  const len = Buffer.alloc(4); len.writeUInt32BE(data.length);
  const td = Buffer.concat([Buffer.from(type), data]);
  const crc = Buffer.alloc(4); crc.writeUInt32BE(crc32(td));
  return Buffer.concat([len, td, crc]);
}
const ihdr = Buffer.alloc(13);
ihdr.writeUInt32BE(S, 0); ihdr.writeUInt32BE(S, 4);
ihdr[8] = 8; ihdr[9] = 6; // 8-bit RGBA
return Buffer.concat([
  Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]),
  chunk("IHDR", ihdr),
  chunk("IDAT", zlib.deflateSync(raw, { level: 9 })),
  chunk("IEND", Buffer.alloc(0)),
]);
}

/* ICO wrapper. Vista+ accepts PNG-compressed entries directly, so each size is
 * just the rendered PNG with a 16-byte directory entry pointing at it. */
function ico(sizes) {
  const images = sizes.map(render);
  const header = Buffer.alloc(6);
  header.writeUInt16LE(0, 0);          // reserved
  header.writeUInt16LE(1, 2);          // type: icon
  header.writeUInt16LE(sizes.length, 4);

  let offset = 6 + sizes.length * 16;
  const entries = sizes.map((size, i) => {
    const e = Buffer.alloc(16);
    e[0] = size >= 256 ? 0 : size;     // 0 means 256
    e[1] = size >= 256 ? 0 : size;
    e[2] = 0;                          // palette colours
    e[3] = 0;                          // reserved
    e.writeUInt16LE(1, 4);             // colour planes
    e.writeUInt16LE(32, 6);            // bits per pixel
    e.writeUInt32LE(images[i].length, 8);
    e.writeUInt32LE(offset, 12);
    offset += images[i].length;
    return e;
  });
  return Buffer.concat([header, ...entries, ...images]);
}

const DIR = process.argv[2] || "src-tauri/icons";
fs.mkdirSync(DIR, { recursive: true });

// Tauri needs the .ico for the Windows resource; the PNGs are for everything else.
const icoSizes = [16, 32, 48, 64, 128, 256];
fs.writeFileSync(`${DIR}/icon.ico`, ico(icoSizes));
console.log(`wrote ${DIR}/icon.ico (${icoSizes.join(", ")})`);

for (const size of [32, 128, 256, 512]) {
  const name = size === 512 ? "icon.png" : `${size}x${size}.png`;
  fs.writeFileSync(`${DIR}/${name}`, render(size));
  console.log(`wrote ${DIR}/${name}`);
}
