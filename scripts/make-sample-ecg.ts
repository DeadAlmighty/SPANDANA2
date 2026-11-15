/** Generate demo ECG files for the browser smoke test (run with bun). */
import { writeFileSync } from "fs";

const rate = 500;
const seconds = 10;
const n = rate * seconds;
const beats = 75; // bpm
const leadProfile = [
  { r: 0.8, s: 0.15 }, // I
  { r: 1.4, s: 0.2 },  // II
  { r: 0.6, s: 0.2 },  // III
  { r: 0.1, s: 0.9 },  // aVR
  { r: 0.5, s: 0.2 },  // aVL
  { r: 1.0, s: 0.15 }, // aVF
  { r: 0.3, s: 1.2 },  // V1
  { r: 0.6, s: 1.4 },  // V2
  { r: 1.1, s: 1.0 },  // V3
  { r: 1.8, s: 0.7 },  // V4
  { r: 3.1, s: 0.4 },  // V5  -> Sokolow-Lyon positive (1.2+3.1=4.3)
  { r: 2.5, s: 0.25 }, // V6
];
const names = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"];

const sample = (i: number, l: number) => {
  const t = i / rate;
  const rr = 60 / beats;
  const phase = t % rr;
  const { r, s } = leadProfile[l];
  const gauss = (c: number, w: number) => Math.exp(-0.5 * Math.pow((phase - c) / w, 2));
  let v = 0.015 * Math.sin(2 * Math.PI * 0.3 * t);
  v += 0.15 * gauss(0.05, 0.02) * (l < 6 ? 1 : 0.6); // P
  v += r * gauss(0.1, 0.011);
  v -= s * gauss(0.128, 0.013);
  v += (0.3 * r) * gauss(0.32, 0.05) * (r > 2 ? -1 : 1); // T (strain on tall-R leads)
  v += 0.006 * Math.sin(i * 0.37 + l);
  return v;
};

// CSV (mV, 12 cols, no header)
const csv = Array.from({ length: n }, (_, i) =>
  Array.from({ length: 12 }, (_, l) => sample(i, l).toFixed(4)).join(","),
).join("\n");
writeFileSync("sample-ecg.csv", csv);

// DAT + HEA (format 16, gain 200, baseline 1024)
const gain = 200, baseline = 1024;
const buf = new ArrayBuffer(n * 12 * 2);
const view = new DataView(buf);
for (let i = 0; i < n; i++)
  for (let l = 0; l < 12; l++)
    view.setInt16((i * 12 + l) * 2, Math.round(baseline + sample(i, l) * gain), true);
writeFileSync("sample-ecg.dat", Buffer.from(buf));
const hea =
  `sample-ecg 12 ${rate} 10:00:00 12/11/2026\n` +
  names.map((nm) => `sample-ecg.dat 16 ${gain} ${baseline} 0 0 0 0 0 0 0 0 0 ,${nm},`).join("\n") +
  "\n";
writeFileSync("sample-ecg.hea", hea);
console.log("created sample-ecg.csv, sample-ecg.dat, sample-ecg.hea");
