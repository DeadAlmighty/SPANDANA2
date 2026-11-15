/**
 * Functional test for SPANDANA in-browser ECG decoders (run with bun).
 * Builds synthetic-but-realistic 12-lead recordings as CSV / DAT+HEA,
 * runs the actual parser code, and asserts measured values + errors.
 */
import {
  parseCsvEcg,
  parseDatFile,
  parseHeaHeader,
  EcgParseError,
} from "../src/lib/ecg/parsers";
import { buildParsedSpec } from "../src/lib/ecg/real-signal";
import { LEAD_NAMES } from "../src/lib/ecg/types";

let pass = 0;
let fail = 0;
function check(name: string, cond: boolean, extra = "") {
  if (cond) {
    pass++;
    console.log(`  ok  ${name}`);
  } else {
    fail++;
    console.error(`FAIL  ${name} ${extra}`);
  }
}

/** Deterministic 12-lead ECG-ish signal: 60 bpm spikes + baseline. */
function makeSignal(rate = 500, seconds = 10, unit: "mV" | "uV" = "mV"): number[][] {
  const n = rate * seconds;
  const cols: number[][] = [];
  const scale = unit === "mV" ? 1 : 1000;
  for (let l = 0; l < 12; l++) {
    const col = new Array<number>(n);
    const rAmp = 0.4 + l * 0.12; // increasing R across leads
    const sAmp = l === 0 ? 1.2 : 0.2; // V1..? big S on lead index 6 handled below
    for (let i = 0; i < n; i++) {
      const t = i / rate;
      const phase = t % 1.0; // 60 bpm
      let v = 0.02 * Math.sin(2 * Math.PI * 0.3 * t);
      if (phase > 0.15 && phase < 0.16) v += rAmp * (1 - Math.abs((phase - 0.155) / 0.005));
      if (phase > 0.18 && phase < 0.19) v -= sAmp * (1 - Math.abs((phase - 0.185) / 0.005));
      col[i] = Math.round(v * scale * 1000) / 1000;
    }
    cols.push(col);
  }
  return cols;
}

console.log("— CSV decode (mV, no header, 12 cols) —");
{
  const cols = makeSignal();
  const csv = cols[0].map((_, i) => cols.map((c) => c[i].toFixed(4)).join(",")).join("\n");
  const sig = parseCsvEcg(csv, "rec.csv");
  check("12 leads parsed", Object.keys(sig.leads).length === 12);
  check("rate inferred 500 Hz", sig.sampleRate === 500, `got ${sig.sampleRate}`);
  check("duration 10 s", Math.abs(sig.durationSec - 10) < 0.01);
  check("unit warning present", sig.warnings.some((w) => w.includes("mV")));
  const spec = buildParsedSpec(sig);
  check("HR detected 60±5", spec.bpm >= 55 && spec.bpm <= 65, `got ${spec.bpm}`);
  check("amplitudes measured", spec.amplitudes["V5"].r > 0.3);
  console.log(`      detail: ${sig.detail}`);
}

console.log("— CSV decode (µV scale, header row, time column) —");
{
  const cols = makeSignal(250, 10, "uV");
  const header = "time,I,II,III,aVR,aVL,aVF,V1,V2,V3,V4,V5,V6";
  const rows = cols[0].map((_, i) =>
    [(i / 250).toFixed(4), ...cols.map((c) => c[i].toFixed(1))].join(","),
  );
  const csv = [header, ...rows].join("\n");
  const sig = parseCsvEcg(csv, "rec.csv");
  check("rate from time column 250", sig.sampleRate === 250, `got ${sig.sampleRate}`);
  check("13→12 columns (time dropped)", sig.warnings.some((w) => w.includes("time column")));
  check("µV→mV scaling", sig.warnings.some((w) => w.includes("µV")));
  check("lead order from header", sig.leads["aVR"] !== undefined);
}

console.log("— CSV error paths —");
{
  try {
    parseCsvEcg("1,2,3\n4,5,6", "short.csv");
    check("short file rejected", false);
  } catch (e) {
    check("short file rejected", e instanceof EcgParseError && e.code === "too-short");
  }
  const cols = makeSignal();
  const csv13 = cols[0].map((_, i) => cols.map((c) => c[i]).join(",") + ",9").join("\n");
  try {
    parseCsvEcg(csv13, "wide.csv");
    check("13 cols w/o time axis rejected", false);
  } catch (e) {
    check("13 cols w/o time axis rejected", e instanceof EcgParseError && e.code === "bad-columns");
  }
}

console.log("— PhysioNet .dat + .hea (format 16, gain 200, baseline 1024) —");
{
  const rate = 500;
  const cols = makeSignal(rate, 10);
  const gain = 200;
  const baseline = 1024;
  const nLeads = 12;
  const n = cols[0].length;
  const buf = new ArrayBuffer(n * nLeads * 2);
  const view = new DataView(buf);
  for (let i = 0; i < n; i++) {
    for (let l = 0; l < nLeads; l++) {
      const raw = Math.round(baseline + cols[l][i] * gain);
      view.setInt16((i * nLeads + l) * 2, raw, true);
    }
  }
  const leadNames = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"];
  const heaLines = [
    `testrec ${nLeads} ${rate} 10:00:00 12/10/2026`,
    ...leadNames.map(
      (nm, l) => `testrec.dat 16 ${gain} ${baseline} 0 0 0 0 0 0 0 0 0 ,${nm},`,
    ),
  ];
  const hea = parseHeaHeader(heaLines.join("\n"), "testrec.hea");
  check("hea nLeads 12", hea.nLeads === 12);
  check("hea freq 500", hea.freq === 500);
  check("hea lead names mapped", hea.leads[6].name === "V1");
  const sig = parseDatFile(buf, "testrec.dat", hea, "testrec.hea");
  check("dat+hea decoded 12 leads", Object.keys(sig.leads).length === 12);
  check("rate from header 500", sig.sampleRate === 500);
  check("gain applied (peak < 3 mV)", sig.warnings.some((w) => w.includes("gain 200")));
  const spec = buildParsedSpec(sig);
  check("HR detected 60±5", spec.bpm >= 55 && spec.bpm <= 65, `got ${spec.bpm}`);
  check("S on lead I measured (≥0.9 mV)", spec.amplitudes["I"].s >= 0.9, `got ${spec.amplitudes["I"].s}`);
  console.log(`      detail: ${sig.detail}`);
}

console.log("— .dat without header (heuristic) —");
{
  const rate = 500;
  const cols = makeSignal(rate, 10);
  const nLeads = 12;
  const n = cols[0].length;
  const buf = new ArrayBuffer(n * nLeads * 2);
  const view = new DataView(buf);
  for (let i = 0; i < n; i++)
    for (let l = 0; l < nLeads; l++)
      view.setInt16((i * nLeads + l) * 2, Math.round(cols[l][i] * 200), true);
  const sig = parseDatFile(buf, "bare.dat");
  check("heuristic 12 leads", Object.keys(sig.leads).length === 12);
  check("gain assumption warning", sig.warnings.some((w) => w.includes("No .hea header")));
  check("rate inferred 500", sig.sampleRate === 500, `got ${sig.sampleRate}`);
}

console.log("— .hea without .dat / wrong lead count —");
{
  try {
    parseDatFile(new ArrayBuffer(100), "x.dat", {
      recordName: "two",
      nLeads: 2,
      freq: 500,
      leads: [],
    });
    check("2-lead record rejected", false);
  } catch (e) {
    check("2-lead record rejected", e instanceof EcgParseError && e.code === "lead-count");
  }
}

console.log(`\n${pass} passed, ${fail} failed`);
if (fail > 0) process.exit(1);
