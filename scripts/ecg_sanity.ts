/**
 * Sanity check: run the synthetic waveform generator + criteria engine
 * and print measured values for the three curated demo cases.
 */
import { createDemoAnalyses } from "../src/lib/demo-data";

const demos = createDemoAnalyses();
for (const a of demos) {
  const sl = a.criteria.sokolowLyon;
  const co = a.criteria.cornell;
  console.log(
    `${a.meta.code} (${a.meta.sex}, ${a.meta.age}) bpm=${a.waveform.bpm}`,
  );
  console.log(
    `  Sokolow-Lyon: SV1=${sl.sv1} RV5/6=${sl.rv56} sum=${sl.sum} >= ${sl.threshold} -> ${sl.positive}`,
  );
  console.log(
    `  Cornell:      RaVL=${co.ravl} SV3=${co.sv3} sum=${co.sum} >= ${co.threshold} -> ${co.positive}`,
  );
  console.log(
    `  Agreement=${a.criteria.agreement} Interpretation=${a.interpretation}`,
  );
  console.log(
    `  Perceptron=${a.predictions[0].probability} (${a.predictions[0].positive}) CNN=${a.predictions[1].probability} (${a.predictions[1].positive})`,
  );
  console.log(
    `  Samples/lead=${a.waveform.leads["II"].length} beats=${a.waveform.beats.length}`,
  );
}
