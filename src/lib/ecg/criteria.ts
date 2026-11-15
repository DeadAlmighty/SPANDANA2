/**
 * Traditional ECG voltage criteria for Left Ventricular Hypertrophy (LVH).
 *
 * References (real literature):
 *  - Sokolow M, Lyon TP. The ventricular complex in left ventricular
 *    hypertrophy. Am Heart J. 1949;37(1):161–186.
 *  - Casale PN, Devereux RB, Alonso DR, Campo E, Kligfield P. Improved
 *    sex-specific criteria of left ventricular hypertrophy for
 *    electrocardiography and echocardiography. Circulation. 1987;75(3):565–572.
 */

import type {
  CriteriaResult,
  EcgWaveform,
  LeadName,
  Sex,
} from "./types";

export const SOKOLOW_LYON_THRESHOLD = 3.5; // mV (35 mm)
export const CORNELL_THRESHOLD = { Male: 2.8, Female: 2.0 }; // mV

export const SOKOLOW_LYON_REF =
  "Sokolow M, Lyon TP. Am Heart J. 1949;37(1):161–186.";
export const CORNELL_REF =
  "Casale PN, Devereux RB, et al. Circulation. 1987;75(3):565–572.";

export function computeCriteria(
  waveform: EcgWaveform,
  sex: Sex,
): CriteriaResult {
  const amps = waveform.amplitudes;

  const sv1 = amps["V1"].s;
  const rv5 = amps["V5"].r;
  const rv6 = amps["V6"].r;
  const rv56 = Math.max(rv5, rv6);
  const slSum = round3(sv1 + rv56);

  const ravl = amps["aVL"].r;
  const sv3 = amps["V3"].s;
  const cornellThreshold = CORNELL_THRESHOLD[sex];
  const cornellSum = round3(ravl + sv3);

  const sokolowLyon = {
    sv1: round3(sv1),
    rv56: round3(rv56),
    sum: slSum,
    threshold: SOKOLOW_LYON_THRESHOLD,
    positive: slSum >= SOKOLOW_LYON_THRESHOLD,
    reference: SOKOLOW_LYON_REF,
  };

  const cornell = {
    ravl: round3(ravl),
    sv3: round3(sv3),
    sum: cornellSum,
    threshold: cornellThreshold,
    positive: cornellSum >= cornellThreshold,
    reference: CORNELL_REF,
  };

  const agreement: CriteriaResult["agreement"] =
    sokolowLyon.positive && cornell.positive
      ? "both-positive"
      : !sokolowLyon.positive && !cornell.positive
        ? "both-negative"
        : "discordant";

  return { sokolowLyon, cornell, agreement };
}

/**
 * Headline traditional-criteria verdict: positive when at least one
 * standard criterion is met (typical screening practice), labelled clearly
 * in the UI as a voltage-based research output.
 */
export function criteriaHeadline(criteria: CriteriaResult): "positive" | "negative" {
  return criteria.sokolowLyon.positive || criteria.cornell.positive
    ? "positive"
    : "negative";
}

function round3(x: number): number {
  return Math.round(x * 1000) / 1000;
}

/** Short human explanation of a criterion, used in reports and tooltips. */
export function criterionExplanations(sex: Sex): { sl: string; cornell: string } {
  return {
    sl: "Sokolow–Lyon: S in V1 + R in V5 or V6 ≥ 3.5 mV (35 mm) suggests LVH.",
    cornell: `Cornell: R in aVL + S in V3 ≥ ${CORNELL_THRESHOLD[sex].toFixed(1)} mV for ${sex.toLowerCase()}s suggests LVH.`,
  };
}

export function leadLabel(lead: LeadName): string {
  return lead;
}
