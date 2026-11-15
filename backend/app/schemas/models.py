"""Pydantic request/response schemas for the SPANDANA API.

The JSON screening contract matches src/lib/api.ts exactly:
  request : { patient: {code, age, sex}, waveform: {sampleRate, leads} }
            (the key "signal" is accepted as an alias)
  response: { predictions: [ {model, probability, positive, threshold,
                              latencyMs, simulated, contributions} ],
              criteria: {...}, meta: {...} }
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

LeadArrays = dict[str, list[float]]


class PatientIn(BaseModel):
    code: str | None = None
    age: int | None = Field(default=None, ge=0, le=130)
    sex: Literal["male", "female", "unknown", None] = None


class WaveformIn(BaseModel):
    sampleRate: float = Field(gt=1, le=10_000)
    durationSec: float | None = Field(default=None, gt=0)
    leads: LeadArrays = Field(min_length=1)


class ScreenRequest(BaseModel):
    patient: PatientIn | None = None
    waveform: WaveformIn | None = None
    signal: WaveformIn | None = None  # alias accepted for forward-compat

    def get_waveform(self) -> WaveformIn:
        wf = self.waveform or self.signal
        if wf is None:
            raise ValueError("provide 'waveform' (or 'signal') with the decoded leads")
        return wf


class Contribution(BaseModel):
    feature: str
    weight: float
    value: float


class PredictionOut(BaseModel):
    model: Literal["perceptron", "cnn"]
    probability: float
    positive: bool
    threshold: float
    latencyMs: float
    simulated: bool = False
    contributions: list[Contribution] = []


class CriteriaOut(BaseModel):
    sokolowLyon: dict
    cornell: dict
    agreement: str
    features: dict


class MetaOut(BaseModel):
    fs_used: int
    samples_used: int
    input_fs: float
    unit: str
    models_dir: str


class ScreenResponse(BaseModel):
    predictions: list[PredictionOut]
    criteria: CriteriaOut
    meta: MetaOut
    simulated: bool = False


class MetricOut(BaseModel):
    model: str
    family: Literal["criterion", "ml"]
    accuracy: float
    precision: float
    sensitivity: float
    specificity: float
    f1: float
    rocAuc: float
    prAuc: float | None = None
    threshold: float | None = None


class MetricsResponse(BaseModel):
    metrics: list[MetricOut]
    meta: dict = {}


class HealthResponse(BaseModel):
    status: str
    models: list[str]
    version: str
    metrics_available: bool = False
