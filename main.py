"""
Relevant Priors API
Challenge: relevant-priors-v1

Predicts whether a prior radiology examination is relevant to a current study.
Uses anatomical region + modality heuristics derived from study descriptions.
"""

import re
import logging
from typing import Any
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Relevant Priors API", version="1.0.0")

# ── Anatomical region keyword mapping ────────────────────────────────────────
BODY_REGIONS: dict[str, list[str]] = {
    "BRAIN":    ["BRAIN", "HEAD", "CRANIAL", "NEURO", "STROKE"],
    "CHEST":    ["CHEST", "LUNG", "THORAX", "PULM"],
    "ABDOMEN":  ["ABDOMEN", "ABD "],
    "PELVIS":   ["PELVIS", "PELVIC"],
    "ABDPELV":  ["ABDOMEN AND PELVIS", "ABD/PELV", "ABD & PELV",
                  "ABD AND PELV", "ABDOMEN/PELVIS"],
    "SPINE":    ["SPINE", "LUMBAR", "CERVICAL", "THORACIC", "SPINAL", "SCOLIOSIS"],
    "CARDIAC":  ["CARDIAC", "HEART", "CORONARY", "CARDIO", "MYO PERF",
                  "MYOCARD", "ECHO"],
    "BREAST":   ["BREAST", "MAMMO", "MAM ", "MAMMOGRAPHY", "TOMO"],
    "NECK":     ["NECK", "THYROID"],
    "KNEE":     ["KNEE"],
    "HIP":      ["HIP"],
    "SHOULDER": ["SHOULDER"],
    "ANKLE":    ["ANKLE"],
    "WRIST":    ["WRIST"],
    "ELBOW":    ["ELBOW"],
    "FOOT":     ["FOOT", " FEET"],
    "HAND":     ["HAND"],
    "LIVER":    ["LIVER", "HEPAT"],
    "KIDNEY":   ["KIDNEY", "RENAL"],
    "PROSTATE": ["PROSTATE"],
    "COLON":    ["COLON"],
}

MODALITY_ALIASES = {"MAMMO", "MAM"}

NOISE_WORDS = {
    "WITH", "WITHOUT", "AND", "THE", "W", "CNTRST", "CONTRAST", "LIMITED",
    "COMPLETE", "LEFT", "RIGHT", "RT", "LT", "DIAGNOSTIC", "VIEW", "PA",
    "LAT", "AP", "BILATERAL", "BILAT", "SCREEN", "SCREENING",
}


def extract_region(desc: str) -> str:
    d = desc.upper()
    for region, keywords in BODY_REGIONS.items():
        for kw in keywords:
            if kw in d:
                return region
    return "OTHER"


def extract_modality(desc: str) -> str:
    d = desc.upper()
    if "MRI" in d or d.startswith("MR "):
        return "MRI"
    if d.startswith("CT ") or " CT " in d or "/CT" in d:
        return "CT"
    if d.startswith("XR ") or "XRAY" in d:
        return "XR"
    if "MAMMO" in d or d.startswith("MAM "):
        return "MAMMO"
    if d.startswith("US ") or "ULTRASOUND" in d:
        return "US"
    if d.startswith("NM ") or "SPECT" in d or "PET" in d:
        return "NM"
    if "DEXA" in d or "DXA" in d:
        return "DEXA"
    return "OTHER"


def predict_is_relevant(current_desc: str, prior_desc: str) -> bool:
    curr_reg = extract_region(current_desc)
    prior_reg = extract_region(prior_desc)

    # Primary signal: same anatomical region
    if curr_reg != "OTHER" and curr_reg == prior_reg:
        return True

    # Both unclassified: fall back to meaningful token overlap
    if curr_reg == "OTHER" and prior_reg == "OTHER":
        curr_words = set(re.findall(r"[A-Z]+", current_desc.upper())) - NOISE_WORDS
        prior_words = set(re.findall(r"[A-Z]+", prior_desc.upper())) - NOISE_WORDS
        if len(curr_words & prior_words) >= 2:
            return True

    # Cross-modality breast studies (mammogram ↔ breast US)
    curr_d = current_desc.upper()
    prior_d = prior_desc.upper()
    curr_mod = extract_modality(current_desc)
    prior_mod = extract_modality(prior_desc)
    if curr_mod == "MAMMO" and "BREAST" in prior_d:
        return True
    if prior_mod == "MAMMO" and "BREAST" in curr_d:
        return True
    if "BREAST" in curr_d and ("BREAST" in prior_d or "MAMMO" in prior_d or "MAM " in prior_d):
        return True

    return False


# ── Pydantic models ───────────────────────────────────────────────────────────

class CurrentStudy(BaseModel):
    study_id: str
    study_description: str
    study_date: str

class PriorStudy(BaseModel):
    study_id: str
    study_description: str
    study_date: str

class Case(BaseModel):
    case_id: str
    patient_id: str | None = None
    patient_name: str | None = None
    current_study: CurrentStudy
    prior_studies: list[PriorStudy]

class PredictRequest(BaseModel):
    challenge_id: str | None = None
    schema_version: int | None = None
    generated_at: str | None = None
    cases: list[Case]

class Prediction(BaseModel):
    case_id: str
    study_id: str
    predicted_is_relevant: bool

class PredictResponse(BaseModel):
    predictions: list[Prediction]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
async def predict(request: Request, body: PredictRequest):
    logger.info(
        "Request received: %d case(s), %d total prior studies",
        len(body.cases),
        sum(len(c.prior_studies) for c in body.cases),
    )

    predictions: list[Prediction] = []

    for case in body.cases:
        curr_desc = case.current_study.study_description
        for prior in case.prior_studies:
            relevant = predict_is_relevant(curr_desc, prior.study_description)
            predictions.append(
                Prediction(
                    case_id=case.case_id,
                    study_id=prior.study_id,
                    predicted_is_relevant=relevant,
                )
            )

    logger.info(
        "Returning %d predictions (%d relevant)",
        len(predictions),
        sum(1 for p in predictions if p.predicted_is_relevant),
    )
    return PredictResponse(predictions=predictions)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error: %s", exc)
    return JSONResponse(status_code=500, content={"error": str(exc)})
