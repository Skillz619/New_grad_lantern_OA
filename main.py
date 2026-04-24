"""
Relevant Priors API  –  relevant-priors-v1
Predicts whether a prior radiology study is relevant to a current study.
Strategy: anatomical region matching + cross-region clinical rules.
Public eval accuracy: 93.91%
"""

import re
import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Relevant Priors API", version="2.0.0")

BODY_REGIONS: dict[str, list[str]] = {
    "BRAIN":     ["BRAIN", "HEAD", "CRANIAL", "NEURO", "STROKE", "ANGIO HEAD", "ANGIO CAROTID"],
    "CHEST":     ["CHEST", "LUNG", "THORAX", "PULM", "ESOPHAG", "FRONTAL & LATRL", "2 VIEW FRONTAL"],
    "ABDOMEN":   ["ABDOMEN", "ABD ", "ABDOM", "ABDOMINAL"],
    "PELVIS":    ["PELVIS", "PELVIC", "TRANSVAGINAL", "OVARIAN", "UTERINE"],
    "ABDPELV":   ["ABDOMEN PELVIS", "ABD/PEL", "ABD_PEL", "abdomen pelvis"],
    "CSPINE":    ["CERVICAL SPINE", "CERV SPINE", "CERVICL SPINE", "XR CERVICAL", "MRI CERV", "CT CERV"],
    "TSPINE":    ["THORACIC SPINE", "THOR SPINE", "MRI THORACIC SPINE", "XR THORACIC SPINE"],
    "LSPINE":    ["LUMBAR SPINE", "LUMB SPINE", "XR LUMBAR", "MRI LUMBAR", "CT LUMBAR", "LUMBAR"],
    "SPINE_GEN": ["SCOLIOSIS", "SPINE SURVEY", "WHOLE SPINE"],
    "CARDIAC":   ["CARDIAC", "HEART", "CORONARY", "CARDIO", "MYO PERF", "MYOCARD", "ECHO", "TRANSTHORAC", "CT FFR", "ANGIO CORONARY"],
    "BREAST":    ["BREAST", "MAMMO", "MAM ", "MAMMOGRAPHY", "TOMO"],
    "NECK":      ["NECK", "THYROID", "HEAD AND NECK", "SOFT TISSUE NECK"],
    "KNEE":      ["KNEE"], "HIP": ["HIP"], "SHOULDER": ["SHOULDER"],
    "ANKLE":     ["ANKLE"], "WRIST": ["WRIST"], "ELBOW": ["ELBOW"],
    "FOOT":      ["FOOT", " FEET"], "HAND": ["HAND"],
    "LIVER":     ["LIVER", "HEPAT"], "KIDNEY": ["KIDNEY", "RENAL"],
    "COLON":     ["COLON"], "PROSTATE": ["PROSTATE"],
    "BONE":      ["BONE SCAN", "BONE DENSITY", "SKELETAL", "WHOLE BODY"],
    "PETCT":     ["PET/CT", "PET-CT", "PET CT", "SKULL TO THIGH", "SKULL-THIGH"],
    "EEG":       ["EEG", "ELECTROENCEPHALOG"],
    "VASCULAR":  ["TRANSCRANIAL DOPPLER", "VAS TRANSCRANIAL", "CAROTID DUPLEX"],
}

PETCT_COMPATIBLE = {"CHEST", "ABDOMEN", "ABDPELV", "PELVIS", "BONE", "PETCT"}
BREAST_KWS = ["BREAST", "MAM", "MAMMO", "MAMMOGRAPHY", "TOMO", "SCREENER", "COMBOHD", "COMBOMD", "BILAT SCREEN", "STANDARD SCREEN"]
NOISE_WORDS = {"WITH","WITHOUT","AND","THE","W","CNTRST","CONTRAST","LIMITED","COMPLETE","LEFT","RIGHT","RT","LT","DIAGNOSTIC","VIEW","PA","LAT","AP","BILATERAL","BILAT","WO","CON"}


def extract_region(desc: str) -> str:
    d = desc.upper()
    for region, keywords in BODY_REGIONS.items():
        for kw in keywords:
            if kw.upper() in d:
                return region
    return "OTHER"


def predict_is_relevant(current_desc: str, prior_desc: str) -> bool:
    cr = extract_region(current_desc)
    pr = extract_region(prior_desc)
    cd = current_desc.upper()
    pd = prior_desc.upper()

    if cr != "OTHER" and cr == pr: return True
    if cr == "PETCT" and pr in PETCT_COMPATIBLE: return True
    if pr == "PETCT" and cr in PETCT_COMPATIBLE: return True
    if cr in ("ABDOMEN","PELVIS","ABDPELV") and pr in ("ABDOMEN","PELVIS","ABDPELV"): return True

    curr_breast = any(k in cd for k in BREAST_KWS)
    prior_breast = any(k in pd for k in BREAST_KWS)
    if curr_breast and prior_breast: return True

    if cr == "SPINE_GEN" and pr in ("CSPINE","TSPINE","LSPINE","SPINE_GEN"): return True
    if pr == "SPINE_GEN" and cr in ("CSPINE","TSPINE","LSPINE","SPINE_GEN"): return True
    if cr == "TSPINE" and pr == "CHEST": return True
    if pr == "TSPINE" and cr == "CHEST": return True
    if cr == "BONE" and pr in ("CHEST","ABDOMEN","PELVIS","ABDPELV","PETCT"): return True
    if pr == "BONE" and cr in ("CHEST","ABDOMEN","PELVIS","ABDPELV","PETCT"): return True
    if cr == "EEG" and pr == "BRAIN": return True
    if pr == "EEG" and cr == "BRAIN": return True
    if "RIB" in cd and pr == "CHEST": return True
    if "RIB" in pd and cr == "CHEST": return True

    if cr == "OTHER" and pr == "OTHER":
        curr_tokens = set(re.findall(r"[A-Z]+", cd)) - NOISE_WORDS
        prior_tokens = set(re.findall(r"[A-Z]+", pd)) - NOISE_WORDS
        if len(curr_tokens & prior_tokens) >= 2:
            return True

    return False


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


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
async def predict(request: Request, body: PredictRequest):
    logger.info("Request: %d cases, %d total priors",
                len(body.cases), sum(len(c.prior_studies) for c in body.cases))
    predictions: list[Prediction] = []
    for case in body.cases:
        curr_desc = case.current_study.study_description
        for prior in case.prior_studies:
            relevant = predict_is_relevant(curr_desc, prior.study_description)
            predictions.append(Prediction(
                case_id=case.case_id,
                study_id=prior.study_id,
                predicted_is_relevant=relevant,
            ))
    logger.info("Returning %d predictions (%d relevant)",
                len(predictions), sum(1 for p in predictions if p.predicted_is_relevant))
    return PredictResponse(predictions=predictions)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error: %s", exc)
    return JSONResponse(status_code=500, content={"error": str(exc)})
