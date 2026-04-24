"""
Local evaluation script against the public JSON split.
Run: python evaluate.py
"""
import json
import sys
import re

# Mirror the prediction logic from main.py
BODY_REGIONS = {
    "BRAIN":    ["BRAIN", "HEAD", "CRANIAL", "NEURO", "STROKE"],
    "CHEST":    ["CHEST", "LUNG", "THORAX", "PULM"],
    "ABDOMEN":  ["ABDOMEN", "ABD "],
    "PELVIS":   ["PELVIS", "PELVIC"],
    "ABDPELV":  ["ABDOMEN AND PELVIS", "ABD/PELV", "ABD & PELV", "ABD AND PELV", "ABDOMEN/PELVIS"],
    "SPINE":    ["SPINE", "LUMBAR", "CERVICAL", "THORACIC", "SPINAL", "SCOLIOSIS"],
    "CARDIAC":  ["CARDIAC", "HEART", "CORONARY", "CARDIO", "MYO PERF", "MYOCARD", "ECHO"],
    "BREAST":   ["BREAST", "MAMMO", "MAM ", "MAMMOGRAPHY", "TOMO"],
    "NECK":     ["NECK", "THYROID"],
    "KNEE": ["KNEE"], "HIP": ["HIP"], "SHOULDER": ["SHOULDER"],
    "ANKLE": ["ANKLE"], "WRIST": ["WRIST"], "ELBOW": ["ELBOW"],
    "FOOT": ["FOOT", " FEET"], "HAND": ["HAND"],
    "LIVER": ["LIVER", "HEPAT"], "KIDNEY": ["KIDNEY", "RENAL"],
    "PROSTATE": ["PROSTATE"], "COLON": ["COLON"],
}
NOISE_WORDS = {"WITH","WITHOUT","AND","THE","W","CNTRST","CONTRAST","LIMITED",
               "COMPLETE","LEFT","RIGHT","RT","LT","DIAGNOSTIC","VIEW","PA",
               "LAT","AP","BILATERAL","BILAT","SCREEN","SCREENING"}

def extract_region(desc):
    d = desc.upper()
    for region, kws in BODY_REGIONS.items():
        for kw in kws:
            if kw in d: return region
    return "OTHER"

def extract_modality(desc):
    d = desc.upper()
    if "MRI" in d or d.startswith("MR "): return "MRI"
    if d.startswith("CT ") or " CT " in d or "/CT" in d: return "CT"
    if d.startswith("XR "): return "XR"
    if "MAMMO" in d or d.startswith("MAM "): return "MAMMO"
    if d.startswith("US ") or "ULTRASOUND" in d: return "US"
    if d.startswith("NM ") or "SPECT" in d or "PET" in d: return "NM"
    return "OTHER"

def predict(curr, prior):
    cr, pr = extract_region(curr), extract_region(prior)
    if cr != "OTHER" and cr == pr: return True
    if cr == "OTHER" and pr == "OTHER":
        cw = set(re.findall(r"[A-Z]+", curr.upper())) - NOISE_WORDS
        pw = set(re.findall(r"[A-Z]+", prior.upper())) - NOISE_WORDS
        if len(cw & pw) >= 2: return True
    cd, pd = curr.upper(), prior.upper()
    cm = extract_modality(curr)
    pm = extract_modality(prior)
    if cm == "MAMMO" and "BREAST" in pd: return True
    if pm == "MAMMO" and "BREAST" in cd: return True
    if "BREAST" in cd and ("BREAST" in pd or "MAMMO" in pd or "MAM " in pd): return True
    return False


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "relevant_priors_public.json"
    with open(path) as f:
        data = json.load(f)

    truth = {(t["case_id"], t["study_id"]): t["is_relevant_to_current"] for t in data["truth"]}
    tp = fp = tn = fn = 0

    for case in data["cases"]:
        curr_desc = case["current_study"]["study_description"]
        for prior in case["prior_studies"]:
            key = (case["case_id"], prior["study_id"])
            if key not in truth:
                continue
            true_label = truth[key]
            pred = predict(curr_desc, prior["study_description"])
            if pred and true_label:     tp += 1
            elif pred and not true_label: fp += 1
            elif not pred and true_label: fn += 1
            else:                         tn += 1

    total = tp + fp + tn + fn
    acc = (tp + tn) / total
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec  = tp / (tp + fn) if (tp + fn) else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0

    print(f"Total predictions : {total}")
    print(f"Accuracy          : {acc*100:.2f}%")
    print(f"Precision         : {prec:.4f}")
    print(f"Recall            : {rec:.4f}")
    print(f"F1                : {f1:.4f}")
    print(f"TP:{tp}  FP:{fp}  TN:{tn}  FN:{fn}")


if __name__ == "__main__":
    main()
