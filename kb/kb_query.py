# kb_query.py
import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Dict, Any
from datetime import datetime
from kb.claim_extraction import normalize_numbers, normalize_dates  # reuse your functions

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL = SentenceTransformer(MODEL_NAME)

KB_FOLDER = "kb_index"
INDEX_PATH = os.path.join(KB_FOLDER, "kb.index")
META_PATH = os.path.join(KB_FOLDER, "metadata.json")
EMB_PATH = os.path.join(KB_FOLDER, "kb_embeddings.npy")

# -------------------------
# helper: load KB
# -------------------------
def load_kb(folder=KB_FOLDER):
    if not os.path.exists(os.path.join(folder, "kb.index")):
        raise FileNotFoundError("FAISS index not found. Build KB first.")
    index = faiss.read_index(os.path.join(folder, "kb.index"))
    with open(os.path.join(folder, "metadata.json"), "r", encoding="utf-8") as f:
        metadata = json.load(f)
    embeddings = None
    if os.path.exists(os.path.join(folder, "kb_embeddings.npy")):
        embeddings = np.load(os.path.join(folder, "kb_embeddings.npy"))
    return index, metadata, embeddings

# -------------------------
# score helpers
# -------------------------
def structured_number_score(claim_numbers, passage_text):
    """
    If claim contains numbers and passage contains numbers,
    reward close matches, penalize large contradictions.
    """
    if not claim_numbers:
        return 0.0
    passage_nums = normalize_numbers(passage_text)
    if not passage_nums:
        return 0.0
    # compare first numeric item for simplicity
    c = claim_numbers[0]["value"]
    p = passage_nums[0]["value"]
    if c == 0 or p == 0:
        return 0.0
    rel_diff = abs(c - p) / max(abs(c), abs(p))
    if rel_diff < 0.05:
        return 0.35
    if rel_diff < 0.2:
        return 0.15
    if rel_diff > 0.5:
        return -0.5
    return 0.0

def structured_date_score(claim_dates, passage_text):
    if not claim_dates:
        return 0.0
    # naive: check if any claim date string appears in passage
    for cd in claim_dates:
        if cd in passage_text:
            return 0.25
    return 0.0

# -------------------------
# evidence scoring for a single passage
# -------------------------
def score_passage(similarity_score, passage_meta, claim_obj):
    # similarity_score = inner product (cosine) from FAISS; expected in [0,1]
    sim = float(similarity_score)
    prov_weight = 0.8
    src = passage_meta.get("source", "")
    if src == "google_factcheck":
        prov_weight = 1.0
    elif src == "data_commons":
        prov_weight = 0.95
    elif src == "wikipedia":
        prov_weight = 0.85
    # base semantic support
    semantic_support = max(0.0, (sim - 0.35) / (1.0 - 0.35))  # thresholded mapping
    evidence = semantic_support * prov_weight

    # structured bonuses/penalties
    claim_nums = claim_obj.get("numbers", [])
    evidence += structured_number_score(claim_nums, passage_meta.get("text", ""))

    claim_dates = claim_obj.get("dates", [])
    evidence += structured_date_score(claim_dates, passage_meta.get("text", ""))

    # factcheck verdict influence (not override here, just evidence tweak)
    verdict = passage_meta.get("verdict")
    if verdict:
        v = verdict.lower()
        if "true" in v or "correct" in v or "true" == v:
            evidence += 0.6
        elif "false" in v or "incorrect" in v:
            evidence -= 0.9
        elif "mixed" in v or "partly" in v:
            evidence -= 0.1

    # clip
    evidence = max(-1.0, min(1.0, evidence))
    return evidence

# -------------------------
# aggregate evidence -> KB score
# -------------------------
def aggregate_evidence(evidence_list, similarity_list):
    # weighted sum where weight = sim^2 to favor high-sim items
    if not evidence_list:
        return 0.0
    weights = [s*s for s in similarity_list]
    num = sum(w*e for w,e in zip(weights, evidence_list))
    den = sum(weights) + 1e-9
    raw = num / den
    # normalize from [-1,1] to [0,1]
    kb_score = (raw + 1.0) / 2.0
    return kb_score

# -------------------------
# main: compare a list of claims against KB
# -------------------------
def evaluate_claims_against_kb(claims: List[Dict[str,Any]], top_k=8):
    index, metadata, _ = load_kb()
    results = []
    for claim in claims:
        text = claim.get("claim_text")
        # encode
        qvec = MODEL.encode([text], normalize_embeddings=True)
        sims, idxs = index.search(qvec.astype("float32"), top_k)
        sims = sims[0].tolist()
        idxs = idxs[0].tolist()

        evidence_scores = []
        supporting = []
        contradicting = []
        factcheck_override = None

        for sim, idx in zip(sims, idxs):
            if idx < 0:
                continue
            meta = metadata[idx]
            # score passage
            ev = score_passage(sim, meta, claim)
            evidence_scores.append(ev)

            # collect supporting/contradicting lists for explanation
            if ev > 0.25:
                supporting.append({"score": ev, "sim": sim, "passage": meta})
            if ev < -0.25:
                contradicting.append({"score": ev, "sim": sim, "passage": meta})

            # check for authoritative factcheck override
            if meta.get("source") == "google_factcheck" and meta.get("verdict"):
                # require reasonable semantic match to apply override
                if sim >= 0.7:
                    fv = meta.get("verdict").lower()
                    if "false" in fv or "incorrect" in fv:
                        factcheck_override = {"label": "false", "meta": meta}
                    elif "true" in fv or "correct" in fv:
                        factcheck_override = {"label": "true", "meta": meta}
                    # once we find a strong factcheck match we can break or keep scanning to gather explanation

        # aggregation
        kb_score = aggregate_evidence(evidence_scores, sims)

        # apply factcheck override logic
        if factcheck_override:
            if factcheck_override["label"] == "true":
                kb_score = max(kb_score, 0.95)
            elif factcheck_override["label"] == "false":
                kb_score = min(kb_score, 0.05)

        # interpret
        if kb_score >= 0.75:
            verdict = "SUPPORTED"
        elif kb_score >= 0.45:
            verdict = "AMBIGUOUS"
        else:
            verdict = "NOT_SUPPORTED"

        results.append({
            "claim": claim,
            "kb_score": kb_score,
            "verdict": verdict,
            "top_supporting": supporting[:5],
            "top_contradicting": contradicting[:5],
            "factcheck_override": factcheck_override
        })
    return results

# -------------------------
# Example CLI usage
# -------------------------
if __name__ == "__main__":
    # Load claims file produced by Step1
    claims_file = "extracted_claims/claims_latest.json"  # replace
    if not os.path.exists(claims_file):
        # pick a random file from folder
        fls = os.listdir("extracted_claims")
        if not fls:
            raise SystemExit("No extracted_claims JSON found; run Step1 first.")
        claims_file = os.path.join("extracted_claims", fls[-1])

    with open(claims_file, "r", encoding="utf-8") as f:
        claims = json.load(f)

    # Build KB first? If you haven't built KB, run kb_builder.build_kb_from_claims(...)
    # Here we assume kb_index exists
    res = evaluate_claims_against_kb(claims, top_k=8)
    print(json.dumps(res, indent=2, ensure_ascii=False))
