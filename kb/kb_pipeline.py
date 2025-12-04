# # main.py

# import json
# import os

# from claim_extraction import extract_and_normalize_claims
# from embed_claims import process_claim_file
# from kb_builder import build_kb_from_claims
# from kb_query import evaluate_claims_against_kb  # STEP 4


# # -----------------------------------------------------
# # INPUT ARTICLE
# # -----------------------------------------------------
# text = """
# <your entire leukemia article here>
# """


# # =====================================================================
# # STEP 1 — Extract Claims
# # =====================================================================
# claims, claims_json_path = extract_and_normalize_claims(text)
# print("\n[STEP 1 COMPLETED] Saved to:", claims_json_path)


# # =====================================================================
# # STEP 2 — Embed Claims
# # =====================================================================
# print("\n[STEP 2 STARTED] Embedding extracted claims...")
# emb_path, meta_path = process_claim_file(claims_json_path)
# print("[STEP 2 COMPLETED] Embeddings saved.")


# # =====================================================================
# # STEP 3 — Build KB Index (Wikipedia + FactCheck + DataCommons + PolitiFact)
# # =====================================================================
# print("\n[STEP 3 STARTED] Building Knowledge Base...")

# with open(claims_json_path, "r", encoding="utf-8") as f:
#     claim_metadata = json.load(f)

# passages, kb_vectors, kb_index = build_kb_from_claims(
#     claim_metadata,
#     output_folder="kb_index"
# )

# print("[STEP 3 COMPLETED] KB Index Ready.")


# # =====================================================================
# # STEP 4 — Query KB and Decide if Claims Are True or False
# # =====================================================================
# print("\n[STEP 4 STARTED] Verifying the claims using the KB...")

# results = evaluate_claims_against_kb(claim_metadata, top_k=8)

# print("\n[STEP 4 COMPLETED] KB Verdicts:\n")

# for res in results:
#     print("CLAIM:", res["claim"]["claim_text"])
#     print("KB SCORE:", res["kb_score"])
#     print("FINAL VERDICT:", res["verdict"])
#     print("FACTCHECK OVERRIDE:", res["factcheck_override"])
#     print("----------")


# # =====================================================================
# # STEP 5 — FUSED KB SCORE (ACROSS ALL CLAIMS)
# # =====================================================================
# print("\n[STEP 5 STARTED] Computing fused KB score...")

# # Extract KB scores for all claims
# kb_scores = [r["kb_score"] for r in results]

# if not kb_scores:
#     fused_kb_score = 0.0
# else:
#     # Weighted fused KB score
#     weights = [max(0.05, s) for s in kb_scores]  # avoid zero-weight claims
#     fused_kb_score = sum(w * s for w, s in zip(weights, kb_scores)) / sum(weights)

# print("\n===== FUSED KB SCORE =====")
# print("Individual claim KB scores:", kb_scores)
# print("Fused KB Score:", fused_kb_score)


# print("\nPipeline Finished ✔")


import json
import os

from kb.claim_extraction import extract_and_normalize_claims
from kb.embed_claims import process_claim_file
from kb.kb_builder import build_kb_from_claims
from kb.kb_query import evaluate_claims_against_kb  


def run_kb_pipeline(article_text: str):
    """
    Runs the entire Knowledge Base fact-checking pipeline:
    
    1. Claim extraction
    2. Embedding
    3. KB building (Wikipedia / FactCheck / DataCommons / PolitiFact)
    4. Claim verification
    5. Computes fused KB score
    
    Returns:
        {
            "claims": list,
            "results": list,
            "fused_kb_score": float
        }
    """

    # -------------------------------
    # STEP 1 — Extract Claims
    # -------------------------------
    claims, claims_json_path = extract_and_normalize_claims(article_text)

    # -------------------------------
    # STEP 2 — Embed Claims
    # -------------------------------
    emb_path, meta_path = process_claim_file(claims_json_path)

    # -------------------------------
    # STEP 3 — Build KB Index
    # -------------------------------
    with open(claims_json_path, "r", encoding="utf-8") as f:
        claim_metadata = json.load(f)

    passages, kb_vectors, kb_index = build_kb_from_claims(
        claim_metadata,
        output_folder="kb_index"
    )

    # -------------------------------
    # STEP 4 — Evaluate Claims
    # -------------------------------
    results = evaluate_claims_against_kb(claim_metadata, top_k=8)

    # -------------------------------
    # STEP 5 — Compute fused KB score
    # -------------------------------
    kb_scores = [r["kb_score"] for r in results]

    if not kb_scores:
        fused_kb_score = 0.0
    else:
        weights = [max(0.05, s) for s in kb_scores]
        fused_kb_score = sum(w * s for w, s in zip(weights, kb_scores)) / sum(weights)

    # -------------------------------
    # FINAL OUTPUT
    # -------------------------------
    return {
        "claims": claims,
        "results": results,
        "fused_kb_score": float(fused_kb_score)
    }


# For testing only (not used in integration)
if __name__ == "__main__":
    sample = "Leukemia is a group of blood cancers."
    out = run_kb_pipeline(sample)
    print(json.dumps(out, indent=4))
