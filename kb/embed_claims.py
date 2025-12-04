import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# Load SBERT model
sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Folder for storing embeddings
EMBED_FOLDER = "embeddings"
os.makedirs(EMBED_FOLDER, exist_ok=True)


# ------------------------------------------------------
# Load claims JSON produced from Step 1
# ------------------------------------------------------
def load_claims(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------
# Encode claims using SBERT (normalized)
# ------------------------------------------------------
def embed_claims(claim_list, batch_size=16):
    texts = [c["claim_text"] for c in claim_list]

    vectors = sbert.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True  # IMPORTANT: allows dot-product = cosine similarity
    )

    return vectors


# ------------------------------------------------------
# Save embeddings and metadata for vector DB use
# ------------------------------------------------------
def save_embeddings(vectors, claim_list):
    embedding_path = os.path.join(EMBED_FOLDER, "claim_embeddings.npy")
    meta_path = os.path.join(EMBED_FOLDER, "claim_metadata.json")

    # Save vectors
    np.save(embedding_path, vectors)

    # Store claim_text and indexes
    meta = []
    for idx, c in enumerate(claim_list):
        meta.append({
            "id": idx,
            "claim_text": c["claim_text"],
            "numbers": c["numbers"],
            "dates": c["dates"],
            "entities": c["entities"]
        })

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return embedding_path, meta_path


# ------------------------------------------------------
# FULL PIPELINE (Step 2)
# ------------------------------------------------------
def process_claim_file(json_path):
    claims = load_claims(json_path)
    print(f"[INFO] Loaded {len(claims)} claims.")

    vectors = embed_claims(claims)
    print("[INFO] Embeddings generated.")

    emb_path, meta_path = save_embeddings(vectors, claims)
    print(f"[INFO] Saved embeddings to {emb_path}")
    print(f"[INFO] Saved metadata to {meta_path}")

    return emb_path, meta_path
