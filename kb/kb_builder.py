# import os
# import json
# import time
# import hashlib
# import requests
# import numpy as np
# import faiss
# from pathlib import Path
# from typing import List, Dict, Any, Optional
# from urllib.parse import quote

# # PolitiFact scraping deps
# from ddgs import DDGS
# from bs4 import BeautifulSoup


# # -------------------------
# # Import claim helpers
# # -------------------------
# try:
#     from claim_extraction import normalize_numbers, normalize_dates, extract_entities
# except:
#     normalize_numbers = lambda x: []
#     normalize_dates = lambda x, y=None: []
#     extract_entities = lambda x: []


# # -------------------------
# # SBERT Model
# # -------------------------
# from sentence_transformers import SentenceTransformer

# MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# MODEL = SentenceTransformer(MODEL_NAME)
# EMBED_DIM = MODEL.get_sentence_embedding_dimension()

# KB_FOLDER = "kb_index"
# os.makedirs(KB_FOLDER, exist_ok=True)


# # ===========================================================
# # SAFE CACHE KEY (Fixes Windows FileNotFoundError)
# # ===========================================================
# def make_cache_key(prefix: str, text: str):
#     digest = hashlib.md5(text.encode("utf-8")).hexdigest()
#     return f"{prefix}_{digest}"


# # ===========================================================
# # HEADERS (Wikipedia requires real User-Agent)
# # ===========================================================
# HEADERS = {
#     "User-Agent": "FakeNewsDetector/1.0 (https://github.com/nikhilsai5; contact: nikhilsaimanam5@gmail.com)"
# }

# # Cache folder
# CACHE_DIR = Path(".kb_cache")
# CACHE_DIR.mkdir(exist_ok=True)


# # ===========================================================
# # SAFE GET with retries and User-Agent
# # ===========================================================
# def safe_get(url, params=None, timeout=12, retries=3):
#     for attempt in range(retries):
#         try:
#             r = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
#             r.raise_for_status()
#             return r
#         except requests.HTTPError as e:
#             if r.status_code in [400, 401, 403]:
#                 raise e
#             time.sleep(0.7 * (attempt + 1))
#     raise RuntimeError(f"Failed GET {url}")


# def cache_load(key):
#     fp = CACHE_DIR / f"{key}.json"
#     if fp.exists():
#         try:
#             return json.loads(fp.read_text())
#         except:
#             return None
#     return None


# def cache_save(key, data):
#     fp = CACHE_DIR / f"{key}.json"
#     fp.write_text(json.dumps(data, indent=2))


# # ===========================================================
# # WIKIPEDIA SEARCH + EXTRACT (cached, safe)
# # ===========================================================
# def fetch_wikipedia_passages(entity: str, max_pages=2) -> List[Dict[str, Any]]:
#     entity = entity.strip()
#     if not entity:
#         return []

#     # hashed key
#     cache_key = make_cache_key("wiki", entity)
#     cached = cache_load(cache_key)
#     if cached:
#         return cached

#     url = "https://en.wikipedia.org/w/api.php"

#     params = {
#         "action": "query",
#         "list": "search",
#         "srsearch": entity,
#         "format": "json",
#         "utf8": 1,
#         "srlimit": max_pages
#     }

#     try:
#         r = safe_get(url, params=params)
#         data = r.json()
#     except Exception as e:
#         print("[WARN] Wikipedia search failed:", e)
#         return []

#     passages = []
#     search_items = data.get("query", {}).get("search", [])

#     for s in search_items:
#         pageid = s["pageid"]

#         extract_params = {
#             "action": "query",
#             "prop": "extracts",
#             "pageids": pageid,
#             "explaintext": True,
#             "format": "json"
#         }

#         try:
#             r2 = safe_get(url, params=extract_params)
#             extract = r2.json()["query"]["pages"][str(pageid)].get("extract", "")
#         except:
#             continue

#         for para in extract.split("\n"):
#             para = para.strip()
#             if len(para.split()) >= 10:
#                 passages.append({
#                     "source": "wikipedia",
#                     "text": para,
#                     "date": None,
#                     "verdict": None
#                 })

#     cache_save(cache_key, passages)
#     return passages


# # ===========================================================
# # GOOGLE FACTCHECK API
# # ===========================================================
# FACTCHECK_KEY = os.environ.get("FACTCHECK_API_KEY", "").strip()

# def fetch_factcheck(claim: str):
#     if not FACTCHECK_KEY:
#         return []

#     url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
#     params = {"query": claim, "key": FACTCHECK_KEY}

#     try:
#         r = safe_get(url, params=params)
#     except Exception as e:
#         print("[WARN] FactCheck fetch failed:", e)
#         return []

#     data = r.json()
#     out = []

#     for c in data.get("claims", []):
#         text = c.get("text", "")
#         for review in c.get("claimReview", []):
#             verdict = review.get("textualRating")
#             pub = review.get("publisher", {}).get("name")
#             out.append({
#                 "source": "google_factcheck",
#                 "text": f"{text} | Verdict: {verdict} ({pub})",
#                 "verdict": verdict,
#                 "date": review.get("reviewDate")
#             })

#     return out


# # ===========================================================
# # DATA COMMONS (basic)
# # ===========================================================
# def fetch_datacommons(entity: str):
#     vars = ["Count_Person", "Amount_GrossDomesticProduct_USD"]
#     out = []

#     for sv in vars:
#         url = "https://api.datacommons.org/stat/value"
#         params = {"stat_var": sv, "place": entity}

#         try:
#             r = safe_get(url, params=params)
#             data = r.json()
#             if "value" in data:
#                 out.append({
#                     "source": "data_commons",
#                     "text": f"{entity} {sv} = {data['value']}",
#                     "verdict": None,
#                     "date": data.get("date")
#                 })
#         except:
#             pass

#     return out


# # ===========================================================
# # POLITIFACT INTEGRATION
# # ===========================================================

# def search_politifact_urls(query: str, max_results: int = 5) -> List[str]:
#     urls = []
#     try:
#         with DDGS() as ddgs:
#             for result in ddgs.text(f"{query} site:politifact.com", max_results=max_results):
#                 url = result.get("href") or result.get("url")
#                 if url and "politifact.com" in url:
#                     urls.append(url)
#     except Exception as e:
#         print("[WARN] PolitiFact search failed:", e)
#         return []

#     return list(dict.fromkeys(urls))


# def scrape_politifact_page(url: str) -> Optional[Dict[str, Any]]:
#     try:
#         r = safe_get(url)
#         soup = BeautifulSoup(r.text, "html.parser")
#     except Exception as e:
#         print("[WARN] Could not scrape PolitiFact:", url, e)
#         return None

#     ruling_tag = soup.select_one(".meter .meter__rating")
#     ruling = ruling_tag.get_text(strip=True) if ruling_tag else None

#     claim_tag = soup.select_one(".m-statement__quote")
#     claim_text = claim_tag.get_text(strip=True) if claim_tag else None

#     summary_tag = soup.select_one(".short-on-time")
#     summary = summary_tag.get_text(strip=True) if summary_tag else None

#     date_tag = soup.select_one(".m-statement__date")
#     date = date_tag.get_text(strip=True) if date_tag else None

#     if not ruling:
#         return None

#     return {
#         "source": "politifact",
#         "text": f"PolitiFact ruling: {ruling}. Claim: {claim_text or ''}. Summary: {summary or ''}",
#         "verdict": ruling,
#         "date": date,
#         "url": url
#     }


# def fetch_politifact_factchecks(claim: str, max_results: int = 5) -> List[Dict[str, Any]]:
#     urls = search_politifact_urls(claim, max_results)
#     results = []
#     for url in urls:
#         fc = scrape_politifact_page(url)
#         if fc:
#             results.append(fc)
#         time.sleep(0.2)
#     return results


# # ===========================================================
# # GATHER PASSAGES FOR CLAIM
# # ===========================================================
# def gather_passages(claim_text, entity_list):
#     combined = []

#     # PolitiFact 🔥
#     try:
#         combined += fetch_politifact_factchecks(claim_text)
#     except Exception as e:
#         print("[WARN] PolitiFact failed:", e)

#     # Google FactCheck
#     combined += fetch_factcheck(claim_text)

#     # Wikipedia + DataCommons from entities
#     for ent in entity_list:
#         ent = ent.replace("’", "'")
#         combined += fetch_wikipedia_passages(ent)
#         combined += fetch_datacommons(ent)
#         time.sleep(0.1)

#     # Fallback: Wikipedia search on whole claim text
#     if not combined:
#         combined += fetch_wikipedia_passages(claim_text)

#     # Dedupe
#     seen = set()
#     final = []
#     for p in combined:
#         key = p["text"][:200]
#         if key not in seen:
#             seen.add(key)
#             final.append(p)

#     return final


# # ===========================================================
# # BUILD KB (FAISS INDEX)
# # ===========================================================
# def build_kb_from_claims(claims, output_folder="kb_index"):
#     passages = []

#     for c in claims:
#         text = c.get("claim_text", "")
#         ents = [e["text"] for e in c.get("entities", [])]
#         p = gather_passages(text, ents)
#         for x in p:
#             x["_claim"] = text
#         passages.extend(p)

#     # Global dedupe
#     uniq = {}
#     final_passages = []
#     for p in passages:
#         key = p["text"][:200]
#         if key not in uniq:
#             uniq[key] = True
#             final_passages.append(p)

#     if not final_passages:
#         raise RuntimeError("[ERROR] No KB passages collected.")

#     print(f"[INFO] Collected {len(final_passages)} KB passages.")

#     # Encode
#     texts = [p["text"] for p in final_passages]
#     vectors = MODEL.encode(texts, normalize_embeddings=True, batch_size=16).astype("float32")

#     # Build FAISS
#     dim = vectors.shape[1]
#     index = faiss.IndexFlatIP(dim)
#     index.add(vectors)

#     os.makedirs(output_folder, exist_ok=True)

#     faiss.write_index(index, f"{output_folder}/kb.index")
#     np.save(f"{output_folder}/kb_embeddings.npy", vectors)

#     with open(f"{output_folder}/metadata.json", "w", encoding="utf-8") as f:
#         json.dump(final_passages, f, indent=2, ensure_ascii=False)

#     print("[INFO] KB saved →", output_folder)
#     return final_passages, vectors, index


# # ===========================================================
# # TEST
# # ===========================================================
# if __name__ == "__main__":
#     test_claims = [
#         {"claim_text": "Joe Biden said inflation is zero", "entities": [{"text":"Biden"}]},
#     ]
#     build_kb_from_claims(test_claims)


# kb_builder.py
import os
import json
import time
import hashlib
import requests
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Any, Optional
from urllib.parse import quote

# -------------------------
# Import claim helpers
# -------------------------
try:
    from claim_extraction import normalize_numbers, normalize_dates, extract_entities
except:
    normalize_numbers = lambda x: []
    normalize_dates = lambda x, y=None: []
    extract_entities = lambda x: []


# -------------------------
# SBERT Model
# -------------------------
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL = SentenceTransformer(MODEL_NAME)
EMBED_DIM = MODEL.get_sentence_embedding_dimension()

KB_FOLDER = "kb_index"
os.makedirs(KB_FOLDER, exist_ok=True)


# ===========================================================
# SAFE CACHE KEY
# ===========================================================
def make_cache_key(prefix: str, text: str):
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return f"{prefix}_{digest}"


# ===========================================================
# HEADERS
# ===========================================================
HEADERS = {
    "User-Agent": "FakeNewsDetector/1.0 (https://github.com/nikhilsai5; contact: nikhilsaimanam5@gmail.com)"
}

CACHE_DIR = Path(".kb_cache")
CACHE_DIR.mkdir(exist_ok=True)


# ===========================================================
# SAFE GET
# ===========================================================
def safe_get(url, params=None, timeout=12, retries=3):
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
            r.raise_for_status()
            return r
        except requests.HTTPError as e:
            if r.status_code in [400, 401, 403]:
                raise e
            time.sleep(0.7 * (attempt + 1))
    raise RuntimeError(f"Failed GET {url}")


def cache_load(key):
    fp = CACHE_DIR / f"{key}.json"
    if fp.exists():
        try:
            return json.loads(fp.read_text())
        except:
            return None
    return None


def cache_save(key, data):
    fp = CACHE_DIR / f"{key}.json"
    fp.write_text(json.dumps(data, indent=2))


# ===========================================================
# WIKIPEDIA SEARCH + PASSAGES
# ===========================================================
def fetch_wikipedia_passages(entity: str, max_pages=2) -> List[Dict[str, Any]]:
    entity = entity.strip()
    if not entity:
        return []

    cache_key = make_cache_key("wiki", entity)
    cached = cache_load(cache_key)
    if cached:
        return cached

    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": entity,
        "format": "json",
        "utf8": 1,
        "srlimit": max_pages
    }

    try:
        r = safe_get(url, params=params)
        data = r.json()
    except Exception as e:
        print("[WARN] Wikipedia search failed:", e)
        return []

    passages = []
    search_items = data.get("query", {}).get("search", [])

    for s in search_items:
        pageid = s["pageid"]
        extract_params = {
            "action": "query",
            "prop": "extracts",
            "pageids": pageid,
            "explaintext": True,
            "format": "json"
        }

        try:
            r2 = safe_get(url, params=extract_params)
            extract = r2.json()["query"]["pages"][str(pageid)].get("extract", "")
        except:
            continue

        for para in extract.split("\n"):
            para = para.strip()
            if len(para.split()) >= 10:
                passages.append({
                    "source": "wikipedia",
                    "text": para,
                    "date": None,
                    "verdict": None
                })

    cache_save(cache_key, passages)
    return passages


# ===========================================================
# GOOGLE FACTCHECK API
# ===========================================================
FACTCHECK_KEY = os.environ.get("FACTCHECK_API_KEY", "").strip()

def fetch_factcheck(claim: str):
    if not FACTCHECK_KEY:
        return []

    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    params = {"query": claim, "key": FACTCHECK_KEY}

    try:
        r = safe_get(url, params=params)
    except Exception as e:
        print("[WARN] FactCheck fetch failed:", e)
        return []

    data = r.json()
    out = []

    for c in data.get("claims", []):
        text = c.get("text", "")
        for review in c.get("claimReview", []):
            verdict = review.get("textualRating")
            pub = review.get("publisher", {}).get("name")
            out.append({
                "source": "google_factcheck",
                "text": f"{text} | Verdict: {verdict} ({pub})",
                "verdict": verdict,
                "date": review.get("reviewDate")
            })

    return out


# ===========================================================
# DATA COMMONS
# ===========================================================
def fetch_datacommons(entity: str):
    vars = ["Count_Person", "Amount_GrossDomesticProduct_USD"]
    out = []

    for sv in vars:
        url = "https://api.datacommons.org/stat/value"
        params = {"stat_var": sv, "place": entity}

        try:
            r = safe_get(url, params=params)
            data = r.json()
            if "value" in data:
                out.append({
                    "source": "data_commons",
                    "text": f"{entity} {sv} = {data['value']}",
                    "verdict": None,
                    "date": data.get("date")
                })
        except:
            pass

    return out


# ===========================================================
# GATHER PASSAGES (NO POLITIFACT)
# ===========================================================
def gather_passages(claim_text, entity_list):
    combined = []

    # Google FactCheck
    combined += fetch_factcheck(claim_text)

    # Wikipedia + DataCommons
    for ent in entity_list:
        ent = ent.replace("’", "'")
        combined += fetch_wikipedia_passages(ent)
        combined += fetch_datacommons(ent)
        time.sleep(0.1)

    # Fallback: Wikipedia search on full claim
    if not combined:
        combined += fetch_wikipedia_passages(claim_text)

    # Dedupe
    seen = set()
    final = []
    for p in combined:
        key = p["text"][:200]
        if key not in seen:
            seen.add(key)
            final.append(p)

    return final


# ===========================================================
# BUILD KB (FAISS INDEX)
# ===========================================================
def build_kb_from_claims(claims, output_folder="kb_index"):
    passages = []

    for c in claims:
        text = c.get("claim_text", "")
        ents = [e["text"] for e in c.get("entities", [])]
        p = gather_passages(text, ents)

        for x in p:
            x["_claim"] = text

        passages.extend(p)

    uniq = {}
    final_passages = []
    for p in passages:
        key = p["text"][:200]
        if key not in uniq:
            uniq[key] = True
            final_passages.append(p)

    if not final_passages:
        raise RuntimeError("[ERROR] No KB passages collected.")

    print(f"[INFO] Collected {len(final_passages)} KB passages.")

    texts = [p["text"] for p in final_passages]
    vectors = MODEL.encode(texts, normalize_embeddings=True, batch_size=16).astype("float32")

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    os.makedirs(output_folder, exist_ok=True)
    faiss.write_index(index, f"{output_folder}/kb.index")
    np.save(f"{output_folder}/kb_embeddings.npy", vectors)

    with open(f"{output_folder}/metadata.json", "w", encoding="utf-8") as f:
        json.dump(final_passages, f, indent=2, ensure_ascii=False)

    print("[INFO] KB saved →", output_folder)
    return final_passages, vectors, index


# ===========================================================
# TEST
# ===========================================================
if __name__ == "__main__":
    test_claims = [
        {"claim_text": "Joe Biden said inflation is zero", "entities": [{"text":"Biden"}]},
    ]
    build_kb_from_claims(test_claims)
