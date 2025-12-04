# # ============================================================
# # START SERVER
# # ============================================================
# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5001, debug=True)


from flask import Flask, request, jsonify
from flask_cors import CORS
import traceback
import sys
import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from bs4 import Comment

# ----------------------------------------------------
# Allow imports from deep_learning/ and kb/
# ----------------------------------------------------
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# from deep_learning.classifier import predict_fake_probability

# from deep_learning.hf_classifier import predict_fake_probability
from deep_learning.classifier_roberta import predict_fake_probability

from kb.kb_pipeline import run_kb_pipeline


app = Flask(__name__)
CORS(app)   # Allow Chrome extension calls


# ============================================================
# GLOBAL LOG COLLECTOR
# ============================================================
def create_logger():
    logs = []

    def log(msg):
        print(msg)
        logs.append(msg)
    return logs, log


# ============================================================
# TEXT SCRAPER
# ============================================================
def scrape_url(url: str, log):
    log("[INFO] Fetching article…")

    headers = {
        "User-Agent":
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
    }

    response = requests.get(url, headers=headers, timeout=10)
    if response.status_code != 200:
        raise RuntimeError(f"Bad response: {response.status_code}")

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove junk tags
    for tag_name in ["script", "style", "header", "footer", "nav", "aside"]:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # Remove junk classes
    junk_classes = [
        "advert", "ads", "sponsored", "share", "social", "promo",
        "related-articles", "recommended", "cookie", "newsletter"
    ]
    for cls in junk_classes:
        for tag in soup.find_all(class_=lambda x: x and cls in x.lower()):
            tag.decompose()

    # Remove comments
    for element in soup(text=lambda text: isinstance(text, Comment)):
        element.extract()

    # 1. ARTICLE TAG
    article_tags = soup.find_all("article")
    if article_tags:
        text = " ".join(a.get_text(" ", strip=True) for a in article_tags)
        if len(text.split()) > 80:
            log("[INFO] Extracted article from <article> tag")
            return clean_text(text)

    # 2. MAIN CLASSES
    main_classes = [
        "content", "article", "story", "main", "post", "post-content",
        "entry-content", "body-copy", "article-body", "article__body"
    ]
    for cls in main_classes:
        elems = soup.find_all(class_=lambda x: x and cls in x.lower())
        if elems:
            text = " ".join(e.get_text(" ", strip=True) for e in elems)
            if len(text.split()) > 80:
                log(f"[INFO] Extracted article from class '{cls}'")
                return clean_text(text)

    # 3. FALLBACK
    log("[INFO] Extracted article from <p> tags fallback")
    p_tags = soup.find_all("p")
    text = " ".join(p.get_text(" ", strip=True) for p in p_tags)
    return clean_text(text)


def clean_text(text: str) -> str:
    import re
    text = re.sub(r"http\S+|www\.\S+", "", text)

    junk_patterns = [
        r"click here",
        r"read more",
        r"follow us",
        r"advertisement",
        r"sponsored content",
        r"subscribe to our newsletter",
        r"share this article",
    ]

    for pattern in junk_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    text = re.sub(r"\s+", " ", text).strip()
    return text


# ============================================================
# SCORE FUSION
# ============================================================
def fuse_scores(p_fake: float, kb_score: float) -> float:
    """
    Improved fusion formula with dynamic weighting and confidence adjustment.
    Returns the final Trust Score in the range [0, 1].
    """

 
    bert_conf = abs(p_fake - 0.5) * 2    

 
    w_bert = 0.2 + 0.8 * bert_conf        
    w_kb   = 1.0 - bert_conf              
    true_by_bert = (1 - p_fake) ** 0.7    
    true_by_kb   = kb_score              

    
    trust_score = (w_bert * true_by_bert + w_kb * true_by_kb) / (w_bert + w_kb)

    return trust_score




def classify_verdict(trust_score: float) -> str:
    if trust_score >= 0.75:
        return "Likely True"
    elif trust_score >= 0.45:
        return "Uncertain"
    else:
        return "Fake"


# ============================================================
# FULL FAKE NEWS CHECK PIPELINE
# ============================================================
def check_article(text: str, log):
    log("[INFO] Running BERT classifier…")
    p_fake = predict_fake_probability(text)
    log(f"[INFO] BERT fake probability: {p_fake:.3f}")

    log("[INFO] Running KB pipeline…")
    kb_output = run_kb_pipeline(text)
    log("[INFO] KB pipeline completed")

    kb_score = kb_output["fused_kb_score"]
    log(f"[INFO] KB Fused Score: {kb_score:.3f}")

    trust_score = fuse_scores(p_fake, kb_score)
    log(f"[INFO] Final Trust Score: {trust_score:.3f}")

    verdict = classify_verdict(trust_score)
    log(f"[INFO] Verdict: {verdict}")

    return {
        "bert_probability_fake": float(p_fake),
        "kb_score": float(kb_score),
        "trust_score": float(trust_score),
        "verdict": verdict,
        "claims": kb_output["claims"],
        "kb_results": kb_output["results"]
    }


# ============================================================
# API ENDPOINT
# ============================================================
@app.route("/api/verify", methods=["POST"])
def verify_article():
    logs, log = create_logger()

    try:
        data = request.json
        url = data.get("url")

        if not url:
            return jsonify({"error": "URL is required"}), 400

        log(f"[INFO] Received URL: {url}")

        # Scrape article
        article_text = scrape_url(url, log)

        if len(article_text.split()) < 50:
            log("[ERROR] Article too short")
            return jsonify({"error": "Article too short", "logs": logs}), 400

        # Full pipeline
        result = check_article(article_text, log)

        return jsonify({
            "logs": logs,
            "result": result
        })

    except Exception as e:
        log(f"[ERROR] {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e), "logs": logs}), 500


# ============================================================
# START SERVER
# ============================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
