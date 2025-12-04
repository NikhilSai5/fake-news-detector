import sys
import os

import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from bs4 import Comment


# Fix path so deep_learning/ and kb/ can be imported from root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from deep_learning.classifier import predict_fake_probability
from kb.kb_pipeline import run_kb_pipeline



def scrape_url(url: str) -> str:
    """
    Fetch, extract, and clean readable article text from a URL.
    Removes ads, links, JS, navigation, social widgets, and junk.
    """
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

    # -------------------------------------------------------
    # 1. REMOVE USELESS TAGS COMPLETELY
    # -------------------------------------------------------
    for tag_name in ["script", "style", "header", "footer", "nav", "aside"]:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # Remove common ad / junk containers
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

    # -------------------------------------------------------
    # 2. TRY ARTICLE TAG FIRST
    # -------------------------------------------------------
    article_tags = soup.find_all("article")
    if article_tags:
        text = " ".join(a.get_text(" ", strip=True) for a in article_tags)
        if len(text.split()) > 80:
            return clean_text(text)

    # -------------------------------------------------------
    # 3. TRY MAIN CONTENT CLASSES
    # -------------------------------------------------------
    main_classes = [
        "content", "article", "story", "main", "post", "post-content",
        "entry-content", "body-copy", "article-body", "article__body"
    ]

    for cls in main_classes:
        elems = soup.find_all(class_=lambda x: x and cls in x.lower())
        if elems:
            text = " ".join(e.get_text(" ", strip=True) for e in elems)
            if len(text.split()) > 80:
                return clean_text(text)

    # -------------------------------------------------------
    # 4. FALLBACK → ALL PARAGRAPHS
    # -------------------------------------------------------
    p_tags = soup.find_all("p")
    text = " ".join(p.get_text(" ", strip=True) for p in p_tags)

    return clean_text(text)


def clean_text(text: str) -> str:
    """
    Remove URLs, leftover junk phrases, excessive whitespace,
    social prompts, read-more links, etc.
    """
    import re

    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)

    # Remove common junk phrases
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

    # Remove multiple spaces / newlines
    text = re.sub(r"\s+", " ", text).strip()

    return text


# -------------------------------
# Fusion logic
# -------------------------------
def fuse_scores(p_fake: float, kb_score: float) -> float:
    """
    Weighted fusion of BERT and KB scores.
    p_fake   → probability (0–1) that text is fake
    kb_score → confidence from knowledge base (0–1)
    """
    return 0.4 * kb_score + 0.6 * (1 - p_fake)


# -------------------------------
# Final verdict logic
# -------------------------------
def classify_verdict(trust_score: float) -> str:
    if trust_score >= 0.75:
        return "Likely True"
    elif trust_score >= 0.45:
        return "Uncertain"
    else:
        return "Fake"


# -------------------------------
# Unified fake news pipeline
# -------------------------------
def check_article(text: str):
    # 1. BERT prediction
    p_fake = predict_fake_probability(text)

    # 2. KB pipeline
    kb_output = run_kb_pipeline(text)
    fused_kb_score = kb_output["fused_kb_score"]

    # 3. Fusion
    trust_score = fuse_scores(p_fake, fused_kb_score)

    # 4. Final verdict
    verdict = classify_verdict(trust_score)

    return {
        "bert_probability_fake": float(p_fake),
        "kb_score": float(fused_kb_score),
        "trust_score": float(trust_score),
        "verdict": verdict,
        "claims": kb_output["claims"],
        "kb_results": kb_output["results"]
    }


# -------------------------------
# Run script manually
# -------------------------------
if __name__ == "__main__":
    url = "https://abc.com/news/b5fba32b-5fc8-40e8-92a9-4a461adb3f05/category/1078476"
    article_text = scrape_url(url)
    # print(article_text)
    result = check_article(article_text)

    print("\n===== FINAL OUTPUT =====\n")
    print("BERT Fake Probability:", result["bert_probability_fake"])
    print("KB Score:", result["kb_score"])
    print("Trust Score:", result["trust_score"])
    print("Verdict:", result["verdict"])
    # print("\nExtracted Claims:")
    # for c in result["claims"]:
    #     print("-", c)
    # print("\nKB Verification Results:")
    # for r in result["kb_results"]:
    #     print(r)

