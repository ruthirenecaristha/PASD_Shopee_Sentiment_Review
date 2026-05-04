"""
app.py — Flask Web Application
Shopee Customer Sentiment Analysis
"""

import os
import uuid
import re
from flask import Flask, render_template, request, redirect, url_for, session
from sentiment_engine import ShopeeAnalysisEngine

app    = Flask(__name__)
app.secret_key = "shopee-sentiment-2025-telkom"

CSV_PATH = os.path.join(os.path.dirname(__file__), "Shopee_Sampled_Reviews.csv")
engine   = ShopeeAnalysisEngine(CSV_PATH)

# Cache sederhana di memory (production: gunakan Redis)
_cache: dict = {}


def is_valid_shopee_url(url: str) -> bool:
    """[Pure Function — RT] Validasi URL Shopee."""
    pattern = r"(shopee\.(co\.id|com)|shp\.ee)"
    return bool(re.search(pattern, url, re.IGNORECASE)) or url.strip() != ""


# ── Routes ─────────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/analisis", methods=["GET", "POST"])
def analisis():
    error = None
    if request.method == "POST":
        url = request.form.get("product_url", "").strip()
        if not url:
            error = "URL produk tidak boleh kosong."
        elif not is_valid_shopee_url(url):
            error = "Masukkan URL produk Shopee yang valid."
        else:
            # Jalankan analisis
            try:
                result     = engine.analyze_url(url)
                session_id = str(uuid.uuid4())
                _cache[session_id] = result
                return redirect(url_for("hasil", sid=session_id))
            except Exception as e:
                error = f"Terjadi kesalahan saat analisis: {str(e)}"

    return render_template("analisis.html", error=error)


@app.route("/hasil/<sid>")
def hasil(sid):
    result = _cache.get(sid)
    if not result:
        return redirect(url_for("analisis"))
    return render_template("hasil.html", r=result)


@app.route("/tentang")
def tentang():
    return render_template("tentang.html")


if __name__ == "__main__":
    app.run(debug=True, port=5000)
