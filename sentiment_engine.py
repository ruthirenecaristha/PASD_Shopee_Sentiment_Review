"""
sentiment_engine.py
====================
Backend Analisis Sentimen — Shopee Sentiment Analysis App
Paradigma: OOP (Encapsulation, Inheritance, Polymorphism)
         + FP  (Immutability, Referential Transparency, HOF)
"""

import re
import base64
import random
import io
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from dataclasses  import dataclass, field
from typing       import List, Dict, Tuple, Callable, Optional
from collections  import Counter
from functools    import reduce
from datetime     import datetime

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════
#  DATA MODELS
# ══════════════════════════════════════════════════════════════════

@dataclass
class Review:
    review_id  : str
    user_name  : str
    content    : str
    score      : int
    thumbs_up  : int
    created_at : str
    reply      : str = ""
    sentiment  : str = "neutral"


@dataclass
class AnalysisResult:
    product_url         : str
    product_name        : str
    total_reviews       : int
    average_rating      : float
    rating_distribution : Dict[int, int]
    sentiment_counts    : Dict[str, int]
    sentiment_pct       : Dict[str, float]
    top_keywords        : List[str]
    common_praises      : List[str]
    common_complaints   : List[str]
    recommendation      : str          # "LAYAK", "PERTIMBANGKAN", "TIDAK LAYAK"
    recommendation_msg  : str
    recommendation_color: str          # hex color
    confidence_score    : float        # 0–100
    chart_rating        : str          # base64 PNG
    chart_sentiment     : str          # base64 PNG
    chart_trend         : str          # base64 PNG
    sample_positives    : List[str]    # contoh ulasan positif
    sample_negatives    : List[str]    # contoh ulasan negatif
    generated_at        : str = field(
        default_factory=lambda: datetime.now().strftime("%d %B %Y, %H:%M WIB")
    )


# ══════════════════════════════════════════════════════════════════
#  FUNCTIONAL PROGRAMMING LAYER — Pure Functions
# ══════════════════════════════════════════════════════════════════

# ── Immutable data ────────────────────────────────────────────────
STOPWORDS_ID: frozenset = frozenset({
    "yang", "dan", "di", "ke", "dari", "ini", "itu", "dengan", "untuk",
    "pada", "tidak", "ada", "juga", "saya", "aja", "nya", "ya", "ga",
    "gak", "tapi", "bisa", "sudah", "udah", "lebih", "sangat", "banget",
    "sih", "deh", "lah", "kan", "nih", "dong", "kok", "kak", "sama",
    "jadi", "kalau", "kalo", "atau", "karena", "saat", "jika", "agar",
    "supaya", "maka", "namun", "tetapi", "akan", "telah", "pun", "hanya",
    "bukan", "bahwa", "oleh", "lagi", "belum", "masih", "baru", "lama",
    "setelah", "sebelum", "ketika", "hingga", "semua", "beberapa", "setiap",
    "mau", "saja", "app", "shopee", "aplikasi", "tolong", "mohon", "harap",
    "update", "versi", "hp", "handphone", "android", "ios", "play", "store",
})

POSITIVE_WORDS: frozenset = frozenset({
    "bagus", "baik", "mantap", "oke", "ok", "recommended", "suka", "puas",
    "sempurna", "top", "keren", "cepat", "aman", "terpercaya", "murah",
    "lengkap", "original", "asli", "senang", "happy", "hebat", "memuaskan",
    "best", "worth", "solid", "kualitas", "mudah", "canggih", "lancar",
    "nyaman", "responsif", "terbaik", "luar biasa", "istimewa", "menarik",
    "sesuai", "tepat", "benar", "jujur", "aman", "terpercaya", "andal",
    "recommended", "rekomendasi", "recommend", "puas", "pasti", "beli lagi",
})

NEGATIVE_WORDS: frozenset = frozenset({
    "buruk", "jelek", "rusak", "lambat", "lama", "tipu", "palsu", "kecewa",
    "mahal", "error", "eror", "lemot", "lag", "crash", "gagal", "cacat",
    "bohong", "sampah", "terburuk", "mengecewakan", "menipu", "penipuan",
    "curang", "unfair", "parah", "susah", "ribet", "bingung", "boros",
    "tidak bisa", "tidak jalan", "zonk", "scam", "penipu", "nyesal",
    "menyesal", "jangan beli", "tidak sesuai", "berbeda", "beda", "beda",
})


# ── Pure Functions (Immutability) ─────────────────────────────────

def clean_text(text: str) -> str:
    if not isinstance(text, str): return ""
    t = text.lower()
    t = re.sub(r"http\S+|www\S+", "", t)
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def tokenize(text: str) -> List[str]:
    return text.split()


def remove_stopwords(tokens: List[str]) -> List[str]:
    return [t for t in tokens if t not in STOPWORDS_ID and len(t) > 2]


# ── Pure Functions (Referential Transparency) ─────────────────────

def score_to_sentiment(score: int) -> str:
    if score <= 2: return "negative"
    if score == 3: return "neutral"
    return "positive"


def keyword_score(tokens: List[str]) -> Tuple[int, int]:
    pos = sum(1 for t in tokens if t in POSITIVE_WORDS)
    neg = sum(1 for t in tokens if t in NEGATIVE_WORDS)
    return pos, neg


def calc_rating_stats(scores: List[int]) -> Dict:
    arr = np.array(scores)
    return {"mean": round(float(arr.mean()), 2), "std": round(float(arr.std()), 2)}


def rating_dist(scores: List[int]) -> Dict[int, int]:
    c = Counter(scores)
    return {k: c.get(k, 0) for k in range(1, 6)}


# ── Higher-Order Functions ─────────────────────────────────────────

def make_classifier(threshold: int = 2) -> Callable:
    def classify(score: int, tokens: List[str]) -> str:
        base = score_to_sentiment(score)
        pos, neg = keyword_score(tokens)
        delta = pos - neg
        if base == "neutral":
            if delta >= threshold:  return "positive"
            if delta <= -threshold: return "negative"
        return base
    return classify


def preprocess_pipeline(text: str) -> List[str]:
    pipeline = [clean_text, tokenize, remove_stopwords]
    return reduce(lambda acc, fn: fn(acc), pipeline, text)


def extract_keywords(token_lists: List[List[str]], n: int = 8) -> List[str]:
    all_t = [t for tl in token_lists for t in tl]
    return [w for w, _ in Counter(all_t).most_common(n)]


def filter_by_sentiment(
    reviews: List[Review],
    sent: str,
    fn: Callable,
) -> List:
    return list(map(fn, [r for r in reviews if r.sentiment == sent]))


# ══════════════════════════════════════════════════════════════════
#  OOP LAYER
# ══════════════════════════════════════════════════════════════════

class BaseAnalyzer:
    """Parent class — Inheritance + Polymorphism."""

    def __init__(self, reviews: List[Review]) -> None:
        self._reviews = reviews                # Encapsulation

    def analyze(self) -> List[Review]:
        raise NotImplementedError

    def sentiment_counts(self) -> Dict[str, int]:
        c = Counter(r.sentiment for r in self._reviews)
        return {"positive": c.get("positive", 0),
                "neutral" : c.get("neutral",  0),
                "negative": c.get("negative", 0)}

    def sentiment_pct(self) -> Dict[str, float]:
        counts = self.sentiment_counts()
        total  = sum(counts.values()) or 1
        return {k: round(v / total * 100, 1) for k, v in counts.items()}


class RatingAnalyzer(BaseAnalyzer):
    """Child 1 — rating-only classification."""
    def analyze(self) -> List[Review]:
        for r in self._reviews:
            r.sentiment = score_to_sentiment(r.score)
        return self._reviews


class LexiconAnalyzer(BaseAnalyzer):
    """Child 2 — rating + lexicon classification."""
    def __init__(self, reviews, threshold=2):
        super().__init__(reviews)
        self._classify = make_classifier(threshold)   # HOF

    def analyze(self) -> List[Review]:
        for r in self._reviews:
            tokens      = preprocess_pipeline(r.content)
            r.sentiment = self._classify(r.score, tokens)
        return self._reviews


class HybridAnalyzer(LexiconAnalyzer):
    """Grandchild — Inheritance bertingkat."""
    def analyze(self) -> List[Review]:
        super().analyze()
        return self._reviews


# ── Data Loader ────────────────────────────────────────────────────

class DataLoader:
    """Encapsulation: menyembunyikan logika loading CSV."""

    _REQUIRED = frozenset({"reviewId", "userName", "content",
                           "score", "thumbsUpCount", "at"})

    def __init__(self, filepath: str) -> None:
        self._filepath = filepath

    def load(self) -> List[Review]:
        df = pd.read_csv(self._filepath).fillna("")
        missing = self._REQUIRED - set(df.columns)
        if missing:
            raise ValueError(f"Kolom hilang: {missing}")
        return list(map(self._row_to_review, df.to_dict("records")))

    @staticmethod
    def _row_to_review(row: Dict) -> Review:
        return Review(
            review_id  = str(row.get("reviewId", "")),
            user_name  = str(row.get("userName", "Anonim")),
            content    = str(row.get("content",  "")),
            score      = int(row.get("score",     3)),
            thumbs_up  = int(row.get("thumbsUpCount", 0)),
            created_at = str(row.get("at", "")),
            reply      = str(row.get("replyContent", "")),
        )


# ── Mock Scraper (simulasi scraping Shopee) ────────────────────────

class ShopeScraper:
    """
    Encapsulation: mensimulasikan scraping review Shopee dari URL.
    Dalam produksi nyata, ganti _fetch() dengan Selenium/Playwright.
    """

    def __init__(self, csv_path: str) -> None:
        self._csv_path   = csv_path
        self._all_reviews: Optional[List[Review]] = None

    def _load_all(self) -> List[Review]:
        if self._all_reviews is None:
            loader = DataLoader(self._csv_path)
            self._all_reviews = loader.load()
        return self._all_reviews

    def scrape(self, url: str, n_reviews: int = 300) -> Tuple[List[Review], str]:
        """
        Simulasi scraping: ambil sampel acak dari dataset.
        Kembalikan (reviews, product_name).
        """
        all_rev      = self._load_all()
        n            = min(n_reviews, len(all_rev))
        sampled      = random.sample(all_rev, n)
        product_name = self._extract_name(url)
        return sampled, product_name

    @staticmethod
    def _extract_name(url: str) -> str:
        """[Pure-like] Ekstrak nama produk dari URL Shopee."""
        try:
            slug = url.rstrip("/").split("/")[-1]
            slug = re.sub(r"-i\.\d+\.\d+$", "", slug)
            slug = slug.replace("-", " ").replace("_", " ")
            words = [w.capitalize() for w in slug.split() if len(w) > 1]
            name  = " ".join(words[:7])
            return name if name else "Produk Shopee"
        except Exception:
            return "Produk Shopee"


# ── Chart Generator ────────────────────────────────────────────────

class ChartGenerator:
    """Encapsulation: semua logika pembuatan grafik dikemas di sini."""

    ORANGE    = "#EE4D2D"
    GREEN     = "#26AA99"
    GRAY      = "#F5F5F5"
    TEXT      = "#212121"
    LIGHT     = "#FFF3F0"

    @staticmethod
    def _to_b64(fig) -> str:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight",
                    facecolor="white", dpi=130)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)
        return b64

    def rating_bar(self, dist: Dict[int, int]) -> str:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        stars   = list(range(5, 0, -1))
        counts  = [dist.get(s, 0) for s in stars]
        total   = sum(counts) or 1
        colors  = [self.ORANGE if s >= 4 else
                   ("#FFA800" if s == 3 else "#BDBDBD") for s in stars]

        bars = ax.barh([f"{s}★" for s in stars], counts,
                       color=colors, height=0.55, edgecolor="none")

        for bar, cnt in zip(bars, counts):
            pct = cnt / total * 100
            ax.text(bar.get_width() + max(counts) * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{cnt:,} ({pct:.1f}%)",
                    va="center", ha="left",
                    fontsize=9, color=self.TEXT)

        ax.set_xlabel("Jumlah Ulasan", fontsize=9, color=self.TEXT)
        ax.set_title("Distribusi Rating", fontsize=12,
                     fontweight="bold", color=self.TEXT, pad=12)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(left=False)
        ax.set_facecolor("white")
        fig.patch.set_facecolor("white")
        ax.set_xlim(0, max(counts) * 1.25)
        plt.tight_layout()
        return self._to_b64(fig)

    def sentiment_donut(self, counts: Dict[str, int]) -> str:
        labels = ["Positif", "Netral", "Negatif"]
        values = [counts.get("positive", 0),
                  counts.get("neutral",  0),
                  counts.get("negative", 0)]
        colors = [self.GREEN, "#FFA800", self.ORANGE]

        fig, ax = plt.subplots(figsize=(5, 4))
        wedges, texts, autotexts = ax.pie(
            values, labels=None, colors=colors,
            autopct=lambda p: f"{p:.1f}%" if p > 3 else "",
            startangle=90, pctdistance=0.75,
            wedgeprops=dict(width=0.55, edgecolor="white", linewidth=3),
        )
        for at in autotexts:
            at.set_fontsize(10)
            at.set_fontweight("bold")
            at.set_color("white")

        legend_patches = [
            mpatches.Patch(color=c, label=f"{l}: {v:,}")
            for c, l, v in zip(colors, labels, values)
        ]
        ax.legend(handles=legend_patches, loc="lower center",
                  bbox_to_anchor=(0.5, -0.08), ncol=3,
                  fontsize=9, frameon=False)

        total = sum(values) or 1
        ax.text(0, 0, f"{sum(values):,}\nUlasan",
                ha="center", va="center",
                fontsize=11, fontweight="bold", color=self.TEXT)
        ax.set_title("Distribusi Sentimen", fontsize=12,
                     fontweight="bold", color=self.TEXT, pad=10)
        plt.tight_layout()
        return self._to_b64(fig)

    def trend_line(self, reviews: List[Review]) -> str:
        """Tren rata-rata rating dari waktu ke waktu (simulasi)."""
        scores_ts = [r.score for r in reviews]
        n         = len(scores_ts)
        window    = max(1, n // 20)

        # rolling average
        smoothed = []
        for i in range(0, n, window):
            chunk = scores_ts[i:i + window]
            smoothed.append(np.mean(chunk))

        x = list(range(len(smoothed)))

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.fill_between(x, smoothed, alpha=0.15, color=self.ORANGE)
        ax.plot(x, smoothed, color=self.ORANGE, linewidth=2.5, marker="o",
                markersize=3.5)
        ax.axhline(y=np.mean(scores_ts), color=self.GREEN,
                   linestyle="--", linewidth=1.2, label=f"Rata-rata: {np.mean(scores_ts):.2f}")

        ax.set_ylim(1, 5.2)
        ax.set_ylabel("Rating", fontsize=9)
        ax.set_xlabel("Periode Ulasan", fontsize=9)
        ax.set_title("Tren Rating dari Waktu ke Waktu", fontsize=12,
                     fontweight="bold", color=self.TEXT, pad=10)
        ax.legend(fontsize=9, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_facecolor("white")
        fig.patch.set_facecolor("white")
        plt.tight_layout()
        return self._to_b64(fig)


# ── Recommender ────────────────────────────────────────────────────

class Recommender:
    """
    Encapsulation: logika rekomendasi "layak/tidak layak dibeli".
    Menggunakan multiple pure functions untuk skor akhir.
    """

    @staticmethod
    def _positive_ratio(pct: Dict[str, float]) -> float:
        return pct.get("positive", 0) / 100

    @staticmethod
    def _rating_score(avg: float) -> float:
        return (avg - 1) / 4          # normalisasi 1-5 → 0-1

    def recommend(
        self,
        avg_rating  : float,
        sentiment_pct: Dict[str, float],
        total_reviews: int,
    ) -> Tuple[str, str, str, float]:
        """
        Hitung skor dan kembalikan (label, pesan, warna, confidence).
        Higher-Order: menggunakan komposisi skor dari pure functions.
        """
        pos_ratio  = self._positive_ratio(sentiment_pct)
        rat_score  = self._rating_score(avg_rating)

        # Weighted score: 60% sentiment + 40% rating
        score      = 0.6 * pos_ratio + 0.4 * rat_score
        confidence = round(score * 100, 1)

        # Volume bonus: lebih banyak ulasan = lebih terpercaya
        if total_reviews >= 200:  confidence = min(confidence + 3, 100)
        elif total_reviews < 50:  confidence = max(confidence - 5, 0)

        neg_pct = sentiment_pct.get("negative", 0)

        if score >= 0.65 and neg_pct < 30:
            return (
                "LAYAK DIBELI",
                f"Produk ini mendapat respons sangat positif dari pembeli. "
                f"{sentiment_pct.get('positive', 0):.1f}% ulasan bersifat positif "
                f"dengan rata-rata rating {avg_rating:.1f}/5.0.",
                "#26AA99",
                confidence,
            )
        elif score >= 0.45 or neg_pct < 45:
            return (
                "PERTIMBANGKAN",
                f"Produk ini memiliki ulasan campuran. Bacalah ulasan negatif "
                f"({neg_pct:.1f}%) secara seksama sebelum memutuskan pembelian. "
                f"Rating rata-rata {avg_rating:.1f}/5.0.",
                "#FFA800",
                confidence,
            )
        else:
            return (
                "HATI-HATI",
                f"Produk ini menerima banyak keluhan dari pembeli "
                f"({neg_pct:.1f}% ulasan negatif). "
                f"Sangat disarankan untuk mencari alternatif produk lain.",
                "#EE4D2D",
                confidence,
            )


# ══════════════════════════════════════════════════════════════════
#  FACADE — ShopeeAnalysisEngine
# ══════════════════════════════════════════════════════════════════

class ShopeeAnalysisEngine:
    """
    Facade: satu titik masuk untuk seluruh pipeline analisis.
    Mengorkestrasikan Scraper → Analyzer → Summarizer → Chart → Recommender.
    """

    def __init__(self, csv_path: str) -> None:
        self._scraper    = ShopeScraper(csv_path)
        self._charts     = ChartGenerator()
        self._recommender= Recommender()

    def analyze_url(self, url: str) -> AnalysisResult:
        """Pipeline end-to-end dari URL ke AnalysisResult."""

        # 1. Scraping (simulasi)
        reviews, product_name = self._scraper.scrape(url, n_reviews=300)

        # 2. Analisis sentimen (Polymorphism: bisa ganti analyzer)
        analyzer = LexiconAnalyzer(reviews, threshold=2)
        reviews  = analyzer.analyze()

        # 3. Statistik
        scores    = [r.score for r in reviews]
        avg       = round(float(np.mean(scores)), 2)
        dist      = rating_dist(scores)
        s_counts  = analyzer.sentiment_counts()
        s_pct     = analyzer.sentiment_pct()

        # 4. Keywords (HOF: map + filter)
        all_tokens = list(map(preprocess_pipeline, [r.content for r in reviews]))
        keywords   = extract_keywords(all_tokens, 10)

        pos_tokens = list(map(preprocess_pipeline,
                              filter_by_sentiment(reviews, "positive", lambda r: r.content)))
        neg_tokens = list(map(preprocess_pipeline,
                              filter_by_sentiment(reviews, "negative", lambda r: r.content)))
        praises    = extract_keywords(pos_tokens, 6)
        complaints = extract_keywords(neg_tokens, 6)

        # 5. Sample ulasan
        pos_samples = [r.content for r in reviews
                       if r.sentiment == "positive" and len(r.content) > 20][:3]
        neg_samples = [r.content for r in reviews
                       if r.sentiment == "negative" and len(r.content) > 20][:3]

        # 6. Rekomendasi
        label, msg, color, conf = self._recommender.recommend(avg, s_pct, len(reviews))

        # 7. Charts
        chart_r = self._charts.rating_bar(dist)
        chart_s = self._charts.sentiment_donut(s_counts)
        chart_t = self._charts.trend_line(reviews)

        return AnalysisResult(
            product_url          = url,
            product_name         = product_name,
            total_reviews        = len(reviews),
            average_rating       = avg,
            rating_distribution  = dist,
            sentiment_counts     = s_counts,
            sentiment_pct        = s_pct,
            top_keywords         = keywords,
            common_praises       = praises,
            common_complaints    = complaints,
            recommendation       = label,
            recommendation_msg   = msg,
            recommendation_color = color,
            confidence_score     = conf,
            chart_rating         = chart_r,
            chart_sentiment      = chart_s,
            chart_trend          = chart_t,
            sample_positives     = pos_samples,
            sample_negatives     = neg_samples,
        )
