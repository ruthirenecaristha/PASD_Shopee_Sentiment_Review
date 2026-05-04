# 🛍️ ShopeeSentiment — Aplikasi Analisis Sentimen Ulasan Shopee

Aplikasi web Python untuk menganalisis sentimen ulasan produk Shopee secara otomatis.
Dikembangkan untuk mata kuliah **Perancangan Aplikasi Sains Data**, Telkom University.

---

## 🚀 Cara Menjalankan

### 1. Install dependensi
```bash
pip install flask pandas numpy matplotlib
```

### 2. Pastikan dataset tersedia
Letakkan file `Shopee_Sampled_Reviews.csv` di folder yang sama dengan `app.py`.

### 3. Jalankan server
```bash
python app.py
```

### 4. Buka browser
```
http://localhost:5000
```

---

## 📂 Struktur Proyek

```
shopee_web/
├── app.py                  ← Flask web application (routes)
├── sentiment_engine.py     ← Backend OOP+FP sentiment analysis
├── Shopee_Sampled_Reviews.csv
└── templates/
    ├── base.html           ← Layout dasar + navbar + footer
    ├── home.html           ← Halaman beranda
    ├── analisis.html       ← Form input URL produk
    ├── hasil.html          ← Halaman hasil analisis
    └── tentang.html        ← Tentang aplikasi
```

---

## 🗺️ Halaman Aplikasi

| Route | Halaman | Deskripsi |
|---|---|---|
| `/` | Beranda | Landing page dengan fitur dan cara kerja |
| `/analisis` | Analisis | Form input URL produk Shopee |
| `/hasil/<id>` | Hasil | Laporan lengkap hasil analisis |
| `/tentang` | Tentang | Deskripsi teknis aplikasi |

---

## 🏗️ Konsep OOP yang Digunakan

| Konsep | Implementasi |
|---|---|
| **Encapsulation** | `DataLoader._filepath`, `BaseAnalyzer._reviews` — atribut private, akses via metode publik |
| **Inheritance** | `BaseAnalyzer` → `RatingAnalyzer`, `LexiconAnalyzer` → `HybridAnalyzer` |
| **Polymorphism** | `analyze()` dipanggil sama pada semua analyzer, perilaku berbeda |

## ⚡ Konsep FP yang Digunakan

| Konsep | Implementasi |
|---|---|
| **Immutability** | `frozenset` untuk kamus kata; pure function selalu return objek baru |
| **Referential Transparency** | `score_to_sentiment(5)` selalu `"positive"` |
| **Higher-Order Functions** | `make_classifier()` mengembalikan fungsi; `preprocess_pipeline()` via `reduce`; `map/lambda` |

---

## 📊 Fitur Aplikasi

- ⭐ Statistik rating (rata-rata, distribusi bintang)
- 💬 Klasifikasi sentimen (positif / netral / negatif)
- 📈 Grafik tren rating dari waktu ke waktu
- 🔑 Ekstraksi kata kunci pujian dan keluhan
- 🎯 Rekomendasi: Layak Dibeli / Pertimbangkan / Hati-Hati
- 📝 Cuplikan ulasan positif dan negatif

---

## ⚠️ Catatan

Aplikasi ini menggunakan dataset CSV lokal sebagai simulasi scraping.
Untuk produksi nyata, ganti `ShopeScraper._fetch()` dengan `selenium`/`playwright`.
