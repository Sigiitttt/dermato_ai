# 🔬 DermatoAI — Sistem Deteksi Kanker Kulit Berbasis Deep Learning

<p align="center">
  <img src="demo_preview.png" alt="Tampilan Antarmuka DermatoAI" width="100%" style="border-radius: 12px; box-shadow: 0 8px 30px rgba(0,0,0,0.5);" />
</p>

DermatoAI adalah sistem aplikasi diagnosis citra dermoskopi berbasis Deep Learning dengan arsitektur **ConvNeXt-Tiny** dan segmentasi otomatis **U-Net (ResNet-34)**. Sistem ini membedakan 3 kelas lesi kulit (Melanoma, Nevus, dan Seborrheic Keratosis) dengan akurasi uji **83,87%**, Macro F1-Score **0,7867**, dan AUC **0,9360**.

Aplikasi ini dilengkapi alur prapemrosesan medis standar (*Dull Razor*, *ROI Cropping 15% margin*, *CLAHE*, dan *Letterbox Padding 224×224*) serta visualisasi interpretasi klinis interaktif (*Circular Confidence Gauge*, *Melanoma Risk Spectrum*, dan *Grad-CAM Heatmap*).

---

## 🚀 Fitur Utama
1. **Diagnosis 3 Kelas Lesi Kulit:**
   - **Melanoma**: Kanker kulit ganas berisiko tinggi.
   - **Nevus**: Tahi lalat jinak, dipantau berkala (metode ABCDE).
   - **Seborrheic Keratosis**: Tumor jinak non-kanker.
2. **Prapemrosesan Medis Otomatis:**
   - *Dull Razor*: Menghilangkan gangguan rambut pada permukaan lesi.
   - *Segmentasi U-Net ResNet-34*: Menandai area lesi secara presisi (dengan fallback metode Otsu jika model belum dimuat).
   - *ROI Bounding Box (+15% margin kulit sehat)*: Memotong lesi dengan mempertahankan konteks batas tepi.
   - *Contrast Enhancement (CLAHE LAB)*: Mempertajam pola pigmen dermatoskopi.
   - *Letterbox Reflect Padding*: Standardisasi ukuran ke 224×224 piksel tanpa distorsi rasio aspek.
3. **Interpretasi & Transparansi Model:**
   - *Grad-CAM Attention Map*: Peta panas yang menunjukkan area piksel yang paling memengaruhi keputusan model.
   - *Melanoma Risk Gauge*: Spektrum risiko klinis instan.
   - *Telemetri Proses*: Pengukuran waktu prapemrosesan, inferensi, dan perangkat komputasi (CPU/GPU).

---

## 📁 Struktur Direktori Repositori

```text
dermato_ai/
├── app.py                     # Aplikasi utama Streamlit (siap deploy)
├── requirements.txt           # Dependensi pustaka Python
├── demo_preview.png           # Tangkapan layar antarmuka DermatoAI
├── samples/                   # Contoh citra dermatoskopi demo paten
│   ├── melanoma.jpg
│   ├── nevus.jpg
│   └── seborrheic_keratosis.jpg
├── notebooks/                 # Notebook eksperimen komparasi & pelatihan
│   ├── 8w12p.ipynb            # Prapemrosesan terdistribusi Spark on YARN (8W12P)
│   └── Irisan_convnext_tiny_tanpa_spark (1).ipynb  # Baseline sekuensial (1 core) & pelatihan
└── README.md
```

---

## 💻 Cara Menjalankan Aplikasi Secara Lokal

### 1. Kloning Repositori
```bash
git clone https://github.com/Sigiitttt/dermato_ai.git
cd dermato_ai
```

### 2. Pasang Dependensi
```bash
pip install -r requirements.txt
```

### 3. Jalankan Aplikasi Streamlit
```bash
streamlit run app.py
```
Aplikasi akan otomatis terbuka pada browser di `http://localhost:8501`.

---

## 📊 Notebook Eksperimen Komputasi

Folder `notebooks/` memuat replikasi eksperimen perbandingan kinerja prapemrosesan pada 24.466 citra dermatoskopi:
1. **`8w12p.ipynb`**:
   - Eksperimen pemrosesan terdistribusi menggunakan **Apache Spark on YARN** dengan skenario 8 worker dan 12 partisi.
   - Menghasilkan waktu prapemrosesan **560,09 detik** (throughput **43,68 citra/detik**, efisiensi **60,9%**, percepatan **4,87x**).
2. **`Irisan_convnext_tiny_tanpa_spark (1).ipynb`**:
   - Baseline prapemrosesan sekuensial (1 core) dengan waktu **2.727,86 detik** (throughput **8,97 citra/detik**).
   - Pelatihan model klasifikasi ConvNeXt-Tiny pada GPU (19.713 citra latih, 2.343 citra uji).

---

## ⚖️ Penafian Medis (*Disclaimer*)
Aplikasi ini dikembangkan sebagai luaran riset dan alat bantu skrining awal (*decision support tool*). Aplikasi ini **bukan pengganti** diagnosis resmi dari dokter spesialis dermatologi atau pemeriksaan histopatologi (biopsi).
