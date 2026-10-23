# Modul Aplikasi Web Streamlit (DermatoAI)

Folder ini berisi seluruh berkas implementasi aplikasi antarmuka web interaktif **DermatoAI**.

---

## 📂 Struktur Berkas

* [`app.py`](app.py): Skrip utama Streamlit yang memuat antarmuka pengguna, pipeline segmentasi lesi, visualisasi Grad-CAM, dan inferensi klasifikasi 3 kelas.
* [`requirements.txt`](requirements.txt): Daftar dependensi Python untuk menjalankan aplikasi web.
* [`samples/`](samples/): Kumpulan citra dermatoskopi uji cepat untuk demonstrasi:
  * `melanoma.jpg` (Kanker kulit ganas / Malignant)
  * `nevus.jpg` (Tahi lalat jinak / Benign)
  * `seborrheic_keratosis.jpg` (Keratosis seboroik jinak / Benign)

---

## 🚀 Menjalankan Aplikasi Secara Lokal

```bash
cd app
pip install -r requirements.txt
streamlit run app.py
```
*(Aplikasi juga dapat dijalankan langsung dari direktori root repositori melalui `streamlit run app.py`).*
