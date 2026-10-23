# Komparasi Spark on YARN vs Sekuensial (Baseline)

Folder ini berisi notebook eksperimen prapemrosesan terdistribusi citra skala besar (24.466 citra irisan 3 kelas) dan pelatihan model klasifikasi **ConvNeXt-Tiny**.

---

## 📂 Berkas Notebook

| Notebook | Skenario Komputasi | Waktu Prapemrosesan | Throughput | Keterangan |
| :--- | :--- | :---: | :---: | :--- |
| [`8w12p.ipynb`](8w12p.ipynb) | **Apache Spark on YARN** (8 Worker, 12 Partisi) | **560,09 detik** | **43,68 citra/detik** | Prapemrosesan paralel skala kluster terdistribusi |
| [`Irisan_convnext_tiny_tanpa_spark (1).ipynb`](<Irisan_convnext_tiny_tanpa_spark (1).ipynb>) | **Sekuensial** (Baseline, 1 Core CPU) | **2.727,86 detik** | **8,97 citra/detik** | Eksekusi serial standar tanpa akselerasi Spark |

---

## ⚡ Hasil Kinerja Utama

* **Percepatan (*Speedup*):** **4,87x lebih cepat** menggunakan Spark 8W12P dibandingkan sekuensial.
* **Penghematan Waktu:** Mengurangi durasi prapemrosesan sebesar **79,5%** (selisih 2.167,77 detik / ~36,1 menit).
* **Akurasi Model ConvNeXt-Tiny Uji:** **83,87%** (makro AUC 0,9426, F1-Score 76,01%).

---

## 📦 Dataset Terkait di Kaggle

* [nadiraanindita/ham10k-and-isic-2019](https://www.kaggle.com/datasets/nadiraanindita/ham10k-and-isic-2019)
* [nadiraanindita/dataset-riset-ham10k](https://www.kaggle.com/datasets/nadiraanindita/dataset-riset-ham10k)
* [nadiraanindita/dataset-riset-isic-2019](https://www.kaggle.com/datasets/nadiraanindita/dataset-riset-isic-2019)
