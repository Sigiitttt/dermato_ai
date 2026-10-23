# Modul Segmentasi Lesi Kulit (U-Net)

Folder ini berisi kode dan notebook eksperimen pelatihan model segmentasi otomatis lesi kulit berbasis arsitektur **U-Net** dengan encoder backbone **ResNet-34**.

---

## 📂 Berkas Utama

| Berkas | Deskripsi |
| :--- | :--- |
| [`00_train_unet_segmentasi_ham10k.ipynb`](00_train_unet_segmentasi_ham10k.ipynb) | Notebook pelatihan model U-Net segmentasi lesi kulit menggunakan ground truth mask HAM10000. |

---

## 🎯 Tujuan & Peran dalam Pipeline

1. **Ekstraksi Mask Lesi:** Menghasilkan segmentasi biner (*binary mask*) area lesi dari citra dermatoskopi mentah.
2. **Cropping Region of Interest (ROI):** Memotong fokus lesi dengan *padding* dinamis agar bebas dari artefak kulit sehat di sekelilingnya.
3. **Penyedia Mask ISIC 2019:** Digunakan untuk menghasilkan *pseudo-masks* berkualitas tinggi pada dataset ISIC 2019 yang tidak memiliki ground truth segmentasi bawaan.

---

## 📦 Dataset Terkait di Kaggle

* [nadiraanindita/ham-segmentations-lesion-riset-saya](https://www.kaggle.com/datasets/nadiraanindita/ham-segmentations-lesion-riset-saya) (Ground truth mask HAM10000)
* [nadiraanindita/isic2019-pseudo-masks-zip](https://www.kaggle.com/datasets/nadiraanindita/isic2019-pseudo-masks-zip) (Hasil pseudo-mask U-Net untuk ISIC 2019)
