# 🏥 Medicine Detection using YOLOv11s

Object detection model untuk mendeteksi 21 jenis obat-obatan menggunakan YOLOv11.

## 📊 Dataset
- **Jumlah kelas:** 21 obat
- **Total images:** 765 train, 206 validation, 208 test

## 🎯 Model Performance
- **mAP50:** 98.9%
- **mAP50-95:** 96.7%
- **Model:** YOLOv11s

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Run Streamlit App
```bash
streamlit run app.py
```

### Training (Google Colab)
1. Upload notebook `notebooks/Train_YOLO_Models.ipynb` ke Colab
2. Upload dataset ke Google Drive
3. Run semua cells

## 📁 Project Structure
```
├── app.py              # Streamlit app
├── notebooks/          # Training notebook
├── models/             # Model hasil training 
├── data/               # Sample data
└── results/            # Training results
```

## 📝 Classes
- Biogesic Paracetamol
- Decolgen
- Intunal F
- Intunal Kaplet
- Inza
- Mixagrip
- Mixagrip Flu dan Batuk
- Neozep Forte
- Oskadon SP
- Oskadon Sakit Kepala
- Pamol Paracetamol
- Panadol Cold Flu
- Panadol Extra Paracetamol
- Panadol Paracetamol
- Paramex
- Paramex Flu dan Batuk
- Paramex Nyeri Otot
- Poldan Mig
- Sanaflu
- Ultraflu
- Ultraflu Extr