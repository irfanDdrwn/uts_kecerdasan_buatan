# Gaming Revenue Predictor 🎮

Aplikasi web prediksi revenue game menggunakan 5 algoritma Machine Learning.

## Identitas
- **Nama:** [Nama Mahasiswa]
- **NIM:** [NIM]
- **Mata Kuliah:** Praktikum Kecerdasan Buatan — Semester 4 (Genap) 2025/2026

## Deskripsi
Aplikasi ini memprediksi **Revenue Game (Juta USD)** berdasarkan fitur seperti genre, platform, developer, jumlah pemain, dan Metacritic Score menggunakan dataset Gaming Industry Trends (1000 data).

## Algoritma yang Digunakan
1. **Linear Regression** — Model baseline (scikit-learn)
2. **ANN** — Multilayer Perceptron (TensorFlow/Keras)
3. **RNN/LSTM** — Sequential model (TensorFlow/Keras)
4. **K-Means Clustering** — Segmentasi game (scikit-learn)
5. **Backpropagation Manual** — Implementasi NumPy murni

## Cara Menjalankan

```bash
# Clone repository
git clone [url-repo]
cd gaming_app

# Install dependencies
pip install -r requirements.txt

# Train models (sekali saja)
python train_models.py

# Jalankan aplikasi
python app.py
```

Akses di: http://localhost:5000

## Link
- 🌐 **Demo:** [URL .my.id]
- 📁 **GitHub:** [URL GitHub]
- 🎥 **YouTube:** [URL YouTube]
- 📄 **Laporan:** [URL PDF]

## Struktur Folder
```
gaming_app/
├── data/               # Dataset CSV
├── models/             # Model tersimpan (.pkl, .h5, .json)
├── templates/          # HTML templates (Jinja2)
├── static/             # CSS, JS, assets
├── app.py              # Main Flask app
├── train_models.py     # Script training semua model
├── requirements.txt
└── Procfile
```
# uts_kecerdasan_buatan
