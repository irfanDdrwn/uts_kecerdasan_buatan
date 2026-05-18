# Gaming Revenue Predictor 🎮

Aplikasi web prediksi revenue game menggunakan 5 algoritma Machine Learning.

## Identitas
- **Nama:** [Irfan Dwi Darmawan]
- **NIM:** [301240034]
- **Mata Kuliah:** Praktikum Kecerdasan Buatan

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
# Install dependencies
pip install -r requirements.txt

# Train models (sekali saja)
python train_models.py

# Jalankan aplikasi
python app.py
```

Akses di: http://localhost:5000

## Link
- 🌐 **Demo:** [https://namadomain.my.id]
- 📁 **GitHub:** [https://github.com/username/gaming-revenue-predictor]
- 🎥 **YouTube:** [https://youtu.be/PL8-mBFKPOk]

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
