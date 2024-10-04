# Alzheimer Disease Classification System

Proyek ini adalah sistem klasifikasi untuk penyakit Alzheimer menggunakan model pembelajaran mesin. Proyek ini dibangun menggunakan Flask sebagai framework web dan model CNN (Convolutional Neural Network) dan SVM (Support Vector Machine) untuk ekstraksi fitur dan klasifikasi.

## Prasyarat

Sebelum memulai, pastikan Anda memiliki hal-hal berikut:
- **Python 3.x**: Proyek ini ditulis menggunakan Python, jadi pastikan Anda memiliki versi terbaru terinstal di sistem Anda.
- **Pip**: Pip adalah manajer paket untuk Python yang digunakan untuk menginstal library yang dibutuhkan.

## Instalasi

1. **Clone Repository:**

   Pertama, clone repository ini ke sistem lokal Anda:

   ```bash
   git clone https://github.com/kyorovik22/alzheimer.git
   cd alzheimer

2. **Install Dependencies**
   BUKA CMD --> pip install -r requirements.txt

3. **Start Aplikasi**
   Buka CMD --> python app.py

4. **Akeses Aplikasi**
   Buka Browser --> http://127.0.0.1:5000
   
## Struktur Folder
    alzheimer/
    │
    ├── .vscode/                 # Folder konfigurasi VS Code
    ├── __pycache__/             # Folder cache Python
    ├── models/                  # Folder berisi model machine learning
    ├── static/                  # Folder berisi file statis (CSS, JS, gambar)
    │   ├── css/                 # Folder CSS
    │   ├── fonts/               # Folder font
    │   └── js/                  # Folder JavaScript
    ├── templates/               # Folder berisi template HTML
    ├── uploads/                 # Folder untuk menyimpan gambar yang diunggah
    ├── app.py                   # File utama aplikasi Flask
    └── requirements.txt         # Daftar dependencies Python
