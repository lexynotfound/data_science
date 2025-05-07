# Sistem Prediksi Putus Sekolah Jaya Jaya Institut

## Ringkasan Proyek
Proyek ini menghadirkan solusi lengkap untuk tantangan prediksi putus sekolah di Jaya Jaya Institut. Menggunakan teknik machine learning, proyek ini membantu mengidentifikasi mahasiswa yang berisiko putus sekolah, sehingga institusi dapat memberikan dukungan yang tepat sasaran.

## Komponen yang Telah Dibuat

### 1. Jupyter Notebook
Pipeline data science lengkap yang mencakup:
- Eksplorasi dan visualisasi data
- Rekayasa fitur dan pra-pemrosesan
- Pengembangan dan evaluasi model
- Ekstraksi wawasan

### 2. Aplikasi Streamlit (app.py)
Aplikasi web interaktif dengan:
- Penilaian risiko untuk mahasiswa individual
- Kemampuan prediksi batch
- Indikator risiko dan dashboard visual
- Rekomendasi intervensi yang dipersonalisasi

### 3. README.md
Dokumentasi komprehensif yang mencakup:
- Gambaran umum dan tujuan proyek
- Pendekatan teknis dan metodologi
- Hasil dan rekomendasi
- Petunjuk pengaturan dan penggunaan

### 4. Dashboard Metabase
Instruksi untuk membuat:
- Dashboard gambaran risiko mahasiswa
- Analisis performa akademik
- Visualisasi pemantauan mahasiswa

## Paket Python yang Diperlukan
Berikut daftar paket Python yang diperlukan untuk menjalankan proyek ini:
```
pandas==1.5.3
numpy==1.24.3
matplotlib==3.7.1
seaborn==0.12.2
scikit-learn==1.2.2
streamlit==1.25.0
pillow==9.5.0
pickle-mixin==1.0.2
```

## Cara Membuat requirements.txt

1. Buat file baru bernama `requirements.txt` di direktori utama proyek
2. Salin daftar paket di atas ke dalam file tersebut
3. Simpan file

Atau, Anda bisa menggunakan perintah berikut di terminal untuk membuat file requirements secara otomatis (jika Anda sudah menginstal semua paket yang diperlukan):
```bash
pip freeze > requirements.txt
```

Namun, sebaiknya edit file ini untuk hanya menyertakan paket yang benar-benar digunakan dalam proyek.

## link streamlit
```
https://datasciences.streamlit.app/
```
## Menginstal Semua Paket yang Diperlukan
Setelah Anda memiliki file `requirements.txt`, gunakan perintah berikut untuk menginstal semua paket yang diperlukan:
```bash
pip install -r requirements.txt
```

## Panduan Pengaturan Metabase

### 1. Instalasi Metabase dengan Docker
1. Pastikan Docker sudah terpasang di sistem Anda
2. Jalankan perintah berikut untuk memulai Metabase:
```bash
docker run -d -p 3000:3000 --name metabase metabase/metabase
```
3. Metabase akan dapat diakses di http://localhost:3000

### 2. Pengaturan Awal
1. Akses Metabase di http://localhost:3000
2. Buat akun admin:
   - Email: root@mail.com
   - Password: root123
3. Masukkan nama untuk organisasi Anda (misalnya "Jaya Jaya Institut")

### 3. Menghubungkan ke Data
1. Klik "Tambahkan data Anda" atau navigasi ke Pengaturan admin > Database
2. Klik "Tambah Database"
3. Pilih "CSV" sebagai tipe database
4. Unggah file CSV berikut:
   - student_risk_data.csv (dihasilkan dari notebook.ipynb)
   - data.csv (dataset asli)
5. Klik "Simpan" untuk terhubung ke data

### 4. Membuat Dashboard

#### Dashboard 1: Gambaran Risiko Mahasiswa
1. Buat dashboard baru dengan mengklik "+ Baru" > "Dashboard"
2. Beri nama "Gambaran Risiko Mahasiswa"
3. Tambahkan visualisasi berikut:

##### Visualisasi 1: Distribusi Tingkat Risiko
1. Buat pertanyaan baru
2. Pilih sumber data "student_risk_data.csv"
3. Klik "Visualisasi"
4. Pilih "Diagram lingkaran"
5. Untuk "Data":
   - Kelompokkan berdasarkan: Risk_Level 
   - Ukur: Hitung
6. Untuk "Tampilan":
   - Atur warna yang sesuai untuk tingkat risiko (Rendah: Hijau, Sedang: Kuning, Tinggi: Merah)
7. Simpan visualisasi dan tambahkan ke dashboard

##### Visualisasi 2: Distribusi Risiko berdasarkan Status
1. Buat pertanyaan baru
2. Pilih sumber data "student_risk_data.csv"
3. Klik "Visualisasi"
4. Pilih "Diagram batang"
5. Untuk "Data":
   - Sumbu X: True_Status
   - Sumbu Y: Risk_Score (Rata-rata)
6. Simpan visualisasi dan tambahkan ke dashboard

#### Dashboard 2: Analisis Performa Akademik
Buat dashboard kedua dengan fokus pada performa akademik mahasiswa dengan visualisasi untuk nilai semester pertama dan kedua.

#### Dashboard 3: Dashboard Pemantauan Mahasiswa
Buat dashboard ketiga untuk pemantauan mahasiswa berisiko tinggi.

### 5. Ekspor Database Metabase
Setelah dashboard Anda selesai, Anda perlu mengekspor database Metabase untuk pengiriman:

1. Pastikan container Metabase sedang berjalan
2. Jalankan perintah berikut untuk mengekspor database:
```bash
docker cp metabase:/metabase.db/metabase.db.mv.db ./
```
3. Perintah ini akan menyalin file `metabase.db.mv.db` ke direktori kerja Anda

4. Pastikan file ini disertakan dalam struktur direktori proyek:
```
submission/
├── model/
├── notebook.ipynb
├── app.py
├── README.md
├── requirements.txt
├── metabase.db.mv.db
└── ... (file lainnya)
```

## Pengaturan Lingkungan

### Persyaratan Sistem
- Python 3.8 atau lebih tinggi
- Docker (untuk Metabase)
- Akses internet untuk mengunduh package yang diperlukan
- RAM minimal 4GB untuk pemrosesan data

### Langkah-langkah Pengaturan
1. Clone repositori ini:
   ```bash
   git clone https://github.com/username/jaya-jaya-institut-dropout.git
   cd jaya-jaya-institut-dropout
   ```

2. Buat dan aktifkan virtual environment (opsional tetapi direkomendasikan):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Untuk Linux/Mac
   venv\Scripts\activate     # Untuk Windows
   ```

3. Instal paket-paket yang diperlukan:
   ```bash
   pip install -r requirements.txt
   ```

## Sumber Data

Dataset yang digunakan dalam proyek ini berasal dari dataset mahasiswa yang disediakan oleh Dicoding Academy. Dataset ini dapat diakses melalui tautan berikut:
[https://github.com/dicodingacademy/dicoding_dataset/tree/main/students_performance](https://github.com/dicodingacademy/dicoding_dataset/tree/main/students_performance)

Dataset ini terdiri dari berbagai informasi tentang mahasiswa, termasuk:
- **Data Demografis**: Status perkawinan, jenis kelamin, usia, kebangsaan
- **Latar Belakang Pendidikan**: Kualifikasi sebelumnya, nilai kualifikasi sebelumnya, mode aplikasi
- **Data Akademik**: Mata kuliah yang diikuti, nilai semester, evaluasi
- **Informasi Keuangan**: Status debitur, status pembayaran biaya kuliah, beasiswa
- **Faktor Ekonomi**: Tingkat pengangguran, tingkat inflasi, PDB
- **Target**: Status (mahasiswa yang drop out, lulus, atau masih terdaftar)

Data ini digunakan untuk melatih model machine learning yang dapat memprediksi risiko mahasiswa putus sekolah, sehingga institusi dapat mengidentifikasi mahasiswa yang membutuhkan dukungan tambahan.

## Menjalankan Proyek Lengkap

### 1. Persiapan Data
- Jalankan Jupyter notebook terlebih dahulu untuk memproses data dan melatih model
- Langkah ini akan menghasilkan file pickle yang diperlukan (model, preprocessor, dll.)

### 2. Aplikasi Streamlit
- Setelah menjalankan notebook, mulai aplikasi Streamlit dengan `streamlit run app.py`
- Aplikasi akan menggunakan model terlatih untuk membuat prediksi

### 3. Dashboard Metabase
- Siapkan Metabase menggunakan Docker seperti yang dijelaskan dalam panduan
- Impor CSV yang diekspor dari notebook
- Buat visualisasi seperti yang diuraikan dalam panduan

## Wawasan Utama
Analisis mengungkapkan beberapa faktor kunci yang mempengaruhi risiko putus sekolah:

1. **Performa Akademik**: Nilai semester pertama dan tingkat keberhasilan kursus adalah prediktor kuat
2. **Faktor Keuangan**: Status penghutang dan masalah pembayaran biaya kuliah secara signifikan meningkatkan risiko
3. **Pola Demografis**: Usia saat masuk kuliah menunjukkan korelasi bermakna dengan tingkat putus sekolah
4. **Sistem Dukungan**: Penerima beasiswa cenderung memiliki tingkat putus sekolah yang lebih rendah

## Langkah Selanjutnya untuk Implementasi
Setelah mengimplementasikan solusi, kami merekomendasikan:

1. **Pengujian Pilot**: Mulai dengan sekelompok kecil mahasiswa untuk memvalidasi prediksi
2. **Program Intervensi**: Kembangkan strategi dukungan khusus untuk setiap tingkat risiko
3. **Perbaikan Berkelanjutan**: Perbarui model secara teratur dengan data mahasiswa baru
4. **Siklus Umpan Balik**: Lacak efektivitas intervensi dan sempurnakan strategi secara berkala

Solusi lengkap ini menyediakan Jaya Jaya Institut dengan alat teknis dan wawasan strategis yang diperlukan untuk mengatasi tantangan putus sekolah secara efektif.