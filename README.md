# Sistem Prediksi Putus Sekolah Jaya Jaya Institut

## 1. Business Understanding

Jaya Jaya Institut mengalami tantangan dalam mempertahankan mahasiswa hingga kelulusan. Tingginya angka putus sekolah berdampak pada reputasi dan efisiensi operasional. Oleh karena itu, diperlukan sistem prediktif berbasis data untuk mendeteksi risiko mahasiswa putus sekolah secara dini.

## 2. Permasalahan Bisnis

Bagaimana Jaya Jaya Institut dapat mengidentifikasi mahasiswa yang berisiko tinggi untuk putus sekolah agar dapat dilakukan intervensi sedini mungkin?

## 3. Cakupan Proyek

* Membuat model prediksi risiko putus sekolah
* Membuat aplikasi interaktif berbasis Streamlit
* Membangun dashboard pemantauan menggunakan Metabase
* Memberikan rekomendasi intervensi berbasis hasil prediksi

## 4. Persiapan

### Sumber Data

Dataset berasal dari [Dicoding Academy - students performance](https://github.com/dicodingacademy/dicoding_dataset/tree/main/students_performance), berisi data demografis, akademik, dan finansial mahasiswa.

### Setup Environment

Persyaratan:

* Python >= 3.8
* Docker (untuk Metabase)

Instalasi paket:

```bash
pip install -r requirements.txt
```

## 5. Business Dashboard

Dashboard dibuat menggunakan Metabase dengan 3 kategori utama:

### 5.1 Gambaran Risiko Mahasiswa

* Distribusi tingkat risiko (pie chart)
* Rata-rata skor risiko berdasarkan status aktual (bar chart)

### 5.2 Analisis Performa Akademik

* Visualisasi nilai semester pertama dan kedua
* Korelasi antara nilai dan status putus sekolah

### 5.3 Dashboard Pemantauan Mahasiswa

* Daftar mahasiswa dengan risiko tinggi
* Filter berdasarkan fakultas dan jurusan

## 6. Menjalankan Sistem Machine Learning

### 6.1 Notebook

Berisi pipeline:

* Data understanding dan eksplorasi
* Feature engineering
* Model training dan evaluasi
* Simpan model dan preprocessing dalam file `.pkl`

### 6.2 Aplikasi Streamlit

Aplikasi dapat diakses melalui:
👉 [Streamlit App](https://datasciences.streamlit.app/)

Fitur:

* Prediksi individual
* Prediksi batch
* Klasifikasi risiko: Low, Medium, High
* Rekomendasi intervensi

Cara menjalankan lokal:

```bash
streamlit run app.py
```

## 7. Conclusion

Model prediksi dropout berbasis Gradient Boosting berhasil dibangun dengan akurasi \~85%. Faktor penting dalam prediksi meliputi nilai akademik semester awal, status pembayaran biaya kuliah, dan usia saat masuk kuliah. Aplikasi dan dashboard memungkinkan pengguna non-teknis untuk mengakses hasil prediksi secara langsung.

## 8. Rekomendasi Action Items

1. **Program Intervensi Dini** untuk mahasiswa dengan risiko sedang dan tinggi.
2. **Bantuan Finansial** kepada mahasiswa bermasalah biaya.
3. **Evaluasi Kurikulum** bagi mahasiswa dengan nilai akademik rendah.
4. **Dukungan Khusus** untuk mahasiswa usia dewasa.
5. **Pemantauan Berbasis Data** melalui dashboard rutin setiap semester.
6. **Retraining Model** setiap semester untuk menjaga akurasi model.

---

**Struktur Proyek:**

```
submission/
├── model/
├── notebook.ipynb
├── app.py
├── README.md
├── requirements.txt
├── metabase.db.mv.db
```
