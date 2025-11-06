# **Panduan Deployment Aplikasi Deteksi Fraud ke Streamlit Community Cloud**

Ini adalah panduan langkah-demi-langkah untuk mendeploy aplikasi Streamlit Anda secara gratis.

## **Prasyarat**

1. **Akun GitHub:** Anda memerlukannya untuk menyimpan kode Anda. (gratis)  
2. **Akun Streamlit Community Cloud:** Daftar di [share.streamlit.io](https://share.streamlit.io/) menggunakan akun GitHub Anda. (gratis)  
3. **File Artefak Model:** Anda harus memiliki semua file .pkl yang dihasilkan oleh notebook Colab Anda.

## **Langkah 1: Siapkan Folder Proyek Anda (di Komputer Lokal)**

Buat satu folder baru di komputer Anda, misalnya fraud-detection-app.

Masukkan file-file berikut ke dalam folder tersebut:

1. **app.py** (Kode yang baru saya berikan)  
2. **requirements.txt** (File yang baru saya berikan)  
3. **decision\_tree\_fraud\_model.pkl** (Unduh dari Colab)  
4. **random\_forest\_fraud\_model.pkl** (Unduh dari Colab)  
5. **scaler.pkl** (Unduh dari Colab)  
6. **label\_encoder\_type.pkl** (Unduh dari Colab)  
7. **label\_encoder\_amount\_cat.pkl** (Unduh dari Colab)  
8. **feature\_columns.pkl** (Unduh dari Colab)  
9. **model\_comparison.pkl** (Unduh dari Colab)  
10. **model\_info.pkl** (Unduh dari Colab)  
11. **random\_forest\_model\_info.pkl** (Unduh dari Colab)

**PENTING:** Pastikan semua 11 file (atau lebih) ini berada di *root* (folder utama) yang sama.

## **Langkah 2: Unggah Proyek ke GitHub**

1. Buat **repository baru** di GitHub (misalnya, fraud-app-project).  
2. Unggah *semua* file dari folder lokal Anda (termasuk semua file .pkl) ke repository GitHub ini.

**Catatan:** GitHub mungkin memberi peringatan untuk file .pkl yang besar. Jika ukurannya di atas 100MB (seharusnya tidak), Anda mungkin perlu menggunakan [Git LFS](https://git-lfs.github.com/), tetapi untuk model-model ini, unggahan normal seharusnya sudah cukup.

## **Langkah 3: Deploy di Streamlit Community Cloud**

1. Login ke [share.streamlit.io](https://share.streamlit.io/).  
2. Klik tombol **"New app"** (biasanya di kanan atas).  
3. **"Deploy from GitHub"**:  
   * **Repository:** Pilih repository fraud-app-project Anda.  
   * **Branch:** main (atau master).  
   * **Main file path:** app.py (Streamlit biasanya mendeteksi ini secara otomatis).  
4. Klik **"Deploy\!"**.

## **Langkah 4: Selesai\!**

Streamlit akan:

1. Meng-kloning repository GitHub Anda.  
2. Membaca requirements.txt dan meng-install semua library (pandas, streamlit, openpyxl, dll.).  
3. Menjalankan app.py.

Proses ini mungkin memakan waktu 2-5 menit. Setelah selesai, Anda akan mendapatkan **URL publik** (misalnya: https://nama-anda-fraud-app-project-app-xyz.streamlit.app) yang dapat Anda bagikan **langsung** ke dosen atau mitra bank Anda untuk presentasi.

Aplikasi Anda sudah live\!