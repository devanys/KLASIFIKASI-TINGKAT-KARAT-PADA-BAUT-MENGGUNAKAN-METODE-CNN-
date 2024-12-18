# KLASIFIKASI-TINGKAT-KARAT-PADA-BAUT-MENGGUNAKAN-OPEN-CV
![Screenshot 2024-06-26 033401](https://github.com/user-attachments/assets/75ea8820-99db-48e1-aace-25dbd50b6496)

## Deskripsi Proyek

Program ini digunakan untuk melakukan **klasifikasi objek dalam gambar** menggunakan model yang telah dilatih dengan **Keras** serta mendeteksi objek berdasarkan kontur menggunakan **OpenCV**. Gambar input dianalisis untuk mengidentifikasi objek, menampilkan **kelas objek** dan **tingkat kepercayaan prediksi** dengan visualisasi bounding box berwarna.

---

## Fitur Utama

1. **Memuat Model Keras**  
   Program memuat model `keras_Model.h5` dan label kelas dari file `labels.txt`.

2. **Klasifikasi Gambar Utuh**  
   - Prediksi kelas objek pada gambar secara keseluruhan.
   - Menampilkan **nama kelas** dan **confidence score**.

3. **Deteksi dan Klasifikasi Multi-Objek**  
   - Menggunakan **thresholding Otsu** untuk mendeteksi area objek.
   - Klasifikasi setiap area objek yang terdeteksi (ROI).  
   - Warna bounding box sesuai hasil klasifikasi:
     - **Merah** untuk objek "bad".
     - **Hijau** untuk objek selain "bad".

4. **Visualisasi Hasil**  
   Menampilkan gambar dengan bounding box dan label kelas objek.
