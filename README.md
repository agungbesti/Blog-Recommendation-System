# Blog Recommendation System   - Agung Besti
---
# Project Overview
---
Artikel merupakan bentuk tulisan yang berisi informasi atau pendapat tentang suatu topik tertentu. Artikel sering kali dipublikasikan di media cetak, situs web, blog, atau platform berbagi konten online. Sebagian besar artikel memiliki struktur umum yang terdiri dari judul, perkenalan, tubuh artikel yang berisi informasi rinci atau argumen, dan penutup. Beberapa artikel juga dapat mencakup gambar, grafik, atau kutipan untuk mendukung dan mengilustrasikan poin-poin yang disampaikan. Medium sendiri merupakan sebuah platform penerbitan dan berbagi konten yang memungkinkan penulis dan pembaca untuk terhubung dan berinteraksi. Salah satu permasalahan yang sering dijumpai oleh para pembaca artikel adalah menentukan artikel-artikel yang akan mereka baca selanjutnya. Kesulitan pembaca artikel dalam menentukan artikel yang akan dibaca disebabkan oleh banyaknya jumlah artikel dan beragamnya jumlah artikel yang ada [1].

Salah satu cara untuk membuat akses pengunjung lebih lama dari sebuah website adalah membuat sistem rekomendasi. Sistem rekomendasi dapat memberikan saran atau pilihan artikel yang relevan untuk pengguna dalam mencari artikel teknologi pada sebuah situs Medium [2]. 

**Oleh sebab itu, diperlukan suatu aplikasi yang dapat memberikan rekomendasi secara akurat bagaimana memberikan pembaca sebuah referensi relevan terkait artikel mana yang harus mereka baca selanjutnya, sehingga meningkatkan kepuasan pembaca dalam mengunjungi sebuah website.**

Pada kasus ini, aplikasi machine learning secara spesifik akan memberikan rekomendasi sebuah blog yang akan dibaca oleh pengguna, sehingga dapat memberikan **kepuasan pengguna dalam mengunjungi sebuah situs website** dan juga mampu **meningkatkan kunjungan pada sebuah situs website**.

## Alasan Penting yang Mendasari Proyek ini:
---
Alasan penting yang mendasari bahwa permasalahan rekomendasi blog sangat penting, yaitu sebagai berikut:
- Rekomendasi artikel yang tidak relevan menyebabkan **penurunan kunjungan** pada situs website/blog .
- Informasi dari sebuah artikel pada saat ini sangat banyak dan beragam, hal ini menjadi sebuah tantangan tersendiri ketika **banyak informasi yang dihasilkan** tetapi **bukan informasi yang diingikan**.
- Untuk menyelesaikan masalah tersebut, maka akan dibuat aplikasi yang dapat memberikan rekomendasi blog selanjutnya yang harus dibaca untuk **melengkapi informasi yang lebih detail terkait topik yang sedang diminati**.
- Aplikasi ini akan memanfaatkan teknologi Machine Learning serta bahasa pemrograman Python dalam membuat **sistem rekomendasi blog untuk menentukan referensi artikel selanjutnya untuk dibaca**.

# Business Understanding
---
Dari latar belakang yang telah dijelaskan sebelumnya, maka diperlukan suatu aplikasi atau program yang mampu **memberikan  informasi terkait artikel selanjutnya yang harus dibaca** oleh pengunjung situs blog.
Oleh sebab itu, diperlukan sistem yang mampu memberikan rekomendasi artikel yang relevan sehingga **meningkatkan kepuasan pengunjung** situs blog dan juga **meningkatkan kunjungan** pada situs blog tersebut.
## Problem Statements
---
Berdasarkan penjelasan yang telah disampaikan sebelumnya, maka problem statements (rumusan masalah) yaitu sebagai berikut:
- Bagaimana cara memberikan rekomendasi artikel yang relevan kepada pengunjung situs berdasarkan preferensi dan rating yang diberikan?
- Bagaimana metode Content-Based Filtering dapat memberikan rekomendasi artikel terhadap pengunjung situs?
- Bagaimana metode Collaborative Filtering dapat memberikan rekomendasi artikel terhadap pengunjung situs?
 - Apa metode terbaik yang menghasilkan artikel relevan antara Cosine Similarity dan Euclidean Distance pada pendekatan Content-Based Filtering?

## Goals
---
Tujuan yang ingin dicapai dari pembuatan aplikasi sistem rekomendasi blog ini, yaitu sebagai berikut:
- Menghasilkan rekomendasi artikel yang relevan kepada pengunjung situs berdasarkan preferensi dan rating dengan membuat sistem rekomendasi blog.
- Menghasilkan sejumlah rekomendasi artikel yang dipersonalisasi untuk pengunjung situs dengan teknik Content-Based Filtering.
- Menghasilkan sejumlah rekomendasi artikel dengan preferensi pengunjung situs dengan teknik Collaborative Filtering.
- Membandingkan model hasil rekomendasi artikel dari metode Cosine Similarity dan Euclidean Distance serta menghitung Precision dari setiap artikel yang dihasilkan.

### Solution Approach
---
- Solusi yang dapat dilakukan untuk menangani permasalahan sebagaimana terdapat dalam problem statements, yaitu dengan membuat aplikasi yang dapat memberikan rekomendasi artikel yang relevan. Adapun aplikasi tersebut dibuat dengan menerapkan teknologi machine learning serta bahasa pemrograman python dengan metode pendekatan **Content-Based Filtering** dan **Collaborative Filtering**.
- foto https://miro.medium.com/v2/resize:fit:1200/1*KBriLd3AYrLuULCqdffxCQ.png
- ![image](https://github.com/agungbesti/German_Credit_Risk/assets/35904444/75294c2b-9f6b-457c-93ed-4aaccaf971ff)
###### Gambar 1: Content-Based Filtering dan Collaborative Filtering
- Metode **Content-Based Filtering** bekerja dengan melihat kemiripan artikel baru dengan artikel yang sebelumnya. Content-Based Filtering memberikan rekomendasi berdasarkan kemiripan artikel yang dianalisis dari fitur yang dikandung oleh artikel sebelumnya.
- **Collaborative filtering** merupakan proses penyaringan atau pengevaluasian artikel menggunakan penilaian orang lain sebagai informasi yang baru kepada pengunjung situs yang lainnya.
- Pada model Content-Based Filtering menerapkan metode **Cosine Similarity**  dan **Euclidean Distance** .
- Pada model Collaborative Filtering menerapkan metode yang di kombinasikan dengan deep learning yaitu **RecommenderNet**.

# Data Understanding
---
Data yang digunakan adalah dataset yang bersumber dari situs Kaggle yang berisi informasi blog medium dengan topik teknologi. Dataset tersebut dapat didownload pada link berikut ini:  [Blog Recommendation Data](https://www.kaggle.com/datasets/yakshshah/blog-recommendation-data). 
foto medium
![image](https://github.com/agungbesti/German_Credit_Risk/assets/35904444/75294c2b-9f6b-457c-93ed-4aaccaf971ff)
###### Gambar 1: Sumber Dataset

Dataset tersebut memiliki 3 file yaitu:
- Author Data.csv yang berisi informasi penulis blog yang berjumlah 6868 baris
- Medium Blog Data.csv berisi data blog yang ditulis sebanyak 10467  baris.
- Blog Ratings.csv berisi 3 kolom yaitu blog_id, user_id dan ratings yang berjumlah 200140.


## Sample Data
---
Ada pun sample data yang bisa dilihat seperti dibawah ini.

|#| Age | Sex  | Job | Housing | Saving_accounts | Checking_account | Credit_amount | Duration | Purpose | Risk |
|-|-------------------------------|--------|-----------------------|------------------------|----------------------|-------------|--------------|--------------|-----------|----------|
|1|67 | male| 2 | own | NaN| little | 1169| 6 | radio/TV | good|
|2|22 | female| 2 | own | little| moderate | 5951| 48 | radio/TV | bad|
|3|49 | male| 1 | own | little| NaN | 2096| 12 | education | good|
|4|45 | male| 2 | free | little| little | 7882| 42 | furniture/equipment | good|
|5|53 | male| 2 | free | little| little | 4870| 24 | car | bad|

###### Tabel 1 : Sample Data German Credit Risk
Untuk sample data bisa dilihat seperti pada Tabel 1.
## Variabel-variabel yang terdapat pada dataset German Credit Risk adalah sebagai berikut:
---
- Age = Umur pengaju
- Sex = Jenis kelamin pengaju
- Job = Jenis pekerjaan pengaju
- Housing = Status kepemilikan rumah pengaju
- Saving accounts = Rekening Tabungan pengaju
- Checking account = Rekening Giro pengaju
- Credit amount = Jumlah Pinjaman pengaju
- Duration = Waktu Pinjaman dibayarkan
- Purpose = Alasan peminjaman kredit
- Risk = Tingkat risiko pengajuan

## Langkah-Langkah dalam melakukan Data Understanding
---
Untuk memahami dataset, langkah-langkah yang dilakukan, yaitu sebagai berikut:
- Melakukan load dataset kedalam google colaboratory.
- Melakukan Exploratory data analysis untuk memahami makna-makna variabel yang terdapat dalam dataset.
- menggunakan teknik visualisasi data kategorikal dan non-kategorikal dengan menggunakan library seaborn.
- Memvisualisasikan data dengan menggunakan boxplot untuk mencari outlier.
- Melakukan univariative analysis untuk memahami sebaran data variabel.
- Melakukan multivariative analysis untuk memahami korelasi variabel kategorikal dan non-kategorikal terhadap variabel risk.

### Hasil Visualisasi Exploratory Data Analysis
---
|   #   |    Column     | Non-Null Count |  Dtype  |
|-------|--------------|----------------|---------|
|   0   |     Age     |    1000        | int64  |
|   1   |    Sex     |    1000        | object  |
|   2   |    Job     |    1000        | int64  |
|   3   |   Housing    |    1000        | object   |
|   4   | Saving_accounts  |    817        | object |
|   5   | Checking_account  |    606        | object  |
|   6   | Credit_amount |    1000        | int64   |
|   7   | Duration |    1000        | int64   |
|   8   |  Purpose   |    1000        | object   |
|   9   |     Risk      |    1000        | object   |
###### Tabel 2 : Melihat kolom dan tipe data pada dataset 
Pada Tabel 2 dapat dilihat bahwa data memiliki 6 kolom numerik atau angka sedangkan sisanya non-numerik atau kategorikal.

![image](https://github.com/agungbesti/German_Credit_Risk/assets/35904444/75294c2b-9f6b-457c-93ed-4aaccaf971ff)
###### Gambar 1: Univariate Analysis pada variabel Sex
Pada Gambar 1 dapat dilihat bahwa orang yang berjenis kelamin **laki-laki memiliki distribusi data yang lebih banyak sekitar 69%** dibandingkan perempuan.

![image](https://github.com/agungbesti/German_Credit_Risk/assets/35904444/82d2f3c0-aeaa-4058-90d7-7ef3eccf947b)
###### Gambar 2: Univariate Analysis pada variabel Job
Pada Gambar 2 dapat dilihat bahwa orang yang pekerjaanya **memiliki keahlian khusus memiliki distribusi yang paling banyak** dibandingkan dengan yang lainnya. Hal ini menegaskan bahwa sebagian besar peminjam diisi oleh orang - orang yang sudah mempunyai pekerjaan.

![image](https://github.com/agungbesti/German_Credit_Risk/assets/35904444/c406f9c1-782c-45a2-bedb-52cc19507765)
###### Gambar 3: Univariate Analysis pada variabel Housing
Pada Gambar 3 dapat dilihat bahwa orang yang **status kepemilikan rumahnya telah memiliki** memiliki distribusi yang paling banyak dibandingkan dengan yang lainnya. Hal ini menegaskan bahwa sebagian besar peminjam diisi oleh orang - orang yang sudah memiliki kemampuan ekonomi yang cukup.

![image](https://github.com/agungbesti/German_Credit_Risk/assets/35904444/79709a81-2fbd-4ba7-a475-3a38948509fb)
###### Gambar 4: Univariate Analysis pada variabel Saving_accounts
Pada Gambar 4 dapat dilihat bahwa orang yang melakukan peminjaman kredit diisi oleh orang - orang yang memiliki **saldo rekening yang kecil**. Hal ini menegaskan bahwa sebagian besar peminjam diisi oleh orang - orang yang memiliki kemampuan ekonomi yang cukup dengan pengeluaran yang besar, sehingga tabungan yang tersisa cenderung lebih sedikit.

![image](https://github.com/agungbesti/German_Credit_Risk/assets/35904444/50670479-020c-428f-8fb2-2d0fac784336)
###### Gambar 5: Univariate Analysis pada variabel Checking_account
Pada Gambar 5 dapat dilihat bahwa orang yang melakukan peminjaman kredit sebagian besar tidak memiliki rekening giro, sebaran data ini berlaku bagi yang lainnya.

![image](https://github.com/agungbesti/German_Credit_Risk/assets/35904444/cf5653c6-4b74-41df-8d81-1fbe7ff8ad66)
###### Gambar 6: Univariate Analysis pada variabel Purpose
Pada Gambar 6 dapat dilihat bahwa orang yang melakukan peminjaman kredit paling banyak digunakan untuk melakukan pembelian sebuah **Mobil (car)**. Hal ini menegaskan bahwa sebagian besar peminjam diisi oleh orang - orang yang **menginginkan kebutuhan sekundernya terpenuhi**, dibandingkan dengan kebutuhan primernya yaitu **pendidikan (education)** yang hanya memiliki alasan peminjaman sebanyak 6%.

![image](https://github.com/agungbesti/German_Credit_Risk/assets/35904444/2656663d-b012-415b-ab12-75d6078e6221)
###### Gambar 7: Univariate Analysis pada variabel Age_category
Pada Gambar 7 dapat dilihat bahwa orang yang melakukan peminjaman kredit diisi oleh kelompok anak dewasa akhir[4] rentang usia 25-35 tahun. Hal ini menegaskan bahwa sebagian besar peminjam diisi oleh orang - orang yang **sudah dewasa**, sedangkan untuk orang lansia cenderung tidak ada keinginan untuk melakukan peminjaman.

![image](https://github.com/agungbesti/German_Credit_Risk/assets/35904444/44fc117f-2582-4f14-90c2-7c45ea21798f)
###### Gambar 8: Univariate Analysis pada variabel Credit_category
Pada Gambar 8 dapat dilihat bahwa orang yang akan melakukan peminjaman memiliki nilai yang hampir sama pada setiap distribusinya. Hal ini menegaskan bahwa jumlah pinjaman memiliki distribusi yang seimbang dengan yang lainnya.

![image](https://github.com/agungbesti/German_Credit_Risk/assets/35904444/396b29b4-4895-4220-994b-55fa8d6d4e37)
###### Gambar 9: Univariate Analysis pada variabel Duration_category
Pada Gambar 9 dapat dilihat bahwa orang yang melakukan peminjaman kredit cenderung mengingikan **durasi yang lebih singkat**. Hal ini menegaskan bahwa sebagian besar peminjam menginginkan pinjaman yang relatif singkat untuk dilunasi.

![image](https://github.com/agungbesti/German_Credit_Risk/assets/35904444/98e37f61-0a1d-4d28-a16d-73d6831b70dd)
###### Gambar 10: Univariate Analysis pada variabel Risk
Pada Gambar 10 dapat dilihat bahwa orang yang melakukan peminjaman kredit memiliki history **pembayaran yang cukup baik/tepat waktu sebanyak 70%** dibandingkan dengan gagal bayar/telat bayar. Hal ini menegaskan bahwa sebagian besar **data imbalanced**.

![image](https://github.com/agungbesti/German_Credit_Risk/assets/35904444/e18a225b-e229-4923-8d8a-e37575709eb1)
###### Gambar 11: Multivariete Analysis pada variabel Sex dan Risk
Pada Gambar 11 dapat dilihat bahwa orang yang memiliki risiko tinggi gagal bayar cenderung dialami oleh **wanita sebesar 35%**. Hal ini menegaskan bahwa wanita memiliki **risiko gagal bayar atau telat bayar yang tinggi** dibandingkan pria.

![image](https://github.com/agungbesti/German_Credit_Risk/assets/35904444/12b82885-e801-4384-ba68-8fefe4dffc7f)
###### Gambar 12: Multivariete Analysis pada variabel Job dan Risk
Pada Gambar 12 dapat dilihat bahwa sebaran data yang memiliki kegagalan bayar terhadap jenis pekerjaan cenderung hampir sama. Hal ini menegaskan bahwa jenis pekerjaan bukanlah satu - satunya yang menyebabkan gagal bayar.

![image](https://github.com/agungbesti/German_Credit_Risk/assets/35904444/d31dc61c-6ba9-4715-9bbc-7e838189ebc9)
###### Gambar 13: Multivariete Analysis pada variabel Housing dan Risk
Pada Gambar 13 dapat dilihat bahwa orang yang **tidak memiliki rumah cenderung memiliki risiko tinggi gagal bayar/telat bayar** sebanyak 40%. Hal ini menegaskan bahwa siapapun yang tidak memiliki rumah dalam melakukan pinjaman cenderung memiliki risiko yang tinggi untuk gagal bayar/telat bayar.

![image](https://github.com/agungbesti/German_Credit_Risk/assets/35904444/23eba721-ef92-4900-b823-43ea9e9ab7c2)
###### Gambar 14: Multivariete Analysis pada variabel Saving_account dan Risk
Pada Gambar 14 dapat dilihat bahwa **orang yang memiliki nilai tabungan rekening yang lebih rendah cenderung berpotensi tinggi untuk melakukan gagal bayar/telat bayar** sebesar 35%. Hal ini menegaskan bahwa tabungan yang sedikit cenderung berisiko untuk gagal bayar/telat bayar.

![image](https://github.com/agungbesti/German_Credit_Risk/assets/35904444/27947039-d476-42a5-b53d-3aaa114b3e89)
###### Gambar 15: Multivariete Analysis pada variabel Checking_account dan Risk
Pada Gambar 15 dapat dilihat bahwa **orang yang memiliki rekening giro yang lebih kecil cenderung berpotensi tinggi untuk melakukan gagal bayar/telat bayar** sebesar 49%. Hal ini menegaskan bahwa rekening giro dengan nilai yang lebih kecil berpotensi tinggi untuk melakukan gagal bayar atau telat pelunasan.

![image](https://github.com/agungbesti/German_Credit_Risk/assets/35904444/5cb9a54e-f13d-4104-ba09-08614d2bc696)
###### Gambar 16: Multivariete Analysis pada variabel Purpose dan Risk
Pada Gambar 16 dapat dilihat bahwa **orang yang memiliki alasan peminjaman untuk liburan (vacation) cenderung berpotensi tinggi untuk melakukan gagal bayar/telat bayar** sebesar 41%. Hal ini menegaskan bahwa orang - orang yang memiliki gaya hidup mewah cenderung memiliki risiko kegagalan bayar yang tinggi.

![image](https://github.com/agungbesti/German_Credit_Risk/assets/35904444/f29f1592-1b65-4f87-b933-4236c791a2e9)
###### Gambar 17: Multivariete Analysis pada variabel Age_category dan Risk
Pada Gambar 17 dapat dilihat bahwa **usia anak antara 18-24 tahun (Student) berpotensi tinggi untuk melakukan kegagalan bayar/telat bayar** sebesar 42%. Hal ini menegaskan bahwa orang - orang yang masih remaja cenderung belum bisa bertanggung jawab untuk melakukan pelunasan peminjaman.

![image](https://github.com/agungbesti/German_Credit_Risk/assets/35904444/93ad590e-ba20-4b6b-ab76-859d4dcb6b25)
###### Gambar 18: Multivariete Analysis pada variabel Credit_category dan Risk
Pada Gambar 18 dapat dilihat bahwa **jumlah pinjaman yang tinggi cenderung memiliki potensi untuk kegagalan bayar** sebesar 42%. Hal ini menegaskan bahwa semakin besar nilai pinjaman yang dilakukan maka semakin besar risiko untuk mengalami kegagalan bayar atau pelunasan tepat waktu.

![image](https://github.com/agungbesti/German_Credit_Risk/assets/35904444/52fb96ca-487b-4f53-8668-06b14aab4653)
###### Gambar 19: Multivariete Analysis pada variabel Duration_category dan Risk
Pada Gambar 19 dapat dilihat bahwa **durasi yang lebih lama cenderung memiliki potensi tinggi untuk melakukan kegagalan bayar** sebesar 44%. Hal ini menegaskan bahwa semakin lama durasi yang diambil dalam waktu pelunasan, maka semakin besar pula risiko yang terjadi dalam kegagalan pembayaran atau pelunasan tepat waktu.

![image](https://github.com/agungbesti/German_Credit_Risk/assets/35904444/1235f8b8-0df3-4f78-987e-d93610ee4e76)
###### Gambar 20: Descriptive Analysis pada variabel Purpose dan Credit_amount
Pada Gambar 20 dapat dilihat bahwa ketika orang yang mengambil kredit untuk tujuan tertentu seperti pembelian **Mobil atau Furnitur memiliki ketimpangan mencapai 60-80%** sementara untuk beberapa tujuan seperti pembelian Radio/TV atau Pendidikan masih berada diantara nilai yang sama.

![image](https://github.com/agungbesti/German_Credit_Risk/assets/35904444/bf0efa80-3826-47e7-85ca-16e9a1a274a9)
###### Gambar 21: Descriptive Analysis pada variabel Purpose dan Sex
Pada Gambar 21 dapat dilihat bahwa **lebih banyak pria** mengambil pinjaman untuk mengambil mobil daripada perempuan. Sedangkan untuk pendidikan wanita dan pria hampir sama untuk jumlah orang yang melakukan peminjaman.

![image](https://github.com/agungbesti/German_Credit_Risk/assets/35904444/342f0e5c-c5f9-4a4c-8e54-acfc7ad1ca9e)
###### Gambar 22: Correlation Matrix untuk Fitur Numerik
Pada Gambar 22 dapat dilihat bahwa untuk setiap fitur memiliki korelasi positif dan korelasi negatif yang tidak terlalu tinggi, dimana fitur yang paling berpengaruh yaitu **Job, Duration dan Credit_amount**. Sedangkan pada **fitur Age, memiliki korelasi yang paling kecil diantara fitur yang lainnya**.

![image](https://github.com/agungbesti/German_Credit_Risk/assets/35904444/2136066f-2473-4518-9655-2e645bf3704f)
###### Gambar 23: Chi Squared Tes antara Fitur Risk dan Kategori
Pada Gambar 23 dapat dilihat bahwa ada beberapa fitur kategorikal yang memiliki korelasi yang signifikan terhadap risiko kegagalan bayar.
Fitur - fitur yang memiliki korelasi yang signifikan terhadap Risk:
- Sex
- Housing
- Saving_account
- Checking_account
- Age_category
- Credit_category
- Duration_category

# Data Preparation
---
## Proses Yang dilakukan
--- 
- Proses data preparation dilakukan melalui langkah-langkah, yaitu sebagai berikut: Melakukan load data pada google colaboratory, kemudian melakukan analisis awal terkait variabel yang tidak relevan untuk diproses lebih lanjut. Selanjutnya, memahami makna-makna variabel dengan menerapkan Exploratory Data Analysis dan mengatasi data yang masih kosong, membuat variabel baru dari data numerik seperti fitur age, duration dan amount. Kemudian melakukan visualisasi data untuk mencari outlier dengan menggunakan boxplot dari library seaborn. Selanjutnya, melakukan tahapan univariate analysis, multivariative analysis dan deskritif analysis. Selanjutnya membuat correlation matrix antar fitur numerik, kemudian memlihat korelasi yang signifikan menggunakan chi squared tes antara fitur risk dan kategori, lalu membuang variabel yang memiliki korelasi rendah atau tidak memiliki korelasi yang signifikan terhadap variabel risk.
- Menerapkan One Hot Encoding pada data Categorical dengan menggunakan pandas library pada fungsi pd.get_dummies()
- Menerapkan Oversampling SMOTE pada data yang imbalance
- Membagi data set antara training dan testing dengan library sklearn dengan fungsi train_test_split() dengan perbandingan 80:20 sehingga memiliki 1120 data train dan 280 data testing

## Alasan Penggunaan
--- 
- Melihat korelasi antara fitur numerik ataupun fitur kategorikal terhadap variabel risk, memberikan pemahaman bahwa fitur apa saja yang memiliki korelasi yang signifikan terhadap variabel risk.
- Pembuatan fitur baru dari data yang ada dapat meningkatkan akurasi model yang dihasilkan dengan pengkategorian dari data numerik.
- One-hot-encoding digunakan untuk mengubah variabel kategorikal menjadi representasi numerik yang dapat digunakan dalam model machine learning. serta dapat meningkatan performa model dalam melakukan klasifikasi karena hanya menggunakan nilai biner. Hal ini sangat diperlukan karena algoritma machine learning umumnya membutuhkan data numerik sebagai input.
- Penggunaan SMOTE diterapkan dalam rangka menangani ketidak seimbangan kelas. Teknik ini mensintesis sampel baru dari kelas minoritas untuk menyeimbangkan dataset dengan cara sampling ulang sampel kelas minoritas.
- Membagi dataset kedalam bentuk training dan testing adalah agar model dapat di evaluasi nantinya. Selain itu, pembagian ini dapat juga untuk mendeteksi apakah model mengalami overfitting jika model memiliki performa yang sangat baik pada data pelatihan tetapi performa yang buruk pada data pengujian.

# Modeling
---
Pada proyek ini akan menggunakan model Linear Discriminant Analysis, LightGBM dan XGBoost berdasarkan hasil dari perbandingan top tiga (3) algoritma yang dijalankan menggunakan library scikit-learn.
![image](https://github.com/agungbesti/German_Credit_Risk/assets/35904444/350cab0c-8783-40b2-9972-7bbcb8870c6b)
- Model machine learning yang digunakan adalah Linear Discriminant Analysis, LightGBM dan XGBoost algorithm.
- Linear Discriminant Analysis (LDA) merupakan salah satu metode yang digunakan untuk mengelompokkan data ke dalam beberapa kelas. Penentuan pengelompokan didasarkan pada garis batas (garis lurus) yang diperoleh dari persamaan linear.
- Untuk menggunakan model LDA menggunakan function LinearDiscriminantAnalysis yang merupakan bagian dari library sklearn.discriminant_analysis.
- Parameter yang digunakan dalam model LDA, yaitu sebagai berikut:
    -  solver = Pemecah untuk digunakan, dalam hal ini menggunakan solusi kuadrat terkecil yang dapat dikombinasikan dengan penyusutan atau penaksir kovarians khusus, yaitu lsqr
    -  store_covarience = Secara eksplisit hitung matriks kovarians dalam kelas yang diberi bobot ketika pemecahnya, yaitu True
    -  tol = Ambang batas absolut untuk nilai singular X yang dianggap signifikan, dalam hal ini digunakan 0.01
- Untuk mendapatkan hasil yang terbaik selain menggunakan LDA, digunakan algoritma LightGBM sebagai algoritma pembanding
- LightGBM mengimplementasikan algoritma Gradient Boosting Decision Tree (GBDT) konvensional dengan penambahan dua teknik baru: Gradient based One-Side Sampling (GOSS) dan Exclusive Feature Bundling (EFB)
- Penggunaan model LightGBM menggunakan function LGBMClassifier yang merupakan bagian dari library lightgbm.
- Parameter yang digunakan dalam model LightGBM, yaitu sebagai berikut:
    - learning_rate = parameter yang digunakan untuk mengatur proses training dari algoritma ini. Pada model ini, paramater di isi 1.0.
    - n_estimators = jumlah pohon keputusan (decision tree) yang akan dibuat pada model yang digunakan. Pada model ini n_estimators yang di buat, yaitu 1000.
    - max_depth = maksimal kedalaman dari decision tree yang akan dibuat. Pada model ini kedalaman yang dibuat sampai 5 level.
- Selain menggunakan LDA dan LightGBM, digunakan algoritma yang terakhir, yaitu XGBoost algorithm untuk melihat mana performa algoritma yang terbaik dalam  menghasilkan klasifikasi risiko gagal bayar peminjaman
- XGBoost adalah algoritma pohon yang ditingkatkan gradien. Peningkatan gradien adalah algoritma pembelajaran yang diawasi, yang mencoba memprediksi variabel target secara akurat dengan menggabungkan perkiraan serangkaian model yang lebih sederhana dan lebih lemah.
- Penggunaan model XGBoost menggunakan function XGBClassifier yang merupakan bagian dari library xgboost.

## Kelebihan dan kekurangan masing-masing algoritma
--- 
- Setelah melakukan training menggunakan LDA, LightGBM dan XGBoost Algoritma, maka dapat disimpulkan kelebihan dan kekurangan masing-masing, yaitu sebagai berikut:
    - Kelebihan algoritma LDA, yaitu dapat menghasilkan klasifikasi yang seimbang antar kelas.
    - Waktu training yang digunakan pada pembuatan model LDA jauh lebih cepat dibandingkan yang lainnya.
    - Kekurangan LDA , yaitu untuk mendapatkan klasifikasi yang akurat, melakukan tuning parameter saja tidak mampu mendapatkan akurasi yang lebih tinggi tapi diperlukan pengolahan data atau feature engineering yang tepat.
    - Kelebihan LightGBM algoritma, yaitu untuk proses hypertuning parameter dilakukan lebih cepat dibandingkan algoritma Xgboost.
    - Kekurangan LightGBM algoritma, yaitu LightGBM tidak cukup baik dalam melakukan klasifikasi terhadap kelas negatif atau risiko buruk
    - Kelebihan XGBoost algoritma, yaitu akurasi jauh lebih stabil dibandingkan algoritma yang lainnya.
    - Berdasarkan hasil training model, maka ditetapkan bahwa algoritma yang terbaik diantara LDA, LightGBM dan XGBoost Algoritma dalam mengklasifikasi tingkat risiko, yaitu algoritma XGBoost. Alasannya, karena nilai Akurasi yang dihasilkan oleh XGBoost lebih baik dari Algoritma yang lainnya.

# Evaluation
---
## Confusion Matrix
---
Confusion matrix adalah sebuah tabel yang sering digunakan untuk mengukur kinerja dari model klasifikasi di machine learning. Tabel ini menggambarkan lebih detail tentang jumlah data yang diklasifikasikan dengan benar maupun salah.

Ada empat nilai yang dihasilkan di dalam tabel confusion matrix, di antaranya **True Positive** (TP), **False Positive** (FP), **False Negative** (FN), dan **True Negative** (TN). Ilustrasi tabel confusion matrix dapat dilihat pada Gambar berikut.
![image](https://github.com/agungbesti/German_Credit_Risk/assets/35904444/8a16a5f1-836a-48d8-b27b-1b1c7f45c249)


**True Positive (TP)** : Jumlah data yang bernilai Positif dan diprediksi benar sebagai Positif.
Jika kita telah mengklasifikasikan tingkat risiko baik, dan ternyata risikonya baik.

**False Positive (FP)** : Jumlah data yang bernilai Negatif tetapi diprediksi sebagai Positif.
Jika kita telah mengklasifikasikan tingkat risiko baik, dan ternyata risikonya buruk.

**False Negative (FN)** : Jumlah data yang bernilai Positif tetapi diprediksi sebagai Negatif.
Jika kita telah mengklasifikasikan tingkat risiko buruk, dan ternyata risikonya baik.

**True Negative (TN)** : Jumlah data yang bernilai Negatif dan diprediksi benar sebagai Negatif.
Jika kita telah mengklasifikasikan tingkat risiko buruk, dan ternyata risikonya buruk.

### Accuracy
---
Nilai akurasi didapatkan dari jumlah data bernilai positif yang diprediksi positif dan data bernilai negatif yang diprediksi negatif dibagi dengan jumlah seluruh data di dalam dataset.

Rumus Accuracy = $$\frac{TP+TN}{TP+TN+FP+FN}$$

### Precision
---
Precision adalah peluang kasus yang diprediksi positif yang pada kenyataannya termasuk kasus kategori positif.

Rumus Precision = $$\frac{TP}{TP+FP}$$

### Recall
---
Recall adalah peluang kasus dengan kategori positif yang dengan tepat diprediksi positif.
Rumus Recall = $$\frac{TP}{TP+FN}$$

### F1
---
Nilai F1-Score atau dikenal juga dengan nama F-Measure didapatkan dari hasil Precision dan Recall antara kategori hasil prediksi dengan kategori sebenarnya.
Rumus F1-score = $$\frac{2*Precision*Recall}{Precision+Recall}$$ = $$\frac{2*TP}{2*TP+FP+FN}$$

Tabel 2: Hasil Evaluasi Model dengan Menggunakan Confusion Matrix pada Data Testing


Model                           | Precision     | Recall | f1-score | Accuracy  |
--------------------------------| --------------|--------|----------|-----------|
Linear Discriminat Analysisis   |       83%     | 83%    | 83%      |   82%     |
LightGBM                        |       85%     | 82%    | 82%      |   82%     |
XGBoost                         |       92%     | 91%    | 91%      |   91%     |

- Metrik yang digunakan untuk mengukur kinerja hasil model adalah Confusion Matrix.
- Berdasarkan pada data testing, bahwa model XGBoost menghasilan nilai tingkat akurasi sebesar 91%, hal ini menandakan bahwa model yang telah dibangun sudah cukup baik (good fit).
- Berdasarkan hasil training model, maka ditetapkan bahwa algoritma yang terbaik diantara Linear Discriminat Analysisis, LightGBM dan XGBoost Algoritma dalam mengklasifikasikan tingkat risiko peminjaman, yaitu algoritma XGBoost.
Alasannya, karena nilai akurasi yang dihasilkan oleh XGBoost lebih baik dari algoritma yang lainnya.

### Kesimpulan
---
- Berdasarkan hasil training dan test, maka algoritma yang terbaik adalah XGBoost, alasannya karena nilai akurasi yang dihasilkan oleh XGBoost lebih baik dari algoritma yang lainnya.
- Model yang dibangun sudah cukup baik dalam melakukan klasifikasi, alasannya karena nilai akurasi telah mencapai lebih dari 90%

### Referensi
---
[1] [SISTEM REKOMENDASI BUKU MENGGUNAKAN METODE ITEM-BASED COLLABORATIVE FILTERING](https://ejournal.undip.ac.id/index.php/jmasif/article/view/31482) 

[2] [SISTEM REKOMENDASI ARTIKEL BERITA MENGGUNAKAN METODE K-NEAREST NEIGHBOR BERBASIS WEBSITE](http://repository.unmuhjember.ac.id/654/1/journal.pdf)

[3] [Theoretical and Applied Aspects of Bank Credit Risks Minimization](https://ieeexplore.ieee.org/document/9468056)

[4] [KLASIFIKASI KELOMPOK UMUR MANUSIA BERDASARKAN ANALISIS DIMENSIFRAKTAL BOX COUNTING DARI CITRA WAJAH DENGAN DETEKSI TEPI CANNY](https://ejournal.unesa.ac.id/index.php/mathunesa/article/view/19398/17715)

