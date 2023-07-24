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
![image](https://github.com/agungbesti/Blog-Recommendation-System/assets/35904444/043405f5-7717-4503-bff7-99855ce21b24)

###### Gambar 1: Content-Based Filtering dan Collaborative Filtering


- **Content-Based Filtering** bekerja dengan melihat kemiripan artikel baru dengan artikel yang sebelumnya. Content-Based Filtering memberikan rekomendasi berdasarkan kemiripan artikel yang dianalisis dari fitur yang dikandung oleh artikel sebelumnya.
- **Collaborative filtering** merupakan proses penyaringan atau pengevaluasian artikel menggunakan penilaian orang lain sebagai informasi yang baru kepada pengunjung situs yang lainnya.
- Pada model Content-Based Filtering menerapkan metode **Cosine Similarity**  dan **Euclidean Distance** .
- Pada model Collaborative Filtering menerapkan metode yang di kombinasikan dengan deep learning yaitu **RecommenderNet**.

# Data Understanding
---
Data yang digunakan adalah dataset yang bersumber dari situs Kaggle yang berisi informasi blog medium dengan topik teknologi. Dataset tersebut dapat didownload pada link berikut ini:  [Blog Recommendation Data](https://www.kaggle.com/datasets/yakshshah/blog-recommendation-data). 
![image](https://github.com/agungbesti/Blog-Recommendation-System/assets/35904444/25d0f8fc-483f-4ed3-9515-923e95600bf4)

###### Gambar 2: Dataset
Data ini merupakan data yang dikumpulkan dari [Medium.com](https://www.medium.com) sebuah situs yang berisi berbagai informasi mengenai teknologi. Dataset ini memiliki 3 file yaitu:
- Author Data.csv yang berisi informasi penulis blog yang berjumlah 6868 baris
- Medium Blog Data.csv berisi data blog yang ditulis sebanyak 10467  baris.
- Blog Ratings.csv berisi 3 kolom yaitu blog_id, user_id dan ratings yang berjumlah 200140.

## Variabel-variabel yang terdapat pada dataset blog recommendation adalah sebagai berikut:
--- 
1. Author Data.csv
   - author_id : id unik penulis blog
   - author_name : nama penulis blog

2. Medium Blog Data.csv 
   - blog_id : id unik nama blog
   - author_id : id unik penulis blog
   - blog_title : judul blog
   - blog_content : ringkasan isi blog
   - blog_link : link blog
   - blog_img : gambar blog
   - topic : topik blog yang ditulis.
   - scrape_time: Waktu pengambilan data

3. Blog Ratings.csv
   - blog_id : id unik nama blog
   - userId : id pengguna
   - ratings : rating yang diberikan oleh pengguna

## Sample Data
---
## Author Data.csv
Tabel 1 : Sample Data Author Data
|#| author_id | author_name  |
|-|-------------------------------|--------|
|1|1 | yaksh|
|2|2 | XIT|
|3|3 | Daniel Meyer|
|4|4 | Seedify Fund|
|5|5 | Ifedolapo Shiloh Olotu|

## Medium Blog Data.csv 
Tabel 2 : Sample Data Medium Blog Data
|#| blog_id | author_id  | blog_title | blog_content | blog_link | blog_img | topic | scrape_time |
|-|-------------------------------|--------|-----------------------|------------------------|----------------------|-------------|--------------|--------------|
|1|1 | 4| Let’s Dominate The Launchpad Space Again	 | Hello, fam! If you’ve been with us since 2021,... | https://medium.com/@seedifyfund/lets-dominate-...	| https://miro.medium.com/fit/c/140/140/1*nByLJr...	 | ai	|2023-02-27 07:37:48|
|2|3 | 4| Let’s Dominate The Launchpad Space Again	 | Hello, fam! If you’ve been with us since 2021,... | https://medium.com/@seedifyfund/lets-dominate-...	| https://miro.medium.com/fit/c/140/140/1*nByLJr...	 | ai	|2023-02-27 07:41:47|
|3|4 | 7| Using ChatGPT for User Research| Applying AI to 4 common user research activiti... | https://medium.com/ux-planet/using-chatgpt-for...| https://miro.medium.com/fit/c/140/140/1*TZSGnN...| ai	|2023-02-27 07:41:47|
|4|5 | 8| The Automated Stable-Diffusion Checkpoint Merg...| Checkpoint merging is powerful. The power of c... | https://medium.com/@media_97267/the-automated-...| https://miro.medium.com/fit/c/140/140/1*x3N_Hj...| ai	|2023-02-27 07:41:47|
|5|6 |9| The Art of Lazy Creativity: My Experience Co-W...	| I was feeling particularly lazy one day and co... | https://medium.com/@digitalshedmedia/the-art-o...| https://miro.medium.com/fit/c/140/140/0*m2DdeT...| ai	|2023-02-27 07:41:47|

## Blog Ratings.csv
Tabel 3 : Sample Data Blog Ratings
|#| blog_id | userId  |ratings |
|-|-------------------------------|--------|--------|
|1|9025 | 11| 3.5|
|2|9320 | 11|5.0|
|3|9246 | 11| 3.5|
|4|9431 | 11|5.0|
|5|875 | 11|2.0|

## Langkah-Langkah dalam melakukan Data Understanding
---
Untuk memahami dataset, langkah-langkah yang dilakukan, yaitu sebagai berikut:
- Melakukan load dataset kedalam google colaboratory.
- Melakukan Exploratory data analysis untuk memahami makna-makna variabel yang terdapat dalam dataset.
- menggunakan teknik visualisasi data kategorikal dan non-kategorikal dengan menggunakan library seaborn.
- Melakukan univariative analysis untuk memahami sebaran data variabel.

### Hasil Visualisasi Exploratory Data Analysis
---

# Data Preparation
---
## Proses Yang dilakukan
--- 

## Alasan Penggunaan
--- 

# Modeling
---

## Kelebihan dan kekurangan masing-masing algoritma
--- 

# Evaluation
---

### Kesimpulan
---

### Referensi
---
[1] [SISTEM REKOMENDASI BUKU MENGGUNAKAN METODE ITEM-BASED COLLABORATIVE FILTERING](https://ejournal.undip.ac.id/index.php/jmasif/article/view/31482) 

[2] [SISTEM REKOMENDASI ARTIKEL BERITA MENGGUNAKAN METODE K-NEAREST NEIGHBOR BERBASIS WEBSITE](http://repository.unmuhjember.ac.id/654/1/journal.pdf)


