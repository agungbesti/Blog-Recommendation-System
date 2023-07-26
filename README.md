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

 <p align="center">
 <img src="https://github.com/agungbesti/Blog-Recommendation-System/assets/35904444/043405f5-7717-4503-bff7-99855ce21b24" alt="Content-Based Filtering dan Collaborative Filtering">
 Gambar 1: Content-Based Filtering dan Collaborative Filtering
 </p>

- **Content-Based Filtering** bekerja dengan melihat kemiripan artikel baru dengan artikel yang sebelumnya. Content-Based Filtering memberikan rekomendasi berdasarkan kemiripan artikel yang dianalisis dari fitur yang dikandung oleh artikel sebelumnya.
- **Collaborative filtering** merupakan proses penyaringan atau pengevaluasian artikel menggunakan penilaian orang lain sebagai informasi yang baru kepada pengunjung situs yang lainnya.
- Pada model Content-Based Filtering menerapkan metode **Cosine Similarity**  dan **Euclidean Distance** .
- Pada model Collaborative Filtering menerapkan metode yang di kombinasikan dengan deep learning yaitu **RecommenderNet**.

# Data Understanding
---
Data yang digunakan adalah dataset yang bersumber dari situs Kaggle yang berisi informasi blog medium dengan topik teknologi. Dataset tersebut dapat didownload pada link berikut ini:  [Blog Recommendation Data](https://www.kaggle.com/datasets/yakshshah/blog-recommendation-data). 

 <p align="center">
 <img src="https://github.com/agungbesti/Blog-Recommendation-System/assets/35904444/25d0f8fc-483f-4ed3-9515-923e95600bf4" alt="Dataset Blog">
 Gambar 2: Dataset Blog
 </p>

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
- menggunakan teknik visualisasi data kategorikal dan non-kategorikal dengan menggunakan library matplotlib.
- Melakukan univariete analysis untuk memahami sebaran data variabel.

### Hasil Visualisasi Exploratory Data Analysis
---
Tabel 4 : Melihat kolom dan tipe data pada dataset Author
|   #   |    Column     | Non-Null Count |  Dtype  |
|-------|--------------|----------------|---------|
|   0   |     author_id     |    6868        | int64  |
|   1   |    author_name     |    6868        | object  |

Pada Tabel 4 dapat dilihat bahwa data memiliki 1 kolom numerik atau angka sedangkan sisanya non-numerik atau kategorikal.

Tabel 5 : Melihat kolom dan tipe data pada dataset Blog
|   #   |    Column     | Non-Null Count |  Dtype  |
|-------|--------------|----------------|---------|
|   0   |blog_id|10467| int64  |
|   1   |author_id|10467| int64  |
|   2   |blog_title|10467| object  |
|   3   |blog_content|10467| object|
|   4   | blog_link|10467| object |
|   5   | blog_img|10467| object|
|   6   | topic |10467 | object |
|   7   | scrape_time |10467 | object|

Pada Tabel 5 dapat dilihat bahwa data memiliki 2 kolom numerik atau angka sedangkan sisanya non-numerik atau kategorikal.

Tabel 6 : Melihat kolom dan tipe data pada dataset Blog
|   #   |    Column     | Non-Null Count |  Dtype  |
|-------|--------------|----------------|---------|
|   0   |blog_id|200140| int64  |
|   1   |userId|200140| int64  |
|   2   |ratings|200140| float64  |

Pada Tabel 6 dapat dilihat bahwa data memiliki 3 kolom numerik atau angka.

 <p align="center">
 <img src="https://github.com/agungbesti/Blog-Recommendation-System/assets/35904444/d57e03ea-89ec-401b-ae1e-75930092c2bf" alt="Univariate Analysis pada variabel Topic">
 Gambar 3: Univariate Analysis pada variabel Topic
 </p>
Pada Gambar 3 dapat dilihat bahwa penulisan artikel dengan topik **AI (artificial inteligence) memiliki distribusi data yang lebih banyak** dibandingkan dengan topik yang lainnya. Hal ini disebabkan karena topik ai merupakan salah satu topic yang paling banyak diminati untuk saat ini.

Tabel 7 : Jumlah total penulisan artikel berdasarkan topik
|#| Topic | Total|
|-|------|--------|
|1|ai | 736 |
|2|blockchain | 644|
|3|cybersecurity | 642 |
|4|web-development | 635 |
|5|data-analysis | 594|
|6|cloud-computing | 589 |
|7|security | 527 |
|8|web3 | 471 |
|9|machine-learning | 467 |
|10|nlp | 453 |
|11|data-science  | 444 |
|12|deep-learning | 430|
|13|android | 426 |
|14|dev-ops | 384 |
|15|information-security | 374|
|16|image-processing | 354 |
|17|flutter | 343 |
|18|backend | 341|
|19|cloud-services | 339|
|20|Cryptocurrency | 331|
|21|app-development | 322|
|22|backend-development | 312|
|23|Software-Development | 309 |

Pada Tabel 7 diurutkan jumlah artikel yang paling banyak ditulis sampai yang terendah berdasarkan topik. Terlihat yang paling banyak ditulis adalah dengan topik **AI** dan yang terendah dengan penulisan artikel bertopik **Software-Development**.

 <p align="center">
 <img src="https://github.com/agungbesti/Blog-Recommendation-System/assets/35904444/bf909ecc-5f99-4d16-a050-e100dd2a0ec1" alt="Univariate Analysis pada variabel Ratings">
 Gambar 4: Univariate Analysis pada variabel Ratings
 </p>

Pada Gambar 4 dapat dilihat bahwa artikel yang ditulis **memiliki distribusi yang paling banyak pada skor 5** dibandingkan dengan yang lainnya. Hal ini menegaskan bahwa sebagian besar penulisan blog sudah sesuai dengan keinginan para pembaca blog.

# Data Preparation
--- 
##### 1. Menghapus kolom yang tidak diperlukan pada variabel blog
Langkah pertama yang dilakukan adalah melakukan pembersihan pada data blog dengan menghapus kolom tertentu, seperti:
   - author_id
   - blog_link
   - blog_img
   - scrape_time

##### 2. Menghapus blog yang duplikat
Langkah kedua yang dilakukan adalah melakukan penghapusan data blog yang terduplikasi pada bagian kolom blog_tile dan juga blog_content

##### 3. Melakukan Preprocessing Text Data
Langkah ketiga yang dilakukan adalah Melakukan praproses pada text data untuk menghapus stopwords dari konten blog dan juga menerapkan lemmatization untuk mengembalikan semua kata ke bentuk kata dasar.

Tabel 8 : Sample Data setelah Preprocessing Text Data
|#| blog_id | blog_title  | blog_content | topic | clean_blog_content |
|-|-------------------------------|--------|-----------------------|------------------------|----------------------|
|1|1 | Let’s Dominate The Launchpad Space Again| Hello, fam! If you’ve been with us since 2021,...	 | ai	| hello fam youve u since 2021 probably remember...|
|2|4 | Using ChatGPT for User Research	| Applying AI to 4 common user research activiti...		 | ai	| applying ai 4 common user research activity us...|
|3|5 | The Automated Stable-Diffusion Checkpoint Merg...	| Checkpoint merging is powerful. The power of c...| ai	| checkpoint merging powerful power checkpoint m...|
|4|6 | The Art of Lazy Creativity: My Experience Co-W..| I was feeling particularly lazy one day and co...	 | ai	|feeling particularly lazy one day couldnt both... |
|5|7 | LLaMA: Everything you want to know about Meta’...| Facebook’s Parent Company Just Released a Game...	 | ai	|facebooks parent company released gamechanging... |

##### 4. Melakukan Data Preprocessing untuk Content-Based Filtering
Setelah melakukan peghapusan kolom yang tidak diperlukan, penghapusan blog yang duplikat dan melakukan Preprocessing Text data langkah selanjutnya adalah melakukan Preprocessing untuk data yang akan digunakan pada pelatihan model. Dikarenakan ada 2 pendekatan yang dilakukan, maka tiap proses akan dipisah berdasarkan pendekatannya, Untuk model dengan pendekatan Content-Based Filtering data yang akan digunakan adalah data dari variabel blog.

##### 5. Melakukan Data Preprocessing untuk Collaborative Based Filtering
Untuk preprocessing data dengan pendekatan Collaborative Based Filtering, data yang akan digunakan adalah data dari variabel blog yang akan digabungkan dengan data dari variabel ratings. Pada pendekatan ini, data yang akan dijadikan patokan adalah **userId**, **id_blog** dan **ratings** yang diberikan oleh pengguna tersebut.

Dikarenakan itu, ada 4 tahapan yang akan dilakukan, yaitu:
1. Encode fitur userId dan blog_id
2. Konversi fitur rating
3. Mendapatkan jumlah data userId, blog_id, serta nilai ratings minimum dan maksimum.
4. Melakukan pembagian untuk variabel x (userId, blog_id) dan variabel y (ratings) dengan proporsi rasio 80% untuk data training dan 20% untuk data validasi.

###### A. Encode fitur userId dan blog_id
Proses ini dilakukan untuk membuat seluruh data tersebut bisa digunakan pada saat proses pembuatan model. Encode akan memiliki dua format, yaitu **_id to encoded_** dan **_encoded to id_**.

Setelah dilakukan encode, hasil encode id to encoded dilakukan mapping dan simpan ke variabel blog dengan nama kolom blog dan user. Sehingga data variabel blog akan menjadi sebagai berikut.

Tabel 9: Sample Data setelah encode fitur userId dan blog_id
|#| blog_id | blog_title  | blog_content | topic | clean_blog_content | userId | ratings | user |blog |
|-|---------|--------|-------|----------|----------|---------|-----------|-----------|--------------|
|1|8501| Why NodeJS is the Top Choice for Scalable and ...| NodeJS is a popular open-source framework that...		 | app-development | nodejs popular opensource framework allows dev...		| 2772	 | 5.0	|4763| 8453|
|2|2883| The Use of Java In Backend Development	| Introduction Programmers often get confused wh...		 | backend-development	 | introduction programmer often get confused cho...	| 128	 | 5.0		|326| 2849|
|3|1092| Web 3 Industry| Part 11 Emillion Reality journalism of the “Ta...		 | web3 | part 11 emillion reality journalism talent web...	| 1434	 | 3.5	|1522| 1064|
|4|9534| Go Packages and Modules: Building Reusable and...| As software development projects grow in size ...	 | Software-Development | software development project grow size complex...		| 2783	 | 2.0	|2660| 9484|
|5|4228| SupraOracles partners with Syscoin to build in...| SupraOracles is excited to announce its partne...		 | blockchain | supraoracles excited announce partnership sysc...		| 1035	 | 3.5|1187| 4191|

###### B. Konversi fitur Rating
Setelah melakukan encode pada fitur userId dan blog_id, langkah berikutnya adalah melakukan konversi dari nilai rating dari string menjadi float64. Sehingga hasil statistik dari variabel blog menjadi sebagai berikut.
Tabel 10 : Melihat kolom dan tipe data pada dataset Blog
|   #   |    Column     | Non-Null Count |  Dtype  |
|-------|--------------|----------------|---------|
|   0   |blog_id|200130 | int64  |
|   1   |blog_title|200130 | object  |
|   2   |blog_content|200130 | object  |
|   3   |topic|200130 | object|
|   4   | clean_blog_content|200130 | object |
|   5   | userId|200130 | int64|
|   6   | ratings  |200130  | float32 |
|   7   | user |200130  | int64|
|   8   | blog |200130  | int64|

###### C. Mendapatkan jumlah data userId, blog_id, serta nilai ratings minimum dan maksimum.
Langkah berikutnya adalah menghitung total user, judul blog dan nilai rating minimum dan maksimum. Total user dan judul blog akan digunakan untuk mengatur ukuran dari embedding pada model, kemudian nilai minimum dan maksimum rating digunakan untuk normalisasi nilai rating pada tahapan berikutnya.

Hasil dari tahapan ini, dapat disimpulkan bahwa total user adalah 5001 pengguna dengan total judul blog sebanyak 9705, kemudian rating minimum sebesar 0 dan maksimum sebesar 5.0.

###### D. Melakukan pembagian untuk variabel x (userId, blog_id) dan variabel y (ratings) dengan proporsi rasio 80% untuk data training dan 20% untuk data validasi.
Langkah terakhir adalah membagi data tersebut menjadi data latih dan data validasi. Untuk rasio pembagian adalah 80:20 dimana 80% adalah data latih dan 20% merupakan data uji.

Pada saat pembagian dilakukan pengacakan data. Hal ini digunakan agar data yang akan digunakan untuk latih dan uji bervariasi sehingga sampel 5 data pertama pada variabel blog  menjadi sebagai berikut.

Tabel 11: Sample Data setelah diacak
|#| blog_id | blog_title  | blog_content | topic | clean_blog_content | userId | ratings | user |blog |
|-|---------|--------|-------|----------|----------|---------|-----------|-----------|--------------|
|1|8501| Why NodeJS is the Top Choice for Scalable and ...| NodeJS is a popular open-source framework that...		 | app-development | nodejs popular opensource framework allows dev...		| 2772	 | 5.0	|4763| 8453|
|2|2883| The Use of Java In Backend Development	| Introduction Programmers often get confused wh...		 | backend-development	 | introduction programmer often get confused cho...	| 128	 | 5.0		|326| 2849|
|3|1092| Web 3 Industry| Part 11 Emillion Reality journalism of the “Ta...		 | web3 | part 11 emillion reality journalism talent web...	| 1434	 | 3.5	|1522| 1064|
|4|9534| Go Packages and Modules: Building Reusable and...| As software development projects grow in size ...	 | Software-Development | software development project grow size complex...		| 2783	 | 2.0	|2660| 9484|
|5|4228| SupraOracles partners with Syscoin to build in...| SupraOracles is excited to announce its partne...		 | blockchain | supraoracles excited announce partnership sysc...		| 1035	 | 3.5|1187| 4191|

Setelah diacak, lakukan pembagian dengan mengambil data dari fitur UserId, blog dan rating, kemudian untuk fitur rating akan di normalisasi dengan rumus sebagai berikut.

$$Rating_{norm} = \dfrac{rating - min(rating)}{max(rating) - min(rating)}$$



# Modeling
---
Pada proyek ini akan menggunakan 2 pendekatan tipe model, yaitu **Content-Based Filtering** dan **Collaborative Based Filtering**.

## A. Content-Based Filtering
Content-Based Filtering adalah metode dalam sistem rekomendasi yang menggunakan informasi konten atau fitur dari item (artikel, produk, film, lagu, dll.) dan preferensi pengguna untuk memberikan rekomendasi yang sesuai. Pendekatan ini mencocokkan preferensi pengguna dengan fitur-fitur item yang relevan.

Kelebihan dari Content Based Filtering adalah
1. Tidak memerlukan data pengguna lain atau kolaboratif.
2. Dapat memberikan rekomendasi personal yang disesuaikan dengan preferensi pengguna.
3. Dapat memanfaatkan fitur-fitur detail dari item untuk memberikan rekomendasi yang lebih spesifik.
4. Mampu menangani keadaan baru, di mana item baru dapat direkomendasikan berdasarkan fitur-fitur yang relevan.

Namun terdapat beberapa kelemahannya, yaitu
1. Rentan terhadap overfitting, di mana rekomendasi dapat menjadi terlalu spesifik dan kurang variasi.
2. Terbatas pada informasi konten yang tersedia untuk menggambarkan item.
3. Tidak dapat menangkap preferensi pengguna yang kompleks atau berubah seiring waktu.
4. Tidak mampu merekomendasikan item baru yang tidak ada dalam data pelatihan.

Pada metode ini, model yang dikembangkan akan menggunakan fitur topik dimana hasilnya akan merekomendasikan artikel berdasarkan kemiripan topic. Akan ada 2 teknik perhitungan similarity yang akan digunakan, yaitu **Cosine Similarity** dan **Euclidean Distance**.

### B. Menggunakan TF-IDF
TF-IDF, kependekan dari Term Frequency-Inverse Document Frequency, adalah teknik yang banyak digunakan dalam pemrosesan bahasa alami dan pengambilan informasi untuk mengukur pentingnya suatu istilah dalam dokumen dalam kumpulan dokumen. TF-IDF menggabungkan dua faktor: **Term Frequency (TF) dan Inverse Document Frequency (IDF)**.
- **Term Frequency (TF)**: TF mengukur frekuensi istilah dalam dokumen. Ini menghitung berapa kali suatu istilah muncul dalam dokumen dan mewakilinya sebagai hitungan mentah atau nilai yang dinormalisasi. Alasan di balik TF adalah bahwa istilah yang lebih sering muncul dalam dokumen cenderung lebih penting atau relevan dengan dokumen tersebut.
- **Inverse Document Frequency (IDF)**: IDF mengukur signifikansi suatu istilah di seluruh kumpulan dokumen. Ini menghitung logaritma fraksi terbalik dari jumlah dokumen yang mengandung istilah tersebut. Ide di balik IDF adalah bahwa istilah yang muncul di sejumlah kecil dokumen lebih informatif dan berharga daripada istilah yang muncul di sejumlah besar dokumen.

Perhitungan TF-IDF dilakukan dengan mengalikan nilai TF dan IDF secara bersamaan. Skor yang dihasilkan merepresentasikan pentingnya suatu istilah dalam dokumen dalam konteks keseluruhan kumpulan dokumen. Skor yang lebih tinggi menunjukkan bahwa suatu istilah lebih relevan atau berbeda dengan dokumen tertentu. 
Rumus perhitungan TF-IDF untuk term (t) dalam dokumen (d) dalam kumpulan dokumen adalah sebagai berikut:
 <p align="center">
 <img src="https://github.com/agungbesti/Blog-Recommendation-System/assets/35904444/aff977e6-a9d8-4104-9686-9454910d17a6" alt="Rumus perhitungan TF-IDF">
 Gambar 5: Perhitungan TF-IDF
 </p>

Pada bagian ini, TF-IDF akan diterapkan untuk kolom topic. Langkah yang dilakukan untuk menerapkan Pada tahapan ini, tokenizer yang akan digunakan adalah dengan split pada data kolom tersebut. Hal ini digunakan agar data topic akan diproses dalam keadaan utuh, seperti pada suatu artikel dengan topic "Ai, development, cybersecurity", maka setelah dilakukan vectonizer menjadi ['ai', 'development', 'cybersecurity']. Setelah itu lakukan perhitungan IDF pada data topic. Kemudian jika di-mapping, maka hasilnya akan sebagai berikut.
```
array(['ai', 'image-processing', 'Cryptocurrency', 'data-science', 'dev-ops', 'security', 'android', 'cloud-computing', 'nlp', 'cloud-services', 'flutter', 'web3', 'cybersecurity', 'information-security', 'blockchain', 'machine-learning' 'deep-learning' 'data-analysis' 'backend', 'backend-development', 'app-development', 'web-development', 'Software-Development'], dtype=object)
```
Setelah itu, lakukan proses fit dan transformasikan ke dalam bentuk matriks. Sehingga hasil ukuran matriks yang terbentuk adalah 10466 x 28 dengan 10 sampel hasil nya adalah sebagai berikut:

Tabel 12: Sampel hasil TF-IDF
| blog_title | development | machine | ai | cybersecurity | blockchain | services | nlp | web | cryptocurrency |
|:-------------------------------------------|----------:|----------------:|----------:|---------:|---------:|---------:|----------:|----------:|--------:|
| 2023’s Top IoT Device Management Software and Platforms | 0 | 0 | 0 | 0 | 0 | 0.7912	 | 0 | 0 | 0 |
| Creating a Conversational AI Chatbot with Python: A Comprehensive Guide | 0 | 0 | 0 | 0 | 0 | 0 | 1.0 | 0 | 0 | 0 |
| Unveiling the Surprising Results of Testing the World’s Top AI Models' Causal Understanding!| 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| InfoSecSherpa’s News Roundup for Wednesday, February 8, 2023 | 0 | 0 | 0 | 0 | 0 | 0 | 1.0 | 0 | 0 | 0 |
|Deploy multiple apps in Aptible with a reverse proxy| 0 | 0 | 0 | 0 | 0 | 0 | 0.7912 | 0 | 0 | 0 |
| Leaders in Cybersecurity to Follow| 0 | 0 | 0 | 1.0 | 0 | 0 | 1.0 | 0 | 0 | 0 |
| Tech and the Human Experience: How Big Tech Misses the Mark on Cultural Competency.	| 0 | 0 | 0 | 0 | 0 | 0 | 1.0 | 0 | 0 | 0 |
| Web Stack Weekly — Issue#61 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Dependency Injection in Spring Boot: A Beginner’s Guide | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| At the Intersection of Machine Learning and Image Analysis: Insights from a Data Scientist	 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

### C. Menggunakan Cosine Similarity
Cosine Similarity adalah ukuran yang digunakan untuk menentukan kesamaan antara dua vektor dalam ruang multidimensi. Ini menghitung cosinus sudut antara vektor, yang menunjukkan seberapa dekat hubungan vektor dalam hal orientasi dan arahnya. Dalam cosine similarity, kemiripan antara dua vektor diukur berdasarkan sudut antara vektor-vektor tersebut. Nilai cosine similarity berkisar antara -1 hingga 1, di mana nilai 1 menunjukkan kedua vektor memiliki arah yang sama atau sangat mirip, nilai 0 menunjukkan tidak ada kemiripan, dan nilai -1 menunjukkan arah yang berlawanan atau sangat berbeda.

Berikut adalah rumus untuk menghitung Cosinus Similarity:

 <p align="center">
 <img src="https://github.com/agungbesti/Blog-Recommendation-System/assets/35904444/9510067a-5fc3-4e86-a6f7-7e1a65c9330a" alt="Rumus Cosine Similarity">
 Gambar 6: Rumus Cosine Similarity
 </p>
 
Di mana:
- A dan B adalah dua vektor yang akan dibandingkan.
- A . B adalah hasil perkalian dot (inner product) antara vektor A dan B.
- ||A|| dan ||B|| adalah panjang (magnitude) dari vektor A dan B.

 <p align="center">
 <img src="https://github.com/agungbesti/Blog-Recommendation-System/assets/35904444/ddf22025-9e03-420f-ab67-54b072926a39" alt="Cosine Similarity Concept">
 Gambar 9. Grafik jarak antar 2 vektor di <i>Cosine Similarity</i> 
 </p>

Untuk penerapan pada proyek ini, dapat menggunakan fungsi cosine_similarity dari library sklearn. Pada tahapan ini, kita akan menghitung Cosine Similarity pada data hasil tf-idf sebelumnya. Sehingga hasil dari proses ini dapat dilihat di tabel sampel berikut ini.

Tabel 13: Sampel hasil perhitungan jarak menggunakan Cosine Similarity
| blog_title | What are the most notable findings and recommendations from Jordan’s national cybersecurity strategy? | Day 6: Improving Security and Error Handling in My Web Application | Traditions Create Stability For Kids | Type-Safe Builders in Kotlin: The Secret Sauce of Happy Developers | SQL Injection ( SQLi ) |
|:-------------------------------------------|----------:|----------------:|----------:|---------:|---------:|
| Extending the Android SDK — SDKExtensions | 0 | 0 | 0 | 1.0 | 0 |
| Python and Interpreter— QuickStart | 0 | 0 | 0 | 0 | 0 |
| Useful GitHub repositories to build a better App UI. | 0 | 0.328412 | 0 | 0 | 0 |
| Telco Churn Prediction With Machine Learning	 | 0 | 0 | 0 | 0 | 0 |
| Kubernetes Architecture : Powerful and simple yet complex at the same time | 0 | 0 | 0 | 0 | 0 |
| Solar Cells Image Classification using DNN Deep Neural Networks | 0 | 0 | 0 | 0 | 0 |
| The 5 Most Active Blockchains | 0 | 0 | 0 | 0 | 0 |
| Subdomain Take Over on Azurewebsite	 | 0 | 0 | 0.623386	 | 0 | 0 |
| Sydney Houston Group Announces Partnership with Creaticles | 0 | 0 | 0 | 0 | 0 |
| SkyLine — An Informative Document | 1.0 | 0 | 0 | 0 | 1.0 |

### D. Menggunakan Euclidean Distance
Euclidean Distance adalah sebuah metode yang digunakan untuk mengukur jarak antara dua titik dalam ruang berdimensi banyak. Metode ini sering digunakan dalam analisis data, pengelompokan data, dan pengenalan pola.

Dalam Euclidean Distance, jarak antara dua titik dihitung sebagai panjang garis lurus yang menghubungkan kedua titik tersebut. Jarak ini dihitung berdasarkan perbedaan koordinat antara dua titik pada setiap dimensi. Semakin besar nilainya, maka semakin jauh perbedaannya.

Rumusnya adalah sebagai berikut:

$$Euclidean Distance_{(A,B)} = \sqrt{(x2 - x1)^2 + (y2 - y1)^2}$$

Di mana:

-   A dan B adalah dua titik yang akan diukur jaraknya.
-   (x1, y1) dan (x2, y2) adalah koordinat dari titik A dan B pada setiap dimensi.

 <p align="center">
 <img src="https://github.com/agungbesti/Blog-Recommendation-System/assets/35904444/6ba0073a-2fc8-41d8-8476-f9a229e28909" alt="euclidean Distance">
 Gambar 10. Grafik jarak antara 2 vektor di <i>euclidean Distance</i>  dimana konsep pengukurannya sama dengan perhitungan sisi miring Pythagoras
 </p>

Untuk penerapan pada proyek ini, dapat menggunakan fungsi euclidean_distance dari library sklearn. Pada tahapan ini, kita akan menghitung euclidean distance pada data hasil tf-idf sebelumnya. Sehingga hasil dari proses ini dapat dilihat di tabel sampel berikut ini.

Tabel 14: Sampel hasil perhitungan jarak menggunakan Euclidean Distance
| blog_title | Joining Data with Pandas — The Important Concepts (Part 1) | The Future of Artificial Intelligence: Exploring the Latest Advancements. | NFT Breaking News — Bringing joy in Africa through the WaxSnkrs Project	 | ctproc | Elon Musk’s Twitter Ad Revenue Initiative Boosts Dogecoin Price|
|:-------------------------------------------|----------:|----------------:|----------:|---------:|---------:|
| The Key to Finding Common Ground: Asking the Right Questions | 0 | 0 | 0 | 0 | 0 |
|8 Week SQL Challenge Case Study #1 — Danny’s Diner | 1.0 | 0 | 0 | 0 | 0 |
| Engaging in the remarkable — An exploration of Intelligence	 | 0 | 0 | 0 | 0 | 0 |
| Steel threads are a technique that will make you a better engineer	 | 0 | 0 | 0 | 0 | 0 |
| Boosting Your JavaScript Performance: PNPM, Deep Cloning, and JSON Circular Reference Solutions | 0 | 0 | 0 | 0 | 0 |
| 5 Crypto Trading Strategies for Beginner and Intermediate Traders | 0 | 0 | 0 | 0 | 0 |
| Jibu Labs | 0 | 0 | 0 | 0 | 0 |
| Does NIST CSF v1.1 play nicely with the EU GDPR?	 | 0 | 0 | 0	 | 0 | 0 |
|How to Claim a .Celo Domain and Mint a Prosperity Passport	 | 0 | 0 | 1.0 | 0 | 1.0 |
| Presearch’s Meme Backgrounds: A Game-Changer in Search Engine Experiences | 0 | 0 | 1.0 | 0 | 1.0 |

### E. Hasil Rekomendasi Content-Based Filtering
Setelah menerapkan kedua metode pengukuran kemiripan diatas, langkah terakhir adalah membuat sebuah fungsi untuk menghasilkan hasil prediksi yang disimpan pada sebuah dataframe. Fungsi tersebut memiliki 1 parameter wajib dan 4 parameter opsional dengan sebagai berikut:
- blog_title - Judul artikel blog yang menjadi patokan untuk direkomendasikan (Wajib)
- similarity_data - dataframe dari hasil perhitungan similarity. Secara bawaan, dataframe cosine similarity yang digunakan
- items - Mengandung kedua nama dan fitur lainnya yang digunakan untuk mendefinisikan kemiripan
- k - Banyaknya jumlah rekomendasi yang diberikan

Cara kerja algoritma ini adalah sebagai berikut:
1. Mengambil data dengan menggunakan argpartition untuk melakukan partisi secara tidak langsung sepanjang sumbu yang diberikan. 
2. Mengambil data dengan similarity terbesar dari index yang ada.
3. Drop blog_title agar nama blog yang dicari tidak muncul dalam daftar rekomendasi
4. Tampilkan daftar rekomendasi

Pada contoh ini, saya akan memasukkan title salah satu blog yang berjudul "Relation between cloud computing and artificial intelligence" dengan informasi artikel tersebut sebagai berikut.

Tabel 15: Judul artikel yang akan dijadikan salah satu contoh
|#| blog_id | blog_title  | blog_content | topic | clean_blog_content |
|-|---------|--------|-------|----------|----------|
|2628|2655|Relation between cloud computing and artificial intelligence| Cloud computing is a fairly new service that o...	| cloud-services| cloud computing fairly new service organizatio...|

Kemudian hasil dari daftar 10 artikel yang direkomendasi berdasarkan metode Cosine Similarity adalah sebagai berikut:

Tabel 16: Hasil 10 artikel yang direkomendasikan menggunakan pendekatan Cosine Similarity
|#| blog_title | topic  | clean_blog_content | score |
|-|---------|--------|-------|-------|
|1|Cloud Databases: Modern Approach in IT. Part 1...	|cloud-services	| almost technology project requires database th...|1|
|2|Cloud Services : SaaS vs PaaS vs IaaS vs STaaS...|cloud-services	| quick guide understanding different type cloud...|1|
|3|How to create your first AWS EKS cluster using...|cloud-services	| running software container environment like aw...|1|
|4|Getting started with Cloud Computing	|cloud-services	| cloud computing cloud computing model deliveri...|1|
|5|A Beginner’s Guide To Cloud|cloud-services	| find extensive list definition across platform...|1|
|6|Why I recommend everyone to dump torrent clien...|cloud-services	| hi today reviewing seedr website allows downlo...|1|
|7|Open-source Data Tools will provide a way out ...|cloud-services	| need know use production many business acquire...|1|
|8|ECS Deep Dive, why does this container orchest...|cloud-services	| im working startup first priority minimize ope...|1|
|9|Happy Super Bowl! Share The Winning Team in ...|cloud-services	| football fan day come two best team compete fa..|1|
|10|Securing the Cloud: Best Practices for Protect...	|cloud-services	| article talk best practice securing cloud prot...|1|

Sedangkan untuk hasil dari daftar 10 artikel yang direkomendasi berdasarkan metode Euclidean Distance adalah sebagai berikut:

Tabel 14: Hasil 10 artikel yang direkomendasikan menggunakan pendekatan Euclidean Distance
|#| blog_title | topic  | clean_blog_content | score|
|-|---------|--------|-------|-------|
|1|Amazon DevOps Guru|cloud-services| guru devops became one fastest methodology get...|0|
|2|Multicloud Strategy!	|cloud-services| whats multicloud organization opting multiclou...	|0|
|3|Using Cloudinary Gem with Ruby on Rails|cloud-services|youre ready push rail app heroku everything go...	|0|
|4|Using API Gateway with Lambda to Send a Messag...|cloud-services|api gateway provide unified entry point across...|0|
|5|Using LiteFS with Bun on Fly.io|cloud-services|neither bun litefs recommended production yet ...|0|
|6|What is Azure Sentinel|cloud-services|introduction cybersecurity evergrowing concern...|0|
|7|Why is the service mesh a critical component o..|cloud-services|introduction cloudnative environment becoming ..|0|
|8|Eight Critical Success Factors for Cloud Migra..|cloud-services|migrating cloud requires substantial time comm...|0|
|9|Cloud Computing ☁️☁️|cloud-services|youre seeking expand knowledge cloud computing...|0|
|10|ZenDesk vs. Salesforce Service Cloud: A Compar...|cloud-services|delivering exceptional customer service choosi..|0|

## F. Collaborative Based Filtering
Collaborative Based Filtering adalah metode dalam sistem rekomendasi yang mengandalkan informasi dari pengguna lain untuk memberikan rekomendasi. Pendekatan ini mencocokkan preferensi pengguna dengan preferensi pengguna lain yang memiliki profil atau perilaku serupa. 

Kelebihan dari metode ini yaitu:
1. Mampu menangani kompleksitas preferensi pengguna yang tidak dapat dilihat dari informasi konten item.
2. Dapat merekomendasikan item baru yang tidak ada dalam data pelatihan berdasarkan perilaku pengguna lain.
3. Tidak terbatas pada fitur-fitur item, sehingga dapat merekomendasikan item yang berbeda tetapi relevan.

Namun terdapat beberapa kekurangan yaitu:
1. Bergantung pada ketersediaan data pengguna yang cukup untuk membuat rekomendasi yang akurat.
2. Membutuhkan pemrosesan yang lebih intensif dan kompleks, terutama dalam skala besar.
3. Rentan terhadap cold start di mana rekomendasi sulit untuk pengguna baru yang memiliki sedikit atau tanpa data perilaku.

### G. Membangun Model
Untuk metode ini, algoritma model yang akan digunakan adalah algoritma **_RecommenderNet_**.
_RecommenderNet_ menggunakan 4 buah node embeddings. Embedding adalah sebuah hidden layers di sebuah neural network. Tujuan dari embedding adalah melakukan mapping dari data dengan dimensi tinggi ke dimensi yang lebih rendah, sehingga model dapat lebih mudah mempelajari hubungan antara input dan proses data lebih efisien.

 <p align="center">
 <img src="https://github.com/agungbesti/Blog-Recommendation-System/assets/35904444/239917ad-636a-4d45-8fa5-eca305657430" alt="RecommenderNet">
 Gambar 10.Gambaran dari Embedding Layers
 </p>
 
 Embedding layer pada model ini ada 2 tipe, yaitu embedding untuk fitur dan bias layer dimana embedding fitur akan memiliki parameter regularizer l2 dengan nilai 0.000001 dan ukurannya adalah total fitur x 50.

Sedangkan bias akan memiliki ukuran total fitur x 1.

Hasil dari embedding fitur user dan anime kemudian dilakukan operasi perkalian matriks menggunakan tensordot. Kemudian hasil perkalian tersebut baru dijumlahkan dengan bias dari user dan anime. Setelah itu dimasukkan ke fungsi aktivasi sigmoid.

### H. Compile Model dan Buat Callbacks
Setelah membangun model, model kemudian dilakukan kompilasi. Pada bagian ini, juga akan ditentukan loss function, optimizer, dan metrics. Untuk **loss function** dan **metrics** akan menggunakan **BinaryCrossentropy** dan **RootMeanSquare Error**. Sedangkan untuk optimizer,  menggunakan 2 metode, yaitu **Adam lr.0.001** dan **RMSprop lr.0.0001**.

Kemudian dibuat dua callback yaitu **EarlyStopping** yang bertujuan untuk menghentikan proses latih jika nilai metrik tertentu tidak mengalami perubahan ke yang lebih baik. Selain itu teknik ini bisa digunakan untuk menyimpan model dengan hasil terbaik, sehingga ketika pelatihan selesai, model yang dijalankan di memori adalah model dengan weight dan kualitas terbaiknya. Pada EarlyStopping ini, nilai patience adalah 1 yang menunjukkan jika 1 epoch berikutnya tidak ada peningkatan kualitas, maka proses latih dihentikan. Kemudian parameter restore_best_weight menjadi True dan monitor diatur menjadi **"val_root_mean_squared_error"**, sehingga hasil RMSE validasi yang akan dijadikan patokan.

Selain itu juga diterapkan callback ModelCheckpoint. Tujuan dari callback ini adalah agar setiap epochs yang memberikan hasil terbaik, model tersebut langsung disimpan. Model dapat disimpan dalam format **h5, hdf, pb** dan **ckpt**.

### I. Training Model 
Setelah itu, maka dijalankan proses pelatihan dengan jumlah batch sebesar 1024 dengan jumlah epochs sebesar 100. Lama proses pelatihan sangat bervariasi. Untuk model dengan **_Optimizer Adam_**, proses pelatihan cukup singkat dengan jumlah epoch adalah 2 epoch, sedangkan untuk **_Optimizer RMSprop_** sebesar 2 epoch.

Setelah dilakukan perbandingan, maka model dengan **_Optimizer RMSprop_** memberikan hasil yang lebih baik dibandingkan menggunakan Adam dimana penjelasan lebih detail akan dibahas di bagian Evaluation.

### I. Hasil Rekomendasi Collaborative Based Filtering
Setelah melakukan pelatihan, langkah terakhir adalah membuat sebuah fungsi untuk menghasilkan hasil rekomendasi ke pengguna. Langkah yang perlu dilakukan adalah sebagai berikut:
1. Masukkan userID pengguna. Pada bagian ini, akan memilih secara acak userID yang akan diberikan rekomendasi
2. Mengambil data blog yang pernah dibaca oleh pengguna
3. Berdasarkan data blog yang pernah dibaca, ambil data blog yang belum pernah dibaca
4. Ambil encoded blog dari daftar blog belom pernah dibaca
5. Ambil encoded userID yang akan diberikan rekomendasi
6. Buat sebuah data awal yang akan digunkan model untuk memprediksi berdasarkan encode userID sekarang dengan encode blog yang belum pernah dibaca
7. Jalankan prediksi model berdasarkan data tersebut
8. Ambil hasil dengan rating tertinggi kemudian ambil id blog tersebut
9. Tampilkan hasil rekomendasi blog yang akan dibaca
 
Pada contoh ini, sistem rekomendasi akan memberikan rekomendasi blog untuk pengguna dengan userID 71 dimana 5 blog yang paling tinggi rating yang ia berikan adalah sebagai berikut:

Tabel 15 : Hasil 5 blog dari pengguna 71 dengan rating tertinggi

|#| Title Blog | Topic|
|-|------|--------|
|1|Kotlin loop best practice with example| android |
|2|5 Reasons You Should Buy An Android Phone Even If You’re A Die Hard iPhone User | android|
|3|How To Use Fetch API and Redux in a React Native Functional Component:[Android] | android|
|4|Make Gradle Dependencies Management better | android|
|5|From Shared Preferences to SQLite: Understanding Data Persistence in Flutter | flutter|

Dan setelah dilakukan prediksi, maka 10 blog yang direkomendasikan untuk pengguna 71 adalah sebagai berikut:

Tabel 16 : Hasil rekomendasi 10 blog untuk pengguna 71

|#| Title Blog | Topic|
|-|------|--------|
|1|The Power of Python’s Frequency Aggregation: Unlocking Valuable Insights from Your Data| data-analysis |
|2|Beginners guide to data analysis using Python. Part 2| data-analysis|
|3|Data Analysis Project Using SQL and Power BI — Analysis of Supermarket Sales| data-analysis|
|4|Image Processing with Python: Frequency Domain Filtering for Noise Reduction and Image Enhancement| image-processing|
|5|Documents Image Quality Assessment Part -1 | image-processing|
|6|Predict customer behavior? 7 Simple Steps to Begin Utilizing Natural Language Processing in Data Science | nlp|
|7|The real role of a SaaS CEO | Software-Development|
|8|What’s Kubernetes? Why open source technology gets a lot of tractions from AWS | Software-Development|
|9|Redis vs. Other Databases: An In-Depth Comparison of SQL and NoSQL Solutions | Software-Development|
|10|Power Up Spawning | Software-Development|



# Evaluation
---

Pada proyek ini, ada dua perhitungan metrik yang digunakan berdasarkan metode pendekatan yang digunakan, yaitu Precision untuk pendekatan Content-Based Filtering dan Root Mean Square Error untuk Collaborative Based Filtering.

Pada pendekatan ini, metrik yang digunakan adalah **Precision**. Sebelum masuk ke penjelasan mengenai Precission, maka kita harus mengetahui mengenai confusion matrix.

## A. Confusion Matrix
Confusion matrix adalah sebuah tabel yang sering digunakan untuk mengukur kinerja dari model klasifikasi di machine learning. Tabel ini menggambarkan lebih detail tentang jumlah data yang diklasifikasikan dengan benar maupun salah.

Ada empat nilai yang dihasilkan di dalam tabel confusion matrix, di antaranya **True Positive** (TP), **False Positive** (FP), **False Negative** (FN), dan **True Negative** (TN). Ilustrasi tabel confusion matrix dapat dilihat pada Gambar berikut.

 <p align="center">
 <img src="https://github.com/agungbesti/German_Credit_Risk/assets/35904444/8a16a5f1-836a-48d8-b27b-1b1c7f45c249" alt="Confusion Matrix">
 Gambar 11. Confusion Matrix
 </p>
 
**True Positive (TP)** : Jumlah data yang bernilai Positif dan diprediksi benar sebagai Positif.
Jika kita telah mengklasifikasikan tingkat risiko baik, dan ternyata risikonya baik.

**False Positive (FP)** : Jumlah data yang bernilai Negatif tetapi diprediksi sebagai Positif.
Jika kita telah mengklasifikasikan tingkat risiko baik, dan ternyata risikonya buruk.

**False Negative (FN)** : Jumlah data yang bernilai Positif tetapi diprediksi sebagai Negatif.
Jika kita telah mengklasifikasikan tingkat risiko buruk, dan ternyata risikonya baik.

**True Negative (TN)** : Jumlah data yang bernilai Negatif dan diprediksi benar sebagai Negatif.
Jika kita telah mengklasifikasikan tingkat risiko buruk, dan ternyata risikonya buruk.

Dari Confusion Matrix ini dapat ditentukan nilai Accuracy, Precission, Recall dan F1-Score. Karena pendekatan model ini menggunakan Precission, maka yang akan dijelaskan disini adalah Precission.

## B. Precision
---
Precision adalah peluang kasus yang diprediksi positif yang pada kenyataannya termasuk kasus kategori positif. Precission salah satu metrik evaluasi yang umum digunakan untuk mengukur sejauh mana model klasifikasi benar dalam mengidentifikasi secara akurat. Precission memberikan gambaran tentang seberapa baik model dalam menghindari memberikan hasil False Positive. Precission yang tinggi menunjukkan bahwa model memiliki sedikit jumlah kesalahan dalam mengklasifikasikan data dengan kelas negatif menjadi positif.

Rumus dalam perhitungan precission matiks adalah sebagai berikut.

$$Precission = \dfrac{TP}{TP + FP}$$

Pada proyek ini, penulis akan melakukan percobaan untuk mendapatkan rekomendasi blog dimana ada 1 judul blog yang akan dijadikan bahan percobaan. Untuk nilai TP dapat diberikan jika topic tersebut memiliki relevansi dengan topic dari blog yang jadi patokan. Suatu judul dinyatakan relevan jika memenuhi minimal 2 topic yang ada pada blog patokannya.


Tabel 17 : Judul blog yang akan dijadikan patokan untuk mendapatkan daftar rekomendasi blog
|#| blog_id | blog_title  | blog_content | topic | clean_blog_content |
|-|---------|--------|-------|----------|----------|
|2628|2655|Relation between cloud computing and artificial intelligence| Cloud computing is a fairly new service that o...	| cloud-services| cloud computing fairly new service organizatio...|

Selain itu penulis akan membandingkan hasil rekomendasi yang menggunakan cosine similarity dan euclidean distance dimana hasil nya untuk blog pertama pertama adalah sebagai berikut.

<table>
<thead>
  <tr>
    <th colspan="3">Cosine Similarity</th>
    <th></th>
    <th colspan="4">Euclidean Distance </th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>No</td>
    <td>Blog Title</td>
    <td>Topic</td>
    <td>Score</td>
    <td>No</td>
    <td>Blog Title</td>
    <td>Topic</td>
    <td>Score</td>
  </tr>
  <tr>
    <td>1</td>
    <td>Cloud Databases: Modern Approach in IT. Part 1..</td>
    <td>cloud-services</td>
    <td>1</td>
    <td>1</td>
    <td>Amazon DevOps Guru</td>
    <td>cloud-services</td>
    <td>0</td>
  </tr>
  <tr>
    <td>2</td>
    <td>Cloud Services : SaaS vs PaaS vs IaaS vs STaaS...</td>
    <td>cloud-services</td>
    <td>1</td>
    <td>2</td>
    <td>Multicloud Strategy!</td>
    <td>cloud-services</td>
    <td>0</td>
  </tr>
  <tr>
    <td>3</td>
    <td>How to create your first AWS EKS cluster using...</td>
    <td>cloud-services</td>
    <td>1</td>
    <td>3</td>
    <td>Using Cloudinary Gem with Ruby on Rails</td>
    <td>cloud-services</td>
    <td>0</td>
  </tr>
  <tr>
    <td>4</td>
    <td>Getting started with Cloud Computing</td>
    <td>cloud-services</td>
    <td>1</td>
    <td>4</td>
    <td>Using API Gateway with Lambda to Send a Messag…</td>
    <td>cloud-services</td>
    <td>0</td>
  </tr>
  <tr>
    <td>5</td>
    <td>A Beginner’s Guide To Cloud</td>
    <td>cloud-services</td>
    <td>1</td>
    <td>5</td>
    <td>Using LiteFS with Bun on Fly.io</td>
    <td>cloud-services</td>
    <td>0</td>
  </tr>
  <tr>
    <td>6</td>
    <td>Why I recommend everyone to dump torrent clien...</td>
    <td>cloud-services</td>
    <td>1</td>
    <td>6</td>
    <td>What is Azure Sentinel</td>
    <td>cloud-services</td>
    <td>0</td>
  </tr>
  <tr>
    <td>7</td>
    <td>Open-source Data Tools will provide a way out ..</td>
    <td>cloud-services</td>
    <td>1</td>
    <td>7</td>
    <td>Why is the service mesh a critical component o…</td>
    <td>cloud-services</td>
    <td>0</td>
  </tr>
  <tr>
    <td>8</td>
    <td>ECS Deep Dive, why does this container orchest...</td>
    <td>cloud-services</td>
    <td>1</td>
    <td>8</td>
    <td>Eight Critical Success Factors for Cloud Migra…</td>
    <td>cloud-services</td>
    <td>0</td>
  </tr>
  <tr>
    <td>9</td>
    <td>🏈 Happy Super Bowl! Share The Winning Team in ..</td>
    <td>cloud-services</td>
    <td>1</td>
    <td>9</td>
    <td>Cloud Computing ☁️☁️</td>
    <td>cloud-services</td>
    <td>0</td>
  </tr>
  <tr>
    <td>10</td>
    <td>Securing the Cloud: Best Practices for Protect...</td>
    <td>cloud-services</td>
    <td>1</td>
    <td>10</td>
    <td>ZenDesk vs. Salesforce Service Cloud: A Compar…</td>
    <td>cloud-services</td>
    <td>0</td>
  </tr>
</tbody>
</table>

### Kesimpulan
---


### Referensi
---
[1] [SISTEM REKOMENDASI BUKU MENGGUNAKAN METODE ITEM-BASED COLLABORATIVE FILTERING](https://ejournal.undip.ac.id/index.php/jmasif/article/view/31482) 

[2] [SISTEM REKOMENDASI ARTIKEL BERITA MENGGUNAKAN METODE K-NEAREST NEIGHBOR BERBASIS WEBSITE](http://repository.unmuhjember.ac.id/654/1/journal.pdf)

[3] [Theoretical and Applied Aspects of Bank Credit Risks Minimization](https://ieeexplore.ieee.org/document/9468056)

[4] [KLASIFIKASI KELOMPOK UMUR MANUSIA BERDASARKAN ANALISIS DIMENSIFRAKTAL BOX COUNTING DARI CITRA WAJAH DENGAN DETEKSI TEPI CANNY](https://ejournal.unesa.ac.id/index.php/mathunesa/article/view/19398/17715)

