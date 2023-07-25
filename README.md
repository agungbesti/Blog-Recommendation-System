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

###### Gambar 2: Dataset Blog
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


![image](https://github.com/agungbesti/Blog-Recommendation-System/assets/35904444/d57e03ea-89ec-401b-ae1e-75930092c2bf)
###### Gambar 3: Univariate Analysis pada variabel Topic
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

![image](https://github.com/agungbesti/Blog-Recommendation-System/assets/35904444/bf909ecc-5f99-4d16-a050-e100dd2a0ec1)
###### Gambar 4: Univariate Analysis pada variabel Ratings
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

## Content-Based Filtering
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

### Menggunakan TF-IDF Vectorizer to Vectorize pada blog topic
TF-IDF, kependekan dari Term Frequency-Inverse Document Frequency, adalah teknik yang banyak digunakan dalam pemrosesan bahasa alami dan pengambilan informasi untuk mengukur pentingnya suatu istilah dalam dokumen dalam kumpulan dokumen. TF-IDF menggabungkan dua faktor: **Term Frequency (TF) dan Inverse Document Frequency (IDF)**.
- **Term Frequency (TF)**: TF mengukur frekuensi istilah dalam dokumen. Ini menghitung berapa kali suatu istilah muncul dalam dokumen dan mewakilinya sebagai hitungan mentah atau nilai yang dinormalisasi. Alasan di balik TF adalah bahwa istilah yang lebih sering muncul dalam dokumen cenderung lebih penting atau relevan dengan dokumen tersebut.
- **Inverse Document Frequency (IDF)**: IDF mengukur signifikansi suatu istilah di seluruh kumpulan dokumen. Ini menghitung logaritma fraksi terbalik dari jumlah dokumen yang mengandung istilah tersebut. Ide di balik IDF adalah bahwa istilah yang muncul di sejumlah kecil dokumen lebih informatif dan berharga daripada istilah yang muncul di sejumlah besar dokumen.

Perhitungan TF-IDF dilakukan dengan mengalikan nilai TF dan IDF secara bersamaan. Skor yang dihasilkan merepresentasikan pentingnya suatu istilah dalam dokumen dalam konteks keseluruhan kumpulan dokumen. Skor yang lebih tinggi menunjukkan bahwa suatu istilah lebih relevan atau berbeda dengan dokumen tertentu. 
Rumus perhitungan TF-IDF untuk term (t) dalam dokumen (d) dalam kumpulan dokumen adalah sebagai berikut:
![image](https://github.com/agungbesti/Blog-Recommendation-System/assets/35904444/aff977e6-a9d8-4104-9686-9454910d17a6)
###### Gambar 5: Perhitungan TF-IDF

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

### Menggunakan Cosine Similarity
Cosine Similarity adalah ukuran yang digunakan untuk menentukan kesamaan antara dua vektor dalam ruang multidimensi. Ini menghitung cosinus sudut antara vektor, yang menunjukkan seberapa dekat hubungan vektor dalam hal orientasi dan arahnya. Dalam cosine similarity, kemiripan antara dua vektor diukur berdasarkan sudut antara vektor-vektor tersebut. Nilai cosine similarity berkisar antara -1 hingga 1, di mana nilai 1 menunjukkan kedua vektor memiliki arah yang sama atau sangat mirip, nilai 0 menunjukkan tidak ada kemiripan, dan nilai -1 menunjukkan arah yang berlawanan atau sangat berbeda.

Berikut adalah rumus untuk menghitung Cosinus Similarity:
![image](https://github.com/agungbesti/Blog-Recommendation-System/assets/35904444/9510067a-5fc3-4e86-a6f7-7e1a65c9330a)
###### Gambar 6: Rumus Cosine Similarity
Di mana:
- A dan B adalah dua vektor yang akan dibandingkan.
- A . B adalah hasil perkalian dot (inner product) antara vektor A dan B.
- ||A|| dan ||B|| adalah panjang (magnitude) dari vektor A dan B.

![image](https://github.com/agungbesti/Blog-Recommendation-System/assets/35904444/ddf22025-9e03-420f-ab67-54b072926a39)
 <p align="center">
 <img src="https://github.com/agungbesti/Blog-Recommendation-System/assets/35904444/ddf22025-9e03-420f-ab67-54b072926a39" alt="Cosine Similarity Concept">
 Gambar 9. Grafik jarak antar 2 vektor di <i>Cosine Similarity</i> 
 </p>

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

