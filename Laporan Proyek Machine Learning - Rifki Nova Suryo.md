# Laporan Proyek Machine Learning - Rifki Nova Suryo

## Domain Proyek

Kesadaran tentang kesehatan mental pada generasi muda belakangan sangat meningkat. Namun, hal ini sering dianggap remeh oleh generasi sebelumnya. Padahal sadar akan kesehatan mental sangat diperlukan terlebih pada kalangan mahasiswa yang memiliki beban yang besar dari sisi akademik maupun non akademik. Dilansir pada halaman suara.com yang berjudul 'Alert! Depresi di kalangan mahasiswa Meroket 135 persen' mengakatan para peneliti di Universitas Boston menemukan bahwa tingkat depresi dikalangan mahasiswa menungkat hampir 135% selama delapan tahun, sementara kecemasan melonjak 110%. Selain itu, laman tersebut mengutip perkataan salah satu  profesor kebijakan kesehatan di Sekolah Kesehatan Masyarakat Universitas Boston dan penulis utama studi, yaitu Sarah Lipson di NY post yang mengatakan bahwa faktor linkungan baru dan jauh dari rumah seringkali dapat menciptakan stress. Studi yang diterbitkan dalam Journal of Affective Disorders pada bulan Juni, menganalisis data dari lebih dari 350.000 siswa di 373 kampus, yang dikumpulkan oleh Health Minds Network antara 2013 hingga 2021. Masalah ini perlu diatasi agar mahasiswa tidak terganggu dalam hal akademik maupun kehidupan sosialnya. Dengan mengetahui gejala-gejala mengenai depresi dan sosialisasi untuk datang ke psikolog dapat mengatasi masalah ini.

## Business Understanding

### Problem Statements
- Apa saja faktor yang mempengaruhi depresi pada mahasiswa?
- Bisakah melakukan prediksi sedini mungkin berdasarkan faktor yang mempengaruhi depresi menggunakan model machine learning?

### Goals
- Mengetahui faktor-faktor yang mempengaruhi depresi pada mahasiswa 
- Mengetahui apakah model machine learning dapat memprediksi orang terkena depresi berdasarkan faktor yang mempengaruhi depresi

### Solution statements
- Menggunakan 3 algoritma machine learning yaitu Logistic Regression, Gradient Boosting Classifier dan Decision Tree Classifier untuk melatih model.
- Menggunakan accuracy sebagai metrik evaluasi untuk menilai model telah belajar dengan baik atau tidak terhadap dataset yang digunakan.

## Data Understanding
Dataset ini mengumpulkan berbagai informasi yang bertujuan untuk memahami, menganalisis, dan memprediksi tingkat depresi di kalangan mahasiswa. Dataset ini dirancang untuk penelitian di bidang psikologi, ilmu data, dan pendidikan, memberikan wawasan tentang faktor-faktor yang berkontribusi terhadap tantangan kesehatan mental siswa dan membantu dalam perancangan strategi intervensi dini. Dataset ini berasal dari OpenML.org dan diposting ulang pada https://www.kaggle.com/datasets/adilshamim8/student-depression-dataset. 

Dataset ini berjumlah 27901 data yang terdiri dari 18 kolom dan 27901 baris dengan 3 tipw data int64(2),float64(7) dan object(9) . Dataset ini tidak memiliki missing value, data duplikat dan outlier tetapi persebaran dataset tidak terdistribusi normal.

### Variabel-variabel pada student depression dataset adalah sebagai berikut:
- id : nomer unik yang diberikan pada setiap record mahasiswa pada dataset
- Gender : Hal ini membantu dalam menganalisis tren spesifik gender dalam kesehatan mental.
- Age : Usia mahasiswa dalam tahun
- City : Wilayah tempat tinggal para mahasiswa
- Profession  : Bidang pekerjaan atau studi siswa, yang dapat memberikan wawasan tentang faktor stres pekerjaan atau akademik.
- Academic Pressure : Ukuran yang menunjukkan tingkat tekanan yang dihadapi siswa dalam lingkungan akademik. Hal ini dapat mencakup tekanan dari ujian, tugas, dan ekspektasi akademik secara keseluruhan.
- Work Pressure  : Ukuran tekanan yang terkait dengan pekerjaan atau tanggung jawab pekerjaan, relevan untuk siswa yang bekerja di samping studi mereka.
- Study Satisfaction : Indikator seberapa puas mahasiswa terhadap studi mereka, yang dapat berkorelasi dengan kesejahteraan mental.
- Work Satisfaction : Ukuran kepuasan siswa dengan pekerjaan atau lingkungan kerja mereka, jika ada.
- Sleep Duration : Jumlah rata-rata jam tidur siswa per hari, yang merupakan faktor penting dalam kesehatan mental.
- Dietary Habits  : Penilaian terhadap pola makan dan kebiasaan nutrisi siswa, yang berpotensi memengaruhi kesehatan dan suasana hati secara keseluruhan.
- Have you ever had suicidal thoughts? : Indikator biner (Ya/Tidak) yang mencerminkan apakah siswa pernah mengalami keinginan untuk bunuh diri
- Work/Study Hours : Jumlah rata-rata jam per hari yang didedikasikan mahasiswa untuk bekerja atau belajar, yang dapat mempengaruhi tingkat stres.
- Family History of Mental Illness : Menunjukkan apakah ada riwayat penyakit mental dalam keluarga (Ya/Tidak), yang dapat menjadi faktor penting dalam kecenderungan kesehatan mental.
- Depression : Variabel target yang mengindikasikan apakah siswa mengalami depresi (Ya/Tidak). Ini adalah fokus utama dari analisis.
- Stres Finansial : Ukuran stres yang dialami karena masalah keuangan, yang dapat mempengaruhi kesehatan mental.
- Degree : Gelar atau program akademis yang sedang ditempuh siswa.
- CGPA: Indeks Prestasi Kumulatif (IPK) kumulatif mahasiswa, yang mencerminkan kinerja akademik secara keseluruhan.

Untuk memahami dataset yang digunakan ada beberapa hal yang dilakukan seperti:
- Melakukan analisis statistik menggunakan describe() dan membuat visualisasi untuk melihat persebaran dataset
- Melakukan pengecekan format data menggunakan info() dan melakukan pengecekan kehilangan data pada dataset menggunakan isna() dan sum()
- Melakukan visualisasi menggunakan diagram batang untuk melihat distribusi data depresi dan tidak depresi
- Melakukan visualisasi menggunakan diagram batang untuk melihat distribusi data depresi dan tidak depresi pada jenis kelamin
- Melakukan visualisasi korelasi setiap kolom untuk melihat korelasi antar kolom.

## Data Preparation
Karen pada dataset ini tidak ada data yang hilang dan format data pada tiap kolom sudah sesuai maka dataset tidak perlu melakukan data cleaning. Tetapi pada dataset ini melakukan beberapa data preparation seperti:
- Data Splitting atau pemisahan data : pada tahap ini dataset dipecah atau dibagi menjadi data train dan data test yang bertujuan agar model dapat belajar dengan baik tidak hanya pada data yang dilatih saja tetapi juga belajar dengan baik pada data test yang belum pernah dilatih.
- Transformation Data atau transformasi data : pada tahap ini dataset yang bertipe teks seperti pada kolom gender, city dll dirubah menjadi angka atau numerik. Selain itu, dataset juga dilakukan standarisasi menggunakan StandardScaler() untuk menyamakan skala fitur-fitur kolom numerik pada dataset. Alasan dilakukan transformasi data ini adalah agar machine learning bisa belajar dengan lebih efektif jika skala pada data sama rata dan tidak semua algoritma machine learning dapat belajar pada data teks atau kategorikal sehingga dilakukan perubahan pada data teks ke numerik.

## Modeling
Pada proyek ini menggunakan 3 algoritma machine learning yaitu Logistic Regression, Gradient Boosting Classifier dan Decision Tree Classifier. Ketiga algoritma yang digunakan pada proyek ini menggunakan parameter default atau paramater yang bawaan dari ketiga algoritma tersebut. 
Decision Tree Classifier adalah algoritma machine learning yang memiliki struktur yang mirip dengan bentuk pohon dengan setiap cabang mewakili keputusan atau percabangan dari data berdasarkan fitur-fitur yang ada. Kelebihan algortima ini adalah model yang dibuat mudah dipahami dan ditafsirkan, dapat menangani data ketegorikal dan numerik, tidak memerlukan skala fitur, dan fleksibel. Algoritma ini juga memiliki kekurangan sensitif pada noise, jika tak dikendalikan pohon akan terlalu besar sehingga menjadi rumit, pohon tidak stabil. 

Logistic Regression adalah salah satu teknik pemodelan statistik yang digunakan untuk memprediksi hasil biner. Berbeda dengan regresi linear yang digunakan untuk memprediksi nilai numerik, regresi logistik digunakan untuk memodelkan probabilitas bahwa suatu kejadian akan terjadi (hasil biner). Kelebihan algoritma ini adalah model yang dibuat sederhana dan proses training cepat, karena hasilnya adalah probabilitas maka mudah untuk diinterpretasi. Algoritma ini juga memiliki kelemahan meski sering digunakan untuk melakukan klasifikasi tetapi algoritma ini tentu saja berbasis linear sehingga kurang efektif untuk data yang tidak linear, sensitif terhadap fitur yang memiliki korelasi lebih dari dua fitur atau biasa disebut multikolinearitas. 

Gradient Boosting Classifier algoritma ensemble berbasis pohon keputusan yang membangun model secara bertahap, di mana setiap model baru memperbaiki kesalahan dari model sebelumnya. Kelebihan dari algoritma ini adalah memiliki nilai akurasi yang tinggi, dapat menangani data non-linear, toleran terhadap outlier dan dapat menangani berbagai tipe data. Algoritma ini juga memiliki kelemahan yaitu proses training lambat, dan susah diinterpreyasikan. 

Berdasarkan hasil yang didapatkan model Gradient Boosting Classifier dapat menjadi solusi karena memiliki accuracy sebesar 84% yang lebih besar sedikit dari model Logistic Regression yang memilki accuracy sebesar 83% dan lebih besar dari model Descision Tree Classifier yang memiliki accuracy 76%.

## Evaluation
Karena pada proyek ini melakukan klasifikasi maka matriks evaluasi yang digunakan adalah akurasi. Akurasi adalah metrik yang paling sederhana dan sering digunakan untuk mengukur kinerja model klasifikasi. Akurasi dihitung sebagai proporsi dari prediksi benar (baik positif maupun negatif) terhadap seluruh prediksi yang dilakukan oleh model. Cara kerja dari akurasi ini adalah jika model yang sudah dibuat lalu melakukan prediksi sebanyak 100 prediksi dan 90 yang benar, akurasi model itu akan 90%. Jika dituliskan dengan rumus akan sebagai berikut:
Accuracy = (TP + TN)/(TP + TN + FP + FN)
dimana 
TP (True Positives): model benar memprediksi orang depresi.
TN (True Negatives): model benar memprediksi orang tidak depresi sebagai orang tidak depresi.
FP (False Positives): model salah memprediksi orang yang tidak depresi menjadi depresi dan sebaliknya.
FN (False Negatives): model salah memprediksi orang yang depresi menjadi tidak depresi.
​
Berdasarkan hasil akurasi pada ketiga algoritma machine learning yang digunakan di mana Logistic Regression memiliki akurasi sebesar 83%, Gradient Boosting 84% dan Decision Tree 76%. Model Gradient Boosting dapat dikatakan terbaik antara kedua algoritma lainnya. 

Setelah melakukan pelatihan model machine learning dapat dikatakan model machine learning dapat belajar dan menangkap pola orang terkena depresi berdasarkan faktor yang mempengaruhi depresi pada mahasiswa ini sangat menjawab problem statment dan goals dari proyek ini. Ketiga model yang digunakan memiliki dampak yang cukup baik pada proyek ini meskipun nilai akurasi model yang didapatkan belum menyentuh 95% tetapi ketiga model sudah belajar cukup baik terhadap dataset yang memberi gambaran bawha machine learning dapat membuat prediksi orang yang terkena depresi pada mahasiswa berdasarkan faktor-faktor yang mempengaruhinya meskipun perlu analisa lebih lanjut oleh psikolog.


