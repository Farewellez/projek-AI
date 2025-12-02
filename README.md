# How it works?
## 1. train.py
Script ini berisi code yang digunakan untuk memproses model yang ada agar dapat menghandle gambar-gambar yang ada.

### import library
```
import os
import cv2
import mediapipe as mp
import numpy as np
import pickle
import warnings
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
```

- os disini untuk menjelajahi folder dataset yang ada agar bisa membaca file dan juga nama kelas

- cv2  digunakan untuk memproses vision dan membaca citra dari sebuah gambar ASL (American Sign Language) yaitu komunitas tuli dan keras pendengaran di Amerika Serikat

- mediapipe berfungsi untuk mengambil 21 titik landmark yang ada di tangan. 21 titik Landmark sendiri merupakan standard dari mediapipe yang dimana merupakan titik-titik yang signifikan dan unik yang ada pada struktur tangan manusia. untuk informasi 21 landmark sendiri adalah sebagai berikut:

```
1. Wrist (pergelangan tangan)
2. Thumb CMC (sendi dasar ibu jari)
3. Thumb MCP (sendi tengah ibu jari)
4. Thumb IP (sendi atas ibu jari)
5. Thumb TIP (ujung ibu jari)
6. Index Finger MCP (sendi dasar jari telunjuk)
7. Index Finger PIP (sendi tengah jari telunjuk)
8. Index Finger DIP (sendi atas jari telunjuk)
9. Index Finger TIP (ujung jari telunjuk)
10. Middle Finger MCP (sendi dasar jari tengah)
11. Middle Finger PIP (sendi tengah jari tengah)
12. Middle Finger DIP (sendi atas jari tengah)
13. Middle Finger TIP (ujung jari tengah)
14. Ring Finger MCP (sendi dasar jari manis)
15. Ring Finger PIP (sendi tengah jari manis)
16. Ring Finger DIP (sendi atas jari manis)
17. Ring Finger TIP (ujung jari manis)
18. Pinky MCP (sendi dasar jari kelingking)
19. Pinky PIP (sendi tengah jari kelingking)
20. Pinky DIP (sendi atas jari kelingking)
21. Pinky TIP (ujung jari kelingking)
``` 
- numpy sendiri merupakan library yang biasanya digunakan untuk operasi dalam sebuah proses matematik yang disini nantinya akan digunakan untuk mengubah landmark yang didapat tadi menjadi array numerik yang bakal masuk di model. caranya sendiri adalah mengambil titik kordinat yang didapat dari mediapipe. titik kordinat 21 landmark yang dideteksi mediapipe terdiri dari titik x, y  dan z sehingga total ada 21 x 3 = 63 angka. numpy mengubahnya menjadi array yang awalnya 3D yang kemudian di normalisasikan menjadi 1D list. dari sini SVM lah yang nantinya mempelajari pola dari 63 angka ini

- pickle digunakan untuk menyimpan model hasil training ke file .pkl. hasil training akan menjadi sebuah model dengan bentuk objek python. isinya mirip json (list array) namun json sendiri tidak bisa menyimpan objek machine learning jadi pickle lebih cocok

- sklearn.svm.SVC disini digunakan untuk mengambil class svm dari sklearn yang merupakan library machine learning. SVC sendiri merupakan klasifikasi dari SVM. (Untuk SVM dan algoritma lain akan dibahas lebih lanjut)

- train_test_split ini digunakan untuk pembagian data. semisal ada 80% data yang dilatih namun 20% tetap digunakan untuk diuji. pembagian acak namun tetap seimbang.

- accuracy_score untuk mempermudah perhitungan tingkat keakuratan prediksi data yaitu data_benar/seluruh_data

- LabelEncoder untuk melakukan encoding pada label huruf pada dataset karena model machine learning hanya mengerti angka

###