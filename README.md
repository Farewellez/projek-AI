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

### Konfigurasi dan Inisialisasi MediaPipe Hands
Pada bagian kode ini

```
# --- Fix TQDM (Optional Progress Bar) ---
try:
    from tqdm import tqdm
except ImportError:
    # Jika tqdm tidak ada, buat fungsi dummy agar tidak error
    def tqdm(iterable):
        return iterable

# --- Konfigurasi ---
DATA_DIR = "data/asl_alphabet_train"
MODEL_PATH = "models/rf_model.pkl" # Nama file tetap rf_model.pkl biar backend gak perlu ubah config

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
```

jadi library tqdm ini digunakan untuk menampilkan progress bar atau loading bar agar lebih terlihat visual atau UI nya meskipun versi CLI. Dalam projek ini digunakan agar gambar yang diproses itu terlihat sudah sampai berapa gambar semisal 500/1500 gambar dan kurang berapa persen lagi. try except disini untuk error handling jika library tqdm belum terinstall yaitu dengan membuat dummy function yang me return parameternya yang merupakan iterable asli tanpa menampilkan progress bar.

DATA_DIR digunakan untuk menyimpan path ke data data train yang berisi 29 class dengan tiap class ada 3000 image file. Hasil dari training nanti akan menghasilkan sebuah model yang merupakan objek python dan dikemas menggunakan pickle library dengan output file rf_model.pkl yang awalnya berniat menggunakan random forest algorithm namun diganti SVM.

setelah dapat datanya selanjutnya perlu untuk membaca gestur tangan berdasarkan 21 titik landmark dari standard library mediapipe. modul spesifik dari mediapipe untuk detektor tangan yaitu solutions.hands yang digunakan untuk detektor bahasa tangan nanti. parameter dari objek detektor tangan ada 3:
> static_image_mode=True memberitahu mediapipe bahwa gambar yang diinput adalah gambar statis bukan video

> max_num_hands=1 digunakan untuk deteksi hanya satu tangan karena itu sudah cukup untuk ASL

> min_detection_confidence=0.5 disini semakin tinggi persentasenya maka semakin stricth terhadap noise karena hanya mempertimbangkan deteksi tangan kalau condifence nya diatas 50% atau 0.5

### Fungsi normalize_landmarks
Fungsi ini yang digunakan untuk proses ekstraksi fitur. pre-processing sendiri disini adalah proses dimana data mentah (gambar tangan) akan diolah hingga menjadi dalam bentuk angka yang lebih bersih, efisien dan dapat dimengerti untuk diolah oleh algoritma Machine Learning. Tujuan akhir dari proses ini yaitu data yang hanya berisi informasi relevan yaitu tentang 21 titik landmarks tadi atau 63 angka kordinat tadi tanpa adanya noise.

Bafian kodenya ada di sini

```
def normalize_landmarks(landmarks):
    """
    Mengubah koordinat menjadi relatif terhadap wrist DAN invarian terhadap skala (jarak).
    Sama persis dengan logika Backend.
    """
    points = np.array([[p.x, p.y, p.z] for p in landmarks])
    base = points[0]
    points = points - base
    max_value = np.max(np.abs(points))
    if max_value > 0:
        points = points / max_value
    return points.flatten().tolist()
```

jadi variable points itu akan menerima nilai dari 21 landmarks dari nilai parameter dan mengubahnya menjadi array NumPy berdimensi (21, 3). base variable akan mengambil element array NumPy yang pertama yaitu titik kordinat dari landmark bagian wrist atau pergelangan tangan. setelah mendapat base point maka lanjut membuat semua landmark relatif dengan base point dengan cara mengurangi semua landmark dengan kordinat dari base point. untuk menjaga agar semua fitur berada pada rentang -1.0 dan 1.0 maka perlu normalisasi atau penskalaan dengan membagi semua kordinat relatif dengan absolut maximum. terakhir function akan me return nilai points atau titik-titik landmark dengan flatten method yang mengubah array (21,3) tadi menjadi satu list panjang (63 fitur)

