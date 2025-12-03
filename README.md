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

### Fungsi load_dataset
Fungsi ini yang akan mengumpulkan semua gambar, eksekusi ekstraksi fitur menggunakan mediapipe dan normalize_landmarks function sebelumnya dan beberapa batasan untuk model svm. kodenya seperti ini

```
def load_dataset():
    X, y_raw = [], []
    
    # ... (Pengecekan path data) ...

    labels = sorted([f for f in os.listdir(DATA_DIR_FIX) if os.path.isdir(os.path.join(DATA_DIR_FIX, f))])
    print(f"Detected {len(labels)} classes.")

    for label in labels:
        folder_path = os.path.join(DATA_DIR_FIX, label)
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # --- PENTING: LIMIT DATA UNTUK SVM ---def load_dataset():
    X, y_raw = [], []
    
    # ... (Pengecekan path data) ...

    labels = sorted([f for f in os.listdir(DATA_DIR_FIX) if os.path.isdir(os.path.join(DATA_DIR_FIX, f))])
    print(f"Detected {len(labels)} classes.")

    for label in labels:
        folder_path = os.path.join(DATA_DIR_FIX, label)
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # --- PENTING: LIMIT DATA UNTUK SVM ---
        # SVM sangat lambat (O(n^2)) jika datanya terlalu banyak.
        # Kita ambil 1500 sampel per kelas agar keburu deadline (total ~43k data).
        files = files[:1500] 

        print(f"Processing '{label}' ({len(files)} images)...")

        for img_name in tqdm(files):
            img_path = os.path.join(folder_path, img_name)
            try:
                img = cv2.imread(img_path)
                if img is None: continue

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                result = hands.process(img_rgb)

                if result.multi_hand_landmarks:
                    lm = result.multi_hand_landmarks[0]
                    features = normalize_landmarks(lm.landmark)
                    X.append(features)
                    y_raw.append(label)
            except Exception:
                continue

    return np.array(X), np.array(y_raw)
```

yang terjadi disini di awal adalah mempersiapkan wadah untuk variabel X yang menampung fitur dan variabel y_raw sebagai label mentah. skrip ini akan membaca semua subfolder setelah itu tiap kelas akan di looping untuk dibaca. files variable akan menggunakan 1500/3000 data saja untuk di training. tiap image files nantinya akan dibaca oleh cv2 (img = cv2.imread(img_path)) kemudian image color akan diubah warnanya agar mengurangi noise dan MediaPipe bekerja lebih optimal. jika mediapipe berhasil mendeteksi tangan di gambar maka landmark yang berada dalam list python yang berisi objek landmarks (multi_hand_landmarks) maka landmarks akan di tambpung di variable lm dan variable feature akan menampung nilai return dari landmark yang telah di normalisasi. function ini kemudian akan mengembalikkan array X untuk fitur dan array y_raw untuk label mentah (list of string)

### Eksekusi Utama dan Pelatihan Model
di bagian kode ini model mulai belajar dan diuji.

```
# --- Main Execution ---
if __name__ == "__main__":
    print("--- Memulai Ekstraksi Fitur (IsyaratKu Engine - SVM Version) ---")
    X, y_raw = load_dataset()

    if len(X) == 0:
        print("Data kosong. Pastikan dataset sudah benar.")
        exit()

    # Encode Labels
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    
    print(f"\nTotal Sampel: {len(X)}")
    print(f"Kelas: {le.classes_}")

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\n--- Training SVM (Support Vector Machine) ---")
    print("WARNING: Proses ini akan memakan waktu 15-30 menit. Mohon bersabar...")
    
    # Inisialisasi SVM
    # kernel='rbf' -> Cocok untuk data non-linear (gerakan tangan kompleks)
    # probability=True -> WAJIB agar backend bisa menampilkan % confidence
    clf = SVC(
        kernel='rbf', 
        probability=True, 
        verbose=True, 
        random_state=42
    )
    
    clf.fit(X_train, y_train)

    # Evaluasi
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {acc:.4f} ({acc*100:.2f}%)")

    # Simpan Model
    # Struktur dictionary disamakan dengan RF agar backend tidak error
    os.makedirs("models", exist_ok=True)
    save_data = {
        'model': clf,
        'encoder': le,
        'classes': le.classes_
    }
    
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(save_data, f)
    
    print(f"\nModel SVM berhasil disimpan ke: {MODEL_PATH}")
    print("Langkah selanjutnya: Copy file ini ke backend/models/rf_model.pkl")
```

jadi pada bagian awal, variable le akan menyimpan nilai dari class labelencoder. le ini yang kemudian akan mengubah y_raw menjadi angka integer yang akan diproses SVM dan disimpan di variable y (label yang sudah di encode dan bersih). setelah itu dilakukan split data untuk menentukan berapa persen data yang diuji dan berapa persen data yang dilatih. disini kita mengambil 20% data untuk diuji dan sisanya untuk dilatih ke model. train disini menggunakan 20% sample karena dianggap seimbang. pada bagian  clf.fit(X_train, y_train) data pelatihan di fetch dan model SVC akan dioptimalkan. penggunaan model SVC yang merupakan implementasi spesifik dari SVM yang dirancang untuk melakukan tugas klasifikasi menjadi alasan pemilihannya.