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

def load_dataset():
    X, y_raw = [], []
    
    if not os.path.exists(DATA_DIR):
        # Coba cek path alternatif jika user menjalankan dari folder berbeda
        if os.path.exists(f"../{DATA_DIR}"):
            DATA_DIR_FIX = f"../{DATA_DIR}"
        else:
            print(f"Error: Folder {DATA_DIR} tidak ditemukan!")
            return [], [], None
    else:
        DATA_DIR_FIX = DATA_DIR

    labels = sorted([f for f in os.listdir(DATA_DIR_FIX) if os.path.isdir(os.path.join(DATA_DIR_FIX, f))])
    print(f"Detected {len(labels)} classes.")

    for label in labels:
        folder_path = os.path.join(DATA_DIR_FIX, label)
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # --- PENTING: LIMIT DATA UNTUK SVM ---
        # SVM sangat lambat (O(n^2)) jika datanya terlalu banyak.
        # Kita ambil 1500 sampel per kelas agar keburu deadline (total ~43k data).
        # Random Forest tadi kuat makan semua data, tapi SVM butuh diet.
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