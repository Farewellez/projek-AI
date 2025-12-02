import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import threading
import time
import cv2
import pickle
import mediapipe as mp
import numpy as np
import cairo
import os
from collections import Counter

# --- Konfigurasi ---
ASSETS_DIR = "assets"
MODEL_PATH = "models/rf_model.pkl"
CONF_THRESHOLD = 0.75       # Ambang batas confidence
PRED_SMOOTH_FRAMES = 5      # Buffer smoothing (voting)
INTERVAL_TIME = 0.8         # Jeda waktu antar input huruf (detik)

# --- Helper: Normalisasi Fitur (SINKRON dengan train.py) ---
def normalize_landmarks(landmarks):
    """
    Mengubah koordinat menjadi relatif terhadap wrist DAN invarian terhadap skala.
    Hapus logika mirroring agar konsisten dengan training data.
    """
    points = np.array([[p.x, p.y, p.z] for p in landmarks])
    
    # Translasi ke Wrist (0,0,0)
    base = points[0]
    points = points - base
    
    # Scaling (Scale Invariant)
    max_value = np.max(np.abs(points))
    if max_value > 0:
        points = points / max_value

    return points.flatten().tolist()

# ---------------- Camera Worker ----------------
class CameraWorker(threading.Thread):
    def __init__(self, label_video, label_pred, label_sentence, model_path=MODEL_PATH, cam_index=0):
        super().__init__()
        self.label_video = label_video
        self.label_pred = label_pred
        self.label_sentence = label_sentence
        self.model_path = model_path
        self.cam_index = cam_index
        self._running = False
        self.sentence = ""
        
        # Buffer untuk smoothing prediksi
        self.pred_buffer = [] 

        # --- Load Model & Encoder ---
        try:
            with open(model_path, "rb") as f:
                data = pickle.load(f)
            
            if isinstance(data, dict):
                self.model = data['model']
                self.encoder = data.get('encoder')
                self.classes = data.get('classes', [])
            else:
                self.model = data
                self.encoder = None
                self.classes = []
                print("Warning: Format model lama.")
        except Exception as e:
            print(f"Critical Error loading model: {e}")
            self.model = None

        # --- Inisialisasi MediaPipe ---
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.last_time = time.time()

    def run(self):
        self.cap = cv2.VideoCapture(self.cam_index)
        self._running = True
        
        while self._running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            # Flip horizontal (Mirror effect untuk UX)
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            current_pred = "-"
            current_conf = 0.0
            
            # --- Proses Inference ---
            if results.multi_hand_landmarks and self.model:
                lm = results.multi_hand_landmarks[0]
                self.mp_draw.draw_landmarks(frame, lm, self.mp_hands.HAND_CONNECTIONS)

                try:
                    # 1. Pipeline Normalisasi
                    features = normalize_landmarks(lm.landmark)
                    feats_np = np.array(features, dtype=np.float32).reshape(1, -1)

                    # 2. Prediksi
                    if hasattr(self.model, "predict_proba"):
                        proba = self.model.predict_proba(feats_np)[0]
                        idx = np.argmax(proba)
                        current_conf = float(proba[idx])
                        
                        if self.encoder:
                            current_pred = self.encoder.inverse_transform([idx])[0]
                        elif len(self.classes) > 0:
                            current_pred = self.classes[idx]
                        else:
                            current_pred = str(idx)
                    else:
                        current_pred = str(self.model.predict(feats_np)[0])
                        current_conf = 1.0

                except Exception as e:
                    print(f"Inference Error: {e}")

                # --- 3. Logic Smoothing & Timer ---
                # Hanya masukkan ke buffer jika confidence cukup tinggi
                if current_conf >= CONF_THRESHOLD:
                    self.pred_buffer.append(current_pred)
                
                # Jaga ukuran buffer tetap kecil (sliding window)
                if len(self.pred_buffer) > PRED_SMOOTH_FRAMES:
                    self.pred_buffer.pop(0)

                # Cek apakah buffer sudah penuh untuk voting
                if len(self.pred_buffer) == PRED_SMOOTH_FRAMES:
                    # Ambil huruf yang paling sering muncul di buffer (Voting)
                    most_common, count = Counter(self.pred_buffer).most_common(1)[0]
                    
                    # Jika konsisten (mayoritas frame setuju)
                    if count >= (PRED_SMOOTH_FRAMES - 1): 
                        # Cek Timer
                        if (time.time() - self.last_time) >= INTERVAL_TIME:
                            self.process_input(most_common)
                            self.last_time = time.time()
                            self.pred_buffer = [] # Reset buffer setelah input

            else:
                # Jika tangan hilang, reset buffer agar tidak ada input "sisa"
                self.pred_buffer = []

            # --- Update UI ---
            self.update_ui_frame(frame, current_pred, current_conf)
            time.sleep(0.01)

        self.cap.release()
        self.hands.close()

    def process_input(self, char):
        """Menangani logika spasi, delete, dan huruf"""
        if char == "space": 
            self.sentence += " "
        elif char == "del":
            self.sentence = self.sentence[:-1]
        elif char == "nothing":
            pass # Abaikan kelas 'nothing' jika ada
        else:
            self.sentence += char
        
        # Update text UI di main thread melalui variabel yang di-bind (opsional) atau update langsung
        # Di sini kita update di loop UI nanti, tapi variable self.sentence sudah terupdate

    def update_ui_frame(self, frame, pred, conf):
        # Resize video agar pas di UI
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb).resize((800, 600))
        imgtk = ImageTk.PhotoImage(image=img_pil)
        
        # Update Video Label
        self.label_video.imgtk = imgtk
        self.label_video.config(image=imgtk)

        # Update Prediction Label (Cairo)
        cairo_img = render_cairo(pred, f"Conf: {conf:.2f}")
        cairo_img = cairo_img.resize((360, 220))
        pred_tk = ImageTk.PhotoImage(cairo_img)
        self.label_pred.imgtk = pred_tk
        self.label_pred.config(image=pred_tk)

        # Update Sentence Label
        self.label_sentence.config(text="Sentence: " + self.sentence)

    def stop(self):
        self._running = False

    def reset_sentence(self):
        self.sentence = ""
        self.pred_buffer = []
        self.last_time = time.time()

# ---------------- Cairo Renderer ----------------
def render_cairo(main_text, sub_text=None, width=360, height=220):
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    ctx = cairo.Context(surf)
    ctx.set_source_rgba(0,0,0,0) 
    ctx.paint()

    # Font Style
    ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    main_size = int(height*0.5)
    ctx.set_font_size(main_size)
    
    # Shadow
    ctx.set_source_rgba(0,0,0,0.5)
    ctx.move_to(23, main_size+13)
    ctx.show_text(str(main_text))
    
    # Text Color Logic
    is_high_conf = False
    if "Conf" in str(sub_text):
        try:
            if float(sub_text.split(":")[1]) >= CONF_THRESHOLD:
                is_high_conf = True
        except: pass

    if is_high_conf:
        ctx.set_source_rgba(0.2, 1, 0.4, 1) # Hijau jika yakin
    else:
        ctx.set_source_rgba(1, 1, 1, 1)     # Putih jika ragu

    ctx.move_to(20, main_size+10)
    ctx.show_text(str(main_text))

    # Subtext (Confidence)
    if sub_text:
        ctx.set_font_size(int(main_size*0.25))
        ctx.move_to(20, main_size + int(main_size*0.45))
        ctx.set_source_rgba(1,1,1,0.7)
        ctx.show_text(str(sub_text))

    buf = surf.get_data()
    arr = np.frombuffer(buf, dtype=np.uint8).reshape((height,width,4))
    img = Image.fromarray(arr, 'RGBA')
    # Fix channel swap
    b,g,r,a = img.split()
    return Image.merge("RGBA", (r,g,b,a))

# ---------------- Main UI Class ----------------
class ASLApp:
    def __init__(self, root):
        self.root = root
        root.title("IsyaratKu - Smart Engine")
        root.geometry("1200x720")
        root.configure(bg="#1e1e2f")

        # --- Layout UI ---
        
        # 1. Video Frame (Kiri)
        self.label_video = tk.Label(root, bg="#000", borderwidth=2, relief="sunken")
        self.label_video.place(x=20, y=20, width=800, height=600)

        # 2. Prediction Panel (Kanan Atas)
        self.label_pred = tk.Label(root, bg="#111", borderwidth=2, relief="raised")
        self.label_pred.place(x=840, y=20, width=340, height=220)

        # 3. Sentence Panel (Kanan Bawah)
        self.label_sentence = tk.Label(
            root, 
            text="Sentence: ", 
            wraplength=320, 
            justify="left", 
            bg="#2a2a40", 
            fg="#00ffcc", 
            font=("Consolas", 14), 
            anchor="nw", 
            padx=10, pady=10,
            borderwidth=1, relief="solid"
        )
        self.label_sentence.place(x=840, y=260, width=340, height=360)

        # 4. Buttons (Bawah Kiri)
        btn_style = {"font": ("Segoe UI", 11, "bold"), "bd": 0, "cursor": "hand2"}
        
        self.btn_reset = tk.Button(root, text="🔄 RESET", command=self.on_reset, bg="#ff4444", fg="white", **btn_style)
        self.btn_reset.place(x=20, y=640, width=120, height=50)

        self.btn_save = tk.Button(root, text="💾 SAVE", command=self.on_save, bg="#4444ff", fg="white", **btn_style)
        self.btn_save.place(x=160, y=640, width=120, height=50)

        self.btn_bg = tk.Button(root, text="🖼️ BG", command=self.change_bg, bg="#555", fg="white", **btn_style)
        self.btn_bg.place(x=300, y=640, width=80, height=50)

        # Start System
        self.worker = CameraWorker(self.label_video, self.label_pred, self.label_sentence)
        self.worker.start()

    def on_reset(self):
        self.worker.reset_sentence()
        self.label_sentence.config(text="Sentence: ")

    def on_save(self):
        text = self.worker.sentence
        if not text: return
        path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files","*.txt")])
        if path:
            with open(path,"w",encoding="utf-8") as f:
                f.write(text)

    def change_bg(self):
        path = filedialog.askopenfilename(filetypes=[("Images","*.png *.jpg *.jpeg")])
        if not path: return
        try:
            bg_img = Image.open(path).resize((1200,720))
            bg_tk = ImageTk.PhotoImage(bg_img)
            
            # Buat label background paling belakang
            bg_label = tk.Label(self.root, image=bg_tk)
            bg_label.image = bg_tk
            bg_label.place(x=0, y=0, relwidth=1, relheight=1)
            bg_label.lower() # Kirim ke layer paling bawah
        except Exception as e:
            print(f"Failed to load BG: {e}")

    def close(self):
        self.worker.stop()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = ASLApp(root)
    root.protocol("WM_DELETE_WINDOW", app.close)
    root.mainloop()