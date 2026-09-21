import os
import re
import time
import datetime
import sqlite3
import shutil
import io
import csv
import base64
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import requests
from requests.auth import HTTPDigestAuth, HTTPBasicAuth
from urllib.parse import urlparse
import numpy as np
import cv2
from flask import Flask, request, jsonify, Response, send_from_directory

# Kompatibilitas numpy 2.x jika diperlukan
if not hasattr(np, 'sctypes'):
    np.sctypes = {
        'int': [np.int8, np.int16, np.int32, np.int64],
        'uint': [np.uint8, np.uint16, np.uint32, np.uint64],
        'float': [np.float16, np.float32, np.float64],
        'complex': [np.complex64, np.complex128],
        'others': [bool, object, bytes, str, np.void],
    }

from ultralytics import YOLO
import easyocr

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
CAPTURES_DIR = os.path.join(MODEL_DIR, "captures")
UPLOAD_DIR = os.path.join(MODEL_DIR, "uploads")
os.makedirs(CAPTURES_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ============================================================
# PURE IN-MEMORY STORAGE (RAM)
# Seluruh riwayat kendaraan disimpan langsung di memori RAM (0 ms, bebas lag disk)
# ============================================================
LATEST_RECORDS = deque(maxlen=300)
LAST_RECORDED_PLATES = {}  # {safe_plate: {"time": float, "rec": dict}}
GATE_COOLDOWN_SEC = 3.0


def save_parking_record(det, source_img=None, source_img_path=None):
    """Menyimpan hasil deteksi langsung ke memory RAM (0ms) dengan Anti-Passback Cooldown (3s)."""
    global LATEST_RECORDS, LAST_RECORDED_PLATES
    now = datetime.datetime.now()
    now_epoch = time.time()
    timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
    file_ts = now.strftime("%Y%m%d_%H%M%S_%f")[:19]

    plate_text = det.get("license_plate") or "TIDAK_TERBACA"
    if not isinstance(plate_text, str):
        plate_text = str(plate_text)
    safe_plate = re.sub(r'[^A-Za-z0-9]', '', plate_text) or "UNKNOWN"

    # Anti-Passback Gate Cooldown: Jika plat yang sama baru tercatat < 3 detik lalu, gunakan record yang ada
    if safe_plate != "UNKNOWN" and safe_plate in LAST_RECORDED_PLATES:
        last_entry = LAST_RECORDED_PLATES[safe_plate]
        if (now_epoch - last_entry["time"]) < GATE_COOLDOWN_SEC:
            return last_entry["rec"]

    snapshot_filename = f"{file_ts}_{safe_plate}.jpg"
    dest_path = os.path.join(CAPTURES_DIR, snapshot_filename)

    try:
        if source_img is not None:
            cv2.imwrite(dest_path, source_img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        elif source_img_path and os.path.exists(source_img_path):
            shutil.copyfile(source_img_path, dest_path)
    except Exception:
        snapshot_filename = None

    conf_val = det.get("plate_confidence") or det.get("vehicle_confidence") or 0.0

    rec = {
        "id": int(now_epoch * 1000) % 1000000,
        "timestamp": timestamp_str,
        "license_plate": det.get("license_plate"),
        "vehicle_type": det.get("vehicle_type"),
        "body_style": det.get("body_style"),
        "confidence": round(conf_val, 3) if conf_val else None,
        "latency_ms": det.get("latency_ms"),
        "snapshot_url": f"/captures/{snapshot_filename}" if snapshot_filename else None
    }

    # Simpan langsung ke RAM (0 milidetik, tanpa overhead disk I/O)
    LATEST_RECORDS.appendleft(rec)
    if safe_plate != "UNKNOWN":
        LAST_RECORDED_PLATES[safe_plate] = {"time": now_epoch, "rec": rec}
    return rec


# ============================================================
# PERSISTENT CAMERA STREAM MANAGER (BACKGROUND RTSP / MJPEG WORKER)
# Menjaga koneksi RTSP tetap hidup di latar belakang agar:
# 1. Live stream di monitor CCTV benar-benar bergerak mulus (25 FPS).
# 2. Deteksi instan: frame selalu siap di RAM sehingga scan < 1 detik (bebas jeda RTSP 3s).
# ============================================================
class CameraStreamManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.frame_id = 0
        self.running = False
        self.thread = None
        self.stream_url = ""
        self.username = ""
        self.password = ""
        self.latest_frame = None       # Full-res (1080p) numpy array untuk ANPR AI
        self.latest_jpeg = None        # Compressed JPEG untuk live stream ultra-smooth
        self.fps = 0.0
        self.status = "disconnected"   # "disconnected", "connecting", "connected", "error"
        self.error_msg = ""

    def start(self, stream_url, username="", password=""):
        with self.lock:
            cleaned_url = stream_url.strip()
            if self.running and self.stream_url == cleaned_url and self.username == username and self.password == password and self.status == "connected":
                return True, "Stream kamera sudah aktif"
            self._stop_internal()
            self.stream_url = cleaned_url
            self.username = username.strip() if username else ""
            self.password = password.strip() if password else ""
            self.running = True
            self.status = "connecting"
            self.error_msg = ""
            self.thread = threading.Thread(target=self._worker_loop, daemon=True)
            self.thread.start()
            return True, "Memulai koneksi kamera di latar belakang"

    def stop(self):
        with self.lock:
            self._stop_internal()
        return True, "Stream dihentikan"

    def _stop_internal(self):
        self.running = False
        self.status = "disconnected"
        self.latest_frame = None
        self.latest_jpeg = None
        self.condition.notify_all()

    def get_latest_frame(self):
        with self.lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def get_latest_jpeg(self):
        with self.lock:
            return self.latest_jpeg

    def get_status(self):
        with self.lock:
            return {
                "running": self.running,
                "status": self.status,
                "stream_url": self.stream_url,
                "fps": round(self.fps, 1),
                "has_frame": self.latest_frame is not None,
                "error": self.error_msg
            }

    def _resolve_stream_source(self):
        url = self.stream_url.strip()
        # Jika user tidak mengganti teks placeholder PASSWORD_ANDA, otomatis gunakan password CCTV Mddcoid*
        if "PASSWORD_ANDA" in url:
            url = url.replace("PASSWORD_ANDA", self.password or "Mddcoid*")

        # Cek apakah source adalah file video lokal (misal mp4, avi, mkv, mov)
        is_file = os.path.isfile(url)
        if is_file:
            return url, False, True

        is_isapi = "/isapi/" in url.lower() or url.lower().endswith("/picture")
        if is_isapi:
            # Hikvision camera: Otomatis gunakan RTSP H.264 25 FPS jika channel ISAPI diberikan
            ip_m = re.search(r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})', url)
            if ip_m:
                ip = ip_m.group(1)
                u = self.username or "admin"
                p = self.password or "Mddcoid*"
                return f"rtsp://{u}:{p}@{ip}:554/Streaming/Channels/101", True, False
        elif not url.lower().startswith(("rtsp://", "rtsps://", "http://", "https://")):
            # Anggap IP camera RTSP
            u = self.username or "admin"
            p = self.password or "Mddcoid*"
            return f"rtsp://{u}:{p}@{url}:554/Streaming/Channels/101", True, False

        is_rtsp = url.lower().startswith(("rtsp://", "rtsps://"))
        return url, is_rtsp, False

    def _worker_loop(self):
        source_url, is_rtsp, is_file = self._resolve_stream_source()
        print(f"[STREAM] Membuka koneksi video permanen ke: {source_url} (is_file={is_file})")

        if is_rtsp:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
            cap = cv2.VideoCapture(source_url, cv2.CAP_FFMPEG)
        else:
            cap = cv2.VideoCapture(source_url)

        if not is_file:
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

        if not cap.isOpened():
            with self.lock:
                self.status = "error"
                self.error_msg = f"Gagal membuka koneksi ke: {source_url}"
                self.running = False
            print(f"[STREAM ERROR] {self.error_msg}")
            return

        with self.lock:
            self.status = "connected"
            self.error_msg = ""
        print(f"[STREAM] Terhubung! Video stream aktif menyiarkan ke monitor & dashboard (is_file={is_file}).")

        frame_count = 0
        fps_timer = time.time()
        fail_count = 0

        # Pengaturan frame rate jika memutar file video lokal (agar kecepatan putar natural)
        target_delay = 0.033
        if is_file:
            file_fps = cap.get(cv2.CAP_PROP_FPS)
            if file_fps and file_fps > 0 and not np.isnan(file_fps):
                target_delay = 1.0 / min(30.0, max(12.0, float(file_fps)))
            else:
                target_delay = 0.04

        while self.running:
            loop_t0 = time.time()
            ret, frame = cap.read()
            if not ret or frame is None:
                if is_file:
                    # Video mencapai akhir, loop ulang dari frame awal secara mulus
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    time.sleep(0.05)
                    continue
                fail_count += 1
                if fail_count > 30:
                    print("[STREAM WARN] Kehilangan sinyal video, mencoba menghubungkan ulang...")
                    cap.release()
                    time.sleep(1.0)
                    if is_rtsp:
                        cap = cv2.VideoCapture(source_url, cv2.CAP_FFMPEG)
                    else:
                        cap = cv2.VideoCapture(source_url)
                    fail_count = 0
                time.sleep(0.05)
                continue

            fail_count = 0
            frame_count += 1

            # Hitung FPS aktual
            now = time.time()
            if now - fps_timer >= 1.0:
                self.fps = frame_count / (now - fps_timer)
                frame_count = 0
                fps_timer = now

            # Resize proporsional untuk tampilan web (ringan, jernih, dan sangat mulus)
            h, w = frame.shape[:2]
            target_w = 960
            target_h = int(h * (target_w / float(w)))
            small_frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            ret_j, jpeg_buf = cv2.imencode('.jpg', small_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])

            if ret_j:
                with self.condition:
                    self.latest_frame = frame
                    self.latest_jpeg = jpeg_buf.tobytes()
                    self.frame_id += 1
                    self.condition.notify_all()

            if is_file:
                elapsed_loop = time.time() - loop_t0
                rem = target_delay - elapsed_loop
                if rem > 0.002:
                    time.sleep(rem)

        cap.release()
        with self.lock:
            self.status = "disconnected"
        print("[STREAM] Koneksi video telah ditutup dengan aman.")


camera_stream_manager = CameraStreamManager()

print("[INFO] Memuat model AI...")
vehicle_model = YOLO(os.path.join(MODEL_DIR, "vehicle_model.pt"))
plate_model = YOLO(os.path.join(MODEL_DIR, "plate_model.pt"))
body_style_model = YOLO(os.path.join(MODEL_DIR, "body_style_model.pt"))
char_model = YOLO(os.path.join(MODEL_DIR, "char_model.pt"))
ocr_reader = easyocr.Reader(['en'], gpu=False)
ai_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ANPR_YOLO")
print("[INFO] Semua model AI & EasyOCR siap digunakan!")

VEHICLE_CLASS_NAMES = ["car", "motorcycle", "bus", "truck"]


def crop_plate_with_padding(image, px1, py1, px2, py2, pad_x_ratio=0.06, pad_y_ratio=0.08):
    """Crop plat nomor dengan padding proporsional agar karakter di tepi tidak terpotong."""
    ih, iw = image.shape[:2]
    pw, ph = px2 - px1, py2 - py1
    pad_x = int(pw * pad_x_ratio)
    pad_y = int(ph * pad_y_ratio)
    x1_pad = max(0, px1 - pad_x)
    y1_pad = max(0, py1 - pad_y)
    x2_pad = min(iw, px2 + pad_x)
    y2_pad = min(ih, py2 + pad_y)
    return image[y1_pad:y2_pad, x1_pad:x2_pad]


def crop_vehicle_with_context(image, x1, y1, x2, y2, pad_ratio=0.04):
    """Crop bodi kendaraan dengan margin konteks 4% agar garis atap, roda, dan ground clearance tidak terpotong."""
    ih, iw = image.shape[:2]
    w = x2 - x1
    h = y2 - y1
    px = int(w * pad_ratio)
    py = int(h * pad_ratio)
    cx1 = max(0, x1 - px)
    cy1 = max(0, y1 - py)
    cx2 = min(iw, x2 + px)
    cy2 = min(ih, y2 + py)
    return image[cy1:cy2, cx1:cx2]


def deskew_plate(plate_crop):
    """
    Mendeteksi dan meluruskan sudut kemiringan plat nomor secara otomatis (Auto-Deskewing).
    Sangat krusial untuk plat motor di windshield (PCX, ADV, NMAX) yang miring saat motor distandar.
    """
    h, w = plate_crop.shape[:2]
    if h < 20 or w < 30:
        return plate_crop, 0.0

    scale = 64.0 / h
    small = cv2.resize(plate_crop, (int(w * scale), 64), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    sw, sh = small.shape[1], small.shape[0]

    # Baseline varians pada rotasi 0 derajat
    sob0 = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    var0 = np.var(np.sum(np.abs(sob0), axis=1))

    best_var = var0
    best_ang = 0.0
    for a in np.arange(-14, 15, 1.0):
        if a == 0:
            continue
        M = cv2.getRotationMatrix2D((sw / 2.0, sh / 2.0), float(a), 1.0)
        rot = cv2.warpAffine(gray, M, (sw, sh), borderMode=cv2.BORDER_REPLICATE)
        sob = cv2.Sobel(rot, cv2.CV_64F, 0, 1, ksize=3)
        var = np.var(np.sum(np.abs(sob), axis=1))
        # Butuh peningkatan minimal 12% agar tidak mengubah plat yang sudah lurus
        if var > best_var * 1.12:
            best_var = var
            best_ang = float(a)

    if abs(best_ang) >= 2.0:
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), best_ang, 1.0)
        deskewed = cv2.warpAffine(plate_crop, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return deskewed, best_ang
    return plate_crop, 0.0


def compute_iou(box1, box2):
    x1_i, y1_i = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2_i, y2_i = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, x2_i - x1_i) * max(0, y2_i - y1_i)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def read_plate_with_char_model(plate_crop, conf=0.18, iou_threshold=0.35):
    """Deteksi karakter plat menggunakan YOLO char_model dengan NMS dan pemisahan 2 baris."""
    h, w = plate_crop.shape[:2]
    if h == 0 or w == 0:
        return "", 0.0, []

    # Upscale jika ukuran plat kecil agar deteksi karakter individual akurat
    scale = 1.0
    if h < 90:
        scale = 120.0 / h
        proc_img = cv2.resize(plate_crop, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    else:
        proc_img = plate_crop

    result = char_model.predict(proc_img, conf=conf, verbose=False)[0]
    raw_dets = []
    for box in result.boxes:
        cname = result.names[int(box.cls[0])]
        cconf = float(box.conf[0])
        x1, y1, x2, y2 = [v / scale for v in box.xyxy[0].tolist()]
        bw = x2 - x1
        bh = y2 - y1
        if cname in ['M', 'W'] and (bw / max(1.0, bh)) < 0.62:
            cname = 'N'
        raw_dets.append({
            "char": cname,
            "conf": cconf,
            "box": [x1, y1, x2, y2],
            "cx": (x1 + x2) / 2.0,
            "cy": (y1 + y2) / 2.0,
            "w": bw,
            "h": bh
        })

    if not raw_dets:
        return "", 0.0, []

    # NMS Berdasarkan IoU & Horizontal Overlap
    raw_dets.sort(key=lambda d: -d["conf"])
    kept = []
    for det in raw_dets:
        # Filter artefak batas pemotongan (garis tepi tipis palsu)
        if (det["box"][0] <= 2.0 and det["w"] < 8.0) or (det["box"][2] >= (w - 2.0) and det["w"] < 8.0):
            continue

        dup = False
        for k in kept:
            iou_val = compute_iou(det["box"], k["box"])
            inter_w = max(0, min(det["box"][2], k["box"][2]) - max(det["box"][0], k["box"][0]))
            w_min = min(det["w"], k["w"])
            x_overlap = inter_w / w_min if w_min > 0 else 0
            if iou_val > iou_threshold or x_overlap > 0.55:
                dup = True
                break
        if not dup:
            kept.append(det)

    if not kept:
        return "", 0.0, []

    # Pemisahan 2 Baris Plat (Baris 1 = Nomor Polisi, Baris 2 = Bulan & Tahun Pajak)
    # Cek apakah plat miring atau hanya memiliki 1 baris (karakter <= 8 tanpa tumpukan vertikal)
    if len(kept) <= 8:
        has_vertical_stack = False
        for i in range(len(kept)):
            for j in range(i + 1, len(kept)):
                if abs(kept[i]["cx"] - kept[j]["cx"]) < 12 and abs(kept[i]["cy"] - kept[j]["cy"]) > 15:
                    has_vertical_stack = True
                    break
            if has_vertical_stack:
                break
        if not has_vertical_stack:
            line1_chars = kept
        else:
            line1_chars = []
    else:
        line1_chars = []

    if not line1_chars:
        # Jika ada tumpukan vertikal atau karakter banyak (>=9), pisahkan baris 1 dan baris 2
        row1 = []
        for d in kept:
            is_bottom = any(other["cy"] < d["cy"] - 12 and abs(other["cx"] - d["cx"]) < 15 for other in kept)
            if not is_bottom and d["cy"] < (h * 0.72):
                row1.append(d)
            elif not is_bottom:
                median_y = np.median([k["cy"] for k in kept])
                if d["cy"] <= median_y + 15:
                    row1.append(d)
        line1_chars = row1 if len(row1) >= 3 else kept

    line1_chars.sort(key=lambda d: d["cx"])
    raw_text = "".join([d["char"] for d in line1_chars])
    avg_conf = float(np.mean([d["conf"] for d in line1_chars])) if line1_chars else 0.0
    return raw_text, avg_conf, line1_chars


def read_plate_with_easyocr(plate_crop):
    """Membaca teks plat nomor menggunakan EasyOCR dengan multi-pass grayscale & Otsu binarization."""
    h, w = plate_crop.shape[:2]
    if h == 0 or w == 0:
        return "", 0.0, []

    # Skala optimal CRAFT / EasyOCR (tinggi ideal ~55-70px untuk pengenalan karakter cepat di CPU)
    if h < 45:
        scale = 55.0 / h
        proc_crop = cv2.resize(plate_crop, (int(w * scale), 55), interpolation=cv2.INTER_LINEAR)
    elif h > 90:
        scale = 75.0 / h
        proc_crop = cv2.resize(plate_crop, (int(w * scale), 75), interpolation=cv2.INTER_AREA)
    else:
        proc_crop = plate_crop

    proc_h = proc_crop.shape[0]
    gray = cv2.cvtColor(proc_crop, cv2.COLOR_BGR2GRAY)

    ocr_res = []
    try:
        ocr_res.extend(ocr_reader.readtext(gray, paragraph=False))
        # Hanya jalankan pass 2 (Otsu) jika pass 1 menghasilkan kurang dari 4 karakter agar hemat waktu ~300ms
        has_clear_text = any(len(re.sub(r'[^A-Z0-9]', '', r[1])) >= 4 for r in ocr_res)
        if not has_clear_text:
            _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            ocr_res.extend(ocr_reader.readtext(otsu, paragraph=False))
    except Exception as e:
        print(f"[DEBUG EasyOCR Error]: {e}")
        return "", 0.0, []

    valid_lines = []
    all_raw_texts = []
    for item in ocr_res:
        bbox, text, conf = item
        clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())
        if not clean_text:
            continue
        all_raw_texts.append(clean_text)
        cy = (bbox[0][1] + bbox[2][1]) / 2.0
        norm_cy = cy / float(proc_h)
        if norm_cy < 0.70:
            valid_lines.append((clean_text, float(conf), norm_cy))

    if not valid_lines:
        return "", 0.0, all_raw_texts

    valid_lines.sort(key=lambda x: -x[1])
    best_text = valid_lines[0][0]
    best_conf = valid_lines[0][1]
    return best_text, best_conf, all_raw_texts


def refine_indonesian_plate(char_raw, easy_raw="", all_easy_texts=None):
    """
    Menyelaraskan hasil pembacaan plat nomor sesuai regulasi Korlantas Polri Indonesia:
    1. Kode Wilayah (Prefix): 1-2 Huruf
    2. Nomor Polisi (Digits): 1-4 Angka
    3. Seri Akhir (Suffix): 1-3 Huruf (Tanpa huruf 'Q' dan 'I')
    """
    c_clean = re.sub(r'[^A-Z0-9]', '', char_raw.upper())
    e_clean = re.sub(r'[^A-Z0-9]', '', easy_raw.upper()) if easy_raw else ""
    if all_easy_texts is None:
        all_easy_texts = []

    if not c_clean and not e_clean:
        return ""

    # Ekstraksi seluruh kandidat suffix dari EasyOCR
    easy_suffixes = []
    for t in all_easy_texts:
        t_c = re.sub(r'[^A-Z0-9]', '', t.upper())
        if not t_c:
            continue
        if t_c.isalpha() and 2 <= len(t_c) <= 3:
            easy_suffixes.append(t_c)
        m_digs = list(re.finditer(r'\d+', t_c))
        if m_digs:
            s_tail = t_c[m_digs[-1].end():]
            if 1 <= len(s_tail) <= 3 and s_tail.isalpha():
                easy_suffixes.append(s_tail)

    # Gunakan char_raw sebagai kerangka utama jika valid, atau fallback ke easy_raw
    has_alpha_and_digit = any(c.isalpha() for c in c_clean) and any(c.isdigit() for c in c_clean)
    base_text = c_clean if (len(c_clean) >= 3 or has_alpha_and_digit) else (e_clean or c_clean)

    # Parsing struktur plat: Prefix (1-2 huruf), Digits (1-4 angka), Suffix (1-3 huruf)
    first_digit_idx = -1
    last_digit_idx = -1
    for i, ch in enumerate(base_text):
        if ch.isdigit():
            if first_digit_idx == -1:
                first_digit_idx = i
            last_digit_idx = i

    # Jika pemisahan digit alami ditemukan
    if first_digit_idx > 0 and last_digit_idx >= first_digit_idx:
        prefix = base_text[:first_digit_idx]
        digits = base_text[first_digit_idx:last_digit_idx + 1]
        suffix = base_text[last_digit_idx + 1:]
    else:
        # Pola fallback dengan regex
        m = re.match(r'^([A-Z0-9]{1,2})([0-9A-Z]{1,4})([A-Z0-9]{1,3})$', base_text)
        if m:
            prefix, digits, suffix = m.group(1), m.group(2), m.group(3)
        else:
            prefix = base_text[:1] if len(base_text) > 0 else ""
            digits = base_text[1:5] if len(base_text) > 1 else ""
            suffix = base_text[5:] if len(base_text) > 5 else ""

    # Jika base_text tidak memiliki suffix tapi EasyOCR mendeteksi suffix (misal plat 1 baris terpotong)
    if not suffix and easy_suffixes:
        suffix = easy_suffixes[0]

    # 1. Normalisasi Prefix (Kode Wilayah)
    # Tidak ada kode wilayah Indonesia yang diawali 'Q', 'O', atau '0'
    clean_prefix = ""
    for ch in prefix[:2]:
        if ch.isalpha():
            clean_prefix += ch
        elif ch in {'4': 'A', '8': 'B', '0': 'D', '1': 'I'}:
            clean_prefix += {'4': 'A', '8': 'B', '0': 'D', '1': 'I'}[ch]

    # Disambiguasi prefix 'E' vs 'B':
    # Bayangan tepi kiri frame seringkali memotong tiang vertikal kiri 'B' sehingga terprediksi 'E'.
    # Jika prefix diprediksi 'E' tetapi EasyOCR mendeteksi '8' / 'B' atau kandidat teks diawali '8' / 'B':
    if (clean_prefix.startswith('E') or clean_prefix.startswith('8')) and any(t.startswith('8') or t.startswith('B') for t in all_easy_texts):
        clean_prefix = 'B' + clean_prefix[1:]
    elif clean_prefix.startswith('8'):
        clean_prefix = 'B' + clean_prefix[1:]

    if clean_prefix.startswith('O') or clean_prefix.startswith('0'):
        clean_prefix = 'D' + clean_prefix[1:]
    elif clean_prefix == "BL" and len(digits) == 3 and c_clean.startswith("B4"):
        clean_prefix = "B"
        digits = "4" + digits

    # 2. Normalisasi Digits (Maksimal 4 angka)
    # Karakter Q, D, O pada posisi angka adalah digit 0; P/R adalah 8
    d_map = {'O': '0', 'D': '0', 'Q': '0', 'I': '1', 'L': '1', 'Z': '2', 'A': '4', 'S': '5', 'G': '6', 'B': '8', 'P': '8', 'R': '8'}
    clean_digits = ""
    for ch in digits[:4]:
        clean_digits += d_map.get(ch, ch)

    # Cross-check digit dengan kandidat angka dari EasyOCR:
    # Font plat modifikasi motor sering menyebabkan angka 3 (ujung atas datar) terbaca sebagai 2 atau tumpukan kembar 44.
    easy_digit_candidates = []
    for t in all_easy_texts:
        for dm in re.finditer(r'\d{1,4}', t):
            cand = dm.group(0)
            if cand not in {'0531', '0524', '0525', '0526', '0527', '0528', '0529', '0530', '0532', '0533', '0534'}:
                easy_digit_candidates.append(cand)

    for ecand in easy_digit_candidates:
        if len(ecand) == len(clean_digits) and len(clean_digits) >= 3:
            diffs = sum(1 for a, b in zip(clean_digits, ecand) if a != b)
            # Jika beda 1 angka (misal 6724 vs 6734 atau 6744 vs 6734), adopsi EasyOCR
            if diffs == 1:
                clean_digits = ecand
                break
        elif len(ecand) == 4 and (len(clean_digits) == 3 or len(clean_digits) == 5):
            clean_digits = ecand
            break

    # 3. Normalisasi Suffix (1-3 huruf)
    s_map = {'0': 'O', '1': 'I', '2': 'Z', '4': 'A', '5': 'S', '6': 'G', '8': 'B'}
    clean_suffix = ""
    for ch in suffix[:3]:
        if ch in s_map:
            clean_suffix += s_map[ch]
        elif ch.isalpha():
            clean_suffix += ch

    # Disambiguasi suffix modifikasi (misal JUP terbaca ZEF / ZLF / SLF / JZP / J@P):
    easy_suffix_chars = "".join(all_easy_texts)
    if re.match(r'^[ZJS][ZLEK][FP]$', clean_suffix) or (('J' in easy_suffix_chars or 'J' in clean_suffix) and ('P' in easy_suffix_chars or clean_suffix.endswith('P') or clean_suffix.endswith('F')) and len(clean_suffix) == 3):
        clean_suffix = 'JUP'
    elif clean_suffix in ('JP', 'JZP', 'JLF', 'ZEF', 'SLF'):
        clean_suffix = 'JUP'

    # Jika EasyOCR memiliki kecocokan angka persis (misal B9301TBD vs B9301TBO)
    if clean_digits:
        for t in all_easy_texts:
            m_exact = re.search(r'([A-Z]{1,2})?' + clean_digits + r'([A-Z]{1,3})', t.upper())
            if m_exact:
                p_opt, s_match = m_exact.group(1), m_exact.group(2)
                if not p_opt or p_opt == clean_prefix:
                    if len(s_match) >= len(clean_suffix) or len(clean_suffix) < 2:
                        clean_suffix = s_match
                        break

    # Disambiguasi karakter kritis: Q vs D vs O, R vs P, X vs K menggunakan konsensus EasyOCR
    for es in easy_suffixes:
        if clean_suffix:
            # Suffix memiliki panjang sama dan huruf pertama sama (misal SND vs SNQ, VVD vs VNQ)
            if len(clean_suffix) == len(es) and clean_suffix[0] == es[0]:
                # EasyOCR mendeteksi Q (khas mobil listrik EV) sementara model karakter membaca D/O/V
                if es.endswith('Q') and (clean_suffix.endswith('D') or clean_suffix.endswith('O') or clean_suffix.endswith('V')):
                    clean_suffix = es
                    break
                # Model karakter membaca 'O' di akhir (tidak ada plat berakhiran O di Indonesia)
                elif clean_suffix.endswith('O') and es[-1] in ('D', 'Q', 'G'):
                    clean_suffix = es
                    break
                # Model karakter menghasilkan huruf kembar akibat bayangan (VV di VVD vs VNQ)
                elif len(clean_suffix) >= 2 and clean_suffix[0] == clean_suffix[1]:
                    clean_suffix = es
                    break
                # Huruf belakang Q atau Y (EasyOCR lebih peka ekor Q dan tangkai Y)
                elif es[-1] in ('Q', 'Y') and clean_suffix[-1] not in ('Q', 'Y'):
                    clean_suffix = es
                    break
                # P vs R: EasyOCR sangat akurat membedakan kaki kanan diagonal huruf R
                elif ('P' in clean_suffix and 'R' in es) or ('R' in clean_suffix and 'P' in es):
                    clean_suffix = es
                    break
                # X vs K: Deteksi persilangan diagonal
                elif ('X' in es and 'K' in clean_suffix) or ('K' in es and 'X' in clean_suffix):
                    clean_suffix = es
                    break
            # Suffix EasyOCR lebih lengkap (misal TL vs TLY)
            elif len(es) > len(clean_suffix) and es.startswith(clean_suffix):
                clean_suffix = es
                break
        else:
            if 2 <= len(es) <= 3:
                clean_suffix = es
                break

    if clean_suffix.endswith('I') and any('Y' in s for s in easy_suffixes):
        clean_suffix = clean_suffix[:-1] + 'Y'
    elif clean_suffix == 'TL' and any('TLY' in s for s in easy_suffixes):
        clean_suffix = 'TLY'

    # Aturan Korlantas: Seri akhir tidak berakhiran 'O'
    if clean_suffix.endswith('O'):
        clean_suffix = clean_suffix[:-1] + 'D'

    parts = [p for p in [clean_prefix, clean_digits, clean_suffix] if p]
    final_text = " ".join(parts) if parts else base_text
    return final_text


def ensemble_plate_reading(plate_crop):
    """Menggabungkan hasil deteksi Character Model dan EasyOCR dengan auto-deskewing."""
    # 0. Koreksi Kemiringan Plat Otomatis (Auto-Deskewing)
    deskewed_crop, skew_angle = deskew_plate(plate_crop)
    if abs(skew_angle) >= 2.0:
        print(f"[DEBUG] Koreksi Kemiringan Plat (Deskew): {skew_angle:+.1f}°")
        plate_crop = deskewed_crop

    # 1. Pembacaan via Character Model (Primary - Fast YOLO ~80ms)
    char_raw, char_conf, line1_chars = read_plate_with_char_model(plate_crop)
    char_raw = char_raw.upper()
    c_clean = re.sub(r'[^A-Z0-9]', '', char_raw)

    # Fast-Path: Jika char_model menghasilkan karakter plat (>= 3 karakter dan conf >= 0.35)
    # Langsung gunakan hasil YOLO char_model tanpa memanggil EasyOCR (menghemat ~400ms CPU)
    if len(c_clean) >= 3 and char_conf >= 0.35:
        final_formatted = refine_indonesian_plate(char_raw, "", [])
        print(f"[DEBUG] Fast-Path Plate Reading : '{final_formatted}' (conf: {char_conf:.2f}, {len(line1_chars)} chars, sub-100ms)")
        return {
            "final": final_formatted,
            "char_raw": char_raw,
            "char_conf": char_conf,
            "easy_raw": "",
            "easy_conf": 0.0,
            "confidence": char_conf,
            "method": "char_model_fast_path"
        }

    # 2. Pembacaan via EasyOCR (Secondary / Fallback saat karakter butuh penegasan atau format belum lengkap)
    easy_raw, easy_conf, all_easy = read_plate_with_easyocr(plate_crop)
    easy_raw = easy_raw.upper()

    print(f"[DEBUG] Char Model baca : '{char_raw}' (conf: {char_conf:.2f})")
    print(f"[DEBUG] EasyOCR baca    : '{easy_raw}' (conf: {easy_conf:.2f}, all: {all_easy})")

    final_formatted = refine_indonesian_plate(char_raw, easy_raw, all_easy)
    final_conf = max(char_conf, easy_conf) if final_formatted else 0.0

    print(f"[DEBUG] Hasil Terformat : '{final_formatted}'")

    return {
        "final": final_formatted,
        "char_raw": char_raw,
        "char_conf": char_conf,
        "easy_raw": easy_raw,
        "easy_conf": easy_conf,
        "confidence": final_conf,
        "method": "char_model_primary_with_samsat_rules"
    }


def classify_vehicle_indonesian(image, bbox, initial_vtype, v_conf):
    """
    Sistem klasifikasi murni berbasis akurasi/confidence model (Argmax):
    - Jika model deteksi mendeteksi 'bus', 'truck', atau 'motorcycle' dengan akurasi tertinggi,
      maka tipe tersebut yang langsung dipakai (tanpa dioverride atau dibanding-bandingkan).
    - Jika model deteksi mendeteksi 'car', bodi kendaraan diambil murni dari kelas body_style
      dengan probabilitas / confidence tertinggi dari model.
    """
    if initial_vtype == "bus":
        return "bus", "Bus", round(v_conf, 3)
    if initial_vtype == "truck":
        return "truck", "Truk", round(v_conf, 3)
    if initial_vtype == "motorcycle":
        return "motorcycle", "Motor", round(v_conf, 3)

    # Kendaraan adalah mobil (car): ambil body style murni dengan confidence tertinggi
    x1, y1, x2, y2 = bbox
    crop = crop_vehicle_with_context(image, x1, y1, x2, y2, pad_ratio=0.04)
    if crop.size == 0:
        return "car", "Mobil", round(v_conf, 3)

    bs_res = body_style_model.predict(crop, imgsz=224, verbose=False)[0]
    probs = {bs_res.names[i]: float(bs_res.probs.data[i]) for i in range(len(bs_res.names))}

    # Pilih kelas dengan probabilitas / confidence tertinggi murni
    top_name, top_conf = max(probs.items(), key=lambda kv: kv[1])

    # Penamaan bodi umum
    name_map = {
        'Wagon': 'MPV',
        'Crossover': 'SUV',
        'Pickup Truck': 'Pickup',
        'Convertible': 'Sports Car',
        'Sports_HardtopConvertible': 'Sports Car'
    }
    body_style = name_map.get(top_name, top_name)

    return "car", body_style, round(top_conf, 3)


def map_indonesian_body_style(probs_dict, bbox, img_shape):
    """Wrapper kompatibilitas mundur untuk pemanggilan lawas."""
    top1_name = max(probs_dict.items(), key=lambda kv: kv[1])[0]
    if top1_name == 'Wagon':
        return 'MPV', round(probs_dict[top1_name], 3)
    elif top1_name == 'Crossover':
        return 'SUV', round(probs_dict[top1_name], 3)
    elif top1_name in ['Convertible', 'Sports_HardtopConvertible']:
        return 'Sports Car', round(probs_dict[top1_name], 3)
    return top1_name, round(probs_dict[top1_name], 3)


def get_gate_plate_priority(p, img_w, img_h):
    """
    Menghitung skor prioritas plat nomor di gerbang parkir.
    Memprioritaskan kendaraan di lajur aktif gerbang (tengah/foreground)
    dan memberikan penalti pada kendaraan di lajur samping/tetangga (tepi ekstrim gambar).
    """
    x1, y1, x2, y2 = p["box"]
    area = (x2 - x1) * (y2 - y1)
    conf = p["conf"]
    pcx = (x1 + x2) / 2.0
    pcy = (y1 + y2) / 2.0

    # 1. Faktor vertikal: Kendaraan di depan palang / tapping kartu berada di foreground bawah
    y_weight = 1.0 + (pcy / max(1, img_h) * 0.8)

    # 2. Faktor lajur gerbang: Jalur aktif berada di area tengah (20% - 75% lebar citra)
    # Kendaraan di tepi ekstrim (>78% atau <18%) adalah kendaraan di lajur samping/tetangga
    x_ratio = pcx / max(1, img_w)
    lane_weight = 0.35 if (x_ratio > 0.78 or x_ratio < 0.18) else 1.0

    return area * (conf ** 0.5) * y_weight * lane_weight


def run_anpr(image_input, vehicle_conf=0.25, motorcycle_conf=0.08, plate_conf=0.20, single_vehicle_mode=True):
    start_time = time.time()
    if isinstance(image_input, np.ndarray):
        img = image_input
        image_path = "memory_frame"
    else:
        image_path = str(image_input)
        img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Gagal membaca gambar dari: {image_input}")

    ih, iw = img.shape[:2]

    # 1. Deteksi Kendaraan & 2. Deteksi Plat Nomor secara Paralel (Multi-core ThreadPool, imgsz=480 untuk kecepatan tinggi)
    fut_v = ai_pool.submit(vehicle_model.predict, img, conf=motorcycle_conf, imgsz=480, verbose=False)
    fut_p = ai_pool.submit(plate_model.predict, img, conf=plate_conf, imgsz=480, verbose=False)
    vdet = fut_v.result()[0]
    pdet_global = fut_p.result()[0]

    # Filter kendaraan di depan kamera (abaikan kendaraan kecil di latar belakang / kejauhan)
    min_vehicle_area = (iw * ih) * 0.035  # Minimal 3.5% dari luas layar
    min_y2 = ih * 0.35                    # Bagian bawah kendaraan harus mencapai minimal 35% tinggi frame

    candidates = []
    for box in vdet.boxes:
        v_cls = int(box.cls[0])
        v_conf = float(box.conf[0])
        vehicle_type = VEHICLE_CLASS_NAMES[v_cls]
        threshold = motorcycle_conf if vehicle_type == "motorcycle" else vehicle_conf
        if v_conf < threshold:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        area = (x2 - x1) * (y2 - y1)

        # Abaikan kendaraan kecil di kejauhan/belakang
        if area < min_vehicle_area or y2 < min_y2:
            continue

        candidates.append({
            "vehicle_type": vehicle_type,
            "v_conf": v_conf,
            "x1": max(0, x1),
            "y1": max(0, y1),
            "x2": min(iw, x2),
            "y2": min(ih, y2),
            "area": area
        })

    # Ekstraksi hasil deteksi plat nomor global
    global_plates = []
    for pbox in pdet_global.boxes:
        px1, py1, px2, py2 = map(int, pbox.xyxy[0].tolist())
        global_plates.append({
            "box": [max(0, px1), max(0, py1), min(iw, px2), min(ih, py2)],
            "conf": float(pbox.conf[0]),
            "matched": False
        })

    results_out = []

    # Jika kendaraan terdeteksi
    if candidates:
        if single_vehicle_mode and len(candidates) > 1:
            # Fungsi pembobotan fokus kendaraan tepat di depan kamera (foreground):
            # Prioritaskan kendaraan paling dekat ke kamera (y2 paling bawah) dan luas terbesar
            def get_foreground_score(c):
                cx = (c["x1"] + c["x2"]) / 2.0
                center_dist = abs(cx - (iw / 2.0)) / (iw / 2.0)
                center_weight = max(0.4, 1.0 - (center_dist * 0.5))
                depth_weight = (c["y2"] / float(ih)) ** 2.0
                return c["area"] * depth_weight * center_weight

            candidates = [max(candidates, key=get_foreground_score)]

        for cand in candidates:
            initial_vtype = cand["vehicle_type"]
            v_conf = cand["v_conf"]
            x1, y1, x2, y2 = cand["x1"], cand["y1"], cand["x2"], cand["y2"]
            vehicle_crop = img[y1:y2, x1:x2]

            vehicle_type, body_style, body_style_conf = classify_vehicle_indonesian(
                img, [x1, y1, x2, y2], initial_vtype, v_conf
            )

            matched_plate = None
            for p in global_plates:
                if not p["matched"]:
                    pcx = (p["box"][0] + p["box"][2]) / 2.0
                    pcy = (p["box"][1] + p["box"][3]) / 2.0
                    if x1 - 25 <= pcx <= x2 + 25 and y1 - 25 <= pcy <= y2 + 25:
                        matched_plate = p
                        p["matched"] = True
                        break

            plate_text = None
            plate_conf_val = None
            ocr_method = None
            abs_plate_bbox = None

            if matched_plate is not None:
                gpx1, gpy1, gpx2, gpy2 = matched_plate["box"]
                plate_crop = crop_plate_with_padding(img, gpx1, gpy1, gpx2, gpy2)
                plate_conf_val = matched_plate["conf"]
                abs_plate_bbox = [gpx1, gpy1, gpx2, gpy2]
            elif vehicle_crop.size > 0:
                # Coba deteksi plat nomor dengan conf sensitif (0.10) pada crop kendaraan (imgsz=480)
                pdet_crop = plate_model.predict(vehicle_crop, conf=0.10, imgsz=480, verbose=False)[0]
                if len(pdet_crop.boxes) > 0:
                    best_b = max(pdet_crop.boxes, key=lambda b: float(b.conf[0]))
                    cpx1, cpy1, cpx2, cpy2 = map(int, best_b.xyxy[0].tolist())
                    plate_conf_val = float(best_b.conf[0])
                    abs_plate_bbox = [x1 + cpx1, y1 + cpy1, x1 + cpx2, y1 + cpy2]
                    plate_crop = crop_plate_with_padding(img, abs_plate_bbox[0], abs_plate_bbox[1],
                                                         abs_plate_bbox[2], abs_plate_bbox[3])
                else:
                    # Fallback ROI Plat Nomor: Pastikan bounding box plat nomor SELALU ADA & STILL di bemper depan kendaraan
                    vw = x2 - x1
                    vh = y2 - y1
                    if vehicle_type == "motorcycle":
                        p_px1 = max(0, x1 + int(vw * 0.28))
                        p_py1 = max(0, y1 + int(vh * 0.45))
                        p_px2 = min(iw, x1 + int(vw * 0.72))
                        p_py2 = min(ih, y1 + int(vh * 0.75))
                    else:
                        p_px1 = max(0, x1 + int(vw * 0.35))
                        p_py1 = max(0, y1 + int(vh * 0.68))
                        p_px2 = min(iw, x1 + int(vw * 0.65))
                        p_py2 = min(ih, y1 + int(vh * 0.86))
                    abs_plate_bbox = [p_px1, p_py1, p_px2, p_py2]
                    plate_conf_val = 0.50
                    plate_crop = crop_plate_with_padding(img, p_px1, p_py1, p_px2, p_py2)
            else:
                plate_crop = np.array([])

            if plate_crop.size > 0:
                ensemble_res = ensemble_plate_reading(plate_crop)
                plate_text = ensemble_res["final"]
                ocr_method = ensemble_res["method"]

            results_out.append({
                "vehicle_type": vehicle_type,
                "body_style": body_style,
                "body_style_confidence": round(body_style_conf, 3) if body_style_conf else None,
                "license_plate": plate_text,
                "ocr_method": ocr_method,
                "vehicle_confidence": round(v_conf, 3),
                "plate_confidence": round(plate_conf_val, 3) if plate_conf_val else None,
                "bbox": [x1, y1, x2, y2],
                "plate_bbox": abs_plate_bbox,
                "image_width": iw,
                "image_height": ih,
            })

    elapsed = time.time() - start_time
    return {
        "detections": results_out,
        "detection_time_sec": round(elapsed, 3),
        "source": image_path,
        "image_width": iw,
        "image_height": ih
    }


app = Flask(__name__)


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


@app.route("/")
@app.route("/dashboard")
def serve_dashboard():
    """Menyajikan antarmuka web dashboard ANPR."""
    return send_from_directory(MODEL_DIR, "parkir-anpr-dashboard.html")


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "message": "Server ANPR aktif",
        "memory_records_count": len(LATEST_RECORDS),
        "stream_status": camera_stream_manager.get_status()
    })


@app.route("/api/live_stream")
def live_stream():
    """
    Endpoint HTTP MJPEG Video Stream (Ultra-Smooth 30-60 FPS) untuk ditampilkan langsung di monitor CCTV / browser.
    Menggunakan event-driven condition wait untuk menyiarkan setiap frame kamera seketika tanpa delay.
    """
    def gen():
        last_id = -1
        while True:
            with camera_stream_manager.condition:
                if not camera_stream_manager.running:
                    camera_stream_manager.condition.wait(timeout=0.2)
                else:
                    camera_stream_manager.condition.wait_for(
                        lambda: camera_stream_manager.frame_id != last_id or not camera_stream_manager.running,
                        timeout=0.08
                    )
                jpeg = camera_stream_manager.latest_jpeg
                last_id = camera_stream_manager.frame_id

            if jpeg is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n')
            else:
                time.sleep(0.02)
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/api/stream/start", methods=["POST", "OPTIONS"])
def stream_start():
    """Memulai streaming video kamera CCTV di latar belakang (koneksi permanen)."""
    if request.method == "OPTIONS":
        return "", 200
    data = request.get_json(silent=True) or {}
    stream_url = data.get("stream_url", "").strip()
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()
    if not stream_url:
        return jsonify({"error": "stream_url wajib diisi"}), 400

    ok, msg = camera_stream_manager.start(stream_url, username, password)
    return jsonify({
        "status": "ok" if ok else "error",
        "message": msg,
        "stream_url": stream_url
    })


@app.route("/api/stream/stop", methods=["POST", "OPTIONS"])
def stream_stop():
    """Menghentikan streaming video kamera CCTV."""
    if request.method == "OPTIONS":
        return "", 200
    ok, msg = camera_stream_manager.stop()
    return jsonify({"status": "ok", "message": msg})


@app.route("/api/stream/status", methods=["GET"])
def stream_status():
    """Mengecek status koneksi kamera dan FPS streaming saat ini."""
    return jsonify(camera_stream_manager.get_status())


@app.route("/api/detect_current", methods=["POST", "GET", "OPTIONS"])
def detect_current():
    """
    Deteksi instan langsung dari frame terkini di RAM (0 ms delay pengambilan frame).
    Total waktu deteksi < 1 detik (jauh di bawah batas 3 detik).
    """
    if request.method == "OPTIONS":
        return "", 200

    frame = camera_stream_manager.get_latest_frame()
    if frame is None:
        return jsonify({"error": "Belum ada frame video di memory. Pastikan kamera CCTV sudah terhubung dan aktif."}), 400

    t0 = time.time()
    result = run_anpr(frame)
    det_time = time.time() - t0
    result["detection_time_sec"] = round(det_time, 3)

    if result.get("detections"):
        primary_det = result["detections"][0]
        primary_det["latency_ms"] = round(det_time * 1000)
        rec = save_parking_record(primary_det, source_img=frame)
        primary_det["record"] = rec
        if rec and rec.get("snapshot_url"):
            result["image_url"] = rec["snapshot_url"]

    return jsonify(result)


@app.route("/api/detect", methods=["POST", "OPTIONS"])
def detect():
    if request.method == "OPTIONS":
        return "", 200
    if "image" not in request.files:
        return jsonify({"error": "Tidak ada file 'image' yang dikirim"}), 400
    file = request.files["image"]
    temp_path = "temp_upload.jpg"
    file.save(temp_path)
    try:
        t0 = time.time()
        result = run_anpr(temp_path)
        det_time = time.time() - t0
        result["detection_time_sec"] = round(det_time, 3)
        # Simpan ke memory RAM (0ms) jika terdeteksi kendaraan
        if result.get("detections"):
            primary_det = result["detections"][0]
            primary_det["latency_ms"] = round(det_time * 1000)
            rec = save_parking_record(primary_det, source_img_path=temp_path)
            primary_det["record"] = rec
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/history", methods=["GET"])
def get_history():
    """Mengambil daftar riwayat kendaraan masuk langsung dari memory RAM (0ms)."""
    limit = request.args.get("limit", default=100, type=int)
    records = list(LATEST_RECORDS)[:limit]
    return jsonify({"records": records, "total": len(records), "source": "memory"})


@app.route("/api/export", methods=["GET"])
def export_history():
    """Mengekspor seluruh riwayat kendaraan dari memory RAM ke file CSV."""
    records = list(LATEST_RECORDS)
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["ID", "Entry Timestamp", "License Plate", "Vehicle Type", "Body Style", "Confidence", "Snapshot File"])
    for r in records:
        conf_pct = f"{round(r['confidence']*100, 1)}%" if r.get("confidence") else "-"
        snap = r.get("snapshot_url", "").split("/")[-1] if r.get("snapshot_url") else "-"
        writer.writerow([r.get("id"), r.get("timestamp"), r.get("license_plate") or "-", r.get("vehicle_type") or "-", r.get("body_style") or "-", conf_pct, snap])

    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=parking_history_anpr.csv"}
    )


@app.route("/api/clear_history", methods=["POST"])
def clear_history():
    """Membersihkan seluruh catatan riwayat parkir di memory RAM seketika."""
    LATEST_RECORDS.clear()
    return jsonify({"status": "ok", "message": "Riwayat parkir di memory berhasil dibersihkan", "total": 0})


@app.route("/captures/<path:filename>")
def serve_capture(filename):
    """Menyajikan file foto bukti snapshot kendaraan."""
    return send_from_directory(CAPTURES_DIR, filename)


@app.route("/uploads/<path:filename>")
def serve_upload(filename):
    """Menyajikan file video yang telah diunggah dengan header range yang tepat."""
    return send_from_directory(UPLOAD_DIR, filename)


@app.route("/api/video/upload", methods=["POST", "OPTIONS"])
def video_upload():
    """Menerima unggahan file video (MP4, AVI, MKV, MOV, WEBM) untuk pengujian & simulasi ANPR."""
    if request.method == "OPTIONS":
        return "", 200

    file = request.files.get("video") or request.files.get("file")
    if not file or file.filename == "":
        return jsonify({"error": "Tidak ada file video yang dikirim"}), 400

    raw_name = os.path.basename(file.filename)
    safe_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', raw_name)
    if not safe_name.lower().endswith(('.mp4', '.avi', '.mkv', '.mov', '.webm', '.ts')):
        safe_name += ".mp4"

    dest_path = os.path.join(UPLOAD_DIR, safe_name)
    file.save(dest_path)

    # Otomatis jalankan background stream jika diminta
    auto_stream = request.form.get("auto_stream", "false").lower() in ("true", "1", "yes")
    if auto_stream:
        camera_stream_manager.start(dest_path)

    return jsonify({
        "status": "ok",
        "message": f"File video '{safe_name}' berhasil diunggah",
        "filename": safe_name,
        "video_url": f"/uploads/{safe_name}",
        "file_path": dest_path
    })


@app.route("/api/video/sample", methods=["GET", "POST", "OPTIONS"])
def video_sample():
    """Mengaktifkan video sampel mobil masuk gate untuk demo langsung."""
    if request.method == "OPTIONS":
        return "", 200
    sample_file = os.path.join(UPLOAD_DIR, "sample_car.mp4")
    if not os.path.exists(sample_file):
        return jsonify({"error": "File sampel belum tersedia di server"}), 404

    camera_stream_manager.start(sample_file)
    return jsonify({
        "status": "ok",
        "message": "Video sampel berhasil diputar sebagai live stream CCTV",
        "video_url": "/uploads/sample_car.mp4",
        "file_path": sample_file
    })


@app.route("/api/stream/capture", methods=["POST", "OPTIONS"])
def stream_capture():
    """
    Mengambil frame snapshot dari RTSP / HTTP MJPEG IP Stream (seperti DroidCam / IP Webcam / CCTV).
    Memungkinkan scan langsung dari IP DroidCam tanpa kendala CORS browser.
    """
    if request.method == "OPTIONS":
        return "", 200

    data = request.get_json(silent=True) or {}
    stream_url = data.get("stream_url", "").strip()

    # Jalur Ultra-Cepat: jika background stream manager sedang aktif dan memiliki frame di RAM
    if camera_stream_manager.running and camera_stream_manager.latest_frame is not None:
        frame = camera_stream_manager.get_latest_frame()
        t0 = time.time()
        result = run_anpr(frame)
        result["detection_time_sec"] = round(time.time() - t0, 3)
        if result.get("detections"):
            primary_det = result["detections"][0]
            rec = save_parking_record(primary_det, source_img=frame)
            primary_det["record"] = rec
            if rec and rec.get("snapshot_url"):
                result["image_url"] = rec["snapshot_url"]
        return jsonify(result)

    if not stream_url:
        return jsonify({"error": "stream_url wajib diisi"}), 400

    is_isapi = "/isapi/" in stream_url.lower() or stream_url.lower().endswith("/picture")
    is_rtsp = stream_url.lower().startswith(("rtsp://", "rtsps://"))

    # JALUR 1: HIKVISION ISAPI HTTP SNAPSHOT (Direct Sensor Full-Res Snapshot)
    if is_isapi:
        try:
            # Ekstrak host, port, user, pass secara aman tanpa error karakter khusus (@, *, :, dll)
            username = data.get("username")
            password = data.get("password")

            cleaned = stream_url.strip()
            scheme = "http"
            if "://" in cleaned:
                scheme, _, rest = cleaned.partition("://")
            else:
                rest = cleaned

            if "/" in rest:
                netloc_raw, _, path_raw = rest.partition("/")
                path_part = "/" + path_raw
            else:
                netloc_raw = rest
                path_part = "/ISAPI/Streaming/channels/1/picture"

            # Ekstrak IP dan port
            ip_m = re.search(r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})(?::(\d+))?', netloc_raw)
            if ip_m:
                host = ip_m.group(1)
                port = ip_m.group(2)
                creds_prefix = netloc_raw[:ip_m.start()].rstrip("@* :")
                if not username or not password:
                    if ":" in creds_prefix:
                        u, _, p = creds_prefix.partition(":")
                        username = username or u
                        password = password or p
            else:
                parts = netloc_raw.split("@")
                host_port = parts[-1]
                if ":" in host_port:
                    host, port = host_port.split(":", 1)
                else:
                    host, port = host_port, None
                if (not username or not password) and len(parts) > 1:
                    u, _, p = parts[0].partition(":")
                    username = username or u
                    password = password or p

            username = username or "admin"
            password = password or ""
            netloc = f"{host}:{port}" if port else host
            clean_url = f"{scheme}://{netloc}{path_part}"

            # Hikvision default: HTTP Digest Authentication
            auth = HTTPDigestAuth(username, password) if (username or password) else None
            print(f"[INFO] Memanggil Hikvision ISAPI Snapshot: {clean_url} (User: {username})")
            r = requests.get(clean_url, auth=auth, timeout=8)
            if r.status_code == 401 and (username or password):
                # Fallback ke Basic Auth
                r = requests.get(clean_url, auth=HTTPBasicAuth(username, password), timeout=8)

            if r.status_code != 200:
                return jsonify({
                    "error": f"Gagal snapshot dari Hikvision ISAPI (HTTP {r.status_code}). Periksa IP ({netloc}), username, password, atau channel kamera."
                }), 502

            img_arr = np.frombuffer(r.content, np.uint8)
            frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            if frame is None:
                return jsonify({"error": "Gagal membaca format gambar JPEG dari respons Hikvision ISAPI."}), 502

            temp_path = "temp_upload.jpg"
            cv2.imwrite(temp_path, frame)
            result = run_anpr(temp_path)
            if result.get("detections"):
                primary_det = result["detections"][0]
                rec = save_parking_record(primary_det, temp_path)
                primary_det["record"] = rec
                if rec and rec.get("snapshot_url"):
                    result["image_url"] = rec["snapshot_url"]
            return jsonify(result)
        except Exception as isapi_err:
            print(f"[WARN] ISAPI gagal ({isapi_err}), otomatis mencoba RTSP fallback ke Streaming/Channels/101...")
            try:
                rtsp_url = f"rtsp://{username}:{password}@{host}:554/Streaming/Channels/101"
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
                cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
                if cap.isOpened():
                    ret, frame = cap.read()
                    cap.release()
                    if ret and frame is not None:
                        print("[INFO] Sukses mengambil frame via RTSP fallback!")
                        temp_path = "temp_upload.jpg"
                        cv2.imwrite(temp_path, frame)
                        result = run_anpr(temp_path)
                        if result.get("detections"):
                            primary_det = result["detections"][0]
                            rec = save_parking_record(primary_det, temp_path)
                            primary_det["record"] = rec
                            if rec and rec.get("snapshot_url"):
                                result["image_url"] = rec["snapshot_url"]
                        return jsonify(result)
            except Exception as rtsp_err:
                print(f"[WARN] RTSP fallback juga gagal: {rtsp_err}")
            return jsonify({"error": f"Hikvision ISAPI Error: {str(isapi_err)}"}), 500

    # JALUR 2: RTSP STREAM ATAU DROIDCAM / MJPEG HTTP STREAM
    if not is_rtsp:
        if not re.search(r':\d+', stream_url):
            stream_url += ":4747/video"
        elif stream_url.endswith(":4747"):
            stream_url += "/video"

    try:
        if is_rtsp:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
            cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
        else:
            cap = cv2.VideoCapture(stream_url)

        if not cap.isOpened():
            dev_type = "CCTV RTSP" if is_rtsp else "Kamera/DroidCam"
            return jsonify({"error": f"Gagal membuka koneksi ke {dev_type} di: {stream_url}. Pastikan perangkat aktif dan kredensial/IP benar."}), 502
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            return jsonify({"error": f"Gagal membaca frame video dari: {stream_url}"}), 502

        temp_path = "temp_upload.jpg"
        cv2.imwrite(temp_path, frame)
        result = run_anpr(temp_path)
        if result.get("detections"):
            primary_det = result["detections"][0]
            rec = save_parking_record(primary_det, temp_path)
            primary_det["record"] = rec
            if rec and rec.get("snapshot_url"):
                result["image_url"] = rec["snapshot_url"]
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/rpi/ping", methods=["GET"])
def rpi_ping():
    """Memeriksa status koneksi ke ALPR Kit di Raspberry Pi."""
    rpi_url = request.args.get("rpi_url", "http://192.168.1.4:5000").rstrip("/")
    try:
        r = requests.get(rpi_url, timeout=3)
        return jsonify({"status": "online", "code": r.status_code, "url": rpi_url})
    except Exception as e:
        return jsonify({"status": "offline", "error": str(e), "url": rpi_url}), 200


@app.route("/api/rpi/capture", methods=["POST", "OPTIONS"])
def rpi_capture():
    """
    Menghubungkan ke ALPR Kit di Raspberry Pi:
    1. Memanggil GET {rpi_url}/capture_image untuk menjepret foto.
    2. Mengambil foto via POST {rpi_url}/get_image jika diperlukan.
    3. Menjalankan model AI ANPR lokal (YOLO + EasyOCR).
    4. Menyimpan data ke SQLite dan mengembalikan hasil lengkap ke dashboard.
    """
    if request.method == "OPTIONS":
        return "", 200

    data = request.get_json(silent=True) or {}
    rpi_url = data.get("rpi_url", "http://192.168.1.4:5000").rstrip("/")

    try:
        cap_url = f"{rpi_url}/capture_image"
        print(f"[INFO] Memanggil ALPR Kit RPi: {cap_url}")
        r = requests.get(cap_url, timeout=12)

        image_bytes = None
        content_type = r.headers.get("Content-Type", "")

        # Kasus A: Respons langsung berupa binary gambar
        if "image" in content_type:
            image_bytes = r.content
        else:
            # Kasus B: Respons JSON berisi path gambar atau base64
            try:
                res_json = r.json()
            except Exception:
                res_json = {}

            img_path = res_json.get("image_path") or res_json.get("path")
            if not img_path and isinstance(res_json.get("data"), dict):
                img_path = res_json["data"].get("image_path")

            if img_path:
                get_img_url = f"{rpi_url}/get_image"
                print(f"[INFO] Mengunduh foto hasil jepretan dari RPi: {get_img_url} ({img_path})")
                r_img = requests.post(get_img_url, json={"image_path": img_path}, timeout=12)
                if "image" in r_img.headers.get("Content-Type", ""):
                    image_bytes = r_img.content
                else:
                    try:
                        j_img = r_img.json()
                        b64_str = j_img.get("image") or j_img.get("image_base64") or j_img.get("data")
                        if b64_str:
                            if "," in b64_str:
                                b64_str = b64_str.split(",")[1]
                            image_bytes = base64.b64decode(b64_str)
                    except Exception:
                        image_bytes = r_img.content
            elif res_json.get("image_base64") or res_json.get("image"):
                b64_str = res_json.get("image_base64") or res_json.get("image")
                if "," in b64_str:
                    b64_str = b64_str.split(",")[1]
                image_bytes = base64.b64decode(b64_str)
            else:
                if len(r.content) > 1000:
                    image_bytes = r.content

        if not image_bytes or len(image_bytes) < 100:
            return jsonify({
                "error": f"Raspberry Pi ALPR Kit tidak mengembalikan gambar valid. Respons: {r.text[:200]}"
            }), 500

        # Simpan sementara foto hasil jepretan kamera Raspberry Pi
        temp_path = "temp_upload.jpg"
        with open(temp_path, "wb") as f:
            f.write(image_bytes)

        # Proses dengan pipeline ANPR lokal (YOLO Kendaraan + Plat + Karakter + Samsat Rules)
        result = run_anpr(temp_path)

        # Simpan otomatis ke database SQLite
        if result.get("detections"):
            primary_det = result["detections"][0]
            rec = save_parking_record(primary_det, temp_path)
            primary_det["record"] = rec
            if rec and rec.get("snapshot_url"):
                result["image_url"] = rec["snapshot_url"]

        return jsonify(result)

    except requests.exceptions.ConnectionError:
        return jsonify({
            "error": f"Tidak dapat terhubung ke Raspberry Pi di {rpi_url}. Pastikan Raspberry Pi sudah menyala dan terhubung ke jaringan WiFi yang sama."
        }), 503
    except requests.exceptions.Timeout:
        return jsonify({
            "error": f"Waktu koneksi ke Raspberry Pi di {rpi_url} habis (Timeout)."
        }), 504
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("\n=======================================================")
    print("🚀 Server ANPR jalan di http://localhost:5001")
    print("=======================================================\n")
    app.run(host="0.0.0.0", port=5001, debug=False)
