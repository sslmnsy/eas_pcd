import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Deteksi Obat AI Live",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 1. SETUP DATABASE (DATA TETAP) ---
MEDICINE_DB = {
    # --- PARACETAMOL (Hijau) 🟢 ---
    "Biogesic Paracetamol": {"name": "Biogesic", "type": "Obat Bebas (Hijau)", "desc": "Meredakan demam dan sakit kepala.", "color": "green"},
    "Pamol Paracetamol": {"name": "Pamol", "type": "Obat Bebas (Hijau)", "desc": "Penurun panas (demam) dan pereda nyeri.", "color": "green"},
    "Panadol Paracetamol": {"name": "Panadol Regular (Biru)", "type": "Obat Bebas (Hijau)", "desc": "Meredakan pusing dan demam. Aman di lambung.", "color": "green"},
    "Panadol Extra Paracetamol": {"name": "Panadol Extra (Merah)", "type": "Obat Bebas (Hijau)", "desc": "Sakit kepala membandel & sakit gigi (Mengandung Kafein).", "color": "green"},

    # --- OBAT FLU & BATUK (Biru) 🔵 ---
    "Decolgen": {"name": "Decolgen", "type": "Obat Bebas Terbatas (Biru)", "desc": "Meredakan sakit kepala, flu, dan hidung tersumbat.", "color": "blue"},
    "Intunal F": {"name": "Intunal F", "type": "Obat Bebas Terbatas (Biru)", "desc": "Gejala flu, demam, batuk, dan hidung meler.", "color": "blue"},
    "Intunal Kaplet": {"name": "Intunal Kaplet", "type": "Obat Bebas Terbatas (Biru)", "desc": "Meredakan flu dan batuk berdahak/kering.", "color": "blue"},
    "Inza": {"name": "Inza", "type": "Obat Bebas Terbatas (Biru)", "desc": "Meringankan gejala flu seperti demam & hidung tersumbat.", "color": "blue"},
    "Mixagrip": {"name": "Mixagrip Flu", "type": "Obat Bebas Terbatas (Biru)", "desc": "Meredakan gejala flu dan bersin-bersin.", "color": "blue"},
    "Mixagrip Flu dan Batuk": {"name": "Mixagrip Flu & Batuk", "type": "Obat Bebas Terbatas (Biru)", "desc": "Meredakan flu disertai batuk.", "color": "blue"},
    "Neozep Forte": {"name": "Neozep Forte", "type": "Obat Bebas Terbatas (Biru)", "desc": "Obat flu berat, hidung tersumbat, dan bersin.", "color": "blue"},
    "Panadol Cold Flu": {"name": "Panadol Cold & Flu (Hijau)", "type": "Obat Bebas Terbatas (Biru)", "desc": "Hidung tersumbat, batuk tidak berdahak, dan demam.", "color": "blue"},
    "Paramex Flu dan Batuk": {"name": "Paramex Flu & Batuk", "type": "Obat Bebas Terbatas (Biru)", "desc": "Meredakan gejala flu dan batuk.", "color": "blue"},
    "Sanaflu": {"name": "Sanaflu", "type": "Obat Bebas Terbatas (Biru)", "desc": "Meringankan gejala flu dan demam.", "color": "blue"},
    "Ultraflu": {"name": "Ultraflu", "type": "Obat Bebas Terbatas (Biru)", "desc": "Meredakan flu, demam, dan sakit kepala.", "color": "blue"},
    "Ultraflu Extra": {"name": "Ultraflu Extra", "type": "Obat Bebas Terbatas (Biru)", "desc": "Formulasi ekstra untuk flu berat dan pusing.", "color": "blue"},

    # --- OBAT SAKIT KEPALA & NYERI (Biru) 🔵 ---
    "Oskadon SP": {"name": "Oskadon SP", "type": "Obat Bebas Terbatas (Biru)", "desc": "Mengurangi nyeri otot, pegal linu, dan sakit pinggang.", "color": "blue"},
    "Oskadon Sakit Kepala": {"name": "Oskadon", "type": "Obat Bebas Terbatas (Biru)", "desc": "Mengurangi sakit kepala dan pusing.", "color": "blue"},
    "Paramex": {"name": "Paramex", "type": "Obat Bebas Terbatas (Biru)", "desc": "Meredakan sakit kepala dan sakit gigi.", "color": "blue"},
    "Paramex Nyeri Otot": {"name": "Paramex Nyeri Otot", "type": "Obat Bebas Terbatas (Biru)", "desc": "Meredakan nyeri otot dan pegal linu.", "color": "blue"},
    "Poldan Mig": {"name": "Poldan Mig", "type": "Obat Bebas Terbatas (Biru)", "desc": "Meredakan sakit kepala sebelah (Migrain).", "color": "blue"}
}

# --- 2. LOAD MODEL ---
@st.cache_resource
def load_model():
    return YOLO("models/best.pt")

try:
    model = load_model()
except Exception as e:
    st.error(f"Error: Tidak bisa load model 'best.pt'. Pastikan file ada di folder yang sama.\n\nDetail: {e}")
    st.stop()

# --- FUNGSI TAMPILKAN INFO OBAT ---
def display_medicine_info(class_name, conf_score):
    """Helper untuk menampilkan kartu informasi obat"""
    if class_name in MEDICINE_DB:
        data = MEDICINE_DB[class_name]
        
        # Styling Badge Warna
        if data['color'] == 'green':
            st.success(f"### 🟢 {data['name']}")
        else:
            st.info(f"### 🔵 {data['name']}")
        
        st.caption(f"Confidence: {conf_score:.2f} | ID: {class_name}")
        st.write(f"**Jenis:** {data['type']}")
        st.write(f"**Fungsi:** {data['desc']}")
    else:
        st.warning(f"⚠️ **{class_name}**")
        st.write("Info detil obat ini belum ada di database.")

# --- 3. UI SIDEBAR & NAVIGASI ---
with st.sidebar:
    st.title("💊 Smart Pharmacy")
    st.write("Aplikasi Deteksi Kemasan Obat")
    st.markdown("---")
    
    # Menu Navigasi
    selected_menu = st.radio(
        "Menu Aplikasi:", 
        ["🚀 Mulai Deteksi", "ℹ️ Panduan / Info", "👥 Tim Pengembang"],
    )
    
    st.markdown("---")
    # Settingan hanya muncul di menu deteksi
    if selected_menu == "🚀 Mulai Deteksi":
        st.subheader("⚙️ Pengaturan")
        confidence = st.slider("Tingkat Keyakinan (Confidence)", 0.0, 1.0, 0.40, 0.05) # Default agak tinggi untuk live
    else:
        confidence = 0.25 

# --- 4. KONTEN HALAMAN UTAMA ---

# === HALAMAN 1: MULAI DETEKSI ===
if selected_menu == "🚀 Mulai Deteksi":
    st.title("Deteksi Obat Cerdas")
    
    # Tab untuk pilihan input
    tab1, tab2 = st.tabs(["📁 Upload Gambar", "🎥 Live Scan (Kamera)"])
    
    # --- TAB 1: UPLOAD FILE (Static) ---
    with tab1:
        st.write("Upload foto untuk deteksi statis.")
        uploaded_file = st.file_uploader("Upload foto kemasan obat", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # PERBAIKAN 1: use_container_width=True
                st.image(image, caption="Gambar Asli", use_container_width=True)
                
            if st.button("🔍 Analisis Foto"):
                with col2:
                    results = model.predict(image, conf=confidence)
                    result = results[0]
                    res_plotted = result.plot()
                    
                    # PERBAIKAN 2: use_container_width=True
                    st.image(res_plotted, caption="Hasil Deteksi", use_container_width=True, channels="BGR")
                    
                    if len(result.boxes) > 0:
                        with st.expander("📝 Hasil Analisis", expanded=True):
                            for box in result.boxes:
                                cls_id = int(box.cls[0])
                                name = model.names[cls_id]
                                conf = float(box.conf[0])
                                display_medicine_info(name, conf)
                                st.markdown("---")
                    else:
                        st.warning("Tidak ada obat terdeteksi.")

    # --- TAB 2: LIVE SCAN (Real-time Loop) ---
    with tab2:
        st.write("Mode ini menggunakan kamera webcam (Dioptimalkan).")
        
        col_btn1, col_btn2 = st.columns([1, 3])
        with col_btn1:
            run_camera = st.checkbox("🔴 Buka Kamera (ON/OFF)")
        
        st_frame = st.empty()
        st_info = st.empty()

        if run_camera:
            cap = cv2.VideoCapture(0)
            
            # OPTIMASI 1: Set Resolusi Kamera lebih rendah (Standard VGA)
            # Semakin kecil resolusi, semakin cepat YOLO bekerja
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            frame_count = 0
            skip_frames = 3  # Jalankan AI hanya setiap 3 frame sekali
            last_result = None # Menyimpan hasil deteksi terakhir

            while run_camera:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # OPTIMASI 2: Frame Skipping Logic
                # Jika frame ke-N, jalankan YOLO. Jika tidak, pakai hasil lama.
                if frame_count % skip_frames == 0:
                    results = model.predict(frame, conf=confidence, verbose=False)
                    last_result = results[0]
                
                # Visualisasi
                if last_result:
                    # Plotting hasil terakhir ke frame yang sekarang (agar kotak tidak hilang)
                    annotated_frame = last_result.plot(img=frame) 
                else:
                    annotated_frame = frame

                # Tampilkan Video
                frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                st_frame.image(frame_rgb, channels="RGB", use_container_width=True)
                
                # Tampilkan Info (Hanya update jika ada hasil baru biar hemat resource)
                if last_result and len(last_result.boxes) > 0:
                    best_box = max(last_result.boxes, key=lambda x: x.conf[0])
                    cls_id = int(best_box.cls[0])
                    name = model.names[cls_id]
                    conf = float(best_box.conf[0])
                    
                    with st_info.container():
                        display_medicine_info(name, conf)

            cap.release()

# === HALAMAN 2: INFO ===
elif selected_menu == "ℹ️ Panduan / Info":
    st.title("Tentang Aplikasi")
    st.info("Aplikasi ini menggunakan teknologi YOLOv11s untuk mengenali kemasan obat secara real-time maupun upload foto.")
    
    st.subheader("Cara Penggunaan Live Scan:")
    st.markdown("""
    1. Masuk ke menu **Mulai Deteksi**.
    2. Pilih Tab **Live Scan (Kamera)**.
    3. Centang kotak **🔴 Buka Kamera (ON/OFF)**.
    4. Arahkan obat ke depan webcam. Penjelasan obat akan muncul di bawah video.
    """)
    
    st.subheader("Warna Kategori Obat:")
    col_a, col_b = st.columns(2)
    with col_a:
        st.success("🟢 **Lingkaran Hijau (Obat Bebas)**")
        st.write("Dapat dibeli tanpa resep dokter secara bebas.")
    with col_b:
        st.info("🔵 **Lingkaran Biru (Bebas Terbatas)**")
        st.write("Dapat dibeli tanpa resep, namun dengan peringatan khusus.")

# === HALAMAN 3: TIM PENGEMBANG ===
elif selected_menu == "👥 Tim Pengembang":
    st.title("Tim Pengembang")
    st.write("Proyek ini didedikasikan untuk membantu identifikasi obat secara digital.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 👩‍💻 Dev 1")
        st.write("Fitri Salwa")
    with col2:
        st.markdown("### 👩‍💻 Dev 2")
        st.write("Salma Nesya Putri Salia")
    