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

# Custom CSS untuk UI yang lebih bagus
st.markdown("""
<style>
    /* Background & Font */
    body {
        background-color: #f8f9fa;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: white !important;
    }
    
    /* Title Styling */
    h1 {
        color: #2d3748;
        border-bottom: 3px solid #667eea;
        padding-bottom: 10px;
    }
    
    h2 {
        color: #4a5568;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Card Styling */
    .medicine-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
        border-left: 4px solid #667eea;
    }
    
    /* Info Box */
    [data-testid="stAlert"] {
        border-radius: 10px;
        padding: 15px;
    }
</style>
""", unsafe_allow_html=True)

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
    st.error(f"❌ Error: Tidak bisa load model 'best.pt'. Pastikan file ada di folder yang sama.\n\nDetail: {e}")
    st.stop()

# --- FUNGSI TAMPILKAN INFO OBAT ---
def display_medicine_info(class_name, conf_score):
    """Helper untuk menampilkan kartu informasi obat"""
    if class_name in MEDICINE_DB:
        data = MEDICINE_DB[class_name]
        
        # Styling Badge Warna
        badge_color = "🟢" if data['color'] == 'green' else "🔵"
        badge_type = "Obat Bebas" if data['color'] == 'green' else "Obat Bebas Terbatas"
        
        st.markdown(f"""
        <div class="medicine-card">
            <h3>{badge_color} {data['name']}</h3>
            <p><strong>Kategori:</strong> {badge_type}</p>
            <p><strong>Kepercayaan:</strong> {conf_score:.1%}</p>
            <p><strong>Fungsi:</strong> {data['desc']}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning(f"⚠️ **{class_name}** - Info detil belum ada di database.")

# --- 3. UI SIDEBAR & NAVIGASI ---
with st.sidebar:
    st.markdown("# 💊 Smart Pharmacy")
    st.markdown("**Aplikasi Deteksi Kemasan Obat**")
    st.markdown("---")
    
    # Menu Navigasi
    selected_menu = st.radio(
        "📍 Menu Aplikasi:", 
        ["🚀 Mulai Deteksi", "ℹ️ Panduan / Info", "👥 Tim Pengembang"],
    )
    
    st.markdown("---")
    # Settingan hanya muncul di menu deteksi
    if selected_menu == "🚀 Mulai Deteksi":
        st.subheader("⚙️ Pengaturan")
        confidence = st.slider("Tingkat Keyakinan", 0.0, 1.0, 0.40, 0.05)
    else:
        confidence = 0.25 

# --- 4. KONTEN HALAMAN UTAMA ---

# === HALAMAN 1: MULAI DETEKSI ===
if selected_menu == "🚀 Mulai Deteksi":
    st.markdown("# 🔍 Deteksi Obat Cerdas")
    st.markdown("Identifikasi kemasan obat dengan teknologi AI terdepan")
    st.markdown("---")
    
    # Tab untuk pilihan input
    tab1, tab2 = st.tabs(["📁 Upload Gambar", "🎥 Live Scan (Kamera)"])
    
    # --- TAB 1: UPLOAD FILE (Static) ---
    with tab1:
        st.markdown("### Unggah Foto Kemasan Obat")
        st.write("Upload foto untuk deteksi statis.")
        
        uploaded_file = st.file_uploader("Pilih gambar:", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📷 Gambar Asli:**")
                st.image(image, use_column_width=True)
                
            if st.button("🔍 Analisis Foto", key="analyze_btn"):
                with col2:
                    st.markdown("**🎯 Hasil Deteksi:**")
                    results = model.predict(image, conf=confidence)
                    result = results[0]
                    res_plotted = result.plot()
                    
                    st.image(res_plotted, use_column_width=True, channels="BGR")
                    
                    if len(result.boxes) > 0:
                        st.markdown("---")
                        st.markdown("### 📝 Informasi Obat Terdeteksi")
                        for idx, box in enumerate(result.boxes, 1):
                            cls_id = int(box.cls[0])
                            name = model.names[cls_id]
                            conf = float(box.conf[0])
                            with st.expander(f"Obat #{idx}: {name}", expanded=True):
                                display_medicine_info(name, conf)
                    else:
                        st.warning("❌ Tidak ada obat terdeteksi. Coba foto yang lebih jelas.")

    # --- TAB 2: LIVE SCAN (Real-time Loop) ---
    with tab2:
        st.markdown("### Live Scan dari Webcam")
        st.write("Mode real-time menggunakan kamera webcam Anda.")
        
        col_btn1, col_btn2 = st.columns([1, 3])
        with col_btn1:
            run_camera = st.checkbox("🔴 Buka Kamera", value=False)
        
        st_frame = st.empty()
        st_info = st.empty()
        status_placeholder = st.empty()

        if run_camera:
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                st.error("❌ Tidak bisa membuka webcam. Pastikan kamera terhubung.")
            else:
                # Optimasi: Set Resolusi Kamera
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)

                frame_count = 0
                skip_frames = 2  # Jalankan AI setiap 2 frame
                last_result = None
                detected_medicines = {}

                status_placeholder.info("🟢 Kamera aktif... Arahkan obat ke depan kamera")

                while run_camera:
                    ret, frame = cap.read()
                    if not ret:
                        status_placeholder.error("❌ Gagal membaca frame dari kamera.")
                        break
                    
                    frame_count += 1
                    
                    # Frame Skipping Logic
                    if frame_count % skip_frames == 0:
                        results = model.predict(frame, conf=confidence, verbose=False)
                        last_result = results[0]
                    
                    # Visualisasi
                    if last_result:
                        annotated_frame = last_result.plot(img=frame) 
                    else:
                        annotated_frame = frame

                    # Tampilkan Video
                    frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    st_frame.image(frame_rgb, use_column_width=True)
                    
                    # Tampilkan Info
                    if last_result and len(last_result.boxes) > 0:
                        best_box = max(last_result.boxes, key=lambda x: x.conf[0])
                        cls_id = int(best_box.cls[0])
                        name = model.names[cls_id]
                        conf = float(best_box.conf[0])
                        
                        with st_info.container():
                            st.markdown("### 🎯 Obat Terdeteksi")
                            display_medicine_info(name, conf)
                    else:
                        with st_info.container():
                            st.info("⏳ Menunggu deteksi... Arahkan obat ke kamera")

                cap.release()
                status_placeholder.success("✅ Kamera ditutup")

# === HALAMAN 2: INFO ===
elif selected_menu == "ℹ️ Panduan / Info":
    st.markdown("# 📚 Tentang Aplikasi")
    
    st.info("🤖 Aplikasi ini menggunakan teknologi **YOLOv11s** untuk mengenali kemasan obat secara real-time maupun upload foto dengan akurasi tinggi.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🚀 Cara Penggunaan Live Scan")
        st.markdown("""
        1. Masuk ke menu **Mulai Deteksi**
        2. Pilih Tab **Live Scan (Kamera)**
        3. Centang **🔴 Buka Kamera**
        4. Arahkan obat ke depan webcam
        5. Tunggu deteksi muncul di bawah video
        """)
    
    with col2:
        st.markdown("### 📸 Cara Penggunaan Upload Gambar")
        st.markdown("""
        1. Masuk ke menu **Mulai Deteksi**
        2. Pilih Tab **Upload Gambar**
        3. Upload foto kemasan obat
        4. Klik **🔍 Analisis Foto**
        5. Lihat hasil deteksi di sebelah kanan
        """)
    
    st.markdown("---")
    st.markdown("### 🎨 Warna Kategori Obat")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.success("""
        ### 🟢 Obat Bebas (Hijau)
        Dapat dibeli **tanpa resep dokter** secara bebas di apotek.
        """)
    with col_b:
        st.info("""
        ### 🔵 Obat Bebas Terbatas (Biru)
        Dapat dibeli **tanpa resep**, namun dengan **peringatan khusus**.
        """)
    
    st.markdown("---")
    st.markdown("### ⚡ Tips Penggunaan")
    st.markdown("""
    - **Pencahayaan yang baik** meningkatkan akurasi deteksi
    - **Posisikan kemasan obat** agar label terlihat jelas
    - **Jarak optimal** 20-50 cm dari kamera
    - Jika deteksi tidak akurat, coba ubah **Tingkat Keyakinan** di Pengaturan
    """)

# === HALAMAN 3: TIM PENGEMBANG ===
elif selected_menu == "👥 Tim Pengembang":
    st.markdown("# 👨‍💻 Tim Pengembang")
    st.markdown("Proyek ini didedikasikan untuk membantu identifikasi obat secara digital dan meningkatkan literasi kesehatan masyarakat.")
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 👩‍💻 Developer 1
        **Salma Nesya Putri Salia**
        
        - Backend Development
        - Model Integration
        """)
    
    with col2:
        st.markdown("""
        ### 👩‍💻 Developer 2
        **Fitri Salwa**
        
        - Frontend Development
        - UI/UX Design
        """)
    
    with col3:
        st.markdown("""
        ### 🎯 Tujuan Proyek
        
        Membantu pengguna mengidentifikasi obat dengan cepat dan akurat melalui teknologi AI.
        """)
    
    st.markdown("---")
    st.markdown("### 🛠️ Tech Stack")
    st.markdown("""
    - **Framework**: Streamlit
    - **AI Model**: YOLOv11s (Ultralytics)
    - **Computer Vision**: OpenCV
    - **Image Processing**: Pillow, NumPy
    """)