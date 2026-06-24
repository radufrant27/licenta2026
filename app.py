import streamlit as st
import sqlite3
import numpy as np
import mediapipe as mp
from PIL import Image, ImageEnhance, ImageOps, ImageDraw

# ==========================================
# --- CONFIGURARE AI ---
# ==========================================
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True, 
    max_num_faces=1, 
    refine_landmarks=True
)

def decupeaza_fata(img_pil):
    """Izolează structura osoasă și normalizează dimensiunea la 256x256"""
    img_array = np.array(img_pil)
    results = face_mesh.process(img_array)
    
    if not results.multi_face_landmarks:
        return None
        
    landmarks = results.multi_face_landmarks[0].landmark
    h, w = img_array.shape[:2]
    
    x_min = min(lm.x for lm in landmarks)
    x_max = max(lm.x for lm in landmarks)
    y_min = min(lm.y for lm in landmarks)
    y_max = max(lm.y for lm in landmarks)
    
    px_min, px_max = int(x_min * w), int(x_max * w)
    py_min, py_max = int(y_min * h), int(y_max * h)
    
    margin_w = int((px_max - px_min) * 0.15)
    margin_h = int((py_max - py_min) * 0.15)
    
    px_min = max(0, px_min - margin_w)
    px_max = min(w, px_max + margin_w)
    py_min = max(0, py_min - margin_h)
    py_max = min(h, py_max + margin_h)
    
    img_crop = img_pil.crop((px_min, py_min, px_max, py_max)).resize((256, 256))
    return img_crop

def extrage_vector(img_pil):
    """Extrage vectorul biometric din imagine"""
    img_array = np.array(img_pil)
    results = face_mesh.process(img_array)
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]
        annotated_image = img_array.copy()
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
        )
        vector = np.array([[lm.x, lm.y] for lm in landmarks.landmark]).flatten()
        return vector, annotated_image
    return None, None

def proceseaza_si_augumenteaza(foto_sursa):
    """Aplică Data Augmentation: Lumini extreme, Simptome și Traumă (Arsuri)"""
    if foto_sursa is None: return None, None
    
    img_base = Image.open(foto_sursa).convert('RGB')
    img_base = decupeaza_fata(img_base)
    
    if img_base is None: return None, None
    
    # 1. Normalizare Bază
    img_base = ImageOps.autocontrast(img_base)
    img_base = ImageEnhance.Sharpness(img_base).enhance(1.5)
    
    # 2. Augmentări de expunere
    img_high_exp = ImageEnhance.Brightness(img_base.copy()).enhance(1.5)
    img_super_low_exp = ImageEnhance.Brightness(img_base.copy()).enhance(0.2) # Super subexpusă
    
    # 3. Augmentări clinice (Culoare)
    # Față Roșie Intensă (Eritem/Febră mare)
    r, g, b = img_base.copy().split()
    r_febra = r.point(lambda i: min(int(i * 1.5), 255))
    img_red = Image.merge('RGB', (r_febra, g, b))
    
    # Față Cianotică (Mov/Albăstrui)
    r, g, b = img_base.copy().split()
    b_cian = b.point(lambda i: min(int(i * 1.4), 255))
    g_cian = g.point(lambda i: int(i * 0.8))
    img_purple = Image.merge('RGB', (r, g_cian, b_cian))
    
    # 4. Simulare Traumă / Arsuri (Tăiere puncte / Ocluzie)
    img_arsuri = img_base.copy()
    draw = ImageDraw.Draw(img_arsuri)
    # Desenăm pete negre/carbonizate pentru a distruge pixelii și a bloca AI-ul
    # Blocăm zona pometelui stâng și o parte din frunte
    draw.ellipse((30, 140, 110, 220), fill=(15, 5, 5)) 
    draw.ellipse((150, 40, 230, 90), fill=(15, 5, 5))
    
    variante_pil = {
        "Bază": img_base,
        "Expunere Max": img_high_exp,
        "Sub-expus Extrem": img_super_low_exp,
        "Eritem (Roșu)": img_red,
        "Cianoză": img_purple,
        "Arsuri (Ocluzie)": img_arsuri
    }
    
    vectori_colectati = []
    imagini_de_afisat = {}
    
    for nume, img in variante_pil.items():
        v, adnotata = extrage_vector(img)
        if v is not None:
            vectori_colectati.append(v)
            imagini_de_afisat[nume] = adnotata
            
    if len(vectori_colectati) > 0:
        vector_robust_final = np.mean(vectori_colectati, axis=0)
        return vector_robust_final, imagini_de_afisat
    else:
        return None, None

# ==========================================
# --- BAZA DE DATE (SQLITE3) ---
# ==========================================
conn = sqlite3.connect('baza_triaj_finala.db', check_same_thread=False)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS pacienti 
             (nume TEXT, cnp TEXT, afectiuni TEXT, 
              bio_deschisi BLOB, bio_inchisi BLOB)''')
conn.commit()

# ==========================================
# --- INTERFAȚĂ STYLING ---
# ==========================================
st.set_page_config(layout="wide", page_title="Sistem Triaj AI", page_icon="🚑")
st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; }
    h1 { color: #2b3a67; }
    .instructiuni-foto { background-color: #e3f2fd; padding: 15px; border-radius: 10px; border-left: 5px solid #1976d2; margin-bottom: 20px;}
    </style>
    """, unsafe_allow_html=True)

st.sidebar.markdown("# 🚑 Meniu Triaj")
meniu = st.sidebar.radio("Navigare:", ["📋 Înregistrare Pacient", "⚕️ Panou Medici"])

# ==========================================
# MODULUL 1: ÎNREGISTRARE PACIENT 
# ==========================================
if meniu == "📋 Înregistrare Pacient":
    st.markdown("<h1 style='text-align: center;'>🚑 Înregistrare Pacient Nou</h1>", unsafe_allow_html=True)
    
    with st.form("reg_form"):
        c1, c2 = st.columns(2)
        with c1:
            nume = st.text_input("Nume Complet Pacient:")
        with c2:
            cnp = st.text_input("CNP (13 cifre):")
        
        afectiuni_standard = st.multiselect("Afecțiuni Cronice / Alergii:", 
            ["Diabet", "Hipertensiune", "Alergie Iod", "Alergie Penicilină", 
             "Pacemaker", "Anticoagulante", "Astm", "Epilepsie"])
        alte_obs = st.text_area("Alte observații medicale:")

        st.write("---")
        st.subheader("📸 Captură Biometrică (Generare Automată Profil Robust)")
        
        st.markdown("""
        <div class="instructiuni-foto">
            <h4>📋 Ghid:</h4>
            Sistemul va izola automat structura osoasă. Vor fi generate 6 variante extreme (inclusiv arsuri care blochează fața și sub-expunere severă) pentru ambele ipostaze.
        </div>
        """, unsafe_allow_html=True)
        
        rand1_col1, rand1_col2 = st.columns(2)

        with rand1_col1:
            st.markdown("### 1. Față (Ochi Deschiși) 👀")
            f1_cam = st.camera_input("📷 Poză Live", key="cam1")
            f1_up = st.file_uploader("📂 Sau încarcă din dispozitiv", type=['jpg', 'jpeg', 'png'], key="up1")
            f1 = f1_cam if f1_cam else f1_up

        with rand1_col2:
            st.markdown("### 2. Față (Ochi Închiși) 😌")
            f2_cam = st.camera_input("📷 Poză Live", key="cam2")
            f2_up = st.file_uploader("📂 Sau încarcă din dispozitiv", type=['jpg', 'jpeg', 'png'], key="up2")
            f2 = f2_cam if f2_cam else f2_up
            
        submit = st.form_submit_button("💾 EXTRAGE CARACTERISTICI ȘI SALVEAZĂ", use_container_width=True)

    if submit:
        if nume and cnp and f1 and f2:
            v1, imgs_dict1 = proceseaza_si_augumenteaza(f1)
            v2, imgs_dict2 = proceseaza_si_augumenteaza(f2)
            
            if v1 is not None and v2 is not None:
                afectiuni_text = ", ".join(afectiuni_standard) + " | " + alte_obs
                
                c.execute("INSERT INTO pacienti VALUES (?,?,?,?,?)", 
                          (nume, cnp, afectiuni_text, v1.tobytes(), v2.tobytes()))
                conn.commit()
                
                st.success(f"✅ Profilul hiper-robust pentru {nume} a fost salvat!")
                
                st.write("### 🧬 Simulări AI (Ochi Deschiși):")
                cols1 = st.columns(len(imgs_dict1))
                for i, (titlu, img) in enumerate(imgs_dict1.items()):
                    cols1[i].image(img, caption=titlu, use_container_width=True)
                
                st.divider()
                
                st.write("### 🧬 Simulări AI (Ochi Închiși):")
                cols2 = st.columns(len(imgs_dict2))
                for i, (titlu, img) in enumerate(imgs_dict2.items()):
                    cols2[i].image(img, caption=titlu, use_container_width=True)
                    
            else:
                st.error("❌ Eșec detecție. Asigurați-vă că fața este vizibilă clar.")
        else:
            st.warning("⚠️ Introduceți datele și furnizați cele 2 imagini frontale.")

# ==========================================
# MODULUL 2: PANOU MEDICI (IDENTIFICARE)
# ==========================================
else:
    st.markdown("<h1>⚕️ Panou Identificare Medicală (UPU)</h1>", unsafe_allow_html=True)
    parola = st.sidebar.text_input("🔑 Parolă Acces Medic:", type="password")
    
    if parola == "licenta2026":
        scan = st.camera_input("Scanați fața pacientului adus la urgențe")
        
        if scan:
            img_pil_scan = Image.open(scan).convert('RGB')
            img_crop_scan = decupeaza_fata(img_pil_scan)
            
            if img_crop_scan is not None:
                v_scan, img_scan_adnotata = extrage_vector(img_crop_scan)
                
                if v_scan is not None:
                    col_img, col_rez = st.columns([1, 2])
                    col_img.image(img_scan_adnotata, caption="Structură Osoasă Extrasă Live", use_container_width=True)
                    
                    c.execute("SELECT nume, cnp, afectiuni, bio_deschisi, bio_inchisi FROM pacienti")
                    db_data = c.fetchall()
                    
                    if db_data:
                        rezultate = []
                        for n, c_p, af, b_d, b_i in db_data:
                            v_db_d = np.frombuffer(b_d, dtype=np.float64)
                            v_db_i = np.frombuffer(b_i, dtype=np.float64)
                            
                            dist_d = np.linalg.norm(v_scan - v_db_d)
                            dist_i = np.linalg.norm(v_scan - v_db_i)
                            
                            dist_minima = min(dist_d, dist_i)
                            rezultate.append((dist_minima, n, c_p, af))
                        
                        rezultate.sort()
                        cea_mai_buna = rezultate[0]
                        
                        with col_rez:
                            st.write(f"📊 **Scor de potrivire (Distanță Geometrică):** `{cea_mai_buna[0]:.4f}`")
                            
                            PRAG = 0.50 
                            
                            if cea_mai_buna[0] < PRAG:
                                st.success(f"### ✅ PACIENT: {cea_mai_buna[1]}")
                                st.metric("CNP", cea_mai_buna[2])
                                st.error(f"⚠️ FIȘĂ MEDICALĂ: {cea_mai_buna[3]}")
                            else:
                                st.error("❌ Pacient necunoscut în sistem.")
                    else:
                        st.warning("Baza de date este goală.")
                else:
                    st.error("Nu s-a detectat geometria osaturii în imaginea izolată.")
            else:
                st.error("AI-ul nu a putut izola fața din fundal.")
    elif parola != "":
        st.error("Parolă incorectă.")