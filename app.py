import streamlit as st
import sqlite3
import numpy as np
import mediapipe as mp
from PIL import Image
import unicodedata

# --- CONFIGURARE AI ---
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True, 
    max_num_faces=1, 
    refine_landmarks=True
)

def proceseaza_vizual(foto):
    if foto is None: return None, None
    try:
        image = Image.open(foto).convert('RGB')
        img_array = np.array(image)
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
    except:
        return None, None

# --- DATABASE ---
conn = sqlite3.connect('baza_triaj_finala.db', check_same_thread=False)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS pacienti 
             (nume TEXT, cnp TEXT, afectiuni TEXT, biometrie BLOB)''')
conn.commit()

# --- INTERFAȚĂ STYLING ---
st.set_page_config(layout="wide", page_title="Sistem Triaj AI", page_icon="🚑")
st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; }
    .stCamera { border: 2px solid #818cd1; border-radius: 15px; overflow: hidden; }
    h1 { color: #2b3a67; }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR (LOGOU ȘI NAVIGARE) ---
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
        st.subheader("📸 Captură Biometrică (Unghiuri Multiple)")
        
        # Cele 3 zone de cameră pe un rând
        col_img1, col_img2, col_img3 = st.columns(3)
        with col_img1:
            f1 = st.camera_input("1. Față", key="cam1")
        with col_img2:
            f2 = st.camera_input("2. Profil Stânga", key="cam2")
        with col_img3:
            f3 = st.camera_input("3. Profil Dreapta", key="cam3")
            
        submit = st.form_submit_button("💾 SALVEAZĂ PACIENT ÎN SISTEM")

    if submit:
        if nume and cnp and f1 and f2 and f3:
            v1, img1 = proceseaza_vizual(f1)
            v2, img2 = proceseaza_vizual(f2)
            v3, img3 = proceseaza_vizual(f3)
            
            if v1 is not None and v2 is not None and v3 is not None:
                v_final = np.mean([v1, v2, v3], axis=0)
                afectiuni_text = ", ".join(afectiuni_standard) + " | " + alte_obs
                
                c.execute("INSERT INTO pacienti VALUES (?,?,?,?)", 
                          (nume, cnp, afectiuni_text, v_final.tobytes()))
                conn.commit()
                
                st.success(f"✅ Pacientul {nume} a fost înregistrat cu succes!")
                st.image([img1, img2, img3], caption=["AI-Față", "AI-Stânga", "AI-Dreapta"], width=250)
            else:
                st.error("❌ AI-ul nu a putut detecta fața în toate cele 3 cadre. Verificați lumina.")
        else:
            st.warning("⚠️ Completați toate câmpurile obligatorii și efectuați toate pozele.")

# ==========================================
# MODULUL 2: PANOU MEDICI (CU PAROLĂ)
# ==========================================
else:
    st.markdown("<h1>⚕️ Panou Identificare Medicală</h1>", unsafe_allow_html=True)
    
    # Parola apare DOAR aici, în sidebar, când e selectat Panoul Medicului
    parola = st.sidebar.text_input("🔑 Parolă Acces Medic:", type="password")
    
    if parola == "licenta2024":
        st.info("Sistemul de scanare este activ. Vă rugăm să scanați pacientul.")
        scan = st.camera_input("Scanare facială pentru identificare instantă")
        
        if scan:
            v_scan, img_scan = proceseaza_vizual(scan)
            
            if v_scan is not None:
                st.image(img_scan, width=400, caption="Masca AI Detectată")
                
                c.execute("SELECT nume, cnp, afectiuni, biometrie FROM pacienti")
                db_data = c.fetchall()
                
                if db_data:
                    rezultate = []
                    for n, c_p, af, bio in db_data:
                        v_db = np.frombuffer(bio, dtype=np.float64)
                        distanta = np.linalg.norm(v_scan - v_db)
                        rezultate.append((distanta, n, c_p, af))
                    
                    rezultate.sort()
                    cea_mai_buna = rezultate[0]
                    
                    # Diagnostic vizual al scorului
                    st.write(f"📊 **Scor de identificare:** `{cea_mai_buna[0]:.4f}`")
                    
                    # PRAG DE CALIBRARE
                    PRAG = 0.45 
                    
                    if cea_mai_buna[0] < PRAG:
                        st.success(f"### ✅ PACIENT IDENTIFICAT: {cea_mai_buna[1]}")
                        col_res1, col_res2 = st.columns(2)
                        col_res1.metric("CNP", cea_mai_buna[2])
                        col_res2.error(f"⚠️ ISTORIC MEDICAL: {cea_mai_buna[3]}")
                    else:
                        st.error("❌ Nu s-a găsit nicio potrivire sigură. Pacientul nu este în sistem.")
                else:
                    st.warning("Baza de date este momentan goală.")
    elif parola == "":
        st.warning("Vă rugăm să introduceți parola în meniul lateral pentru a debloca datele medicale.")
    else:
        st.error("Parolă incorectă. Acces refuzat.")