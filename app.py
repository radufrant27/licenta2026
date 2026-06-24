import streamlit as st
import sqlite3
import numpy as np
import mediapipe as mp
from PIL import Image, ImageEnhance, ImageOps

# ==========================================
# --- CONFIGURARE AI ȘI PREPROCESARE ---
# ==========================================
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True, 
    max_num_faces=1, 
    refine_landmarks=True
)

def proceseaza_vizual(foto_sursa):
    if foto_sursa is None: return None, None
    try:
        # 1. Deschidem imaginea din sursă
        image = Image.open(foto_sursa).convert('RGB')
        
        # 2. Preprocesare digitală pentru îmbunătățirea detaliilor geometrice
        image = ImageOps.autocontrast(image)  # Corectează automat lumina slabă sau umbrele
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(2.0)        # Accentuează marginile pentru o detecție precisă a punctelor
        
        # 3. Conversie în array și procesare cu MediaPipe Face Mesh
        img_array = np.array(image)
        results = face_mesh.process(img_array)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            annotated_image = img_array.copy()
            
            # Desenăm masca de puncte peste imagine pentru feedback-ul vizual al utilizatorului
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
            
            # Extragere vector de caracteristici (Coordonate X și Y aplatizate)
            vector = np.array([[lm.x, lm.y] for lm in landmarks.landmark]).flatten()
            return vector, annotated_image
        return None, None
    except:
        return None, None

# ==========================================
# --- BAZA DE DATE (SQLITE3) ---
# ==========================================
conn = sqlite3.connect('baza_triaj_finala.db', check_same_thread=False)
c = conn.cursor()
# Creăm tabela cu 4 coloane BLOB pentru stocarea celor 4 amprente biometrice distincte
c.execute('''CREATE TABLE IF NOT EXISTS pacienti 
             (nume TEXT, cnp TEXT, afectiuni TEXT, 
              bio_deschisi BLOB, bio_inchisi BLOB, bio_stanga BLOB, bio_dreapta BLOB)''')
conn.commit()

# ==========================================
# --- INTERFAȚĂ ȘI STYLING CSS ---
# ==========================================
st.set_page_config(layout="wide", page_title="Sistem Triaj AI", page_icon="🚑")
st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; }
    .stCamera { border: 2px solid #818cd1; border-radius: 15px; overflow: hidden; }
    h1 { color: #2b3a67; }
    .instructiuni-foto { background-color: #e3f2fd; padding: 15px; border-radius: 10px; border-left: 5px solid #1976d2; margin-bottom: 20px;}
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR (NAVIGARE APLICAȚIE) ---
st.sidebar.markdown("# 🚑 Meniu Triaj")
meniu = st.sidebar.radio("Navigare:", ["📋 Înregistrare Pacient", "⚕️ Panou Medici"])

# ==========================================
# MODULUL 1: ÎNREGISTRARE PACIENT (4 UNGHIURI)
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
        st.subheader("📸 Captură Biometrică Multi-Unghi")
        
        st.markdown("""
        <div class="instructiuni-foto">
            <h4>📋 Ghid pentru înregistrare:</h4>
            Realizați sau încărcați <b>toate cele 4 poze</b> de mai jos pentru a asigura identificarea în orice scenariu medical. Pentru fiecare unghi puteți folosi camera web <b>SAU</b> puteți încărca un fișier preexistent.
        </div>
        """, unsafe_allow_html=True)
        
        # Organizare vizuală pe linii și coloane pentru cele 4 ipostaze
        rand1_col1, rand1_col2 = st.columns(2)
        rand2_col1, rand2_col2 = st.columns(2)

        # 1. Față Ochi Deschiși
        with rand1_col1:
            st.markdown("### 1. Față (Ochi Deschiși) 👀")
            f1_cam = st.camera_input("📷 Poză Live", key="cam1")
            f1_up = st.file_uploader("📂 Sau încarcă din dispozitiv", type=['jpg', 'jpeg', 'png'], key="up1")
            f1 = f1_cam if f1_cam else f1_up

        # 2. Față Ochi Închiși
        with rand1_col2:
            st.markdown("### 2. Față (Ochi Închiși) 😌")
            f2_cam = st.camera_input("📷 Poză Live", key="cam2")
            f2_up = st.file_uploader("📂 Sau încarcă din dispozitiv", type=['jpg', 'jpeg', 'png'], key="up2")
            f2 = f2_cam if f2_cam else f2_up

        st.write("") 
        
        # 3. Profil Stânga
        with rand2_col1:
            st.markdown("### 3. Profil STÂNGA 👈")
            f3_cam = st.camera_input("📷 Poză Live", key="cam3")
            f3_up = st.file_uploader("📂 Sau încarcă din dispozitiv", type=['jpg', 'jpeg', 'png'], key="up3")
            f3 = f3_cam if f3_cam else f3_up

        # 4. Profil Dreapta
        with rand2_col2:
            st.markdown("### 4. Profil DREAPTA 👉")
            f4_cam = st.camera_input("📷 Poză Live", key="cam4")
            f4_up = st.file_uploader("📂 Sau încarcă din dispozitiv", type=['jpg', 'jpeg', 'png'], key="up4")
            f4 = f4_cam if f4_cam else f4_up
            
        submit = st.form_submit_button("💾 SALVEAZĂ PACIENT ÎN SISTEM", use_container_width=True)

    if submit:
        # Validare: Verificăm existența datelor text și a celor 4 surse de imagine
        if nume and cnp and f1 and f2 and f3 and f4:
            v1, img1 = proceseaza_vizual(f1)
            v2, img2 = proceseaza_vizual(f2)
            v3, img3 = proceseaza_vizual(f3)
            v4, img4 = proceseaza_vizual(f4)
            
            # Verificăm dacă MediaPipe a reușit să extragă punctele din toate cele 4 cadre
            if all(v is not None for v in [v1, v2, v3, v4]):
                afectiuni_text = ", ".join(afectiuni_standard) + " | " + alte_obs
                
                # Inserarea celor 4 amprente biometrice serializate în format BLOB
                c.execute("INSERT INTO pacienti VALUES (?,?,?,?,?,?,?)", 
                          (nume, cnp, afectiuni_text, 
                           v1.tobytes(), v2.tobytes(), v3.tobytes(), v4.tobytes()))
                conn.commit()
                
                st.success(f"✅ Pacientul {nume} a fost salvat cu succes în baza de date locală.")
                st.balloons()
                
                # Afișarea rezultatului procesării pentru validarea vizuală a măștii AI
                res_col1, res_col2, res_col3, res_col4 = st.columns(4)
                res_col1.image(img1, caption="Mască Ochi Deschiși", use_column_width=True)
                res_col2.image(img2, caption="Mască Ochi Închiși", use_column_width=True)
                res_col3.image(img3, caption="Mască Profil Stânga", use_column_width=True)
                res_col4.image(img4, caption="Mască Profil Dreapta", use_column_width=True)
            else:
                st.error("❌ Detecție eșuată într-unul din cadre. Asigurați-vă că fața este vizibilă clar, central și că iluminarea este uniformă.")
        else:
            st.warning("⚠️ Formular incomplet. Introduceți datele de identificare text și adăugați imagini pentru toate cele 4 ipostaze.")

# ==========================================
# MODULUL 2: PANOU MEDICI (IDENTIFICARE)
# ==========================================
else:
    st.markdown("<h1>⚕️ Panou Identificare Medicală (Unitate Primiri Urgențe)</h1>", unsafe_allow_html=True)
    
    parola = st.sidebar.text_input("🔑 Parolă Acces Medic:", type="password")
    
    if parola == "licenta2026":
        st.info("Sistemul de scanare la primiri urgențe este activ. Scanați fața pacientului adus.")
        scan = st.camera_input("Scanare facială instantă")
        
        if scan:
            v_scan, img_scan = proceseaza_vizual(scan)
            
            if v_scan is not None:
                col_img_scan, col_rezultate = st.columns([1, 2])
                with col_img_scan:
                    st.image(img_scan, caption="Puncte de interes identificate de AI")
                
                # Preluăm toate înregistrările din baza de date pentru comparare directă
                c.execute("SELECT nume, cnp, afectiuni, bio_deschisi, bio_inchisi, bio_stanga, bio_dreapta FROM pacienti")
                db_data = c.fetchall()
                
                if db_data:
                    rezultate = []
                    
                    for n, c_p, af, b_d, b_i, b_s, b_dr in db_data:
                        # Reconstituim vectorii numerici din datele binare salvate
                        v_db_d = np.frombuffer(b_d, dtype=np.float64)
                        v_db_i = np.frombuffer(b_i, dtype=np.float64)
                        v_db_s = np.frombuffer(b_s, dtype=np.float64)
                        v_db_dr = np.frombuffer(b_dr, dtype=np.float64)
                        
                        # Calculăm distanța geometrică dintre scanarea curentă și fiecare unghi salvat
                        dist_d = np.linalg.norm(v_scan - v_db_d)
                        dist_i = np.linalg.norm(v_scan - v_db_i)
                        dist_s = np.linalg.norm(v_scan - v_db_s)
                        dist_dr = np.linalg.norm(v_scan - v_db_dr)
                        
                        # Cea mai mică distanță reprezintă unghiul cel mai bine potrivit pentru acest pacient
                        dist_minima_pacient = min(dist_d, dist_i, dist_s, dist_dr)
                        
                        rezultate.append((dist_minima_pacient, n, c_p, af))
                    
                    # Sortăm rezultatele (cea mai mică distanță euclidiană ajunge pe prima poziție)
                    rezultate.sort()
                    cea_mai_buna = rezultate[0]
                    
                    with col_rezultate:
                        st.write(f"📊 **Scor minim de distanță calculat:** `{cea_mai_buna[0]:.4f}`")
                        
                        # Prag critic stabilit experimental pentru algoritmul bazat pe repere geometrice
                        PRAG = 0.45 
                        
                        if cea_mai_buna[0] < PRAG:
                            st.success(f"### ✅ PACIENT IDENTIFICAT CU SUCCES: {cea_mai_buna[1]}")
                            st.metric("Cod Numeric Personal (CNP)", cea_mai_buna[2])
                            st.error(f"⚠️ FIȘĂ MEDICALĂ DE URGENȚĂ: {cea_mai_buna[3]}")
                        else:
                            st.error("❌ Pacient necunoscut. Nu s-a putut găsi nicio potrivire sigură în baza de date.")
                else:
                    st.warning("Baza de date a spitalului nu conține pacienți înregistrați.")
            else:
                st.error("AI-ul nu poate distinge o figură umană în cadrul scanat. Ajustați poziția sau lumina.")
    elif parola == "":
        st.warning("Introduceți codul de securitate în meniul lateral pentru accesarea bazei de date.")
    else:
        st.error("Acces neautorizat. Cod incorect.")