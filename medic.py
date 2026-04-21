import streamlit as st
import sqlite3
import numpy as np
from PIL import Image

st.set_page_config(page_title="Terminal Medic UPU", page_icon="⚕️")

def genereaza_amprenta(foto):
    try:
        img = Image.open(foto).convert('L').resize((64, 64))
        return np.array(img).flatten() / 255.0
    except: return None

conn = sqlite3.connect('baza_triaj_comuna.db', check_same_thread=False)
c = conn.cursor()

st.title("⚕️ Panou Identificare Urgențe")
parola = st.sidebar.text_input("Parolă Acces:", type="password")

if parola == "spital2024":
    scan = st.camera_input("Scanează pacient sosit la UPU")
    
    if scan:
        amprenta_noua = genereaza_amprenta(scan)
        c.execute("SELECT nume, cnp, afectiuni, biometrie FROM pacienti")
        db_pacienti = c.fetchall()
        
        if db_pacienti:
            potriviri = []
            for n, cp, af, bio in db_pacienti:
                v_db = np.frombuffer(bio, dtype=np.float64)
                distanta = np.linalg.norm(amprenta_noua - v_db)
                potriviri.append((distanta, n, cp, af))
            
            potriviri.sort()
            cel_mai_bun = potriviri[0]
            
            # Prag de siguranță
            if cel_mai_bun[0] < 15.0:
                st.success(f"### Pacient Identificat: {cel_mai_bun[1]}")
                st.warning(f"**CNP:** {cel_mai_bun[2]}")
                st.error(f"**ALERTE MEDICALE:** {cel_mai_bun[3]}")
            else:
                st.error("Pacientul nu a fost găsit în baza de date centrală.")
        else:
            st.info("Baza de date este goală. Niciun pacient înregistrat de acasă.")
else:
    st.warning("Introduceți parola pentru a activa scanerul biometric.")