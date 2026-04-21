import streamlit as st
import sqlite3
import numpy as np
from PIL import Image

st.set_page_config(page_title="Înregistrare Pacient", page_icon="📝")

# Funcție procesare imagine (Pixel-based pentru stabilitate cloud)
def genereaza_amprenta(foto):
    try:
        img = Image.open(foto).convert('L').resize((64, 64))
        return np.array(img).flatten() / 255.0
    except: return None

# Conectare la baza de date comună
conn = sqlite3.connect('baza_triaj_comuna.db', check_same_thread=False)
c = conn.cursor()
c.execute('CREATE TABLE IF NOT EXISTS pacienti (nume TEXT, cnp TEXT, afectiuni TEXT, biometrie BLOB)')
conn.commit()

st.title("🏥 Înregistrare Pacient (De Acasă)")
st.info("Datele tale vor fi trimise securizat către unitatea de urgențe.")

with st.form("form_acasa"):
    nume = st.text_input("Nume Complet:")
    cnp = st.text_input("CNP:")
    afectiuni = st.multiselect("Afecțiuni cunoscute:", ["Diabet", "Inimă", "Alergii", "Astm"])
    obs = st.text_area("Alte detalii importante:")
    
    st.write("Încarcă o poză clară cu fața ta pentru identificare rapidă în caz de urgență:")
    foto = st.camera_input("Captură foto")
    
    submit = st.form_submit_button("Trimite Datele spre Spital")

if submit:
    if nume and cnp and foto:
        amprenta = genereaza_amprenta(foto)
        istoric = ", ".join(afectiuni) + " | " + obs
        c.execute("INSERT INTO pacienti VALUES (?,?,?,?)", (nume, cnp, istoric, amprenta.tobytes()))
        conn.commit()
        st.success("✅ Datele au fost salvate. În caz de urgență, personalul medical te va identifica prin scanare facială.")
    else:
        st.error("Vă rugăm să completați toate câmpurile și să faceți poza.")