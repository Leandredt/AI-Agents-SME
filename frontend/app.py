# frontend/app.py
import streamlit as st

# Titre de l'application
st.title("📄 Agent IA pour Magazines")
st.markdown("""
Uploadez un fichier (PDF, Docx, Txt) pour le faire relire ou traduire par nos agents IA.
""")

# Upload de fichier
uploaded_file = st.file_uploader(
    "Choisissez un fichier",
    type=["pdf", "docx", "txt"],
    help="Formats supportés : PDF, Word, Texte"
)

if uploaded_file:
    # Affiche le nom du fichier
    st.write(f"Fichier uploadé : {uploaded_file.name}")

    # Choix de l'action
    action = st.radio(
        "Que souhaitez-vous faire ?",
        ["Relecture", "Traduction FR→EN"],
        help="Choisissez entre relecture stylistique ou traduction"
    )

    # Bouton pour lancer le traitement
    if st.button("Lancer le traitement"):
        st.write(f"Traitement en cours pour : {action}...")
        # Ici, tu appelleras tes fonctions d'extraction, relecture, etc.
        # Exemple (à compléter) :
        # text = extract_text(uploaded_file)
        # if action == "Relecture":
        #     corrected_text = relire(text)
        # else:
        #     corrected_text = traduire(text)
        # st.write("Résultat :", corrected_text)
        st.success("Traitement terminé ! (À implémenter)")
