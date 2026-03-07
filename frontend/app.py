import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Magazine Alpin — Agents IA", layout="wide")
st.title("Magazine Alpin — Agents IA")
st.caption("Relecture et traduction pour un magazine de luxe alpin philippin")


def call_process(payload: dict):
    try:
        r = requests.post(f"{API_URL}/process", json=payload, timeout=120)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("Impossible de joindre le backend. Vérifiez que FastAPI tourne sur le port 8000.")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"Erreur serveur : {e.response.text}")
        return None


def call_upload(file_obj, action: str, charte: str, translation_engine: str = "mistral"):
    try:
        r = requests.post(
            f"{API_URL}/upload",
            params={"action": action, "charte": charte or None, "translation_engine": translation_engine},
            files={"file": (file_obj.name, file_obj.getvalue(), file_obj.type)},
            timeout=180
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("Impossible de joindre le backend. Vérifiez que FastAPI tourne sur le port 8000.")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"Erreur serveur : {e.response.text}")
        return None


# ── Tabs ────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["Relecture Française", "Traduction FR → EN"])


# ── Tab 1 : Relecture FR ────────────────────────────────────────────────────
with tab1:
    st.subheader("Agent Relecture Française")
    st.markdown("Corrige un texte en français selon la charte éditoriale du magazine.")

    src_fr = st.radio(
        "Source du texte", ["Saisie directe", "Fichier (PDF / DOCX / TXT)"],
        horizontal=True, key="src_fr"
    )

    text_fr = ""
    upload_fr = None
    if src_fr == "Saisie directe":
        text_fr = st.text_area("Texte à relire", height=220, key="text_fr")
    else:
        upload_fr = st.file_uploader("Uploader un fichier", type=["pdf", "docx", "txt"], key="upload_fr")

    charte_fr = st.text_area(
        "Charte éditoriale (optionnel)", height=80,
        placeholder="Ex : ton élégant, éviter les anglicismes, vouvoiement...",
        key="charte_fr"
    )

    if st.button("Relire en français", type="primary", key="btn_fr"):
        if src_fr == "Saisie directe":
            if not text_fr.strip():
                st.warning("Veuillez saisir un texte.")
            else:
                with st.spinner("Relecture en cours..."):
                    resp = call_process({"text": text_fr, "action": "relecture_fr", "charte": charte_fr or None})
                if resp:
                    st.subheader("Texte corrigé")
                    st.text_area("", value=resp["result"], height=220, key="result_fr")
        else:
            if not upload_fr:
                st.warning("Veuillez uploader un fichier.")
            else:
                with st.spinner("Relecture en cours..."):
                    resp = call_upload(upload_fr, "relecture_fr", charte_fr)
                if resp:
                    st.subheader("Texte corrigé")
                    st.text_area("", value=resp["result"], height=220, key="result_fr_file")
                    if "report_url" in resp:
                        st.info(f"Rapport PDF généré : `{resp['report_url']}`")


# ── Tab 2 : Traduction FR → EN ──────────────────────────────────────────────
with tab2:
    st.subheader("Agent Traduction FR → EN")
    st.markdown("Traduit un texte du français vers l'anglais, avec relecture optionnelle.")

    src_en = st.radio(
        "Source du texte", ["Saisie directe", "Fichier (PDF / DOCX / TXT)"],
        horizontal=True, key="src_en"
    )

    text_en = ""
    upload_en = None
    if src_en == "Saisie directe":
        text_en = st.text_area("Texte à traduire", height=220, key="text_en")
    else:
        upload_en = st.file_uploader("Uploader un fichier", type=["pdf", "docx", "txt"], key="upload_en")

    charte_en = st.text_area(
        "Charte éditoriale (optionnel)", height=80,
        placeholder="Ex : elegant tone, luxury alpine vocabulary, avoid colloquialisms...",
        key="charte_en"
    )

    col1, col2 = st.columns(2)
    with col1:
        engine_label = st.radio(
            "Moteur de traduction",
            ["Mistral (local)", "DeepL", "Comparer les deux"],
            key="engine"
        )
        engine_map = {"Mistral (local)": "mistral", "DeepL": "deepl", "Comparer les deux": "compare"}
        engine = engine_map[engine_label]
    with col2:
        st.markdown("&nbsp;", unsafe_allow_html=True)
        proofread_after = st.checkbox("Relecture EN après traduction", key="proofread_after")

    if st.button("Traduire", type="primary", key="btn_en"):
        # Validation input
        if src_en == "Saisie directe" and not text_en.strip():
            st.warning("Veuillez saisir un texte.")
        elif src_en != "Saisie directe" and not upload_en:
            st.warning("Veuillez uploader un fichier.")
        else:
            with st.spinner("Traduction en cours..."):
                if src_en == "Saisie directe":
                    resp = call_process({
                        "text": text_en,
                        "action": "traduction",
                        "charte": charte_en or None,
                        "translation_engine": engine
                    })
                else:
                    resp = call_upload(upload_en, "traduction", charte_en, translation_engine=engine)

            if resp:
                result = resp["result"]

                # ── Mode comparaison ──────────────────────────────────────
                if engine == "compare":
                    st.subheader("Comparaison des traductions")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**Mistral (local fine-tuné)**")
                        mistral_text = result.get("mistral", "") if isinstance(result, dict) else ""
                        st.text_area("", value=mistral_text, height=220, key="out_mistral")
                    with c2:
                        st.markdown("**DeepL**")
                        deepl_text = result.get("deepl", "") if isinstance(result, dict) else ""
                        st.text_area("", value=deepl_text, height=220, key="out_deepl")

                    if proofread_after:
                        st.subheader("Relecture EN des deux traductions")
                        for label, trad in [("Mistral", mistral_text), ("DeepL", deepl_text)]:
                            with st.spinner(f"Relecture de la traduction {label}..."):
                                r_proof = call_process({
                                    "text": trad,
                                    "action": "relecture_en",
                                    "charte": charte_en or None
                                })
                            if r_proof:
                                st.markdown(f"**Relu — {label}**")
                                st.text_area("", value=r_proof["result"], height=180, key=f"proof_{label}")

                # ── Mode simple (mistral ou deepl) ────────────────────────
                else:
                    st.subheader("Traduction")
                    st.text_area("", value=result, height=220, key="out_translation")

                    if proofread_after:
                        with st.spinner("Relecture EN en cours..."):
                            r_proof = call_process({
                                "text": result,
                                "action": "relecture_en",
                                "charte": charte_en or None
                            })
                        if r_proof:
                            st.subheader("Texte relu en anglais")
                            st.text_area("", value=r_proof["result"], height=220, key="out_proof")

                if "report_url" in resp:
                    st.info(f"Rapport PDF généré : `{resp['report_url']}`")
