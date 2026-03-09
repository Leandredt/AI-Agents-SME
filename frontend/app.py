import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Magazine Alpin — Agents IA", layout="wide")

MAGAZINE_TYPE_MAP = {
    "Luxe Alpin":  "alpine_luxury",
    "Décoration":  "decoration",
    "Gastronomie": "gastronomie",
    "Voyage":      "voyage",
}

MAX_CHARS = 5000


# ── Helpers API ───────────────────────────────────────────────────────────────

def call_health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def call_model_status():
    try:
        r = requests.get(f"{API_URL}/model/status", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def call_process(payload: dict):
    try:
        r = requests.post(f"{API_URL}/process", json=payload, timeout=120)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("Impossible de joindre le backend. Vérifiez que FastAPI tourne sur le port 8000.")
        return None
    except requests.exceptions.Timeout:
        st.error("Le serveur met trop de temps à répondre. Le modèle est peut-être surchargé, réessayez.")
        return None
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code
        try:
            detail = e.response.json().get("detail", e.response.text)
        except Exception:
            detail = e.response.text
        if status == 422:
            st.error(f"Texte invalide : {detail}")
        else:
            st.error(f"Erreur {status} : {detail}")
        return None


def call_upload(file_obj, action: str, charte: str, translation_engine: str = "mistral", magazine_type: str = "alpine_luxury"):
    try:
        r = requests.post(
            f"{API_URL}/upload",
            params={"action": action, "charte": charte or None, "translation_engine": translation_engine, "magazine_type": magazine_type},
            files={"file": (file_obj.name, file_obj.getvalue(), file_obj.type)},
            timeout=180,
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("Impossible de joindre le backend. Vérifiez que FastAPI tourne sur le port 8000.")
        return None
    except requests.exceptions.Timeout:
        st.error("Délai dépassé. Le document est peut-être trop volumineux.")
        return None
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code
        try:
            detail = e.response.json().get("detail", e.response.text)
        except Exception:
            detail = e.response.text
        if status == 422:
            st.error(f"Fichier invalide : {detail}")
        else:
            st.error(f"Erreur {status} : {detail}")
        return None


def show_result(label: str, text: str, key: str):
    st.subheader(label)
    st.text_area("", value=text, height=220, key=key)
    with st.expander("Copier le résultat"):
        st.code(text, language="")


def char_counter(text: str):
    count = len(text)
    color = "red" if count > MAX_CHARS else "gray"
    st.markdown(f"<small style='color:{color}'>{count} / {MAX_CHARS} caractères</small>", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Tableau de bord")

    health = call_health()
    if health and health.get("model_loaded"):
        st.success("Backend en ligne")
    elif health:
        st.warning("Backend en ligne — modèle non chargé")
    else:
        st.error("Backend hors ligne")

    model_info = call_model_status()
    if model_info:
        st.markdown(f"**Modèle :** `{model_info.get('model_name', 'inconnu')}`")
        st.markdown(f"**LoRA :** {'oui' if model_info.get('lora_loaded') else 'non'}")
    else:
        st.markdown("_Informations modèle indisponibles_")

    st.divider()

    magazine_label = st.selectbox("Type de magazine", list(MAGAZINE_TYPE_MAP.keys()), index=0)
    magazine_type = MAGAZINE_TYPE_MAP[magazine_label]


# ── Header ────────────────────────────────────────────────────────────────────

st.title("Magazine Alpin — Agents IA")
st.caption("Relecture et traduction pour un magazine de luxe alpin philippin")

tab1, tab2 = st.tabs(["Relecture Française", "Traduction FR → EN"])


# ── Tab 1 : Relecture FR ──────────────────────────────────────────────────────

with tab1:
    st.subheader("Agent Relecture Française")
    st.markdown("Corrige un texte en français selon la charte éditoriale du magazine.")

    src_fr = st.radio("Source du texte", ["Saisie directe", "Fichier (PDF / DOCX / TXT)"], horizontal=True, key="src_fr")

    text_fr, upload_fr = "", None
    if src_fr == "Saisie directe":
        text_fr = st.text_area("Texte à relire", height=220, key="text_fr")
        char_counter(text_fr)
    else:
        upload_fr = st.file_uploader("Uploader un fichier", type=["pdf", "docx", "txt"], key="upload_fr")

    charte_fr = st.text_area("Charte éditoriale (optionnel)", height=80,
                              placeholder="Ex : ton élégant, éviter les anglicismes, vouvoiement...", key="charte_fr")

    if st.button("Relire en français", type="primary", key="btn_fr"):
        if src_fr == "Saisie directe":
            if not text_fr.strip():
                st.warning("Veuillez saisir un texte.")
            elif len(text_fr) > MAX_CHARS:
                st.error(f"Texte trop long ({len(text_fr)} caractères). Maximum : {MAX_CHARS}.")
            else:
                with st.spinner("Relecture en cours..."):
                    resp = call_process({"text": text_fr, "action": "relecture_fr", "charte": charte_fr or None, "magazine_type": magazine_type})
                if resp:
                    show_result("Texte corrigé", resp["result"], "result_fr")
        else:
            if not upload_fr:
                st.warning("Veuillez uploader un fichier.")
            else:
                with st.spinner("Relecture en cours..."):
                    resp = call_upload(upload_fr, "relecture_fr", charte_fr, magazine_type=magazine_type)
                if resp:
                    show_result("Texte corrigé", resp["result"], "result_fr_file")


# ── Tab 2 : Traduction FR → EN ────────────────────────────────────────────────

with tab2:
    st.subheader("Agent Traduction FR → EN")
    st.markdown("Traduit un texte du français vers l'anglais, avec relecture optionnelle.")

    src_en = st.radio("Source du texte", ["Saisie directe", "Fichier (PDF / DOCX / TXT)"], horizontal=True, key="src_en")

    text_en, upload_en = "", None
    if src_en == "Saisie directe":
        text_en = st.text_area("Texte à traduire", height=220, key="text_en")
        char_counter(text_en)
    else:
        upload_en = st.file_uploader("Uploader un fichier", type=["pdf", "docx", "txt"], key="upload_en")

    charte_en = st.text_area("Charte éditoriale (optionnel)", height=80,
                              placeholder="Ex : elegant tone, luxury alpine vocabulary...", key="charte_en")

    col1, col2 = st.columns(2)
    with col1:
        engine_label = st.radio("Moteur de traduction", ["Mistral (local)", "DeepL", "Comparer les deux"], key="engine")
        engine = {"Mistral (local)": "mistral", "DeepL": "deepl", "Comparer les deux": "compare"}[engine_label]
    with col2:
        st.markdown("&nbsp;", unsafe_allow_html=True)
        proofread_after = st.checkbox("Relecture EN après traduction", key="proofread_after")

    if st.button("Traduire", type="primary", key="btn_en"):
        if src_en == "Saisie directe" and not text_en.strip():
            st.warning("Veuillez saisir un texte.")
        elif src_en == "Saisie directe" and len(text_en) > MAX_CHARS:
            st.error(f"Texte trop long ({len(text_en)} caractères). Maximum : {MAX_CHARS}.")
        elif src_en != "Saisie directe" and not upload_en:
            st.warning("Veuillez uploader un fichier.")
        else:
            with st.spinner("Traduction en cours..."):
                if src_en == "Saisie directe":
                    resp = call_process({"text": text_en, "action": "traduction", "charte": charte_en or None,
                                         "translation_engine": engine, "magazine_type": magazine_type})
                else:
                    resp = call_upload(upload_en, "traduction", charte_en, translation_engine=engine, magazine_type=magazine_type)

            if resp:
                result = resp["result"]

                if engine == "compare":
                    st.subheader("Comparaison des traductions")
                    c1, c2 = st.columns(2)
                    mistral_text = result.get("mistral", "") if isinstance(result, dict) else ""
                    deepl_text = result.get("deepl", "") if isinstance(result, dict) else ""
                    with c1:
                        st.markdown("**Mistral (local fine-tuné)**")
                        st.text_area("", value=mistral_text, height=220, key="out_mistral")
                        with st.expander("Copier — Mistral"):
                            st.code(mistral_text, language="")
                    with c2:
                        st.markdown("**DeepL**")
                        st.text_area("", value=deepl_text, height=220, key="out_deepl")
                        with st.expander("Copier — DeepL"):
                            st.code(deepl_text, language="")

                    if proofread_after:
                        st.subheader("Relecture EN des deux traductions")
                        for label, trad in [("Mistral", mistral_text), ("DeepL", deepl_text)]:
                            with st.spinner(f"Relecture {label}..."):
                                r_proof = call_process({"text": trad, "action": "relecture_en",
                                                        "charte": charte_en or None, "magazine_type": magazine_type})
                            if r_proof:
                                show_result(f"Relu — {label}", r_proof["result"], f"proof_{label}")
                else:
                    show_result("Traduction", result, "out_translation")
                    if proofread_after:
                        with st.spinner("Relecture EN en cours..."):
                            r_proof = call_process({"text": result, "action": "relecture_en",
                                                    "charte": charte_en or None, "magazine_type": magazine_type})
                        if r_proof:
                            show_result("Texte relu en anglais", r_proof["result"], "out_proof")
