from fpdf import FPDF
from difflib import ndiff
import os
import uuid

def generate_report(original_text, corrected_text, reports_dir):
    """Génère un rapport PDF avec les corrections."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Titre
    pdf.cell(0, 10, "Rapport de Relecture/Traduction", ln=True, align="C")

    # Texte original
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Texte Original :", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 5, original_text[:1000] + ("..." if len(original_text) > 1000 else ""))

    # Texte corrigé
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Texte Corrigé/Traduit :", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 5, corrected_text[:1000] + ("..." if len(corrected_text) > 1000 else ""))

    # Différences (si relecture)
    if original_text != corrected_text:
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Différences :", ln=True)
        pdf.set_font("Arial", size=10)
        diff = ndiff(original_text.splitlines(), corrected_text.splitlines())
        pdf.multi_cell(0, 5, "\n".join(list(diff)[:20]))  # Affiche les 20 premières différences

    # Sauvegarde le rapport
    report_filename = f"rapport_{uuid.uuid4().hex}.pdf"
    report_path = os.path.join(reports_dir, report_filename)
    os.makedirs(reports_dir, exist_ok=True)
    pdf.output(report_path)

    return report_path
