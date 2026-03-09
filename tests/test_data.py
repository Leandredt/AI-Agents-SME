import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if __name__ == "__main__":
   from scripts.generate_data import extract_texts_from_pdfs
   texts = extract_texts_from_pdfs('data/pdfs/')
   print(f'{len(texts)} paragraphes extraits')
   print('--- Exemple 1 ---')
   print(texts[0] if texts else 'VIDE')
   print('--- Exemple 2 ---')
   print(texts[1] if len(texts) > 1 else 'VIDE')
   for i in range(min(5, len(texts))):
      print(f'--- Exemple {i+1} ---')
      print(texts[i])
   for i, text in enumerate(texts):
      assert isinstance(text, str) and len(text) > 0, f"Texte {i} est vide ou non valide"
