# Etude de Cas — Analyseur de Plans Techniques Industriels

POC d'une interface Streamlit qui lit des fichiers PDF de plans techniques industriels
(pièces usinées), extrait les informations pertinentes du texte et des schémas, déduit
les dimensions et estime le coût de fabrication.

---

## Architecture

```
app.py                  # Application Streamlit principale
modules/
  pdf_processor.py      # Extraction de texte et d'images via PyMuPDF
  ai_analyzer.py        # Analyse GPT-4o (vision) → dimensions / matière / finitions
  cost_calculator.py    # Modèle de calcul de coût de fabrication
requirements.txt
.env.example
```

## Installation

```bash
# 1. Cloner le dépôt
git clone https://github.com/antoineRgithub/Etude-de-cas.git
cd Etude-de-cas

# 2. Créer et activer un environnement virtuel
python -m venv .venv
source .venv/bin/activate       # Windows : .venv\Scripts\activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Configurer la clé API OpenAI
cp .env.example .env
# Editer .env et renseigner OPENAI_API_KEY=sk-...
```

## Utilisation

```bash
streamlit run app.py
```

Ouvrir http://localhost:8501 dans un navigateur.

### Workflow

1. **Déposer un PDF** — plan technique d'une pièce usinée (tour, fraisage, etc.)
2. **Onglet "Pages"** — visualisation des pages du plan
3. **Onglet "Extracted Text"** — texte brut extrait
4. **Onglet "AI Analysis"** — cliquer *Run AI Analysis* pour envoyer le document à GPT-4o
   - Extracts: nom de la pièce, matière, dimensions + tolérances, état de surface, traitement thermique
5. **Onglet "Cost Estimate"** — décomposition automatique du coût unitaire et total

### Paramètres ajustables (barre latérale)

| Paramètre | Description |
|-----------|-------------|
| OpenAI API Key | Clé API (jamais stockée) |
| CNC Hourly Rate | Taux horaire usinage (€/h) |
| Setup Time | Temps de préparation (h) |
| Overhead Rate | Frais généraux (%) |
| Profit Margin | Marge bénéficiaire (%) |
| Quantity | Quantité à produire |
| Material Override | Forcer la matière (optionnel) |

## Notes

- Le modèle de coût est intentionnellement transparent et conservateur.
- Les estimations sont **indicatives** — ne pas utiliser pour des devis définitifs.
- Fonctionne aussi sur des scans (le modèle s'appuie alors uniquement sur les images).
