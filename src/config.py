# src/config.py
"""
Zentrale Konfiguration für das XAI-Projekt
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Holt den absoluten Pfad zur config.py-Datei
FILE = Path(__file__).resolve()

# Versucht, die Projektwurzel zu finden (2 Ebenen hoch ist am häufigsten)
try:
    PROJECT_ROOT = FILE.parents[1] 
except IndexError:
    # Falls die Umgebung dies falsch interpretiert (Notebooks), geht es 3 Ebenen hoch
    PROJECT_ROOT = FILE.parents[2] 
    
# Fügt das Projekt-Root-Verzeichnis zum Python-Suchpfad hinzu (Import-Fix)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
# -----------------------------------------------------------

# Lade Environment Variables
load_dotenv()

# Projekt-Pfade
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Erstelle Ordner falls nicht vorhanden
for directory in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Roboflow API (aus deiner .env Datei)
ROBOFLOW_API_KEY = os.getenv('ROBOFLOW_KEY')
ROBOFLOW_USERNAME = os.getenv('ROBOFLOW_USERNAME')

# Modell-Konfiguration
MODEL_ID = "football-ball-detection-rejhg/4"
DATASET_PATH = DATA_DIR

# Warnungen für nicht installierte Modelle unterdrücken (optional)
CORE_MODEL_SAM3_ENABLED = os.getenv('CORE_MODEL_SAM3_ENABLED', 'False')
CORE_MODEL_GROUNDINGDINO_ENABLED = os.getenv('CORE_MODEL_GROUNDINGDINO_ENABLED', 'False')