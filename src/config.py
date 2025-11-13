# src/config.py
"""
Zentrale Konfiguration für das XAI-Projekt
"""
import os
from pathlib import Path
from dotenv import load_dotenv

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
DATASET_PATH = DATA_DIR / "football-ball-detection-4"
