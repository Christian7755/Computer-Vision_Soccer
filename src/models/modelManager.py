# src/models/model_manager.py
"""
Modell-Management für XAI-Projekt
"""
import numpy as np
from PIL import Image
from pathlib import Path
import cv2
from inference import get_model
from src.config import ROBOFLOW_API_KEY, MODEL_ID

class ModelManager:
    """
    Verwaltet das Object Detection Modell.
    Nutzt automatisches Caching vom inference SDK.
    """
    
    def __init__(self, model_id=MODEL_ID, api_key=ROBOFLOW_API_KEY):
        self.model_id = model_id
        self.api_key = api_key
        self.model = None
        
    def load_model(self):
        """
        Lädt das Modell. Verwendet automatisch Cache nach erstem Download.
        Kein Training nötig - Modell ist bereits auf Roboflow trainiert!
        """
        if self.model is None:
            print(f"⬇ Lade Modell: {self.model_id}")
            print("  (Verwendet Cache falls bereits heruntergeladen)")
            self.model = get_model(model_id=self.model_id, api_key=self.api_key)
            print("✓ Modell geladen")
        return self.model
    
    
    def infer(self, image):
        """
        Führt Inferenz auf einem Bild durch.
        
        Args:
            image: PIL.Image, numpy array, oder str/Path (Pfad zum Bild)
        """
        if self.model is None:
            self.load_model()
        
        # Fall 1: PIL Image → konvertiere zu numpy array
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Fall 2: Path Objekt → konvertiere zu String
        elif isinstance(image, Path):
            image = str(image)
        
        # Fall 3: String (Dateipfad) → prüfe Existenz
        elif isinstance(image, str):
            if not Path(image).exists():
                raise FileNotFoundError(f"Bild nicht gefunden: {image}")
        
        # Inferenz durchführen
        results = self.model.infer(image)[0]
        
        return results