# src/visualization/visualize.py
"""
Visualisierungsfunktionen für Object Detection
"""
import supervision as sv
import random
import matplotlib.pyplot as plt
from io import BytesIO
import requests
from PIL import Image

def load_image_from_url(url: str):
    """Lädt ein Bild von einer URL"""
    response = requests.get(url)
    response.raise_for_status()
    image = Image.open(BytesIO(response.content))
    return image

def visualize_detections(
    image, 
    results, 
    show_confidence=True,
    labels_on_image=True,
    box_thickness=1,
    text_scale=1.5,
    text_thickness=2,
    text_padding=10,
    smart_position=True
):
    """
    Visualisiert Object Detection Ergebnisse mit Labels
    """
    # Detections erstellen
    detections = sv.Detections.from_inference(results)
    
    # Labels erstellen
    if show_confidence:
        labels = [
            f"{class_name} {confidence:.2f}"
            for class_name, confidence
            in zip(detections['class_name'], detections.confidence)
        ]
    else:
        labels = [f"{class_name}" for class_name in detections['class_name']]
    
    # Annotieren der Box im Bild
    bounding_box_annotator = sv.BoxAnnotator(thickness=box_thickness)

    annotated_image = bounding_box_annotator.annotate(
        scene=image, 
        detections=detections
    )

    if labels_on_image:
        label_annotator = sv.LabelAnnotator(
            text_scale=text_scale,
            text_thickness=text_thickness,
            text_padding=text_padding,
            smart_position=smart_position
        )
        
        annotated_image = label_annotator.annotate(
            scene=annotated_image, 
            detections=detections, 
            labels=labels
        )
        return annotated_image
    else:
        return annotated_image, labels
    
def show_grid(all_images_list, model_manager, fixed_indices=None, rows=3, cols=3):
    """
    Erstellt ein Grid von Bildern mit Inference-Ergebnissen.
    Kombiniert manuell gewählte Indizes mit zufälligen Bildern.
    """
    if fixed_indices is None:
        fixed_indices = []

    num_slots = rows * cols
    
    # 1. Auswahl der Indizes (Mix aus Manuell + Zufall)
    # Erst die manuellen nehmen (soweit im gültigen Bereich)
    current_indices = [i for i in fixed_indices if i < len(all_images_list)]
    
    # Den Rest zufällig auffüllen, ohne Dopplungen
    needed = num_slots - len(current_indices)
    if needed > 0:
        # Wähle aus allen möglichen Indizes, außer denen, die wir schon haben
        available_pool = list(set(range(len(all_images_list))) - set(current_indices))
        
        if len(available_pool) >= needed:
            random_picks = random.sample(available_pool, needed)
            current_indices.extend(random_picks)
        else:
            print("Warnung: Nicht genügend Bilder im Pool, um das Grid zu füllen.")
            current_indices.extend(available_pool) # Nimm was da ist
    
    # Indizes begrenzen (falls man mehr manuelle eingetragen hat als Plätze)
    current_indices = current_indices[:num_slots]

    print(f"✓ Verwendete Indizes: {current_indices}")

    # 2. Plotten
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    # Falls Grid 1x1 ist, ist axes kein Array, daher ensure iterable
    if rows * cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, idx in enumerate(current_indices):
        if i >= len(axes): break # Safety Check
        
        ax = axes[i]
        img_path = all_images_list[idx]
        
        try:
            # Laden & Inference
            img = Image.open(img_path)
            results = model_manager.infer(img)
            
            # Nutzung der internen visualize_detections Funktion
            annotated = visualize_detections(
                img, 
                results, 
                show_confidence=True, 
                labels_on_image=True,
                text_scale=1.0 # Etwas kleiner für das Grid
            )
            
            ax.imshow(annotated)
            
            # Titel Logik
            is_manual = idx in fixed_indices
            status = "★ FIX" if is_manual else "Zufall"
            
            # WICHTIG: Der Index steht fett im Titel
            ax.set_title(f"Index: {idx} ({status})\n{img_path.name[-15:]}", fontsize=11, fontweight='bold')
            ax.axis('off')
            
        except Exception as e:
            ax.text(0.5, 0.5, f"Fehler:\n{e}", ha='center', color='red')
            ax.axis('off')
            print(f"Fehler bei Index {idx}: {e}")

    plt.tight_layout()
    plt.show()
