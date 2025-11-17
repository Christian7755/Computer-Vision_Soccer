# src/visualization/visualize.py
"""
Visualisierungsfunktionen für Object Detection
"""
import supervision as sv
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
