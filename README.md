# Computer-Vision_Soccer
Using Explainability Methods for an Object Detection Modell from Roboflow.

Research Question: Which visual aspects influence the model's decision in ball detection?

Inspired by: https://github.com/roboflow/sports

Dataset: https://universe.roboflow.com/roboflow-jvuqo/football-ball-detection-rejhg/

Modell: YOLOv8


### Lime Analysis:
- Blurring different pixel areas to identify which changes influence the model's decision (model-agnostic).
- Inspired by: https://github.com/marcotcr/lime

### SHAP Analysis
- Theorie: SHAPley-Values: Determining which pixels contribute the most to the model's decision.

### Grad-CAM Analysis
- I had to train the YOLOv8 because the weights are only available in the roboflow premium version
- The required weights are in the folder models/football_gradcam/weights so the training can be skipped

# Setup und Execution
- python environment: >= 3.8 (and <3.13)
- Roboflow: API-Key needed
- for Configuration: see config.py
- UNZIP needed for the download of the data set
- Execution should be performed from the project root directory

Steps:
- run the first cells of setup_model.ipynb
  - Installing Requirements
  - Setup of the Model
  - Download of the Dataset
- Now you're free to run the other notebooks, from the project root