# src/inference/detect_logos.py
import torch
import cv2
import numpy as np
from pathlib import Path
import logging
from torchvision.transforms import functional as F

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(model_path, num_classes=2):
    """
    Carga el modelo entrenado
    """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def detect_logos(image_path: str, model_path: str, confidence_threshold: float = 0.5):
    """
    Detecta logos en una imagen
    """
    # Cargar modelo
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = load_model(model_path)
    model.to(device)
    
    # Preprocesar imagen
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = F.to_tensor(image)
    
    # Inferencia
    with torch.no_grad():
        prediction = model([image_tensor.to(device)])
    
    # Procesar predicciones
    boxes = prediction[0]['boxes'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()
    labels = prediction[0]['labels'].cpu().numpy()
    
    # Filtrar por confianza
    mask = scores > confidence_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    labels = labels[mask]
    
    return boxes, scores, labels