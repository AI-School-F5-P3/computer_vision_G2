#detect_logos.py

import torch
import cv2
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def detect_logos(image_path: str, model_path: str, confidence_threshold: float = 0.5):
    """
    Detecta logos en una imagen
    Args:
        image_path: Ruta a la imagen
        model_path: Ruta al modelo entrenado
        confidence_threshold: Umbral de confianza para detecciones
    Returns:
        List de detecciones (boxes, scores, labels)
    """
    # Cargar modelo
    model = torch.load(model_path)
    model.eval()
    
    # Preprocesar imagen
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
    
    # Inferencia
    with torch.no_grad():
        predictions = model([image_tensor])
    
    # Procesar predicciones
    boxes = predictions[0]['boxes'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()
    labels = predictions[0]['labels'].cpu().numpy()
    
    # Filtrar por confianza
    mask = scores > confidence_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    labels = labels[mask]
    
    return boxes, scores, labels