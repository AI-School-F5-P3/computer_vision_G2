# src/inference/detect_logos.py
from ultralytics import YOLO
import numpy as np
from PIL import Image
import os
import logging

logger = logging.getLogger(__name__)

class LogoDetector:
    def __init__(self, model_path=None):
        """Initialize YOLO model"""
        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
        else:
            # If no model exists, create a new one from YOLO base model
            self.model = YOLO('yolov8n.pt')
            logger.warning("No model found, initialized with base YOLO model")

    def preprocess_image(self, image):
        """
        Preprocesa la imagen de la misma manera que las imágenes de entrenamiento/test
        """
        # Convertir a RGB
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Redimensionar manteniendo el aspect ratio
        width, height = image.size
        new_size = (640, 640)  # mismo tamaño que en entrenamiento
        
        # Calcular el nuevo tamaño manteniendo el aspect ratio
        ratio = min(new_size[0] / width, new_size[1] / height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        
        # Redimensionar
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Crear una nueva imagen con padding negro
        new_image = Image.new('RGB', new_size, (0, 0, 0))
        # Pegar la imagen redimensionada en el centro
        paste_x = (new_size[0] - new_width) // 2
        paste_y = (new_size[1] - new_height) // 2
        new_image.paste(image, (paste_x, paste_y))
        
        return new_image

    def detect(self, image: Image.Image, conf_threshold: float = 0.25) -> tuple:
        """
        Detect logos in image
        """
        # Guardar dimensiones originales
        original_width, original_height = image.size
        target_size = (640, 640)
        
        # Calcular el ratio y padding
        ratio = min(target_size[0] / original_width, target_size[1] / original_height)
        new_width = int(original_width * ratio)
        new_height = int(original_height * ratio)
        
        # Calcular padding
        pad_x = (target_size[0] - new_width) // 2
        pad_y = (target_size[1] - new_height) // 2
        
        # Preprocesar imagen
        preprocessed_img = self.preprocess_image(image)
        
        # Realizar detección
        results = self.model(preprocessed_img, conf=conf_threshold)
        
        boxes = []
        scores = []
        labels = []
        
        for r in results:
            detected_boxes = r.boxes.xyxy.cpu().numpy()
            
            for box in detected_boxes:
                # Restar el padding
                x1 = box[0] - pad_x
                y1 = box[1] - pad_y
                x2 = box[2] - pad_x
                y2 = box[3] - pad_y
                
                # Convertir a coordenadas de la imagen original
                x1 = max(0, min(x1 / ratio, original_width))
                x2 = max(0, min(x2 / ratio, original_width))
                y1 = max(0, min(y1 / ratio, original_height))
                y2 = max(0, min(y2 / ratio, original_height))
                
                boxes.append([x1, y1, x2, y2])
            
            scores.extend(r.boxes.conf.cpu().numpy())
            labels.extend(r.boxes.cls.cpu().numpy())
        
        # Debug info
        print(f"Original size: {original_width}x{original_height}")
        print(f"Padding: x={pad_x}, y={pad_y}")
        print(f"Scale ratio: {ratio}")
        print("Boxes:", boxes)
        
        return np.array(boxes), np.array(scores), np.array(labels)