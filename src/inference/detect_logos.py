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
        Args:
            image: PIL Image
            conf_threshold: Confidence threshold
        Returns:
            tuple: (boxes, scores, labels)
        """
        # Guardar dimensiones originales
        original_width, original_height = image.size
        
        # Preprocesar
        preprocessed_img = self.preprocess_image(image)
        preprocessed_width, preprocessed_height = preprocessed_img.size
        
        # Calcular ratios de escala
        width_ratio = original_width / preprocessed_width
        height_ratio = original_height / preprocessed_height
        
        # Realizar detección
        results = self.model(preprocessed_img, conf=conf_threshold)
        
        boxes = []
        scores = []
        labels = []
        
        for r in results:
            # Obtener cajas en coordenadas de imagen preprocesada
            detected_boxes = r.boxes.xyxy.cpu().numpy()
            
            # Ajustar cada caja a las dimensiones originales
            for box in detected_boxes:
                # Desescalar las coordenadas
                x1 = box[0] * width_ratio
                y1 = box[1] * height_ratio
                x2 = box[2] * width_ratio
                y2 = box[3] * height_ratio
                
                # Asegurar que las coordenadas están dentro de la imagen
                x1 = max(0, min(x1, original_width))
                x2 = max(0, min(x2, original_width))
                y1 = max(0, min(y1, original_height))
                y2 = max(0, min(y2, original_height))
                
                boxes.append([x1, y1, x2, y2])
                
            scores.extend(r.boxes.conf.cpu().numpy())
            labels.extend(r.boxes.cls.cpu().numpy())
        
        print(f"Original size: {image.size}")
        print(f"Preprocessed size: {preprocessed_img.size}")
        print(f"Scale ratios: width={width_ratio}, height={height_ratio}")
        print("Adjusted boxes:", boxes)
            
        return np.array(boxes), np.array(scores), np.array(labels)