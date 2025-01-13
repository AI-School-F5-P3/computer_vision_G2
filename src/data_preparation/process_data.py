#process_data.py

import cv2
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_images(input_dir: str, output_dir: str, target_size=(640, 640)):
    """
    Procesa las imágenes del dataset
    Args:
        input_dir: Directorio con imágenes originales
        output_dir: Directorio donde guardar imágenes procesadas
        target_size: Tamaño objetivo para las imágenes
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    for img_path in Path(input_dir).glob("**/*.jpg"):
        try:
            # Leer imagen
            img = cv2.imread(str(img_path))
            if img is None:
                continue
                
            # Redimensionar
            img_resized = cv2.resize(img, target_size)
            
            # Guardar imagen procesada
            output_path = Path(output_dir) / img_path.name
            cv2.imwrite(str(output_path), img_resized)
            
        except Exception as e:
            logger.error(f"Error procesando {img_path}: {str(e)}")
            continue