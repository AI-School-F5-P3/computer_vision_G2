# src/inference/detect_logos.py
from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2
import os
import logging
from datetime import timedelta

logger = logging.getLogger(__name__)

class LogoDetector:
    def __init__(self, model_path=None):
        """Initialize YOLO model"""
        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
        else:
            self.model = YOLO('yolov8n.pt')
            logger.warning("No model found, initialized with base YOLO model")

    def detect(self, image: Image.Image, conf_threshold: float = 0.25) -> tuple:
        """Detect logos in image"""
        # Tu código actual de detección en imágenes
        results = self.model(image, conf=conf_threshold)
        
        boxes = []
        scores = []
        labels = []
        
        for r in results:
            boxes.extend(r.boxes.xyxy.cpu().numpy())
            scores.extend(r.boxes.conf.cpu().numpy())
            labels.extend(r.boxes.cls.cpu().numpy())
            
        return np.array(boxes), np.array(scores), np.array(labels)

    def process_video(self, video_path: str, conf_threshold: float = 0.25) -> dict:
        """
        Process video and detect logos
        Args:
            video_path: Path to video file
            conf_threshold: Confidence threshold for detections
        Returns:
            dict: Statistics about logo appearances
        """
        # Abrir video
        cap = cv2.VideoCapture(video_path)
        
        # Obtener propiedades del video
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        # Variables para seguimiento
        frame_count = 0
        logo_frames = 0
        logo_appearances = []
        current_appearance = None
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convertir frame a RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            
            # Detectar logos
            boxes, scores, labels = self.detect(frame_pil, conf_threshold)
            
            # Procesar detecciones
            if len(boxes) > 0:
                if current_appearance is None:
                    current_appearance = {
                        'start_frame': frame_count,
                        'start_time': frame_count / fps,
                        'detections': len(boxes)
                    }
                logo_frames += 1
            else:
                if current_appearance is not None:
                    current_appearance['end_frame'] = frame_count - 1
                    current_appearance['end_time'] = (frame_count - 1) / fps
                    current_appearance['duration'] = (
                        current_appearance['end_frame'] - 
                        current_appearance['start_frame'] + 1
                    ) / fps
                    logo_appearances.append(current_appearance)
                    current_appearance = None
            
            frame_count += 1
            
            # Actualizar progreso cada 30 frames
            if frame_count % 30 == 0:
                progress = frame_count / total_frames * 100
                logger.info(f"Procesado: {progress:.1f}%")
        
        # Cerrar última aparición si existe
        if current_appearance is not None:
            current_appearance['end_frame'] = frame_count - 1
            current_appearance['end_time'] = (frame_count - 1) / fps
            current_appearance['duration'] = (
                current_appearance['end_frame'] - 
                current_appearance['start_frame'] + 1
            ) / fps
            logo_appearances.append(current_appearance)
        
        # Cerrar video
        cap.release()
        
        # Calcular estadísticas
        total_logo_time = sum(app['duration'] for app in logo_appearances)
        
        return {
            'total_frames': total_frames,
            'logo_frames': logo_frames,
            'video_duration': duration,
            'total_logo_time': total_logo_time,
            'logo_percentage': (total_logo_time / duration) * 100,
            'appearances': logo_appearances
        }

    def format_time(self, seconds: float) -> str:
        """Format seconds to HH:MM:SS"""
        return str(timedelta(seconds=int(seconds)))